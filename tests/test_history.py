import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent


class HistoryPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.history_path = Path(self.temp_dir.name) / "history.json"
        self.history_patch = patch.object(agent, "HISTORY_PATH", self.history_path)
        self.history_patch.start()
        self.addCleanup(self.history_patch.stop)

    def test_missing_history_returns_system_message(self):
        self.assertEqual(
            agent.load_history(),
            [{"role": "system", "content": agent.SYSTEM_PROMPT}],
        )
        self.assertFalse(self.history_path.exists())

    def test_save_preserves_unicode_and_tool_calls(self):
        messages = [
            {"role": "system", "content": agent.SYSTEM_PROMPT},
            {"role": "user", "content": "请检查文件\n并用中文总结。"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                    },
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "目录：/tmp\n"},
            {"role": "tool", "tool_call_id": "call-2", "content": "文件：你好.txt\n"},
            {"role": "assistant", "content": "检查完成。\n一切正常。"},
        ]

        agent.save_history(messages)

        saved_text = self.history_path.read_text(encoding="utf-8")
        self.assertIn("请检查文件", saved_text)
        self.assertNotIn("\\u8bf7", saved_text)
        self.assertEqual(json.loads(saved_text), messages)
        self.assertFalse(self.history_path.with_name("history.json.tmp").exists())

    def test_load_uses_current_system_prompt(self):
        stored_messages = [
            {"role": "system", "content": "旧提示词"},
            {"role": "user", "content": "之前的问题"},
            {"role": "assistant", "content": "之前的回答"},
        ]
        self.history_path.write_text(json.dumps(stored_messages), encoding="utf-8")

        loaded = agent.load_history()

        self.assertEqual(loaded[0], {"role": "system", "content": agent.SYSTEM_PROMPT})
        self.assertEqual(loaded[1:], stored_messages[1:])
        self.assertEqual(json.loads(self.history_path.read_text(encoding="utf-8")), stored_messages)

    def test_load_rejects_invalid_utf8(self):
        invalid_bytes = b"\xff\xfe"
        self.history_path.write_bytes(invalid_bytes)

        with self.assertRaises((ValueError, RuntimeError)):
            agent.load_history()

        self.assertEqual(self.history_path.read_bytes(), invalid_bytes)

    def test_load_rejects_malformed_history(self):
        invalid_files = [
            "{broken json",
            "{}",
            json.dumps([{"role": "user", "content": "missing system"}]),
            json.dumps([{"role": "system", "content": "prompt"}, {"role": "tool", "content": "no id"}]),
            json.dumps([{"role": "system", "content": "prompt"}, {"role": "assistant", "content": 42}]),
        ]

        for invalid_content in invalid_files:
            with self.subTest(content=invalid_content):
                self.history_path.write_text(invalid_content, encoding="utf-8")
                with self.assertRaises((ValueError, RuntimeError)):
                    agent.load_history()
                self.assertEqual(self.history_path.read_text(encoding="utf-8"), invalid_content)

    def test_save_write_failure_preserves_history(self):
        original = '[{"role":"system","content":"old"}]'
        self.history_path.write_text(original, encoding="utf-8")
        messages = [{"role": "system", "content": agent.SYSTEM_PROMPT}]

        with patch.object(Path, "write_text", side_effect=OSError("write failed")):
            with self.assertRaises(OSError):
                agent.save_history(messages)

        self.assertEqual(self.history_path.read_text(encoding="utf-8"), original)

    def test_save_replace_failure_preserves_history(self):
        original = '[{"role":"system","content":"old"}]'
        self.history_path.write_text(original, encoding="utf-8")
        messages = [{"role": "system", "content": agent.SYSTEM_PROMPT}]

        with patch.object(Path, "replace", side_effect=OSError("replace failed")):
            with self.assertRaises(OSError):
                agent.save_history(messages)

        self.assertEqual(self.history_path.read_text(encoding="utf-8"), original)
        self.assertTrue(self.history_path.with_name("history.json.tmp").exists())


class MainLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)
        self.history_path = self.directory / "history.json"
        self.history_patch = patch.object(agent, "HISTORY_PATH", self.history_path)
        self.history_patch.start()
        self.addCleanup(self.history_patch.stop)

    @staticmethod
    def _response(message):
        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": message}]}

        return Response()

    def test_tool_results_are_sent_and_saved(self):
        first_message = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call-a",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"first"}'},
                },
                {
                    "id": "call-b",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"second"}'},
                },
            ],
        }
        final_message = {"content": "已经完成。\n结果正常。"}
        responses = [self._response(first_message), self._response(final_message)]
        inputs = iter(["开始任务", "", "quit"])
        request_snapshots = []

        def mock_post(*args, **kwargs):
            request_snapshots.append(copy.deepcopy(kwargs["json"]["messages"]))
            return responses[len(request_snapshots) - 1]

        with (
            patch("builtins.input", side_effect=lambda _: next(inputs)),
            patch.object(agent.requests, "post", side_effect=mock_post) as post,
            patch.object(agent, "run_bash", side_effect=["first output", "second output"]),
            patch("builtins.print") as printed,
            patch.object(agent, "Path", return_value=SimpleNamespace(read_text=lambda: "test-key")),
        ):
            agent.main()

        self.assertEqual(post.call_count, 2)
        second_request_messages = request_snapshots[1]
        self.assertEqual([message["role"] for message in second_request_messages], [
            "system", "user", "assistant", "tool", "tool",
        ])
        self.assertEqual(
            [message.get("tool_call_id") for message in second_request_messages if message["role"] == "tool"],
            ["call-a", "call-b"],
        )
        saved_messages = json.loads(self.history_path.read_text(encoding="utf-8"))
        self.assertEqual(saved_messages[-1]["content"], final_message["content"])
        self.assertEqual(saved_messages[2]["tool_calls"], first_message["tool_calls"])
        self.assertEqual(
            [(message["tool_call_id"], message["content"]) for message in saved_messages if message["role"] == "tool"],
            [("call-a", "first output"), ("call-b", "second output")],
        )
        printed.assert_called_once_with("Agent:", final_message["content"])

    def test_new_run_includes_saved_conversation(self):
        snapshots = []
        responses = [
            self._response({"content": "第一轮完成"}),
            self._response({"content": "接着处理"}),
        ]

        def mock_post(*args, **kwargs):
            snapshots.append(copy.deepcopy(kwargs["json"]["messages"]))
            return responses[len(snapshots) - 1]

        with (
            patch("builtins.input", side_effect=["第一条输入", "quit", "继续", "quit"]),
            patch.object(agent.requests, "post", side_effect=mock_post),
            patch("builtins.print"),
            patch.object(agent, "Path", return_value=SimpleNamespace(read_text=lambda: "test-key")),
        ):
            agent.main()
            agent.main()

        self.assertEqual([message["role"] for message in snapshots[1]], ["system", "user", "assistant", "user"])
        self.assertEqual(snapshots[1][0]["content"], agent.SYSTEM_PROMPT)
        self.assertEqual(snapshots[1][1]["content"], "第一条输入")
        self.assertEqual(snapshots[1][2]["content"], "第一轮完成")
        self.assertEqual(snapshots[1][3]["content"], "继续")

    def test_api_failure_does_not_save_incomplete_turn(self):
        original = '[{"role":"system","content":"old"},{"role":"user","content":"已完成"}]'
        self.history_path.write_text(original, encoding="utf-8")

        with (
            patch("builtins.input", return_value="未完成的问题"),
            patch.object(agent.requests, "post", side_effect=RuntimeError("API failed")),
            patch("builtins.print"),
            patch.object(agent, "Path", return_value=SimpleNamespace(read_text=lambda: "test-key")),
        ):
            with self.assertRaises(RuntimeError):
                agent.main()

        self.assertEqual(self.history_path.read_text(encoding="utf-8"), original)

    def test_save_failure_does_not_print_answer(self):
        original = '[{"role":"system","content":"old"}]'
        self.history_path.write_text(original, encoding="utf-8")
        responses = [self._response({"content": "本轮回答"})]

        with (
            patch("builtins.input", return_value="新问题"),
            patch.object(agent.requests, "post", side_effect=responses),
            patch.object(Path, "replace", side_effect=OSError("replace failed")),
            patch("builtins.print") as printed,
            patch.object(agent, "Path", return_value=SimpleNamespace(read_text=lambda: "test-key")),
        ):
            with self.assertRaises(OSError):
                agent.main()

        self.assertEqual(self.history_path.read_text(encoding="utf-8"), original)
        printed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
