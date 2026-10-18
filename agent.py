import json
import subprocess
import readline
from pathlib import Path

import requests


API_URL = "https://api.pinaic.com/v1/chat/completions"
MODEL = "gpt-6-luna"
HISTORY_PATH = Path("history.json")

SYSTEM_PROMPT = """完成任务，可以使用bash，用中文回复。"""

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "在独立，非交互的Bash进程中执行命令，返回退出码和输出。",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


def _validate_history(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("历史文件格式错误：顶层必须是非空消息列表")
    if not isinstance(messages[0], dict) or messages[0].get("role") != "system":
        raise ValueError("历史文件格式错误：首条消息必须是 system 消息")

    valid_roles = {"system", "user", "assistant", "tool"}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"历史文件格式错误：第 {index + 1} 条消息必须是对象")

        role = message.get("role")
        if not isinstance(role, str) or role not in valid_roles:
            raise ValueError(f"历史文件格式错误：第 {index + 1} 条消息的 role 无效")
        if "content" not in message:
            raise ValueError(f"历史文件格式错误：第 {index + 1} 条消息缺少 content")

        content = message["content"]
        if role in {"system", "user", "tool"} and not isinstance(content, str):
            raise ValueError(f"历史文件格式错误：第 {index + 1} 条消息的 content 必须是字符串")
        if role == "assistant" and content is not None and not isinstance(content, str):
            raise ValueError(f"历史文件格式错误：第 {index + 1} 条消息的 content 必须是字符串或 null")

        if role == "tool" and (
            not isinstance(message.get("tool_call_id"), str) or not message["tool_call_id"]
        ):
            raise ValueError(f"历史文件格式错误：第 {index + 1} 条 tool 消息缺少合法的 tool_call_id")

        if "tool_calls" in message:
            tool_calls = message["tool_calls"]
            if role != "assistant" or not isinstance(tool_calls, list):
                raise ValueError(f"历史文件格式错误：第 {index + 1} 条消息的 tool_calls 无效")
            for call_index, tool_call in enumerate(tool_calls):
                function = tool_call.get("function") if isinstance(tool_call, dict) else None
                if (
                    not isinstance(tool_call, dict)
                    or not isinstance(tool_call.get("id"), str)
                    or not tool_call.get("id")
                    or not isinstance(function, dict)
                    or not isinstance(function.get("name"), str)
                    or not function.get("name")
                    or not isinstance(function.get("arguments"), str)
                ):
                    raise ValueError(
                        f"历史文件格式错误：第 {index + 1} 条消息的第 {call_index + 1} 个 tool call 无效"
                    )


def load_history():
    try:
        history_text = HISTORY_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [{"role": "system", "content": SYSTEM_PROMPT}]
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"无法读取历史文件 {HISTORY_PATH}: {error}") from error

    try:
        messages = json.loads(history_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"历史文件 {HISTORY_PATH} 不是有效 JSON: {error}") from error

    _validate_history(messages)
    messages[0]["content"] = SYSTEM_PROMPT
    return messages


def save_history(messages):
    history_text = json.dumps(messages, ensure_ascii=False, indent=2) + "\n"
    temporary_path = HISTORY_PATH.with_name(HISTORY_PATH.name + ".tmp")
    temporary_path.write_text(history_text, encoding="utf-8")
    temporary_path.replace(HISTORY_PATH)


def run_bash(command):

    print(f"\n$ {command}", flush=True)

    try:
        result = subprocess.run(
            ["bash", "-c", command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        status = f"exit_code: {result.returncode}"
        output = result.stdout

    except subprocess.TimeoutExpired as error: 
        
        status = "exit_code: unavailable (timed out after 60 seconds)"
        output = error.output or b""

    output = output.decode("utf-8", errors="replace")

    if len(output) > 12000:
        output = output[:12000] + "\n[output truncated]"
    
    output = f"{status}\n{output}"
    print(output, flush=True)
    return output


def main():
    messages = load_history()
    api_key = Path("api_key.txt").read_text().strip()

    while True:
        user_input = input("You: ").strip()

        if user_input == "quit":
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        while True:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": MODEL, "messages": messages, "tools": [BASH_TOOL]},
                timeout=60,
            )
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            tool_calls = message.get("tool_calls")
            assistant_message = {"role": "assistant", "content": message.get("content")}
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            messages.append(assistant_message)

            if not tool_calls:
                save_history(messages)
                print("Agent:", message.get("content") or "")
                break

            for tool_call in tool_calls:
                arguments = json.loads(tool_call["function"]["arguments"])
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": run_bash(arguments["command"]),
                })


if __name__ == "__main__":
    main()
