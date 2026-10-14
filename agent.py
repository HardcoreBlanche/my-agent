import json
import subprocess
from pathlib import Path

import requests

API_URL = "https://api.pinaic.com/v1/chat/completions"
MODEL = "gpt-6-luna"

api_key = Path("api_key.txt").read_text().strip()

SYSTEM_PROMPT = """帮助用户完成任务，需要执行操作时使用 bash 工具。
根据工具结果继续操作，完成任务后直接回复用户。
每次 Bash 调用都是独立进程，cd 不会影响后续调用；需要时在命令内明确指定目录。"""

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "执行 Bash 命令，返回退出码和合并后的 stdout/stderr。",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


def run_bash(command):
    try:
        result = subprocess.run(
            ["bash", "-c", command],
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
    return f"{status}\n{output}"


messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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
            print("Agent:", message.get("content") or "")
            break

        for tool_call in tool_calls:
            arguments = json.loads(tool_call["function"]["arguments"])
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": run_bash(arguments["command"]),
            })
