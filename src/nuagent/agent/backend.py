"""Installed coding-agent subprocess backends; no model SDK or likelihood loop."""

import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class BackendResult:
    final_text: str
    exit_status: int
    raw_output: str
    stderr: str = ""
    command: tuple[str, ...] = ()
    error: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


class Backend(Protocol):
    name: str

    def run(self, task_dir: Path, max_turns: int) -> BackendResult: ...


class BackendFailure(RuntimeError):
    def __init__(self, result: BackendResult):
        self.result = result
        super().__init__(f"backend failed (exit {result.exit_status})\n{result.raw_output}\n{result.stderr}")


@dataclass(frozen=True)
class SubprocessBackend:
    name: str

    def run(self, task_dir: Path, max_turns: int) -> BackendResult:
        prompt = (task_dir / "TASK.md").read_text()
        if self.name == "claude_code":
            hook = shlex.join([sys.executable, "-m", "nuagent.agent.tool_policy", str(task_dir.resolve())])
            settings = {"hooks": {"PreToolUse": [{"matcher": "", "hooks": [{"type": "command", "command": hook}]}]}}
            command = ["claude", "-p", "--output-format", "json", "--max-turns", str(max_turns),
                       "--restricted", "--tools", "Read,Write,Edit,Bash", "--permission-mode", "dontAsk",
                       "--setting-sources", "", "--settings", json.dumps(settings),
                       "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                       "--disable-slash-commands", "--no-session-persistence"]
        else:
            command = ["codex", "exec", "--sandbox", "workspace-write", "--cd", str(task_dir.resolve()),
                       "--skip-git-repo-check", "--ephemeral", "--json", "--color", "never",
                       "-c", 'approval_policy="never"', "-c", 'sandbox_workspace_write.writable_roots=[]',
                       "-c", "sandbox_workspace_write.network_access=false", "-"]
        environment = os.environ | {"PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
                                    "PYTHONDONTWRITEBYTECODE": "1"}
        process = subprocess.run(command, cwd=task_dir, input=prompt, capture_output=True, text=True, env=environment)
        result = BackendResult("", process.returncode, process.stdout, process.stderr, tuple(command))
        if process.returncode:
            return result
        if self.name == "claude_code":
            payload = json.loads(process.stdout)
            return BackendResult(payload.get("result", ""), process.returncode, process.stdout, process.stderr, tuple(command), bool(payload.get("is_error")))
        events = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
        messages = [event["item"]["text"] for event in events
                    if event.get("type") == "item.completed" and event.get("item", {}).get("type") == "agent_message"]
        failed = any(event.get("type") in ("turn.failed", "error") for event in events)
        return BackendResult("\n".join(messages), process.returncode, process.stdout, process.stderr, tuple(command), failed)


def get_backend(name: str) -> Backend:
    if name not in ("claude_code", "codex"):
        raise ValueError(f"unknown backend {name!r}; choose claude_code or codex")
    return SubprocessBackend(name)
