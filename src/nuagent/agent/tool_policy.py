"""Claude PreToolUse policy for the task's explicit tool restrictions; an agent-loop directory gets the loop rules."""

import json
import sys
from pathlib import Path

from nuagent.agent.task import LOOP_OUTPUT_FILES, LOOP_RECORD, OUTPUT_FILES, validation_commands


def is_loop_directory(task_dir: Path) -> bool:
    return (task_dir / LOOP_RECORD).is_file()


def allowed_tool(name: str, arguments: dict, task_dir: Path) -> bool:
    loop = is_loop_directory(task_dir)
    if name == "Bash":
        return not loop and arguments.get("command", "").strip() in validation_commands()
    if name in ("Read", "Write", "Edit"):
        path = Path(arguments["file_path"])
        target = (task_dir / path).resolve()
        if name == "Read":
            return target.is_relative_to(task_dir.resolve())
        return target in {task_dir.resolve() / output for output in (LOOP_OUTPUT_FILES if loop else OUTPUT_FILES)}
    return False


def deny_reason(task_dir: Path) -> str:
    if is_loop_directory(task_dir):
        return f"Only loop-directory reads and {'/'.join(LOOP_OUTPUT_FILES)} writes are allowed; every shell command is denied."
    return f"Only task reads, {'/'.join(OUTPUT_FILES)} writes and exact nuagent validate commands are allowed."


def main() -> None:
    event = json.load(sys.stdin)
    task_dir = Path(sys.argv[1])
    allowed = allowed_tool(event["tool_name"], event["tool_input"], task_dir)
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": "allow" if allowed else "deny",
        "permissionDecisionReason": deny_reason(task_dir),
    }}))


if __name__ == "__main__":
    main()
