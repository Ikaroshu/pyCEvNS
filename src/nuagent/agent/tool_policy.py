"""Claude PreToolUse policy for the task's explicit tool restrictions."""

import json
import sys
from pathlib import Path

from nuagent.agent.task import OUTPUT_FILES, validation_commands


def allowed_tool(name: str, arguments: dict, task_dir: Path) -> bool:
    if name == "Bash":
        return arguments.get("command", "").strip() in validation_commands()
    if name in ("Read", "Write", "Edit"):
        path = Path(arguments["file_path"])
        target = (task_dir / path).resolve()
        if name == "Read":
            return target.is_relative_to(task_dir.resolve())
        return target in {task_dir.resolve() / output for output in OUTPUT_FILES}
    return False


def main() -> None:
    event = json.load(sys.stdin)
    allowed = allowed_tool(event["tool_name"], event["tool_input"], Path(sys.argv[1]))
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": "allow" if allowed else "deny",
        "permissionDecisionReason": f"Only task reads, {'/'.join(OUTPUT_FILES)} writes and exact nuagent validate commands are allowed.",
    }}))


if __name__ == "__main__":
    main()
