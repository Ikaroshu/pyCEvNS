"""Research reports with the evidence level first and retained CLI provenance."""

import json
import subprocess
from importlib.metadata import version
from pathlib import Path

from nuagent.agent.backend import BackendResult

VERIFICATION = {
    "cross-validated": "Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.",
    "plugin-only": "SM-limit, perturbativity and count checks pass. There is no independent card cross-check; normalization and momentum dependence rest on the derivation.",
    "raw-cross-section": "User-authored complete cross section; perturbativity and count checks pass. The SM-limit and coefficient/card cross-check are skipped; the user owns normalization and the form factor.",
}


def source_provenance() -> dict:
    source = Path(__file__).resolve()
    roots = [parent for parent in source.parents if (parent / ".git").exists()]
    result = {"package": "nuagent", "package_version": version("nuagent"), "source_path": str(source)}
    if roots:
        root = roots[0]
        def git(*arguments: str) -> str:
            return subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True, text=True).stdout.strip()
        result.update(git_revision=git("rev-parse", "HEAD"), git_tree=git("rev-parse", "HEAD^{tree}"),
                      source_dirty=bool(git("status", "--porcelain")))
    else:
        result["git_revision"] = None
    return result


def render_report(description: str, task_dir: Path, backend_name: str, results: list[BackendResult],
                  histories: list[dict], provenance: dict, evaluations: list[dict], level: str | None) -> str:
    first = (f"Verification: **{level}**\n\n{VERIFICATION[level]}" if level else
             "Outcome: **notes-only**\n\nNo physics verification label or predictions: the agent reports missing information or physics outside the supported basis.")
    parts = [first, "## Description\n\n" + description,
             "## Provenance\n\n```json\n" + json.dumps(provenance | {"backend": backend_name}, indent=2) + "\n```"]
    for name in ("plugin.py", "card.json", "notes.md"):
        path = task_dir / name
        if path.is_file():
            parts.append(f"## {name}\n\n```{'python' if name.endswith('.py') else 'json' if name.endswith('.json') else 'text'}\n{path.read_text()}\n```")
    parts.append("## Validation and derivation history")
    for index, history in enumerate(histories, 1):
        parts.append(f"### Attempt {index}\n\n```json\n{json.dumps(history, indent=2)}\n```")
    if evaluations:
        parts.append("## Experiment predictions and likelihood comparison\n\nΔ = −2 ln L(new physics) − (−2 ln L(SM)); positive Δ favors the SM. Timing is integrated out.\n\n```json\n" + json.dumps(evaluations, indent=2, allow_nan=False) + "\n```")
    parts.append("## Backend CLI records\n\nModel, cost and usage fields, when supplied by the CLI, are retained below verbatim. Codex exec has no native max-turns flag; max_turns bounds Claude Code only.")
    for index, result in enumerate(results, 1):
        parts.append(f"### Attempt {index}\n\n```json\n{json.dumps(result.as_dict(), indent=2)}\n```")
    return "\n\n".join(parts) + "\n"
