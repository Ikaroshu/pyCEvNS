"""Research reports with the evidence level or outcome first, the conversation and retained CLI provenance."""

import json
import subprocess
from importlib.metadata import version
from pathlib import Path


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


OUTCOMES = {
    "invalid": "Outcome: **validation-failed**\n\nCorrection was exhausted. No physics verification label or predictions; both validation reports and backend records follow.",
    "questions": "Outcome: **questions**\n\nThe description does not fix the interaction; the agent asks the open questions below. No physics verification label or predictions; answer the questions to continue the task.",
    "unresolved": "Outcome: **unresolved**\n\nThe agent still had questions after the third round, the conversation limit. No physics verification label or predictions; the open questions follow.",
    "notes-only": "Outcome: **notes-only**\n\nNo physics verification label or predictions: the agent reports missing information or physics outside the supported basis.",
}


def answer_line(name: str, value: str | float, unit: str | None) -> str:
    return f"{name}: {value}" + (f" {unit}" if unit else "")


def questions_markdown(questions: list[dict]) -> str:
    lines = []
    for question in questions:
        line = f"- `{question['id']}`" + (f" ({question['unit']})" if question["unit"] else "") + f": {question['question']}"
        if question["suggested"] is not None:
            line += f" Suggested: {question['suggested']}. {question['reason']}"
        lines.append(line)
    return "\n".join(lines)


def conversation_markdown(conversation: dict) -> str:
    """Every round's state, questions and answers in order, from the harness's conversation record."""
    parts = []
    units = {}
    for entry in conversation["rounds"]:
        parts.append(f"### Round {entry['round']}: {entry['state']}")
        if entry.get("questions"):
            parts.append(questions_markdown(entry["questions"]))
            units = {question["id"]: question["unit"] for question in entry["questions"]}
        if "answers" in entry:
            parts.append("Answers:\n\n" + answers_markdown(entry["answers"], units))
    return "\n\n".join(parts)


def answers_markdown(answers: dict[str, str | float], units: dict[str, str | None]) -> str:
    lines = [f"- {answer_line(name, value, units[name])}" for name, value in answers.items() if name != "free_text"]
    if "free_text" in answers:
        lines.append(f"- free text: {answers['free_text']}")
    return "\n".join(lines) or "- none"


LIKELIHOOD_CAPTION = (
    "Δ = −2 ln L(new physics) − (−2 ln L(SM)) on the observed counts; positive Δ favors the SM. Timing is integrated out. "
    "α and β are the profiled nuisance parameters: α scales the signal prediction under its normalization prior, "
    "each β scales one background component under its prior, and each model is profiled separately."
)


def likelihood_table(evaluations: list[dict]) -> str:
    """One Markdown row per experiment: counts, Δ and both models' profiled nuisances."""
    backgrounds = list(evaluations[0]["sm"]["beta"])
    header = ["Experiment", "Observed", "SM signal", "New-physics signal", "Δ", "Signal scale α (new physics / SM)",
              *(f"Background scale β, {name.replace('_', ' ')} (new physics / SM)" for name in backgrounds)]
    rows = [header, ["---"] * len(header)]
    for record in evaluations:
        new, sm = record["new_physics"], record["sm"]
        rows.append([record["experiment"], f"{record['observed_total']:g}", f"{record['sm_predicted_total']:.6g}",
                     f"{record['predicted_total']:.6g}", f"{record['delta']:.6g}", f"{new['alpha']:.4g} / {sm['alpha']:.4g}",
                     *(f"{new['beta'][name]:.4g} / {sm['beta'][name]:.4g}" for name in backgrounds)])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def render_report(conversation: dict, task_dir: Path, provenance: dict, evaluations: list[dict],
                  state: str, level: str | None, questions: list[dict]) -> str:
    """Level or outcome, description and conversation, predictions, notes inline, outputs, histories, provenance last."""
    first = f"Verification: **{level}**\n\n{VERIFICATION[level]}" if level else OUTCOMES[state]
    parts = [first, "## Description\n\n" + conversation["description"],
             "## Conversation\n\n" + conversation_markdown(conversation)]
    if state in ("questions", "unresolved"):
        parts.append("## Open questions\n\n" + questions_markdown(questions))
    if evaluations:
        parts.append("## Experiment predictions and likelihood comparison\n\n" + likelihood_table(evaluations)
                     + "\n\n" + LIKELIHOOD_CAPTION + "\n\nPer-bin predictions and full profiles:\n\n```json\n"
                     + json.dumps(evaluations, indent=2, allow_nan=False) + "\n```")
    if (task_dir / "notes.md").is_file():
        parts.append("## Notes\n\n" + (task_dir / "notes.md").read_text().strip())
    for name in ("plugin.py", "card.json", "questions.json"):
        path = task_dir / name
        if path.is_file():
            parts.append(f"## {name}\n\n```{'python' if name.endswith('.py') else 'json'}\n{path.read_text()}\n```")
    attempts = [(entry["round"], attempt, task_dir / "history" / f"round-{entry['round']}" / f"attempt-{attempt}")
                for entry in conversation["rounds"] for attempt in (1, 2)]
    attempts = [(number, attempt, path) for number, attempt, path in attempts if path.is_dir()]
    parts.append("## Validation and derivation history")
    for number, attempt, path in attempts:
        parts.append(f"### Round {number}, attempt {attempt}\n\n```json\n{(path / 'validation.json').read_text()}\n```")
    parts.append("## Backend CLI records\n\nModel, cost and usage fields, when supplied by the CLI, are retained below verbatim. Codex exec has no native max-turns flag; max_turns bounds Claude Code only.")
    for number, attempt, path in attempts:
        parts.append(f"### Round {number}, attempt {attempt}\n\n```json\n{(path / 'backend.json').read_text()}\n```")
    parts.append(provenance_section(provenance | {"backend": conversation["backend"]}))
    return "\n\n".join(parts) + "\n"


def provenance_section(provenance: dict) -> str:
    return "## Provenance\n\n```json\n" + json.dumps(provenance, indent=2) + "\n```"
