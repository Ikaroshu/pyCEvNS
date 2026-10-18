"""Create isolated translation inputs from installed context resources."""

from collections.abc import Sequence
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

OUTPUT_FILES = ("plugin.py", "card.json", "notes.md", "questions.json")


def copy_context(source: Traversable, destination: Path) -> None:
    destination.mkdir()
    for item in source.iterdir():
        if item.name == "__pycache__":
            continue
        target = destination / item.name
        if item.is_dir():
            copy_context(item, target)
        else:
            target.write_bytes(item.read_bytes())


def validation_commands() -> tuple[str, ...]:
    return (
        "nuagent validate --plugin plugin.py --card card.json",
        "nuagent validate --plugin plugin.py",
        "nuagent validate --card card.json",
    )


def build_task(task_dir: Path, description: str, experiments: Sequence[str]) -> None:
    """Create a fresh task; evaluation answers and repository files are never inputs."""
    task_dir.mkdir(parents=True)
    copy_context(files("nuagent.context"), task_dir / "context")
    (task_dir / "TASK.md").write_text(f"""# Translate a neutrino interaction

## Description

{description}

## Output contract

Read context/conventions.md, context/plugin_api.md, context/card_schema.json,
context/framework.md and the relevant worked examples before writing outputs.
Work only in this task directory. Use your own file tools to read the context
and to write the output files (Read, Write and Edit on Claude Code; file reads
and apply_patch on Codex). The only output files are {', '.join(OUTPUT_FILES)}.
The only shell commands needed are the exact nuagent validate commands below.
Do not inspect the source repository, evaluation cases, expected answers,
observed data or likelihood values.

For a complete description within the vector/scalar basis, write plugin.py with
coefficients, PARAMETERS defaults and COUPLINGS, plus notes.md with the derivation.
Write card.json whenever the schema can express the interaction. A plugin must
compute the coefficients independently of card.json, using exported constants;
do not read files from a plugin. If no card can express the interaction, explain
why in notes.md. Sum contributions before calling Coefficients.set (it overwrites).

If a description within the vector/scalar basis lacks information you need
(the mass, a coupling or the Lorentz structure), write questions.json, notes.md
saying what is missing, and no plugin.py or card.json. questions.json has the shape given in
context/plugin_api.md; suggest a value only when a conventional benchmark exists,
and never present a suggestion as part of the description. If the interaction is
outside the V/S basis, write notes.md explaining the basis limitation and the
user-authored raw-cross-section route, and produce no other file. Never write a
raw cross-section plugin. These question and notes-only outcomes are legitimate
and receive no predictions or physics verification label.

When a plugin is produced, run one of these commands before finishing and repair
validation findings. These commands return validation only, never likelihoods:

    {validation_commands()[0]}
    {validation_commands()[1]}

The harness validates on: {', '.join(experiments)}. Do not tune any parameter
against observations, likelihoods or evaluation answers. Preserve the described
point exactly. Finish with a short description of the files produced.
""")
