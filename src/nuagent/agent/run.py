"""Run conversation rounds in one task directory; each round allows one validation correction, then evaluates without feedback."""

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nuagent.agent.backend import Backend, BackendFailure
from nuagent.agent.report import answer_line, render_report, source_provenance
from nuagent.agent.task import OUTPUT_FILES, build_task
from nuagent.agent.validate import ValidatedModel, ValidationReport, check_plugin, check_questions, validate_model
from nuagent.experiments import registry
from nuagent.experiments.registry import Experiment
from nuagent.rates import predicted_counts
from nuagent.stats.likelihood import profile_record, profiled

CONVERSATION = "conversation.json"
MAX_ROUNDS = 3


class ValidationFailed(RuntimeError):
    def __init__(self, histories: list[dict]):
        self.histories = histories
        super().__init__("agent outputs invalid after two attempts\n" + json.dumps(histories, indent=2))


@dataclass(frozen=True)
class RunResult:
    """The latest round's outcome: `state` is questions, valid, invalid, notes-only or unresolved."""
    report: str
    histories: list[dict]
    signals: dict[str, np.ndarray]
    state: str
    round: int
    questions: list[dict]

    @property
    def valid(self) -> bool:
        return self.histories[-1]["valid"]


def read_conversation(task_dir: Path) -> dict:
    """The harness's own record of the task: description, experiments, backend, max_turns and rounds."""
    return json.loads((task_dir / CONVERSATION).read_text())


def write_conversation(task_dir: Path, conversation: dict) -> None:
    (task_dir / CONVERSATION).write_text(json.dumps(conversation, indent=2))


def clear_outputs(task_dir: Path) -> None:
    for name in OUTPUT_FILES:
        path = task_dir / name
        if path.exists():
            path.unlink()


def evaluate(model: ValidatedModel, experiments: Sequence[Experiment]) -> list[dict]:
    records = []
    for experiment in experiments:
        signal = model.signals[experiment.name]
        new = profiled(experiment, signal)
        standard = profiled(experiment, predicted_counts(experiment))
        records.append({"experiment": experiment.name, "data_release": experiment.data_release,
                        "predicted_counts": signal.tolist(), "predicted_total": float(signal.sum()),
                        "observed_total": float(experiment.observed.sum()),
                        "delta": new.neg2_log_likelihood - standard.neg2_log_likelihood,
                        "new_physics": profile_record(new, experiment), "sm": profile_record(standard, experiment)})
    return records


def validate_outputs(task_dir: Path, experiments: Sequence[Experiment]) -> tuple[ValidationReport, ValidatedModel | None, list[dict]]:
    """Validate one attempt's outputs; return the report, the model when a plugin exists, and valid questions."""
    plugin, card, questions = (task_dir / name for name in ("plugin.py", "card.json", "questions.json"))
    if questions.is_file() and (plugin.is_file() or card.is_file()):
        return ValidationReport(["questions.json beside plugin.py or card.json; a questions outcome omits both"]), None, []
    if questions.is_file():
        report, asked = check_questions(questions)
        if report.valid:
            report.checks.append("questions outcome; no predictions or physics verification label")
        return report, None, asked
    if plugin.is_file():
        model = validate_model(experiments, plugin=plugin, card=card if card.is_file() else None)
        return model.report, model, []
    if card.is_file():
        return ValidationReport(["card.json without required coefficient plugin.py; notes-only outcomes must omit both"]), None, []
    return ValidationReport(checks=["notes-only outcome; no predictions or physics verification label"]), None, []


def run_round(task_dir: Path, conversation: dict, experiments: Sequence[Experiment], backend: Backend, out: Path) -> RunResult:
    """Run the conversation's last round, recorded as running, with two validation attempts."""
    number = conversation["rounds"][-1]["round"]
    histories = []
    for attempt in (1, 2):
        result = backend.run(task_dir, conversation["max_turns"])
        history_dir = task_dir / "history" / f"round-{number}" / f"attempt-{attempt}"
        history_dir.mkdir(parents=True)
        (history_dir / "backend.json").write_text(json.dumps(result.as_dict(), indent=2))
        if result.exit_status or result.error:
            conversation["rounds"][-1]["state"] = "backend-failure"
            write_conversation(task_dir, conversation)
            raise BackendFailure(result)
        notes = task_dir / "notes.md"
        if not notes.is_file() or not notes.read_text().strip():
            raise RuntimeError(f"backend produced no nonempty notes.md\n{result.raw_output}\n{result.stderr}")
        for name in OUTPUT_FILES:
            if (task_dir / name).is_file():
                shutil.copyfile(task_dir / name, history_dir / name)
        validation, model, questions = validate_outputs(task_dir, experiments)
        history = validation.as_dict() | {"notes": notes.read_text()}
        histories.append(history)
        (history_dir / "validation.json").write_text(json.dumps(history, indent=2))
        if validation.valid:
            break
        if attempt == 1:
            clear_outputs(task_dir)
            with (task_dir / "TASK.md").open("a") as prompt:
                prompt.write(f"\n## Validation correction (one attempt)\n\nRepair only the translation using these validation findings. No likelihood feedback is supplied. The previous outputs are preserved in history/round-{number}/attempt-1/. The output files ({', '.join(OUTPUT_FILES)}) have been cleared; write the complete corrected outcome, omitting files when the output contract requires it.\n\n```json\n" + json.dumps(validation.as_dict(), indent=2) + "\n```\n")
    if not validation.valid:
        state = "invalid"
    elif model is not None:
        state = "valid"
    elif questions:
        state = "questions" if number < MAX_ROUNDS else "unresolved"
    else:
        state = "notes-only"
    conversation["rounds"][-1] = {"round": number, "state": state} | ({"questions": questions} if questions else {})
    write_conversation(task_dir, conversation)
    level = ("cross-validated" if model.card else "plugin-only") if state == "valid" else None
    provenance = source_provenance() | {"data_releases": {e.name: e.data_release for e in experiments},
                                        "max_turns": conversation["max_turns"], "rounds": number}
    evaluations = evaluate(model, experiments) if level else []
    report = render_report(conversation, task_dir, provenance, evaluations, state, level, questions)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    return RunResult(report, histories, model.signals if level else {}, state, number, questions)


def run_result(description: str, experiments: Sequence[str], backend: Backend, task_dir: Path, out: Path,
               max_turns: int = 16) -> RunResult:
    """Run round 1; validation and questions are outcomes, unexpected execution failures propagate."""
    selected = [registry.get(name) for name in experiments]
    build_task(task_dir, description, experiments)
    conversation = {"description": description, "experiments": list(experiments), "backend": backend.name,
                    "max_turns": max_turns, "rounds": [{"round": 1, "state": "running"}]}
    write_conversation(task_dir, conversation)
    return run_round(task_dir, conversation, selected, backend, out)


def continue_run(task_dir: Path, answers: Mapping[str, str | float], backend: Backend, out: Path) -> RunResult:
    """Answer the last round's questions and run the next round on the backend recorded at round 1.

    `answers` maps asked question ids to values, plus optional free text under `free_text`.
    """
    conversation = read_conversation(task_dir)
    if backend.name != conversation["backend"]:
        raise ValueError(f"task {task_dir} was recorded with backend {conversation['backend']!r}, not {backend.name!r}")
    last = conversation["rounds"][-1]
    if last["state"] != "questions":
        raise ValueError(f"task {task_dir} is in state {last['state']!r}; only a task in state 'questions' continues")
    asked = {question["id"]: question for question in last["questions"]}
    unknown = sorted(set(answers) - set(asked) - {"free_text"})
    if unknown:
        raise ValueError(f"round {last['round']} did not ask {', '.join(unknown)}; asked ids: {', '.join(asked)} (and free_text)")
    number = last["round"] + 1
    last["answers"] = dict(answers)
    write_conversation(task_dir, conversation)
    lines = [f"- {answer_line(name, value, asked[name]['unit'])}" for name, value in answers.items() if name != "free_text"]
    text = f"\n## Answers (round {number})\n\n" + "\n".join(lines) + "\n"
    if "free_text" in answers:
        text += f"\nFree text:\n\n```text\n{answers['free_text']}\n```\n"
    text += (f"\nThese answers are part of the description. The previous questions and notes are in "
             f"history/round-{number - 1}/. Write questions.json again only if the answers leave the "
             "interaction undetermined.\n")
    with (task_dir / "TASK.md").open("a") as prompt:
        prompt.write(text)
    clear_outputs(task_dir)
    conversation["rounds"].append({"round": number, "state": "running"})
    write_conversation(task_dir, conversation)
    return run_round(task_dir, conversation, [registry.get(name) for name in conversation["experiments"]], backend, out)


def evaluate_point(task_dir: Path, parameters: dict[str, float]) -> tuple[ValidatedModel, list[dict]]:
    """Check and evaluate the task's plugin at another parameter point; the task directory is not changed.

    The result at any point other than the agent's is plugin-only: the card cross-check holds only where the card was written.
    """
    experiments = [registry.get(name) for name in read_conversation(task_dir)["experiments"]]
    model = check_plugin(task_dir / "plugin.py", experiments, parameters=parameters)
    return model, evaluate(model, experiments) if model.report.valid else []


def run(description: str, experiments: Sequence[str], backend: Backend, task_dir: Path, out: Path,
        max_turns: int = 16) -> str:
    """Write a report and raise with both reports after exhausted correction."""
    result = run_result(description, experiments, backend, task_dir, out, max_turns)
    if not result.valid:
        raise ValidationFailed(result.histories)
    return result.report
