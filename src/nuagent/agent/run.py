"""Translate once, allow one validation correction, then evaluate without feedback."""

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nuagent.agent.backend import Backend, BackendFailure, BackendResult
from nuagent.agent.report import render_report, source_provenance
from nuagent.agent.task import build_task
from nuagent.agent.validate import ValidatedModel, ValidationReport, validate_model
from nuagent.experiments import registry
from nuagent.experiments.registry import Experiment
from nuagent.rates import predicted_counts
from nuagent.stats.likelihood import profile_record, profiled


class ValidationFailed(RuntimeError):
    def __init__(self, histories: list[dict]):
        self.histories = histories
        super().__init__("agent outputs invalid after two attempts\n" + json.dumps(histories, indent=2))


@dataclass(frozen=True)
class RunResult:
    report: str
    histories: list[dict]
    signals: dict[str, np.ndarray]

    @property
    def valid(self) -> bool:
        return self.histories[-1]["valid"]


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


def run_result(description: str, experiments: Sequence[str], backend: Backend, task_dir: Path, out: Path,
               max_turns: int = 16) -> RunResult:
    """Return validation as an outcome; unexpected execution failures propagate."""
    selected = [registry.get(name) for name in experiments]
    provenance = source_provenance() | {"data_releases": {e.name: e.data_release for e in selected},
                                        "max_turns": max_turns}
    build_task(task_dir, description, experiments)
    histories = []
    results: list[BackendResult] = []
    model = None
    for attempt in (1, 2):
        result = backend.run(task_dir, max_turns)
        results.append(result)
        history_dir = task_dir / "history" / f"attempt-{attempt}"
        history_dir.mkdir(parents=True)
        (history_dir / "backend.json").write_text(json.dumps(result.as_dict(), indent=2))
        if result.exit_status or result.error:
            raise BackendFailure(result)
        notes = task_dir / "notes.md"
        if not notes.is_file() or not notes.read_text().strip():
            raise RuntimeError(f"backend produced no nonempty notes.md\n{result.raw_output}\n{result.stderr}")
        plugin, card = task_dir / "plugin.py", task_dir / "card.json"
        for name in ("plugin.py", "card.json", "notes.md"):
            if (task_dir / name).is_file():
                shutil.copyfile(task_dir / name, history_dir / name)
        if plugin.is_file():
            model = validate_model(selected, plugin=plugin, card=card if card.is_file() else None)
            validation = model.report
        elif card.is_file():
            validation = ValidationReport(["card.json without required coefficient plugin.py; notes-only outcomes must omit both"])
        else:
            validation = ValidationReport(checks=["notes-only outcome; no predictions or physics verification label"])
        history = validation.as_dict() | {"notes": notes.read_text()}
        histories.append(history)
        (history_dir / "validation.json").write_text(json.dumps(history, indent=2))
        if validation.valid:
            break
        if attempt == 1:
            for name in ("plugin.py", "card.json", "notes.md"):
                path = task_dir / name
                if path.exists():
                    path.unlink()
            with (task_dir / "TASK.md").open("a") as prompt:
                prompt.write("\n## Validation correction (one attempt)\n\nRepair only the translation using these validation findings. No likelihood feedback is supplied. The previous outputs are preserved in history/attempt-1/. The three output files have been cleared; write the complete corrected outcome, omitting plugin.py or card.json when the output contract requires it.\n\n```json\n" + json.dumps(validation.as_dict(), indent=2) + "\n```\n")
    level = ("cross-validated" if model.card else "plugin-only") if validation.valid and plugin.is_file() else None
    evaluations = evaluate(model, selected) if level else []
    report = render_report(description, task_dir, backend.name, results, histories, provenance, evaluations, level)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    return RunResult(report, histories, model.signals if level else {})


def run(description: str, experiments: Sequence[str], backend: Backend, task_dir: Path, out: Path,
        max_turns: int = 16) -> str:
    """Write a report and raise with both reports after exhausted correction."""
    result = run_result(description, experiments, backend, task_dir, out, max_turns)
    if not result.valid:
        raise ValidationFailed(result.histories)
    return result.report
