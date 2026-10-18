"""Repository-only live runner. Only descriptions and case answers cross into task inputs.

A case's answers are task inputs like its description: when the first round ends in questions,
the runner supplies them through continue_run and scores the final round.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from evals.agent.cases import CASES
from evals.agent.scorer import score_case
from nuagent.agent.backend import Backend
from nuagent.agent.run import continue_run, run_result


def run_evaluation(output_root: Path, backend: Backend, experiments: Sequence[str], max_turns: int = 16) -> dict:
    output_root.mkdir(parents=True)
    scores = []
    for case in CASES:
        task_dir, report = output_root / case.case_id, output_root / f"{case.case_id}.md"
        result = run_result(case.description, experiments, backend, task_dir, report, max_turns)
        if result.state == "questions" and case.answers:
            result = continue_run(task_dir, case.answers, backend, report)
        failures = tuple(f"validation attempt {attempt}: {finding}"
                         for attempt, history in enumerate(result.histories, 1)
                         for finding in history["findings"]) if not result.valid else ()
        score = asdict(score_case(case, output_root / case.case_id, validation_failures=failures))
        score["validation_valid"] = result.valid
        score["state"] = result.state
        score["rounds"] = result.round
        scores.append(score)
        (output_root / f"{case.case_id}.score.json").write_text(json.dumps(score, indent=2))
    summary = {"backend": backend.name, "output_root": str(output_root), "cases": scores,
               "physics_accuracy": sum(s["physics_match"] for s in scores) / len(scores),
               "expressibility_recognition": sum(s["expressibility_match"] for s in scores) / len(scores)}
    for label, selected in (
        ("card_expressible", [score for case, score in zip(CASES, scores) if case.card_expected]),
        ("plugin_only", [score for case, score in zip(CASES, scores) if not case.card_expected and case.expected_coefficients is not None]),
        ("no_plugin", [score for case, score in zip(CASES, scores) if case.expected_coefficients is None]),
    ):
        summary[label] = {"count": len(selected), "physics_accuracy": sum(s["physics_match"] for s in selected) / len(selected),
                          "expressibility_recognition": sum(s["expressibility_match"] for s in selected) / len(selected)}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
