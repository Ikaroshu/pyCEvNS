"""Offline scoring of already-produced agent output directories; no backend or harness execution."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evals.agent.cases import CASES, EvalCase
from nuagent.physics.plugin import load_coefficient_plugin


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    physics_match: bool
    expressibility_match: bool
    physics_failures: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationScore:
    cases: tuple[CaseScore, ...]
    physics_accuracy: float
    expressibility_recognition: float


def score_case(case: EvalCase, output_dir: Path) -> CaseScore:
    """Compare plugin defaults with every expected entry, or check a notes-only outcome.

    Missing outputs and coefficient mismatches are scores. Plugin execution and import errors propagate.
    Card contents belong to harness validation; this metric measures card presence only.
    """
    plugin_path = output_dir / "plugin.py"
    expressibility_match = (output_dir / "card.json").is_file() == case.card_expected
    failures = []
    expected = case.expected_coefficients
    if expected is None:
        if plugin_path.exists():
            failures.append("expected no plugin.py")
        notes_path = output_dir / "notes.md"
        if not notes_path.is_file():
            failures.append("missing notes.md")
        else:
            sentences = re.split(r"[.!?;\n]", notes_path.read_text().casefold().replace("ν", "nu"))
            for requirement in case.notes_requirements:
                if not any(all(re.search(pattern, sentence) for pattern in requirement.patterns) for sentence in sentences):
                    failures.append(requirement.label)
    elif not plugin_path.is_file():
        failures.append("missing plugin.py")
    else:
        plugin = load_coefficient_plugin(plugin_path)
        produced = plugin.coefficients(expected.q2, plugin.parameters)
        if produced.values.keys() != expected.values.keys():
            failures.append("coefficient structures/quarks differ")
        else:
            for key, value in expected.values.items():
                actual = produced.values[key]
                if np.shape(actual) != value.shape or not np.allclose(actual, value, rtol=1e-6, atol=0):
                    failures.append(f"coefficient mismatch: {key[0]} {key[1]}")
    return CaseScore(case.case_id, not failures, expressibility_match, tuple(failures))


def score_outputs(output_root: Path, cases: Sequence[EvalCase] = CASES) -> EvaluationScore:
    """Score a nonempty case sequence at output_root/<case_id>/ and return two independent rates."""
    scores = tuple(score_case(case, output_root / case.case_id) for case in cases)
    return EvaluationScore(
        scores,
        sum(score.physics_match for score in scores) / len(scores),
        sum(score.expressibility_match for score in scores) / len(scores),
    )
