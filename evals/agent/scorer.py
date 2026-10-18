"""Offline scoring of already-produced agent output directories; no backend or harness execution."""

import json
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


def notes_clauses(notes: str) -> list[str]:
    """Join soft line breaks and give list items their heading or colon-ended introduction."""
    list_marker = r"(?:[-+*]|\d+[.)])[ \t]+"
    blocks = re.split(
        rf"\n[ \t]*\n|^(?=[ \t]*(?:{list_marker}|\#{{1,6}}[ \t]+))",
        notes.casefold().replace("ν", "nu"),
        flags=re.MULTILINE,
    )
    result = []
    introduction = ""
    for block in blocks:
        text = " ".join(block.split())
        if not text:
            continue
        marker = re.match(list_marker, text)
        content = text[marker.end():] if marker else text
        clauses = [clause.strip() for clause in re.split(r"[.!?;]", content) if clause.strip()]
        if marker:
            result.extend(f"{introduction} {clause}" for clause in clauses)
        else:
            result.extend(clauses)
            is_introduction = text.rstrip("*_").endswith(":") or re.match(r"\#{1,6}\s", text)
            introduction = clauses[-1] if clauses and is_introduction else ""
    return result


def question_clauses(questions: list[dict]) -> list[str]:
    """One clause per question: its id and question joined, casefolded, with ν written nu."""
    return [f"{question['id']} {question['question']}".casefold().replace("ν", "nu") for question in questions]


def score_case(case: EvalCase, output_dir: Path, *, validation_failures: tuple[str, ...] = ()) -> CaseScore:
    """Compare plugin defaults with every expected entry, or check a questions or notes-only outcome.

    Missing outputs and coefficient mismatches are scores. Plugin execution and import errors propagate.
    Card contents belong to harness validation; this metric measures card presence only.
    Exhausted harness validation fails physics without executing a known-invalid plugin.
    """
    plugin_path = output_dir / "plugin.py"
    expressibility_match = (output_dir / "card.json").is_file() == case.card_expected
    if validation_failures:
        return CaseScore(case.case_id, False, expressibility_match, validation_failures)
    failures = []
    expected = case.expected_coefficients
    if expected is None:
        if plugin_path.exists():
            failures.append("expected no plugin.py")
        questions_path = output_dir / "questions.json"
        notes_path = output_dir / "notes.md"
        if case.question_requirements:
            if not questions_path.is_file():
                failures.append("missing questions.json")
            else:
                clauses = question_clauses(json.loads(questions_path.read_text())["questions"])
                failures.extend(requirement.label for requirement in case.question_requirements
                                if not any(all(re.search(pattern, clause) for pattern in requirement.patterns) for clause in clauses))
        elif questions_path.exists():
            failures.append("expected no questions.json")
        if case.notes_requirements and not notes_path.is_file():
            failures.append("missing notes.md")
        elif case.notes_requirements:
            sentences = notes_clauses(notes_path.read_text())
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
