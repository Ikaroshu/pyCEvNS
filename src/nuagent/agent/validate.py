"""Findings at the external model boundary; plugin execution errors propagate."""

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from pydantic_core import SchemaValidator, core_schema

from nuagent.experiments.registry import Experiment
from nuagent.physics.couplings import QUARKS, STRUCTURES, Coefficients, kinetic_mixing_coupling
from nuagent.physics.modelcard import ModelCard
from nuagent.physics.plugin import CoefficientPlugin, RawCrossSectionPlugin, load_module
from nuagent.rates import integrated_counts, predicted_counts

PERTURBATIVE_LIMIT = np.sqrt(4 * np.pi)
CARD_SCHEMA = SchemaValidator(core_schema.json_schema(ModelCard.__pydantic_core_schema__))
JSON_TEXT = SchemaValidator(core_schema.json_schema())
QUESTION_FIELDS = ("id", "question", "unit", "suggested", "reason")
QUESTION_ID = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")


@dataclass
class ValidationReport:
    findings: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.findings

    def as_dict(self) -> dict:
        return {"valid": self.valid, **asdict(self)}


@dataclass
class ValidatedModel:
    report: ValidationReport
    plugin: CoefficientPlugin | RawCrossSectionPlugin | None = None
    card: ModelCard | None = None
    signals: dict[str, np.ndarray] = field(default_factory=dict)


def reference_q2(experiments: Sequence[Experiment]) -> np.ndarray:
    """Union of 33-point isotope grids spanning every selected recoil range."""
    return np.unique(np.concatenate([
        np.linspace(isotope.q2(experiment.recoil_edges_mev()[0]), isotope.q2(experiment.recoil_edges_mev()[-1]), 33)
        for experiment in experiments for isotope in experiment.detector.isotopes
    ]))


def coefficient_findings(c: Coefficients, q2: np.ndarray) -> list[str]:
    """Validate the coefficient container returned across the plugin boundary."""
    if not isinstance(c, Coefficients):
        return ["coefficients must return Coefficients"]
    if not np.array_equal(c.q2, q2):
        return ["coefficient q2 grid differs from the requested grid"]
    if c.values.keys() != {(s, q) for s in STRUCTURES for q in QUARKS}:
        return ["coefficient structures/quarks must be V/S and u/d"]
    findings = []
    for key, value in c.values.items():
        if np.shape(value) != (3, 3, q2.size):
            findings.append(f"{key}: expected shape (3, 3, {q2.size})")
        elif not np.all(np.isfinite(value)):
            findings.append(f"{key}: non-finite coefficients")
        elif not np.allclose(value, np.swapaxes(value.conj(), 0, 1), rtol=1e-12, atol=0):
            findings.append(f"{key}: non-Hermitian coefficients")
    return findings


def count_findings(counts: np.ndarray, experiment: Experiment) -> list[str]:
    return [f"{experiment.name}: non-finite or negative count in bin {i}: {counts[i]}"
            for i in np.flatnonzero(~np.isfinite(counts) | (counts < 0))]


def check_plugin(path: Path, experiments: Sequence[Experiment], *, raw: bool = False,
                 parameters: dict[str, float] | None = None) -> ValidatedModel:
    """Check the plugin at its PARAMETERS defaults, or at `parameters` overriding them."""
    report = ValidationReport()
    module = load_module(path)
    function = "dsigma_dt" if raw else "coefficients"
    for name in (function, "PARAMETERS", "COUPLINGS"):
        if not hasattr(module, name):
            report.findings.append(f"plugin missing {name}")
    if not report.valid:
        return ValidatedModel(report)
    if not callable(getattr(module, function)) or not isinstance(module.PARAMETERS, dict) or not isinstance(module.COUPLINGS, tuple):
        report.findings.append("plugin requires a callable, PARAMETERS dict and COUPLINGS tuple")
        return ValidatedModel(report)
    parameters = module.PARAMETERS | (parameters or {})
    couplings = module.COUPLINGS
    if any(not isinstance(name, str) or name not in parameters for name in couplings):
        report.findings.append("COUPLINGS must name entries in PARAMETERS")
    if any(not isinstance(value, (int, float)) or not np.isfinite(value) for value in parameters.values()):
        report.findings.append("PARAMETERS must contain finite real numbers")
    if not report.valid:
        return ValidatedModel(report)
    for name in couplings:
        if abs(parameters[name]) > PERTURBATIVE_LIMIT:
            report.findings.append(f"non-perturbative coupling {name}={parameters[name]} exceeds sqrt(4 pi)")
    plugin = (RawCrossSectionPlugin(path, module.dsigma_dt, parameters, couplings) if raw else
              CoefficientPlugin(path, module.coefficients, parameters, couplings))
    result = ValidatedModel(report, plugin)
    if not report.valid:
        return result
    report.checks.append("couplings are perturbative")
    if raw:
        report.checks.append("SM-limit check skipped for user-authored raw cross section")
    else:
        q2 = reference_q2(experiments)
        zero = parameters | dict.fromkeys(couplings, 0.0)
        for label, point in (("proposed", parameters), ("zero couplings", zero)):
            report.findings.extend(f"{label}: {finding}" for finding in coefficient_findings(plugin.coefficients(q2, point), q2))
        if not report.valid:
            return result
        for experiment in experiments:
            counts = integrated_counts(experiment, lambda q: plugin.coefficients(q, zero))
            report.findings.extend(count_findings(counts, experiment))
            if not np.allclose(counts, predicted_counts(experiment), rtol=1e-9, atol=0):
                report.findings.append(f"{experiment.name}: SM-limit counts differ at rtol=1e-9 with COUPLINGS zeroed")
        report.checks.append("SM-limit counts checked with COUPLINGS zeroed (rtol=1e-9, atol=0)")
    for experiment in experiments:
        if raw:
            counts = integrated_counts(experiment, raw_cross_section=lambda e, t, f, i: plugin.dsigma_dt(e, t, f, i, parameters))
        else:
            counts = integrated_counts(experiment, lambda q: plugin.coefficients(q, parameters))
        report.findings.extend(count_findings(counts, experiment))
        result.signals[experiment.name] = counts
    report.checks.append("proposed counts checked for finiteness and non-negativity")
    return result


def check_card(path: Path, experiments: Sequence[Experiment]) -> ValidatedModel:
    report = ValidationReport()
    text = path.read_text()
    if not CARD_SCHEMA.isinstance_python(text):
        report.findings.append("card.json is not valid JSON satisfying context/card_schema.json (ModelCard)")
        return ValidatedModel(report)
    card = ModelCard.model_validate_json(text)
    couplings = []
    for mediator in card.mediators:
        couplings.extend(mediator.nu_couplings.matrix().ravel())
        couplings.extend((mediator.quark_couplings.u, mediator.quark_couplings.d, mediator.kinetic_mixing))
        for quark in QUARKS:
            couplings.append(getattr(mediator.quark_couplings, quark) + kinetic_mixing_coupling(mediator.kinetic_mixing, quark))
    for contact in card.contacts:
        couplings.extend(contact.epsilon.matrix().ravel())
    if any(abs(value) > PERTURBATIVE_LIMIT for value in couplings):
        report.findings.append("card has a non-perturbative coupling exceeding sqrt(4 pi)")
    result = ValidatedModel(report, card=card)
    if report.valid:
        report.checks.extend(("ModelCard schema valid", "couplings are perturbative", "card SM limit holds by construction"))
        for experiment in experiments:
            counts = integrated_counts(experiment, card.coefficients)
            report.findings.extend(count_findings(counts, experiment))
            result.signals[experiment.name] = counts
        report.checks.append("card counts checked for finiteness and non-negativity")
    return result


def cross_check(plugin: CoefficientPlugin, card: ModelCard, experiments: Sequence[Experiment]) -> ValidationReport:
    q2 = reference_q2(experiments)
    actual = plugin.coefficients(q2, plugin.parameters)
    expected = card.coefficients(q2)
    report = ValidationReport(coefficient_findings(actual, q2))
    if report.valid:
        for key, value in expected.values.items():
            if not np.allclose(actual.values[key], value, rtol=1e-6, atol=0):
                report.findings.append(f"plugin/card mismatch: {key[0]} {key[1]} (all entries, rtol=1e-6, atol=0)")
        report.checks.append(f"plugin/card cross-check on {q2.size} Q2 points spanning the selected recoil ranges")
    return report


def validate_model(experiments: Sequence[Experiment], *, plugin: Path | None = None, card: Path | None = None,
                   raw: bool = False) -> ValidatedModel:
    result = check_plugin(plugin, experiments, raw=raw) if plugin else check_card(card, experiments)
    if plugin and card:
        card_result = check_card(card, experiments)
        result.card = card_result.card
        result.report.findings.extend(card_result.report.findings)
        result.report.checks.extend(card_result.report.checks)
        if result.report.valid:
            comparison = cross_check(result.plugin, result.card, experiments)
            result.report.findings.extend(comparison.findings)
            result.report.checks.extend(comparison.checks)
    return result


def question_findings(index: int, entry: object) -> list[str]:
    label = f"questions.json question {index}"
    if not isinstance(entry, dict) or set(entry) != set(QUESTION_FIELDS):
        return [f"{label}: must be an object with exactly the fields {', '.join(QUESTION_FIELDS)}"]
    findings = []
    if not isinstance(entry["id"], str) or not QUESTION_ID.fullmatch(entry["id"]):
        findings.append(f"{label}: id must be a short snake_case identifier")
    elif entry["id"] == "free_text":
        findings.append(f"{label}: id free_text is reserved for the user's free-text answer")
    if not isinstance(entry["question"], str) or not entry["question"].strip():
        findings.append(f"{label}: question must be a nonempty sentence")
    if entry["unit"] is not None and not isinstance(entry["unit"], str):
        findings.append(f"{label}: unit must be a string or null")
    suggested = entry["suggested"]
    if isinstance(suggested, bool) or not (suggested is None or isinstance(suggested, str)
                                           or isinstance(suggested, (int, float)) and np.isfinite(suggested)):
        findings.append(f"{label}: suggested must be a finite number (not a boolean), a string or null")
    if entry["reason"] is None and suggested is not None:
        findings.append(f"{label}: reason may be null only when suggested is null")
    elif entry["reason"] is not None and not isinstance(entry["reason"], str):
        findings.append(f"{label}: reason must be a sentence or null")
    return findings


def check_questions(path: Path) -> tuple[ValidationReport, list[dict]]:
    """Check questions.json at the agent-output boundary; return the questions when valid."""
    report = ValidationReport()
    text = path.read_text()
    if not JSON_TEXT.isinstance_python(text):
        report.findings.append("questions.json is not valid JSON")
        return report, []
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != {"questions"} or not isinstance(data["questions"], list) or not data["questions"]:
        report.findings.append('questions.json must be {"questions": [...]} with at least one question')
        return report, []
    questions = data["questions"]
    for index, entry in enumerate(questions, 1):
        report.findings.extend(question_findings(index, entry))
    if report.valid:
        ids = [entry["id"] for entry in questions]
        duplicates = sorted({name for name in ids if ids.count(name) > 1})
        if duplicates:
            report.findings.append(f"questions.json ids must be unique; repeated: {', '.join(duplicates)}")
    if not report.valid:
        return report, []
    report.checks.append(f"questions.json shape valid ({len(questions)} questions)")
    return report, questions


def check_points(path: Path, names: Sequence[str], limit: int) -> tuple[ValidationReport, list[dict[str, float]]]:
    """Check a fit agent's points.json at the agent-output boundary; return the points when valid."""
    report = ValidationReport()
    if not path.is_file():
        report.findings.append("points.json is missing")
        return report, []
    text = path.read_text()
    if not JSON_TEXT.isinstance_python(text):
        report.findings.append("points.json is not valid JSON")
        return report, []
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != {"points"} or not isinstance(data["points"], list):
        report.findings.append('points.json must be {"points": [{name: value, ...}, ...]}')
        return report, []
    points = data["points"]
    if len(points) > limit:
        report.findings.append(f"points.json has {len(points)} points; at most {limit} are allowed")
    for index, point in enumerate(points, 1):
        if not isinstance(point, dict):
            report.findings.append(f"points.json point {index}: must be an object mapping parameter names to numbers")
            continue
        unknown = sorted(set(point) - set(names))
        if unknown:
            report.findings.append(f"points.json point {index}: unknown parameters {', '.join(unknown)}; "
                                   f"plugin parameters: {', '.join(names)}")
        for name, value in point.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                report.findings.append(f"points.json point {index}: {name} must be a finite real number (not a boolean)")
    if not report.valid:
        return report, []
    report.checks.append(f"points.json shape valid ({len(points)} points)")
    return report, points
