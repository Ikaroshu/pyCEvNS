"""Spec test 12: offline scorer, executable seeds, questions and notes-only outcomes."""

import json
from importlib.resources import files
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from evals.agent.cases import CASES, Q2_MEV2, EvalCase, NotesRequirement
from evals.agent.scorer import score_case, score_outputs
from nuagent.physics.couplings import Coefficients, QUARKS, STRUCTURES
from nuagent.physics.modelcard import template_card, u1_card
from nuagent.physics.plugin import load_coefficient_plugin


NUMERIC_CASES = tuple(case for case in CASES if case.expected_coefficients is not None)
MISSING_INFORMATION = r"\b(?:missing|unspecified|not\s+(?:given|specified)|unknown|need(?:ed)?|require[ds]?|provide|specify|what|which)\b"
# The notes rubric that case 14 used before it expected questions.json; it exercises notes_clauses, which case 15 relies on.
NOTES_RUBRIC = EvalCase("notes_rubric", "unused", False, None, (
    NotesRequirement("missing mediator mass", (r"\bmass\b", MISSING_INFORMATION)),
    NotesRequirement("missing neutrino coupling", (r"\b(?:g_nu|neutrino couplings?)\b", MISSING_INFORMATION)),
    NotesRequirement("missing quark coupling", (r"\b(?:g_q|quark couplings?)\b", MISSING_INFORMATION)),
))


def write_coefficients(output_dir: Path, c: Coefficients) -> None:
    """Freeze a produced array in a plugin so these tests exercise scoring rather than matching formulas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = ",\n".join(f"{key!r}: np.array({value.tolist()!r}, dtype=complex)" for key, value in c.values.items())
    (output_dir / "plugin.py").write_text(
        "import numpy as np\nfrom nuagent.physics.couplings import Coefficients\n"
        "PARAMETERS = {}\nCOUPLINGS = ()\n"
        f"def coefficients(q2, params):\n    return Coefficients(q2, {{{arrays}}})\n"
    )


def write_vector_plugin(output_dir: Path, g_nu: float = 7e-5) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plugin.py").write_text(
        "from nuagent.physics.couplings import Coefficients, GF, SQRT2\n"
        f"PARAMETERS = {{'g_nu': {g_nu!r}, 'g_q': 7e-5, 'mass_mev': 45.0}}\n"
        "COUPLINGS = ('g_nu', 'g_q')\n"
        "def coefficients(q2, params):\n"
        "    c = Coefficients.zeros(q2)\n"
        "    value = params['g_nu'] * params['g_q'] / (SQRT2 * GF * (q2 + params['mass_mev'] ** 2))\n"
        "    for quark in ('u', 'd'):\n"
        "        for pair in ('ee', 'mm', 'tt'):\n"
        "            c.set('V', quark, pair, value)\n"
        "    return c\n"
    )


def test_correct_plugin_and_factor_two_coupling(tmp_path):
    case = CASES[0]
    exact, wrong = tmp_path / "exact", tmp_path / "wrong"
    write_vector_plugin(exact)
    (exact / "card.json").write_text("{}")
    assert score_case(case, exact).physics_match
    assert score_case(case, exact).expressibility_match
    write_vector_plugin(wrong, g_nu=14e-5)
    (wrong / "card.json").write_text("{}")
    result = score_case(case, wrong)
    assert not result.physics_match
    assert result.expressibility_match
    assert result.physics_failures == ("coefficient mismatch: V u", "coefficient mismatch: V d")


@pytest.mark.parametrize("case", NUMERIC_CASES, ids=lambda case: case.case_id)
def test_every_numeric_case_accepts_exact_full_precision(case, tmp_path):
    write_coefficients(tmp_path, case.expected_coefficients)
    if case.card_expected:
        (tmp_path / "card.json").write_text("{}")
    result = score_case(case, tmp_path)
    assert result.physics_match and result.expressibility_match
    assert not result.physics_failures


@pytest.mark.parametrize("case", (CASES[0], CASES[11], CASES[12]), ids=lambda case: case.case_id)
def test_wrong_card_presence_is_independent_of_physics(case, tmp_path):
    write_coefficients(tmp_path, case.expected_coefficients)
    if not case.card_expected:
        (tmp_path / "card.json").write_text("{}")
    result = score_case(case, tmp_path)
    assert result.physics_match
    assert not result.expressibility_match


@pytest.mark.parametrize("structure,quark,i,j,k", tuple(product(STRUCTURES, QUARKS, range(3), range(3), range(3))))
def test_every_entry_and_expected_zero_is_checked(structure, quark, i, j, k, tmp_path):
    case = CASES[7]
    expected = case.expected_coefficients
    altered = Coefficients(expected.q2, {key: value.copy() for key, value in expected.values.items()})
    value = altered.values[structure, quark][i, j, k]
    altered.values[structure, quark][i, j, k] = value * 2 if value else 1e-30
    write_coefficients(tmp_path, altered)
    assert not score_case(case, tmp_path).physics_match


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_coefficients_do_not_match(value, tmp_path):
    (tmp_path / "plugin.py").write_text(
        "from nuagent.physics.couplings import Coefficients\n"
        "PARAMETERS = {}\nCOUPLINGS = ()\n"
        "def coefficients(q2, params):\n"
        "    c = Coefficients.zeros(q2)\n"
        f"    c.values['V', 'd'][0, 2] = float('{value}')\n"
        "    return c\n"
    )
    assert not score_case(CASES[7], tmp_path).physics_match


@pytest.mark.parametrize("alteration", ("missing_key", "extra_key", "broadcast_shape"))
def test_incomplete_or_broadcast_matrices_do_not_match(alteration, tmp_path):
    expected = CASES[7].expected_coefficients
    altered = Coefficients(expected.q2, {key: value.copy() for key, value in expected.values.items()})
    if alteration == "missing_key":
        del altered.values["S", "u"]
    elif alteration == "extra_key":
        altered.values["A", "u"] = np.zeros((3, 3, 3))
    else:
        altered.values["S", "u"] = np.zeros((1,))
    write_coefficients(tmp_path, altered)
    assert not score_case(CASES[7], tmp_path).physics_match


def test_relative_tolerance_and_zero_absolute_tolerance(tmp_path):
    expected = CASES[7].expected_coefficients
    altered = Coefficients(expected.q2, {key: value * (1 + 0.5e-6) for key, value in expected.values.items()})
    write_coefficients(tmp_path / "within", altered)
    assert score_case(CASES[7], tmp_path / "within").physics_match
    altered.values["V", "d"] *= 1 + 2e-6
    write_coefficients(tmp_path / "outside", altered)
    assert not score_case(CASES[7], tmp_path / "outside").physics_match


@pytest.mark.parametrize("case", NUMERIC_CASES, ids=lambda case: case.case_id)
def test_missing_required_plugin_fails(case, tmp_path):
    assert not score_case(case, tmp_path).physics_match


def test_plugin_execution_errors_propagate(tmp_path):
    (tmp_path / "plugin.py").write_text("raise RuntimeError('broken plugin')\n")
    with pytest.raises(RuntimeError, match="broken plugin"):
        score_case(CASES[0], tmp_path)


@pytest.mark.parametrize(
    "notes",
    (
        "Missing: mass, g_nu and g_q.",
        "What is the mediator mass? Please provide the neutrino coupling and quark coupling.",
        "The mass is not specified. The neutrino coupling g_ν is unknown. The quark coupling g_q is needed.",
        "Please provide the mediator mass,\nthe neutrino coupling g_nu, and\nthe quark coupling g_q.",
        "Missing information:\n- mediator mass\n- neutrino coupling g_nu\n- quark coupling g_q",
        "**Missing information:**\n\n- Mediator mass.\n- Neutrino coupling g_nu.\n- Quark coupling g_q.",
        "## Missing information\n\n1. Mediator mass.\n2. Neutrino coupling g_nu.\n3. Quark coupling g_q.",
        "Please provide:\n- The mediator\n  mass.\n- The neutrino coupling\n  g_nu.\n- The quark coupling\n  g_q.",
    ),
)
def test_notes_rubric_accepts_missing_information(notes, tmp_path):
    (tmp_path / "notes.md").write_text(notes)
    result = score_case(NOTES_RUBRIC, tmp_path)
    assert result.physics_match and result.expressibility_match


@pytest.mark.parametrize(
    "notes",
    (
        "Cannot calculate this model.",
        "The mass is missing. g_nu and g_q are fixed.",
        "Missing mass and g_nu.",
        "Missing mass and g_q.",
        "Missing g_nu and g_q.",
        "The mass, g_nu and g_q are given. Other information is missing.",
        "The mass is missing.\nThe neutrino coupling g_nu and quark coupling g_q are given.",
        "The mass is missing\n\nThe neutrino coupling g_nu and quark coupling g_q are given",
        "- The mass is missing.\n- The neutrino coupling g_nu and quark coupling g_q are given.",
        "Known inputs:\n- mass\n- g_nu\n- g_q\n\nMissing information:\n- exposure",
    ),
)
def test_notes_rubric_rejects_incomplete_or_unrelated_notes(notes, tmp_path):
    (tmp_path / "notes.md").write_text(notes)
    assert not score_case(NOTES_RUBRIC, tmp_path).physics_match


@pytest.mark.parametrize(
    "notes",
    (
        "The magnetic moment is outside the V, S basis. Use the user-authored raw cross-section route.",
        "This is beyond the vector/scalar basis. A raw cross section written by the user is required.",
        "The magnetic moment is outside the\nV, S basis. Use the user-authored raw\ncross-section route.",
    ),
)
def test_out_of_basis_accepts_limit_and_raw_route(notes, tmp_path):
    (tmp_path / "notes.md").write_text(notes)
    result = score_case(CASES[14], tmp_path)
    assert result.physics_match and result.expressibility_match


@pytest.mark.parametrize(
    "notes",
    (
        "Outside the V, S basis.",
        "Use the user-authored raw cross-section route.",
        "Outside the V, S basis. Use a raw cross-section route.",
        "The V, S basis supports this model. Other physics is outside. Use a user-authored raw cross-section route.",
    ),
)
def test_out_of_basis_rejects_missing_explanation_or_route(notes, tmp_path):
    (tmp_path / "notes.md").write_text(notes)
    assert not score_case(CASES[14], tmp_path).physics_match


def test_notes_only_requires_notes_and_absence_of_plugin_and_questions(tmp_path):
    case = CASES[14]
    assert not score_case(case, tmp_path).physics_match
    (tmp_path / "notes.md").write_text("Outside the V, S basis. Use the user-authored raw cross-section route.")
    assert score_case(case, tmp_path).physics_match
    (tmp_path / "card.json").write_text("{}")
    assert score_case(case, tmp_path).physics_match
    assert not score_case(case, tmp_path).expressibility_match
    (tmp_path / "questions.json").write_text(json.dumps({"questions": question_entries(("mass_mev", "Mediator mass?"))}))
    assert score_case(case, tmp_path).physics_failures == ("expected no questions.json",)
    (tmp_path / "questions.json").unlink()
    (tmp_path / "plugin.py").write_text("# Guessing a plugin is not a valid notes-only outcome.\n")
    assert not score_case(case, tmp_path).physics_match


def question_entries(*pairs: tuple[str, str]) -> list[dict]:
    return [{"id": name, "question": question, "unit": None, "suggested": None, "reason": None} for name, question in pairs]


@pytest.mark.parametrize("pairs", (
    (("quark_couplings", "Which quarks does the boson couple to?"),),
    (("structure", "Is the coupling to nucleons vector-like, with equal u and d couplings?"),),
    (("g_u", "Does it couple to up quarks?"), ("neutrino_flavors", "Does it couple to all three neutrino flavors equally?")),
    (("form", "Does the boson also couple to protons and neutrons?"),),
    (("lorentz", "Is it a vector or a scalar coupling to matter?"),),
    (("quark_ratio", "Are the u and d couplings equal in value, or do they follow B-L charges?"),),
    (("flavors", "Does it couple to all three neutrino flavors with the same strength, and to quarks?"),),
))
def test_under_specified_accepts_form_questions(pairs, tmp_path):
    (tmp_path / "questions.json").write_text(json.dumps({"questions": question_entries(*pairs)}))
    result = score_case(CASES[13], tmp_path)
    assert result.physics_match and result.expressibility_match


@pytest.mark.parametrize("pairs,failures", (
    ((("neutrino_flavors", "Is it flavor universal among neutrinos?"),), ("asks about the quark side or the structure",)),
    ((("quark_couplings", "Which quarks does it couple to?"), ("mass_mev", "Mediator mass in MeV?")), ("asks for a mass",)),
    ((("quark_couplings", "Which quarks does it couple to?"), ("g_nu", "What is the value of g_nu?")), ("asks for a numeric value",)),
    ((("g_q", "How large is the quark coupling?"),), ("asks for a numeric value",)),
    ((("couplings", "What coupling values should be used for the neutrinos and the quarks?"),), ("asks for a numeric value",)),
    ((("g_q", "What is the coupling strength to quarks?"),), ("asks for a numeric value",)),
    ((("quark_couplings", "Which numbers should the quark couplings take?"),), ("asks for a numeric value",)),
    ((("g_q", "What is g_q?"),), ("asks only for a parameter",)),
))
def test_under_specified_rejects_missing_form_or_value_questions(pairs, failures, tmp_path):
    (tmp_path / "questions.json").write_text(json.dumps({"questions": question_entries(*pairs)}))
    assert score_case(CASES[13], tmp_path).physics_failures == failures


def test_under_specified_rejects_plugin_and_notes_only_outputs(tmp_path):
    case = CASES[13]
    (tmp_path / "notes.md").write_text("The quark side of the interaction is not specified.")
    assert score_case(case, tmp_path).physics_failures == ("missing questions.json",)
    (tmp_path / "questions.json").write_text(json.dumps({"questions": question_entries(("quark_couplings", "Which quarks couple?"))}))
    assert score_case(case, tmp_path).physics_match
    (tmp_path / "plugin.py").write_text("# A guessed plugin.\n")
    assert score_case(case, tmp_path).physics_failures == ("expected no plugin.py",)


@pytest.mark.parametrize("index,mass_mev", ((15, 30.0), (16, 10.0)), ids=("answered", "defaults"))
def test_answered_and_defaults_cases_expect_the_universal_point(index, mass_mev, tmp_path):
    case = CASES[index]
    if index == 15:
        assert case.description == CASES[13].description and case.answers.keys() == {"free_text"}
    else:
        assert case.answers is None
    source = files("nuagent.context").joinpath("example_plugin.py").read_text().replace('"mass_mev": 30.0', f'"mass_mev": {mass_mev!r}')
    exact, doubled = tmp_path / "exact", tmp_path / "doubled"
    for directory, text in ((exact, source), (doubled, source.replace('"g_prime": 1e-4', f'"g_prime": {np.sqrt(2) * 1e-4!r}'))):
        directory.mkdir()
        (directory / "plugin.py").write_text(text)
        (directory / "card.json").write_text("{}")
    assert score_case(case, exact).physics_match and score_case(case, exact).expressibility_match
    assert score_case(case, doubled).physics_failures == ("coefficient mismatch: V u", "coefficient mismatch: V d")


def test_aggregate_reports_both_rates(tmp_path):
    cases = (CASES[0], CASES[12])
    for case in cases:
        output_dir = tmp_path / case.case_id
        write_coefficients(output_dir, case.expected_coefficients)
        (output_dir / "card.json").write_text("{}")
    result = score_outputs(tmp_path, cases)
    assert result.physics_accuracy == 1
    assert result.expressibility_recognition == 0.5
    assert [score.case_id for score in result.cases] == [case.case_id for case in cases]
    write_vector_plugin(tmp_path / cases[0].case_id, g_nu=14e-5)
    assert score_outputs(tmp_path, cases).physics_accuracy == 0.5


def test_cases_cover_reference_grid_and_all_coefficients():
    assert len(CASES) == len({case.case_id for case in CASES}) == 17
    assert sum(case.card_expected for case in CASES) == 13
    assert sum(case.expected_coefficients is not None and not case.card_expected for case in CASES) == 2
    np.testing.assert_array_equal(Q2_MEV2, [1238.0, 3714.0, 6190.0])
    for case in NUMERIC_CASES:
        c = case.expected_coefficients
        np.testing.assert_array_equal(c.q2, Q2_MEV2)
        assert set(c.values) == set(product(STRUCTURES, QUARKS))
        for values in c.values.values():
            assert values.shape == (3, 3, 3)
            assert np.all(np.isfinite(values))
            np.testing.assert_array_equal(values, values.swapaxes(0, 1).conj())


def test_card_expressible_answers_match_physics_templates():
    alpha_g = np.sqrt(4 * np.pi * 3.899296105751435e-10)
    cards = (
        template_card("zprime_universal", {"mass_mev": 45, "g_nu": 7e-5, "g_q": 7e-5}),
        template_card("zprime_universal", {"mass_mev": 45, "g_nu": alpha_g, "g_q": alpha_g}),
        u1_card(3e-5, 25, q_nu_e=-1, q_nu_mu=-1, q_nu_tau=-1, q_u=1/3, q_d=1/3),
        template_card("zprime_lmu_ltau", {"mass_mev": 10, "g_prime": 1e-4, "kinetic_mixing": 1e-3}),
        template_card("zprime_lmu_ltau", {"mass_mev": 15, "g_prime": 6e-5}),
        template_card("zprime_lmu_ltau", {"mass_mev": 15, "g_prime": 6e-5, "kinetic_mixing": -5e-4}),
        template_card("scalar_universal", {"mass_mev": 20, "g_nu": 2e-5, "g_d": 2e-5}),
        template_card("contact_nsi", {"eps_d_mm": -0.05, "eps_d_et": 0.02, "eps_d_et_im": 0.03}),
        template_card("zprime_universal", {"mass_mev": 30, "g_nu": 8e-5, "g_q": 8e-5}),
        template_card("zprime_universal", {"mass_mev": 30, "g_nu": 1e-4, "g_q": 1e-4}),
        template_card("zprime_universal", {"mass_mev": 20, "g_nu": 1e-4, "g_u": -1e-4, "g_d": 2e-4}),
        template_card("zprime_universal", {"mass_mev": 30, "g_nu": 1e-4, "g_q": 1e-4}),
        template_card("zprime_universal", {"mass_mev": 10, "g_nu": 1e-4, "g_q": 1e-4}),
    )
    for case, card in zip(CASES[:11] + CASES[15:], cards, strict=True):
        for key, expected in case.expected_coefficients.values.items():
            np.testing.assert_allclose(card.coefficients(Q2_MEV2).values[key], expected, rtol=1e-14, atol=0)


def test_plugin_only_reference_values_have_running_and_dipole_dependence(tmp_path):
    running = CASES[11].expected_coefficients.values["V", "u"][1, 1].real
    constant = template_card("zprime_lmu_ltau", {"mass_mev": 12, "g_prime": 8e-5}).coefficients(Q2_MEV2)
    ratio = running / constant.values["V", "u"][1, 1].real
    assert np.all((ratio > 0.98) & (ratio < 1))
    assert np.all(np.diff(ratio) < 0)
    source = files("nuagent.context").joinpath("examples/06_running_kinetic_mixing.md").read_text()
    plugin_path = tmp_path / "running.py"
    plugin_path.write_text(source.split("```python\n")[1].split("```")[0])
    plugin = load_coefficient_plugin(plugin_path)
    produced = plugin.coefficients(Q2_MEV2, {"mass_mev": 12.0, "g_prime": 8e-5})
    for key, expected in CASES[11].expected_coefficients.values.items():
        np.testing.assert_allclose(produced.values[key], expected, rtol=1e-14, atol=0)
    dipole = CASES[12].expected_coefficients.values["V", "u"][0, 0].real
    universal = CASES[9].expected_coefficients.values["V", "u"][0, 0].real
    np.testing.assert_allclose(dipole / universal, (1 + Q2_MEV2 / 200**2) ** -2, rtol=1e-14)
