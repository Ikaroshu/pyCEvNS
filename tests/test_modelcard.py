"""Spec Verification tests 5 (model card) and 7 (SM limit of every template), and the docs/agent numbers."""

import json
import re
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from nuagent.physics.couplings import (
    E_CHARGE,
    GF,
    KAPPA_LMU_LTAU,
    M_N,
    M_P,
    QUARK_MASS_MEV,
    SCALAR_CONTENT_N,
    SCALAR_CONTENT_P,
    SIN2_THETA_W,
    SQRT2,
    scalar_charge,
    sm,
    vector_charge,
)
from nuagent.physics.modelcard import TEMPLATES, ModelCard, template_card, u1_card
from nuagent.physics.plugin import load_coefficient_plugin
from nuagent.physics.xsec import FLAVOR_ROW, Isotope, dsigma_dt

CONTEXT = Path(__file__).parent.parent / "src" / "nuagent" / "context"
Q2 = np.array([1238.0, 3714.0, 6190.0])
CS133 = Isotope(55, 78, 123800.0, 5.5)


def vector_card(**mediator) -> dict:
    return {"mediators": [{"structure": "V", "mass_mev": 30.0, **mediator}]}


def test_universal_vector_card_matches_derived_mapping():
    g_nu, g_q, mass = 1e-4, 2e-4, 30.0
    c = template_card("zprime_universal", {"g_nu": g_nu, "g_q": g_q, "mass_mev": mass}).coefficients(Q2)
    expected = g_nu * g_q / (SQRT2 * GF * (Q2 + mass**2))
    for quark in ("u", "d"):
        np.testing.assert_allclose(c.values["V", quark], np.eye(3)[:, :, None] * expected, rtol=1e-14)
        assert np.count_nonzero(c.values["S", quark]) == 0


def test_u1_card_from_b_minus_l_charges():
    g = 1e-4
    card = u1_card(g, 10.0, q_nu_e=-1, q_nu_mu=-1, q_nu_tau=-1, q_u=1 / 3, q_d=1 / 3)
    mediator = card.mediators[0]
    np.testing.assert_allclose(mediator.nu_couplings.matrix(), -g * np.eye(3))
    assert (mediator.quark_couplings.u, mediator.quark_couplings.d) == pytest.approx((g / 3, g / 3))
    c = card.coefficients(Q2)
    np.testing.assert_allclose(c.values["V", "u"][1, 1], -g * g / 3 / (SQRT2 * GF * (Q2 + 100.0)), rtol=1e-14)


def test_u1_card_from_arbitrary_charges():
    g = 2e-5
    card = u1_card(g, 15.0, q_nu_e=0.5, q_nu_mu=-2.0, q_nu_tau=3.0, q_u=0.7, q_d=-1.3, kinetic_mixing=1e-3)
    mediator = card.mediators[0]
    np.testing.assert_allclose(mediator.nu_couplings.matrix(), g * np.diag([0.5, -2.0, 3.0]))
    assert (mediator.quark_couplings.u, mediator.quark_couplings.d) == pytest.approx((0.7 * g, -1.3 * g))
    assert mediator.kinetic_mixing == 1e-3


def test_kinetic_mixing_adds_e_epsilon_q():
    eps, g_nu, g_q = 2e-3, 5e-5, 1e-5
    card = ModelCard.model_validate(
        vector_card(nu_couplings={"mm": g_nu}, quark_couplings={"u": g_q, "d": g_q}, kinetic_mixing=eps)
    )
    c = card.coefficients(Q2)
    for quark, charge in (("u", 2 / 3), ("d", -1 / 3)):
        expected = g_nu * (g_q + eps * E_CHARGE * charge) / (SQRT2 * GF * (Q2 + 900.0))
        np.testing.assert_allclose(c.values["V", quark][1, 1], expected, rtol=1e-14)
        assert np.count_nonzero(c.values["V", quark][0, 0]) == 0


def test_complex_off_diagonal_and_contacts():
    card = ModelCard.model_validate(
        {"contacts": [{"fermion": "d", "epsilon": {"mm": -0.05, "et": [0.02, 0.03]}}], "mediators": []}
    )
    c = card.coefficients(Q2)
    np.testing.assert_array_equal(c.values["V", "d"][0, 2], 0.02 + 0.03j)
    np.testing.assert_array_equal(c.values["V", "d"][2, 0], 0.02 - 0.03j)
    np.testing.assert_array_equal(c.values["V", "d"][1, 1], -0.05)
    assert np.count_nonzero(c.values["V", "u"]) == 0


@pytest.mark.parametrize(
    "card",
    [
        vector_card(nu_couplings={"ee": [0.1, 0.2]}),  # complex diagonal: not Hermitian
        vector_card(nu_couplings={"em": 0.1, "me": 0.2}),  # lower-triangle entry: no independent partner
        vector_card(mass_mev=0.0),
        vector_card(mass_mev=-5.0),
        vector_card(structure="A"),
        vector_card(structure="T"),
        vector_card(structure="S", kinetic_mixing=1e-3),
        vector_card(quark_couplings={"s": 1e-4}),
        vector_card(nu_couplings={"ee": float("nan")}),
        {"contacts": [{"fermion": "e", "epsilon": {"ee": 0.1}}]},
    ],
)
def test_invalid_cards_are_rejected(card):
    with pytest.raises(ValidationError):
        ModelCard.model_validate(card)


def test_unknown_template_names_valid_choices():
    with pytest.raises(ValueError, match="zprime_universal, zprime_u1, zprime_lmu_ltau, scalar_universal, contact_nsi"):
        template_card("zprime", {})


def test_unknown_parameter_names_valid_choices():
    with pytest.raises(ValueError, match=r"\['g_prime'\].*g_nu, g_u, g_d, g_q, mass_mev"):
        template_card("zprime_universal", {"g_prime": 1e-4, "mass_mev": 30.0})


def test_lmu_ltau_template_mixing():
    loop = template_card("zprime_lmu_ltau", {"g_prime": 1e-4, "mass_mev": 10.0}).mediators[0]
    assert loop.kinetic_mixing == KAPPA_LMU_LTAU * 1e-4
    assert KAPPA_LMU_LTAU == pytest.approx(0.0144417637, abs=5e-11)
    np.testing.assert_allclose(loop.nu_couplings.matrix(), np.diag([0, 1e-4, -1e-4]))
    assert (loop.quark_couplings.u, loop.quark_couplings.d) == (0, 0)
    given = template_card("zprime_lmu_ltau", {"g_prime": 1e-4, "mass_mev": 10.0, "kinetic_mixing": -5e-4})
    assert given.mediators[0].kinetic_mixing == -5e-4


def test_contact_template_imaginary_part():
    c = template_card("contact_nsi", {"eps_u_ee": 0.1, "eps_d_mt": 0.02, "eps_d_mt_im": 0.03}).coefficients(Q2)
    np.testing.assert_array_equal(c.values["V", "u"][0, 0], 0.1)
    np.testing.assert_array_equal(c.values["V", "d"][2, 1], 0.02 - 0.03j)


# Template points whose couplings are all zero; other parameters (masses, charges) are kept nonzero.
SM_LIMIT_POINTS = {
    "zprime_universal": {"mass_mev": 30.0},
    "zprime_u1": {"mass_mev": 10.0, "q_nu_e": -1.0, "q_nu_mu": -1.0, "q_nu_tau": -1.0, "q_u": 1 / 3, "q_d": 1 / 3},
    "zprime_lmu_ltau": {"mass_mev": 10.0},
    "scalar_universal": {"mass_mev": 10.0},
    "contact_nsi": {},
}


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_template_sm_limit(name):
    params = SM_LIMIT_POINTS[name] | {coupling: 0.0 for coupling in TEMPLATES[name].couplings}
    card = template_card(name, params)
    e_nu = np.linspace(10.0, 52.0, 8)[:, None]
    t = np.linspace(0.004, 0.026, 12)
    q2 = CS133.q2(t)
    for flavor in FLAVOR_ROW:
        new = dsigma_dt(e_nu, t, flavor, CS133, sm(q2) + card.coefficients(q2))
        standard = dsigma_dt(e_nu, t, flavor, CS133, sm(q2))
        np.testing.assert_allclose(new, standard, rtol=1e-9, atol=0)


def test_card_schema_file_is_current():
    """Regenerate with: uv run python -c "import json; from nuagent.physics.modelcard import ModelCard;
    print(json.dumps(ModelCard.model_json_schema(), indent=2, ensure_ascii=False))" > src/nuagent/context/card_schema.json"""
    committed = json.loads((CONTEXT / "card_schema.json").read_text())
    assert committed == ModelCard.model_json_schema()


def example_blocks(path: Path, language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", path.read_text(), flags=re.DOTALL)


def example_plugins(path: Path, tmp_path: Path) -> list:
    plugins = []
    for i, source in enumerate(example_blocks(path, "python")):
        file = tmp_path / f"{path.stem}_{i}.py"
        file.write_text(source)
        plugins.append(load_coefficient_plugin(file))
    return plugins


@pytest.mark.parametrize(
    "stem",
    [
        "01_zprime_universal",
        "02_zprime_b_minus_l",
        "03_zprime_lmu_ltau",
        "04_light_scalar",
        "05_contact_nsi_offdiagonal",
    ],
)
def test_example_cards_and_plugins_agree(stem, tmp_path):
    path = CONTEXT / "examples" / f"{stem}.md"
    cards = [ModelCard.model_validate_json(block) for block in example_blocks(path, "json")]
    plugins = example_plugins(path, tmp_path)
    assert len(cards) == len(plugins) > 0
    for card, plugin in zip(cards, plugins):
        expected = card.coefficients(Q2)
        produced = plugin.coefficients(Q2, plugin.parameters)
        for key, value in expected.values.items():
            np.testing.assert_allclose(produced.values[key], value, rtol=1e-6, atol=1e-300, err_msg=f"{stem} {key}")


def test_example_06_reduces_to_example_03b(tmp_path):
    running = example_plugins(CONTEXT / "examples" / "06_running_kinetic_mixing.md", tmp_path)[0]
    constant = example_plugins(CONTEXT / "examples" / "03_zprime_lmu_ltau.md", tmp_path)[1]
    tiny = np.array([1e-6])
    np.testing.assert_allclose(
        running.coefficients(tiny, running.parameters).values["V", "u"],
        constant.coefficients(tiny, constant.parameters).values["V", "u"],
        rtol=1e-9,
    )


def significant(x: float, digits: int) -> float:
    return float(f"{x:.{digits}g}")


A_CS = 133
Q_SM_TREE = 55 * (0.5 - 2 * 0.2312) + 78 * (-0.5)


def test_conventions_numbers():
    q_sm = vector_charge(sm(Q2), 55, 78)[0, 0, 0].real
    assert round(q_sm, 2) == -38.28
    assert round(Q_SM_TREE, 2) == -36.93
    assert round(0.5 - 4 / 3 * SIN2_THETA_W, 4) == 0.1917
    assert round(-0.5 + 2 / 3 * SIN2_THETA_W, 4) == -0.3459
    c_p = M_P * sum(SCALAR_CONTENT_P[q] / QUARK_MASS_MEV[q] for q in ("u", "d"))
    c_n = M_N * sum(SCALAR_CONTENT_N[q] / QUARK_MASS_MEV[q] for q in ("u", "d"))
    assert (round(c_p, 2), round(c_n, 2), round(c_p, 1), round(c_n, 1)) == (17.08, 17.09, 17.1, 17.1)
    nucleon = np.array(
        [
            [M_P * SCALAR_CONTENT_P[q] / QUARK_MASS_MEV[q] for q in ("u", "d")],
            [M_N * SCALAR_CONTENT_N[q] / QUARK_MASS_MEV[q] for q in ("u", "d")],
        ]
    )
    g_u, g_d = np.linalg.solve(nucleon, [1.0, 1.0])
    assert (round(g_u, 6), round(g_d, 6)) == (0.058970, 0.058122)


def test_example_01_numbers():
    c = template_card("zprime_universal", {"g_nu": 1e-4, "g_q": 1e-4, "mass_mev": 30.0}).coefficients(Q2)
    for quark in ("u", "d"):
        np.testing.assert_allclose(c.values["V", quark][0, 0].real, [2.8355e-01, 1.3139e-01, 8.5506e-02], rtol=5e-5)
    shift = vector_charge(c, 55, 78)[0, 0, 0].real
    assert round(shift, 1) == 113.1
    assert round(shift + Q_SM_TREE, 1) == 76.2
    assert round(((shift + Q_SM_TREE) / Q_SM_TREE) ** 2, 1) == 4.3
    # Delta Q is proportional to g'^2: the coupling with Delta Q = -Q_SM, and with Delta Q = -2 Q_SM.
    q_sm = vector_charge(sm(Q2), 55, 78)[0, 0, 0].real
    cancel = [1e-4 * np.sqrt(-charge / shift) for charge in (Q_SM_TREE, q_sm)]
    assert [significant(g, 3) for g in cancel] == [5.71e-05, 5.82e-05]
    assert [significant(np.sqrt(2) * g, 3) for g in cancel] == [8.08e-05, 8.23e-05]


def test_example_02_numbers():
    card = u1_card(1e-4, 10.0, q_nu_e=-1, q_nu_mu=-1, q_nu_tau=-1, q_u=1 / 3, q_d=1 / 3)
    c = card.coefficients(Q2)
    for quark in ("u", "d"):
        for alpha in range(3):
            np.testing.assert_allclose(
                c.values["V", quark][alpha, alpha].real, [-1.5103e-01, -5.2984e-02, -3.2127e-02], rtol=5e-5
            )
    assert round(vector_charge(c, 55, 78)[0, 0, 0].real, 3) == -60.262


def test_example_03_numbers():
    cases = {
        "A": (
            {"g_prime": 5e-5, "mass_mev": 20.0, "kinetic_mixing": 2e-3},
            [7.4762e-01, 2.9767e-01, 1.8583e-01],
            [-3.7381e-01, -1.4883e-01, -9.2914e-02],
        ),
        "B": (
            {"g_prime": 1e-4, "mass_mev": 10.0},
            [1.3218e-03, 4.6370e-04, 2.8117e-04],
            [-6.6089e-04, -2.3185e-04, -1.4058e-04],
        ),
    }
    for params, up, down in cases.values():
        c = template_card("zprime_lmu_ltau", params).coefficients(Q2)
        for quark, row in (("u", up), ("d", down)):
            np.testing.assert_allclose(c.values["V", quark][1, 1].real, row, rtol=5e-5)
            np.testing.assert_allclose(c.values["V", quark][2, 2].real, -np.array(row), rtol=5e-5)
            assert np.count_nonzero(c.values["V", quark][0, 0]) == 0
    assert significant(KAPPA_LMU_LTAU * 1e-4, 7) == 1.444176e-06


def test_example_04_numbers():
    c = template_card("scalar_universal", {"g_nu": 1e-5, "g_q": 1e-5, "mass_mev": 10.0}).coefficients(Q2)
    for quark in ("u", "d"):
        np.testing.assert_allclose(c.values["S", quark][0, 0].real, [4.5309e-03, 1.5895e-03, 9.6382e-04], rtol=5e-5)
    np.testing.assert_allclose(scalar_charge(c, 55, 78)[0, 0].real, [1.6980e-10, 5.9570e-11, 3.6121e-11], rtol=5e-5)


def test_example_05_numbers():
    c = template_card("contact_nsi", {"eps_u_ee": 0.1, "eps_u_em": 0.05}).coefficients(Q2)
    charge = vector_charge(c, 55, 78)[:, :, 0]
    assert round(charge[0, 0].real, 1) == 18.8
    assert round(charge[0, 0].real + Q_SM_TREE, 2) == -18.13
    assert round(abs(charge[0, 1]), 1) == round(abs(charge[1, 0]), 1) == 9.4


def test_example_06_numbers(tmp_path):
    running = example_plugins(CONTEXT / "examples" / "06_running_kinetic_mixing.md", tmp_path)[0]
    constant = example_plugins(CONTEXT / "examples" / "03_zprime_lmu_ltau.md", tmp_path)[1]
    c = running.coefficients(Q2, running.parameters)
    np.testing.assert_allclose(c.values["V", "u"][1, 1].real, [1.3167e-03, 4.5844e-04, 2.7597e-04], rtol=5e-5)
    np.testing.assert_allclose(c.values["V", "d"][1, 1].real, [-6.5833e-04, -2.2922e-04, -1.3798e-04], rtol=5e-5)
    ratio = c.values["V", "u"][1, 1].real / constant.coefficients(Q2, constant.parameters).values["V", "u"][1, 1].real
    np.testing.assert_array_equal(np.round(ratio, 5), [0.99613, 0.98866, 0.98150])
