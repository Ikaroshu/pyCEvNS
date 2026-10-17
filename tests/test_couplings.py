"""Coefficients container, SM coefficients, and spec Verification test 3 (SM Z anchor)."""

import numpy as np
import pytest

from nuagent.physics.couplings import GF, M_N, M_P, SIN2_THETA_W, SM_VECTOR, SQRT2, Coefficients, sm
from nuagent.physics.formfactor import helm_form_factor_squared
from nuagent.physics.modelcard import ModelCard, template_card
from nuagent.physics.xsec import Isotope, dsigma_dt

Q2 = np.array([1238.0, 3714.0, 6190.0])


def test_set_fills_conjugate_and_overwrites():
    c = Coefficients.zeros(Q2)
    c.set("V", "u", "em", 0.03 + 0.04j)
    c.set("V", "u", "em", 0.05 + 0.04j)
    c.set("S", "d", "tt", Q2 / 1000)
    assert c.values["V", "u"].shape == (3, 3, 3)
    np.testing.assert_array_equal(c.values["V", "u"][0, 1], 0.05 + 0.04j)
    np.testing.assert_array_equal(c.values["V", "u"][1, 0], 0.05 - 0.04j)
    np.testing.assert_array_equal(c.values["S", "d"][2, 2], Q2 / 1000)
    assert np.count_nonzero(c.values["V", "d"]) == np.count_nonzero(c.values["S", "u"]) == 0


def test_set_rejects_complex_diagonal():
    with pytest.raises(ValueError, match="real"):
        Coefficients.zeros(Q2).set("V", "u", "ee", 0.1j)


def test_addition_is_elementwise():
    a = Coefficients.zeros(Q2)
    a.set("V", "u", "ee", 0.1)
    total = sm(Q2) + a
    assert total.values["V", "u"][0, 0, 0] == pytest.approx(SM_VECTOR["u"] + 0.1)
    assert total.values["V", "u"][1, 1, 0] == pytest.approx(SM_VECTOR["u"])
    assert a.values["V", "u"][1, 1, 0] == 0


def test_sm_reproduces_legacy_nucleon_couplings():
    # Legacy writes g_V^p = rho (1/2 - 2 kappa s^2) + 2 l_uL + 2 l_uR + l_dL + l_dR, halved, and likewise for n.
    rho, kappa, lul, ldl, ldr = 1.0086, 0.9978, -0.0031, -0.0025, 7.5e-5
    lur = ldr / 2
    proton = rho * (0.5 - 2 * kappa * 0.2312) + 2 * lul + 2 * lur + ldl + ldr
    neutron = -0.5 * rho + lul + lur + 2 * ldl + 2 * ldr
    assert 2 * SM_VECTOR["u"] + SM_VECTOR["d"] == pytest.approx(proton, rel=1e-14)
    assert SM_VECTOR["u"] + 2 * SM_VECTOR["d"] == pytest.approx(neutron, rel=1e-14)


def test_sm_z_anchor():
    mz_mev = 91187.6
    cos2 = 1 - SIN2_THETA_W
    mw_mev = mz_mev * np.sqrt(cos2)
    g = np.sqrt(4 * SQRT2 * GF * mw_mev**2)  # G_F / sqrt2 = g^2 / (8 M_W^2)
    tree = {"u": 0.5 - 4 / 3 * SIN2_THETA_W, "d": -0.5 + 2 / 3 * SIN2_THETA_W}
    card = template_card(
        "zprime_universal",
        {
            "g_nu": g / (2 * np.sqrt(cos2)),
            "g_u": g * tree["u"] / (2 * np.sqrt(cos2)),
            "g_d": g * tree["d"] / (2 * np.sqrt(cos2)),
            "mass_mev": mz_mev,
        },
    )
    c = card.coefficients(np.array([0.0]))
    for quark in ("u", "d"):
        np.testing.assert_allclose(np.diagonal(c.values["V", quark][:, :, 0]), tree[quark], rtol=1e-6)


def test_scalar_cross_section_with_complex_off_diagonal():
    a, b, g_u, g_d, mass = 2e-5, -3e-5, 1e-5, 4e-5, 20.0
    card = ModelCard.model_validate(
        {
            "mediators": [
                {
                    "structure": "S",
                    "mass_mev": mass,
                    "nu_couplings": {"em": [a, b]},
                    "quark_couplings": {"u": g_u, "d": g_d},
                }
            ]
        }
    )
    isotope = Isotope(55, 78, 123800.0, 5.5)
    t = np.array([0.005, 0.015])
    e_nu = np.array([30.0, 45.0])
    q2 = isotope.q2(t)
    # Legacy nucleon scalar charges: c_N = m_N sum_q g_q f_q^N / m_q.
    c_p = M_P * (g_u * 0.0208 / 2.2 + g_d * 0.0411 / 4.7)
    c_n = M_N * (g_u * 0.0189 / 2.2 + g_d * 0.0451 / 4.7)
    charge = (a + 1j * b) * (55 * c_p + 78 * c_n) / (q2 + mass**2)
    expected = (
        isotope.mass_mev**2
        * t
        * np.abs(charge) ** 2
        * helm_form_factor_squared(np.sqrt(q2), 5.5)
        / (4 * np.pi * e_nu**2)
    )
    for flavor in ("e", "mu", "ebar", "mubar"):
        np.testing.assert_allclose(dsigma_dt(e_nu, t, flavor, isotope, card.coefficients(q2)), expected, rtol=1e-12)
    np.testing.assert_array_equal(dsigma_dt(e_nu, t, "tau", isotope, card.coefficients(q2)), 0)
