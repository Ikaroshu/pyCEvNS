"""Spec Verification test 6 at differential-rate and count levels; `check_plugin` lands with milestone M5b (issue #17)."""

import textwrap

import numpy as np
import pytest
from experiment_fixtures import events_per_mev
from pyCEvNS.detectors import Detector
from pyCEvNS.events import HelmFormFactor, NeutrinoNucleusElasticVector, eff_coherent
from pyCEvNS.flux import NeutrinoFluxFactory
from pyCEvNS.parameters import NSIparameters

from nuagent.experiments.registry import get
from nuagent.physics.modelcard import ModelCard
from nuagent.physics.plugin import load_coefficient_plugin, load_raw_plugin
from nuagent.rates import differential_rate, predicted_counts

COEFFICIENT_PLUGIN = """
import numpy as np
from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_nu": 1e-4, "g_nu_em_im": 3e-5, "g_u": -2e-5, "g_d": 4e-5, "mass_mev": 30.0, "y": 1e-5, "eps_d_mm": 0.05}
COUPLINGS = ("g_nu", "g_nu_em_im", "g_u", "g_d", "y", "eps_d_mm")

def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    prop = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        g_q = params[f"g_{quark}"]
        diagonal = params["g_nu"] * g_q * prop + (params["eps_d_mm"] if quark == "d" else 0.0)
        c.set("V", quark, "ee", params["g_nu"] * g_q * prop)
        c.set("V", quark, "mm", diagonal)
        c.set("V", quark, "em", 1j * params["g_nu_em_im"] * g_q * prop)
        c.set("S", quark, "tt", params["y"] ** 2 / (SQRT2 * GF * (q2 + 100.0)))
    return c
"""

EQUIVALENT_CARD = {
    "mediators": [
        {
            "structure": "V",
            "mass_mev": 30.0,
            "nu_couplings": {"ee": 1e-4, "mm": 1e-4, "em": [0.0, 3e-5]},
            "quark_couplings": {"u": -2e-5, "d": 4e-5},
        },
        {"structure": "S", "mass_mev": 10.0, "nu_couplings": {"tt": 1e-5}, "quark_couplings": {"u": 1e-5, "d": 1e-5}},
    ],
    "contacts": [{"fermion": "d", "epsilon": {"mm": 0.05}}],
}

RAW_SM_PLUGIN = """
from nuagent.physics.couplings import sm
from nuagent.physics.xsec import dsigma_dt as framework_dsigma_dt

PARAMETERS = {}
COUPLINGS = ()

def dsigma_dt(e_nu, t, flavor, isotope, params):
    return framework_dsigma_dt(e_nu, t, flavor, isotope, sm(isotope.q2(t)))
"""


@pytest.fixture(scope="module")
def setup():
    detector = Detector("csi", efficiency=eff_coherent)
    return detector, NeutrinoFluxFactory().get("coherent"), get("coherent_csi_2017")


def test_coefficient_plugin_matches_card(tmp_path, setup):
    path = tmp_path / "plugin.py"
    path.write_text(textwrap.dedent(COEFFICIENT_PLUGIN))
    plugin = load_coefficient_plugin(path)
    assert plugin.couplings == ("g_nu", "g_nu_em_im", "g_u", "g_d", "y", "eps_d_mm")
    card = ModelCard.model_validate(EQUIVALENT_CARD)
    q2 = np.linspace(100.0, 7000.0, 5)
    for key, value in card.coefficients(q2).values.items():
        np.testing.assert_allclose(
            plugin.coefficients(q2, plugin.parameters).values[key], value, rtol=1e-12, atol=1e-300
        )

    detector, _, experiment = setup
    for flavor in ("e", "mu", "mubar"):
        for t in np.linspace(detector.er_min, detector.er_max, 50):
            from_card = differential_rate(t, experiment.flux, experiment.detector, 4466, card.coefficients, flavor=flavor)
            from_plugin = differential_rate(
                t, experiment.flux, experiment.detector, 4466,
                lambda q: plugin.coefficients(q, plugin.parameters), flavor=flavor,
            )
            assert from_plugin == pytest.approx(from_card, rel=1e-12, abs=0)

    np.testing.assert_allclose(
        predicted_counts(experiment, lambda q: plugin.coefficients(q, plugin.parameters)),
        predicted_counts(experiment, card.coefficients), rtol=1e-12, atol=0,
    )


def test_raw_sm_plugin_gives_sm_rates(tmp_path, setup):
    path = tmp_path / "raw.py"
    path.write_text(textwrap.dedent(RAW_SM_PLUGIN))
    plugin = load_raw_plugin(path)
    detector, flux, experiment = setup
    legacy = NeutrinoNucleusElasticVector(NSIparameters(), HelmFormFactor(5.5))
    for flavor in ("e", "mu", "mubar"):
        recoil = np.linspace(detector.er_min, detector.er_max, 50)
        raw = [
            differential_rate(
                t, experiment.flux, experiment.detector, 4466, flavor=flavor,
                raw_cross_section=lambda e, t, f, iso: plugin.dsigma_dt(e, t, f, iso, plugin.parameters),
            )
            for t in recoil
        ]
        framework = [differential_rate(t, experiment.flux, experiment.detector, 4466, flavor=flavor) for t in recoil]
        expected = [legacy.rates(t, flavor, flux, detector) for t in recoil]
        np.testing.assert_array_equal(raw, framework)
        np.testing.assert_allclose(
            np.array(raw), events_per_mev(np.array(expected), detector), rtol=1e-4, atol=1e-6
        )

    np.testing.assert_allclose(
        predicted_counts(experiment, raw_cross_section=lambda e, t, f, iso: plugin.dsigma_dt(e, t, f, iso, plugin.parameters)),
        predicted_counts(experiment), rtol=1e-12, atol=0,
    )
