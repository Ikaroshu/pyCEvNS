"""Spec Verification test 4: the new scalar term against legacy `NeutrinoNucleusElasticScalar`.

Legacy is built with `form_factor_square=HelmFormFactor()` (4.7 fm; its default is no form factor on the
scalar term), and its scalar-only rate is `scalar.rates - scalar.sm_interaction.rates`. Legacy `cu`, `cd`
are the products g_nu g_q of a flavor-universal mediator. The new side folds `xsec.dsigma_dt` with the
scalar card's coefficients alone (no SM), pointwise at 50 recoil energies of the CsI threshold range.
"""

import numpy as np
import pytest
from legacy_folding import detector_isotopes, events_per_mev, fold_coefficients
from pyCEvNS.detectors import Detector
from pyCEvNS.events import HelmFormFactor, NeutrinoNucleusElasticScalar, eff_coherent
from pyCEvNS.flux import NeutrinoFluxFactory

from nuagent.physics.modelcard import template_card

# (g_nu, g_u, g_d, m_phi in MeV), each giving a scalar rate of order the SM rate; legacy cu = g_nu g_u, cd = g_nu g_d.
# The last point has g_u != g_d so that a mix-up of the u and d scalar charges is caught.
SCALAR_POINTS = [
    (3e-5, 3e-5, 3e-5, 5.0),
    (2.5e-5, 2.5e-5, 2.5e-5, 30.0),
    (1.3e-4, 1.3e-4, 1.3e-4, 300.0),
    (3e-5, 1e-5, 5e-5, 30.0),
]


@pytest.mark.parametrize(("g_nu", "g_u", "g_d", "mass_mev"), SCALAR_POINTS)
def test_scalar_rates_match_legacy(g_nu, g_u, g_d, mass_mev):
    detector = Detector("csi", efficiency=eff_coherent)
    flux = NeutrinoFluxFactory().get("coherent")
    isotopes = detector_isotopes(detector, 4.7)
    legacy = NeutrinoNucleusElasticScalar(g_nu * g_u, g_nu * g_d, mass_mev, form_factor_square=HelmFormFactor())
    card = template_card("scalar_universal", {"g_nu": g_nu, "g_u": g_u, "g_d": g_d, "mass_mev": mass_mev})
    recoil = np.linspace(detector.er_min, detector.er_max, 50)
    for flavor in ("e", "mu", "mubar"):
        expected = np.array(
            [
                events_per_mev(
                    legacy.rates(t, flavor, flux, detector) - legacy.sm_interaction.rates(t, flavor, flux, detector),
                    detector,
                )
                for t in recoil
            ]
        )
        new = np.array(
            [
                events_per_mev(fold_coefficients(t, flavor, flux, detector, isotopes, card.coefficients), detector)
                for t in recoil
            ]
        )
        assert np.any(expected > 0)
        np.testing.assert_allclose(new, expected, rtol=1e-4, atol=0, err_msg=flavor)
