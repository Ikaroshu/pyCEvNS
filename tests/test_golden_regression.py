"""Golden regression on the legacy README configuration.

COHERENT CsI detector with the legacy efficiency curve, COHERENT flux, 4466 kg day,
Helm form factor with radius 5.5 fm, Standard Model couplings, six flavors summed.
Production folding is compared with the converged legacy reference.
"""

from dataclasses import replace

import numpy as np
import pytest
from pyCEvNS.detectors import Detector
from pyCEvNS.events import HelmFormFactor, NeutrinoNucleusElasticVector, eff_coherent
from pyCEvNS.flux import NeutrinoFluxFactory
from pyCEvNS.parameters import NSIparameters

EXPOSURE_KG_DAY = 4466
FLAVORS = ("e", "ebar", "mu", "mubar", "tau", "taubar")
CSI_PE_PER_MEV = 0.0878 * 13.348 * 1000  # quenching factor times PE per keVee, as in legacy eff_coherent

# Legacy events() over the whole threshold window; quad under-resolves the 5 and 6 PE efficiency steps.
LEGACY_SINGLE_WINDOW_TOTAL = 135.660528845678
# Sum of legacy events() over 80 equal sub-windows plus edges at the 5 and 6 PE steps.
LEGACY_CONVERGED_TOTAL = 135.9330188425705


@pytest.fixture(scope="module")
def legacy_readme_total():
    detector = Detector("csi", efficiency=eff_coherent)
    flux = NeutrinoFluxFactory().get("coherent")
    interaction = NeutrinoNucleusElasticVector(NSIparameters(), HelmFormFactor(5.5))

    def total(er_min_mev: float, er_max_mev: float) -> float:
        return sum(
            interaction.events(er_min_mev, er_max_mev, flavor, flux, detector, EXPOSURE_KG_DAY)
            for flavor in FLAVORS
        )

    return detector, total


def test_legacy_single_window_total(legacy_readme_total):
    detector, total = legacy_readme_total
    assert total(detector.er_min, detector.er_max) == pytest.approx(LEGACY_SINGLE_WINDOW_TOTAL, rel=1e-9)


def test_legacy_converged_total(legacy_readme_total):
    detector, total = legacy_readme_total
    edges = np.unique(
        np.concatenate(
            [
                np.linspace(detector.er_min, detector.er_max, 81),
                [5 / CSI_PE_PER_MEV, 6 / CSI_PE_PER_MEV],
            ]
        )
    )
    converged = sum(total(a, b) for a, b in zip(edges[:-1], edges[1:]))
    assert converged == pytest.approx(LEGACY_CONVERGED_TOTAL, rel=1e-9)


def test_production_readme_total():
    from nuagent.experiments.registry import get
    from nuagent.rates import predicted_counts

    experiment = replace(get("coherent_csi_2017"), bin_edges=np.array([4.25e-3, 26e-3]), bin_unit="MeV")
    assert predicted_counts(experiment).sum() == pytest.approx(LEGACY_CONVERGED_TOTAL, rel=1e-4)
