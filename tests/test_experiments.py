"""Release conversion, registry provenance and the production folding contract."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nuagent.experiments import registry
from nuagent.experiments.data.convert_coherent_csi import convert
from nuagent.experiments.detector import CSI_PE_PER_MEV, Detector, csi_efficiency_pe
from nuagent.experiments.flux import NeutrinoFlux, coherent_flux
from nuagent.physics.couplings import HBAR_C_MEV_M
from nuagent.physics.modelcard import TEMPLATES, template_card
from nuagent.physics.xsec import Isotope
from nuagent.rates import predicted_counts

DATA = Path(registry.__file__).parent / "data"


def test_registry_data_and_provenance():
    assert registry.names() == ("coherent_csi_2017",)
    experiment = registry.get("coherent_csi_2017")
    assert experiment.data_release == "10.5281/zenodo.1228631"
    assert "https://doi.org/10.5281/zenodo.1228631" in experiment.citations
    np.testing.assert_array_equal(experiment.bin_edges, np.arange(6, 32, 2))
    assert np.all(np.diff(experiment.bin_edges) > 0)
    np.testing.assert_array_equal(experiment.observed, [50, 103, 137, 119, 94, 81, 56, 41, 43, 37, 50, 44])
    np.testing.assert_array_equal(experiment.backgrounds[0].counts, [41, 90, 111, 87, 66, 54, 57, 52, 39, 33, 43, 30])
    assert experiment.observed.sum() == 855
    assert experiment.backgrounds[0].counts.sum() == 703
    assert experiment.backgrounds[0].normalization_prior == pytest.approx(1 / np.sqrt(703))
    assert experiment.backgrounds[1].counts.sum() == pytest.approx(5.419341500706415, rel=1e-14)
    assert experiment.backgrounds[1].normalization_prior == 0.25
    assert experiment.signal_normalization_prior == 0.28
    for background in experiment.backgrounds:
        assert background.counts.shape == experiment.observed.shape == (12,)
    assert experiment.time_window_us == (0, 12)
    assert experiment.exposure_kg_day == 4466
    assert experiment.detector.mass_kg == 14.6
    assert experiment.detector.quenching_factor == 0.0878
    assert experiment.detector.light_yield_pe_per_kev_ee == 13.348
    assert [isotope.helm_radius_fm for isotope in experiment.detector.isotopes] == [5.5, 5.5]
    np.testing.assert_allclose(experiment.recoil_edges_mev(), experiment.bin_edges / CSI_PE_PER_MEV)
    header = (DATA / "coherent_csi_2017.csv").read_text().splitlines()[:9]
    for text in ("10.5281/zenodo.1228631", "2026-10-17", "data_coincidence_beamOn.txt", "data_anticoincidence_beamOn.txt", "promptPDF.txt", "coherent_parameters.yaml", "0 <= time < 12 us", "6 <= PE < 30"):
        assert text in "\n".join(header)


def test_release_conversion_is_reproducible():
    produced = convert(DATA / "coherent_csi_release")
    committed = np.loadtxt(DATA / "coherent_csi_2017.csv", delimiter=",")
    np.testing.assert_allclose(produced, committed, rtol=1e-15, atol=0)
    # One bin checked directly from the original prompt distribution, before rebinning.
    raw = np.loadtxt(DATA / "coherent_csi_release" / "promptPDF.txt", delimiter=",")
    selected = raw[(raw[:, 0] >= 6) & (raw[:, 0] < 8)]
    # Keep the independent arithmetic anchored to the release rows, not copied constants.
    expected_first = sum(count * 0.6655 / (1 + np.exp(-0.4942 * (pe - 10.8507))) * 7.47594 for pe, count in selected)
    assert committed[0, 4] == pytest.approx(expected_first, rel=1e-14)


def test_unknown_experiment_names_valid_choices():
    with pytest.raises(ValueError, match="valid experiments: coherent_csi_2017"):
        registry.get("csi")


def test_efficiency_steps():
    pe = np.array([4.9, 5.0, 5.9, 6.0])
    logistic = 0.6655 / (1 + np.exp(-0.4942 * (pe - 10.8507)))
    np.testing.assert_allclose(csi_efficiency_pe(pe), logistic * [0, 0.5, 0.5, 1])


def test_weighted_flux_interpolates_edges_and_adds_exact_lines():
    flux = NeutrinoFlux(np.array([1., 2., 4.]), {"e": np.array([2., 3., 1.])}, {"e": ((2.5, 3.),)}, 1.)
    normalization = (100 * HBAR_C_MEV_M) ** 2
    # Weighted heights for E^2 are [2,12,16]; interpolated edges at 1.5,3 are 7,14.
    assert flux.integrate(1.5, 3., "e", np.square) / normalization == pytest.approx((7 + 12) * .5 / 2 + (12 + 14) / 2 + 3 * 2.5**2)
    assert flux.integrate(1.25, 1.75, "e", np.square) / normalization == pytest.approx((4.5 + 9.5) * .5 / 2)
    assert flux.integrate(4.5, 5., "e", np.square) == 0
    assert flux.integrate(1., 4., "tau", np.square) == 0
    assert flux.integrate(3., 1., "e", np.square) == 0
    lines_only = replace(flux, continuous={})
    assert lines_only.integrate(2.5, 4., "e", np.square) == 0
    assert lines_only.integrate(1., 2.5, "e", np.square) / normalization == pytest.approx(3 * 2.5**2)


def test_coherent_flux_preserves_legacy_tabulation():
    flux = coherent_flux()
    energy = np.linspace(.001, 52, 100)
    np.testing.assert_array_equal(flux.energy_mev, energy)
    np.testing.assert_array_equal(flux.continuous["e"], (3 * (energy / (2 / 3 * 52))**2 - 2 * (energy / (2 / 3 * 52))**3) / 29.25)
    np.testing.assert_array_equal(flux.continuous["mubar"], (3 * (energy / 52)**2 - 2 * (energy / 52)**3) / 26)
    assert flux.lines == {"mu": ((29., 1.),)}
    assert flux.normalization_cm2_s == 1.13e7


@pytest.fixture(scope="module")
def sm_counts():
    return predicted_counts(registry.get("coherent_csi_2017"))


def test_sm_signal_normalization(sm_counts):
    assert np.all(sm_counts > 0)
    assert 173 - 48 < sm_counts.sum() < 173 + 48


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_all_templates_sm_counts(name, sm_counts):
    parameters = {"mass_mev": 30.} if name != "contact_nsi" else {}
    if name == "zprime_u1":
        parameters |= {"q_nu_e": -1., "q_nu_mu": -1., "q_nu_tau": -1., "q_u": 1 / 3, "q_d": 1 / 3}
    parameters |= {coupling: 0. for coupling in TEMPLATES[name].couplings}
    card = template_card(name, parameters)
    np.testing.assert_allclose(predicted_counts(registry.get("coherent_csi_2017"), card.coefficients), sm_counts, rtol=1e-9, atol=0)


def test_line_endpoints_steps_and_exposure_have_analytic_counts():
    isotopes = (Isotope(1, 1, 1000., 1.), Isotope(1, 2, 2000., 1.))
    detector = Detector(isotopes, (.25, .75), lambda t: np.where(t < .15, .5, 1.), (.15,))
    flux = NeutrinoFlux(np.array([.001, 52.]), {}, {"mu": ((10., 2.),)}, 3.)
    experiment = replace(registry.get("coherent_csi_2017"), flux=flux, detector=detector, bin_unit="MeV", bin_edges=np.array([0., .4]))
    counts = predicted_counts(experiment, raw_cross_section=lambda e, t, f, i: np.ones_like(e))
    recoil_integral = sum(fraction * (.5 * min(.15, endpoint) + max(0., endpoint - .15)) for fraction, endpoint in ((.25, 200 / 1020), (.75, 200 / 2020)))
    expected = recoil_integral * 2 * 3 * (100 * HBAR_C_MEV_M)**2 * 4466 * detector.targets_per_kg() * 86400
    assert counts[0] == pytest.approx(expected, rel=1e-12)
