"""Likelihood, Δ versus the SM, Asimov data and template scans."""

import dataclasses

import numpy as np
import pytest

from nuagent.experiments.registry import get
from nuagent.physics.modelcard import template_card
from nuagent.rates import predicted_counts
from nuagent.stats import asimov, likelihood


@pytest.fixture(scope="module")
def experiment():
    return get("coherent_csi_2017")


@pytest.fixture(scope="module")
def standard(experiment):
    return predicted_counts(experiment)


@pytest.mark.parametrize("observed", [0.0, 7.0])
def test_one_bin_without_nuisances_matches_poisson(experiment, observed):
    one_bin = dataclasses.replace(experiment, observed=np.array([observed]), backgrounds=())
    signal = np.array([4.5])
    poisson = 2 * (4.5 - observed + (observed * np.log(observed / 4.5) if observed else 0.0))
    assert likelihood.neg2_log_likelihood(one_bin, signal, 0.0, ()) == pytest.approx(poisson, rel=1e-12)
    prior = (0.1 / experiment.signal_normalization_prior) ** 2
    shifted = 2 * (4.95 - observed + (observed * np.log(observed / 4.95) if observed else 0.0)) + prior
    assert likelihood.neg2_log_likelihood(one_bin, signal, 0.1, ()) == pytest.approx(shifted, rel=1e-12)


def test_profile_returns_to_zero_on_expected_data(experiment, standard):
    profile = likelihood.profiled(asimov.asimov_experiment(experiment, standard), standard)
    assert profile.neg2_log_likelihood == pytest.approx(0, abs=1e-9)
    np.testing.assert_allclose([profile.alpha, *profile.beta], 0, atol=1e-6)


def test_profile_on_observed_data(experiment, standard):
    profile = likelihood.profiled(experiment, standard)
    nuisances = np.array([profile.alpha, *profile.beta])
    assert profile.neg2_log_likelihood == pytest.approx(
        likelihood.neg2_log_likelihood(experiment, standard, nuisances[0], nuisances[1:]), rel=1e-12
    )
    assert profile.neg2_log_likelihood < likelihood.neg2_log_likelihood(experiment, standard, 0.0, (0.0, 0.0))


def test_sm_against_itself_is_zero(experiment, standard):
    assert likelihood.delta(experiment, standard) == 0
    assert asimov.delta(experiment, standard, "sm") == 0


def test_asimov_truth_at_tested_point_is_not_positive(experiment, standard):
    signal = predicted_counts(experiment, template_card("zprime_universal", {"g_nu": 3e-5, "g_q": 3e-5, "mass_mev": 30.0}).coefficients)
    value = asimov.delta(experiment, signal, signal)
    sm_profile = likelihood.profiled(asimov.asimov_experiment(experiment, signal), standard)
    assert value <= 0
    assert value == pytest.approx(-sm_profile.neg2_log_likelihood, abs=1e-9)
    assert asimov.delta(experiment, signal, "sm") > 0


def test_signal_far_above_sm_gives_finite_delta(experiment, standard):
    value = likelihood.delta(experiment, 100 * standard)
    assert np.isfinite(value) and value > 0


def test_profile_converges_where_signal_dwarfs_data(experiment):
    point = {"g_prime": 0.023713737056616554, "mass_mev": 4.641588833612778}
    signal = predicted_counts(experiment, template_card("zprime_lmu_ltau", point).coefficients)
    profile = likelihood.profiled(experiment, signal)
    assert signal.sum() > 1e5
    assert -1 < profile.alpha < -0.999


def test_scan_grid_and_reference(experiment, standard):
    grid = {"g_prime": np.array([0.0, 1e-3]), "mass_mev": np.array([10.0, 100.0])}
    observed = asimov.scan(experiment, "zprime_lmu_ltau", grid, {}, data="observed")
    assert observed.delta.shape == (2, 2)
    np.testing.assert_allclose(observed.delta[0], 0, atol=1e-9)
    signal = predicted_counts(experiment, template_card("zprime_lmu_ltau", {"g_prime": 1e-3, "mass_mev": 100.0}).coefficients)
    assert observed.delta[1, 1] == pytest.approx(likelihood.delta(experiment, signal), abs=1e-9)
    expected = asimov.scan(experiment, "zprime_lmu_ltau", {"g_prime": [1e-3]}, {"mass_mev": 100.0})
    assert expected.delta.shape == (1,)
    assert expected.delta[0] == pytest.approx(asimov.delta(experiment, signal, "sm"), abs=1e-9)


def test_scan_rejects_unknown_parameter(experiment):
    with pytest.raises(ValueError, match="valid parameters: g_prime, mass_mev, kinetic_mixing"):
        asimov.scan(experiment, "zprime_lmu_ltau", {"g_primes": [1e-3]}, {"mass_mev": 10.0})
