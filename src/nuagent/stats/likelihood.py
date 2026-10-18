"""Binned Poisson likelihood with profiled signal and background normalizations."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from nuagent.experiments.registry import Experiment
from nuagent.rates import predicted_counts


@dataclass(frozen=True)
class Profile:
    """Minimum of −2 ln L over the nuisances and the nuisance values at the minimum."""

    neg2_log_likelihood: float
    alpha: float
    beta: tuple[float, ...]


def neg2_log_likelihood(experiment: Experiment, signal: np.ndarray, alpha: float, beta: np.ndarray) -> float:
    """−2 ln L for signal normalization `alpha` and background normalizations `beta`, in background order."""
    observed = experiment.observed
    expected = (1 + alpha) * signal + sum((1 + b) * background.counts for b, background in zip(beta, experiment.backgrounds))
    log_term = observed * np.log(np.where(observed > 0, observed / expected, 1.0))
    priors = (alpha / experiment.signal_normalization_prior) ** 2 + sum(
        (b / background.normalization_prior) ** 2 for b, background in zip(beta, experiment.backgrounds)
    )
    return float(2 * np.sum(expected - observed + log_term) + priors)


def profiled(experiment: Experiment, signal: np.ndarray) -> Profile:
    """Minimize −2 ln L over alpha and beta from zero, with every normalization factor kept positive.

    L-BFGS-B works on rescaled nuisances z_j = x_j / scale_j, scale_j = min(prior_j, observed total /
    component total). A signal far above the data otherwise leaves the problem so badly scaled that
    the optimizer stops at a non-minimum (for example with every background removed) and reports
    success. A component with zero total keeps its prior as the scale.
    """
    components = np.vstack([signal, *(background.counts for background in experiment.backgrounds)])
    priors = np.array([experiment.signal_normalization_prior, *(b.normalization_prior for b in experiment.backgrounds)])
    totals = components.sum(axis=1)
    data_scale = np.where(totals > 0, experiment.observed.sum() / np.where(totals > 0, totals, 1.0), np.inf)
    scale = np.minimum(priors, data_scale)

    def gradient(x: np.ndarray) -> np.ndarray:
        return 2 * components @ (1 - experiment.observed / ((1 + x) @ components)) + 2 * x / priors**2

    result = minimize(
        lambda z: neg2_log_likelihood(experiment, signal, z[0] * scale[0], z[1:] * scale[1:]),
        np.zeros(len(priors)),
        jac=lambda z: gradient(z * scale) * scale,
        method="L-BFGS-B",
        bounds=[((-1 + 1e-9) / factor, None) for factor in scale],
    )
    if not result.success:
        raise RuntimeError(f"likelihood profile did not converge: {result.message}")
    x = result.x * scale
    return Profile(float(result.fun), float(x[0]), tuple(float(b) for b in x[1:]))


def profile_record(profile: Profile, experiment: Experiment) -> dict:
    """The profile as JSON-ready values, with the background normalizations keyed by background name."""
    return {
        "neg2_log_likelihood": profile.neg2_log_likelihood,
        "alpha": profile.alpha,
        "beta": {background.name: b for background, b in zip(experiment.backgrounds, profile.beta)},
    }


def delta(experiment: Experiment, signal: np.ndarray) -> float:
    """Δ = profiled(new physics) − profiled(SM) on the experiment's observed counts; positive Δ favors the SM."""
    return profiled(experiment, signal).neg2_log_likelihood - profiled(experiment, predicted_counts(experiment)).neg2_log_likelihood
