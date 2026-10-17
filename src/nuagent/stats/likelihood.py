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

    For signals far above the data the signal normalization runs almost to its bound, where the
    problem is ill-conditioned: finite-difference gradients or the default 20 line-search steps
    end in an ABNORMAL line search at points of the observed L_mu - L_tau scan.
    """
    components = np.vstack([signal, *(background.counts for background in experiment.backgrounds)])
    priors = np.array([experiment.signal_normalization_prior, *(b.normalization_prior for b in experiment.backgrounds)])

    def gradient(x: np.ndarray) -> np.ndarray:
        return 2 * components @ (1 - experiment.observed / ((1 + x) @ components)) + 2 * x / priors**2

    result = minimize(
        lambda x: neg2_log_likelihood(experiment, signal, x[0], x[1:]),
        np.zeros(len(priors)),
        jac=gradient,
        method="L-BFGS-B",
        options={"maxls": 100},
        bounds=[(-1 + 1e-9, None)] * len(priors),
    )
    if not result.success:
        raise RuntimeError(f"likelihood profile did not converge: {result.message}")
    return Profile(float(result.fun), float(result.x[0]), tuple(float(b) for b in result.x[1:]))


def delta(experiment: Experiment, signal: np.ndarray) -> float:
    """Δ = profiled(new physics) − profiled(SM) on the experiment's observed counts; positive Δ favors the SM."""
    return profiled(experiment, signal).neg2_log_likelihood - profiled(experiment, predicted_counts(experiment)).neg2_log_likelihood
