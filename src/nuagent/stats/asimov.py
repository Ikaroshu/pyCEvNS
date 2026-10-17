"""Asimov data sets and Δ scans over template parameters."""

import dataclasses
import itertools
from dataclasses import dataclass
from typing import Literal

import numpy as np

from nuagent.experiments.registry import Experiment
from nuagent.physics.modelcard import template_card
from nuagent.rates import predicted_counts
from nuagent.stats.likelihood import profiled


@dataclass(frozen=True)
class Scan:
    """Δ on the Cartesian product of `axes`; `delta` has one dimension per axis, in axis order."""

    axes: dict[str, np.ndarray]
    delta: np.ndarray


def asimov_experiment(experiment: Experiment, truth_signal: np.ndarray) -> Experiment:
    """The experiment with observed counts replaced by the expectation under the truth signal, nuisances at zero."""
    observed = truth_signal + sum(background.counts for background in experiment.backgrounds)
    return dataclasses.replace(experiment, observed=observed)


def delta(experiment: Experiment, signal: np.ndarray, truth: Literal["sm"] | np.ndarray) -> float:
    """Δ = profiled(signal) − profiled(SM) on Asimov data of the truth signal, or of the SM for `"sm"`."""
    standard = predicted_counts(experiment)
    asimov = asimov_experiment(experiment, standard if isinstance(truth, str) else truth)
    return profiled(asimov, signal).neg2_log_likelihood - profiled(asimov, standard).neg2_log_likelihood


def scan(
    experiment: Experiment,
    template: str,
    parameter_grid: dict[str, np.ndarray],
    fixed: dict[str, float],
    data: Literal["asimov", "observed"] = "asimov",
    truth: Literal["sm"] | np.ndarray = "sm",
) -> Scan:
    """Δ of a template over the Cartesian product of `parameter_grid`, other parameters from `fixed`.

    `data="observed"` uses the observed counts; `data="asimov"` uses Asimov data of `truth`.
    """
    standard = predicted_counts(experiment)
    if data == "asimov":
        experiment = asimov_experiment(experiment, standard if isinstance(truth, str) else truth)
    reference = profiled(experiment, standard).neg2_log_likelihood
    axes = {name: np.asarray(values, dtype=float) for name, values in parameter_grid.items()}
    deltas = [
        profiled(experiment, predicted_counts(experiment, template_card(template, fixed | point).coefficients)).neg2_log_likelihood
        - reference
        for point in (
            {name: float(value) for name, value in zip(axes, combination)}
            for combination in itertools.product(*axes.values())
        )
    ]
    return Scan(axes, np.reshape(deltas, tuple(len(axis) for axis in axes.values())))
