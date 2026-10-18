"""Matplotlib figure drawn by the Streamlit page."""

import numpy as np
from matplotlib.figure import Figure

from nuagent.experiments.registry import Experiment


def spectrum_figure(experiment: Experiment, signal: np.ndarray, sm_signal: np.ndarray) -> Figure:
    """New physics and SM, each plus the summed backgrounds, as steps over the bins; observed counts with √N errors."""
    background = sum(b.counts for b in experiment.backgrounds)
    edges = experiment.bin_edges
    centers = (edges[:-1] + edges[1:]) / 2
    figure = Figure(figsize=(7, 4.5))
    axes = figure.subplots()
    axes.stairs(signal + background, edges, label="new physics + background", color="C3", linewidth=1.8, baseline=None)
    axes.stairs(sm_signal + background, edges, label="SM + background", color="C0", linewidth=1.8, linestyle="--", baseline=None)
    axes.stairs(background, edges, label="background", color="0.5", linewidth=1, baseline=None)
    axes.errorbar(centers, experiment.observed, yerr=np.sqrt(experiment.observed), fmt="o", color="k", markersize=4, label="observed")
    axes.set_xlabel(experiment.bin_unit)
    axes.set_ylabel("counts per bin")
    axes.set_title(experiment.name)
    axes.legend()
    figure.tight_layout()
    return figure
