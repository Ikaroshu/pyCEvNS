"""Matplotlib figures drawn by the Streamlit page."""

from typing import Literal

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from nuagent.experiments.registry import Experiment
from nuagent.stats.asimov import Scan

DATA_LABELS = {"asimov": "expected", "observed": "observed"}


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


def contour_figure(scan: Scan, thresholds: list[float], data: Literal["asimov", "observed"], log_axes: tuple[bool, bool]) -> Figure:
    """Contours of Δ over a two-parameter scan, each labeled with its threshold and expected or observed."""
    (x_name, x), (y_name, y) = scan.axes.items()
    levels = sorted(thresholds)
    colors = [f"C{i}" for i in range(len(levels))]
    figure = Figure(figsize=(6.5, 5))
    axes = figure.subplots()
    axes.contour(x, y, scan.delta.T, levels=levels, colors=colors)
    axes.legend(
        [Line2D([], [], color=color) for color in colors],
        [f"Δ = {level:g} ({DATA_LABELS[data]})" for level in levels],
        loc="lower right",
    )
    axes.set_xscale("log" if log_axes[0] else "linear")
    axes.set_yscale("log" if log_axes[1] else "linear")
    axes.set_xlabel(x_name)
    axes.set_ylabel(y_name)
    figure.tight_layout()
    return figure
