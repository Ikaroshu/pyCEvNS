"""Matplotlib figures drawn by the Streamlit page: the spectrum and the agent loop trajectory."""

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


def passed_points(history: list[dict]) -> list[dict]:
    """The agent-loop points that passed the plugin checks; only these have Δ."""
    return [entry for entry in history if entry["evaluations"]]


def delta_axis(axes) -> None:
    # Δ spans from about −0.1 near the best fit to tens far from it; symlog keeps both readable.
    axes.set_yscale("symlog", linthresh=1)
    axes.axhline(0, color="0.6", linewidth=0.8)


def delta_by_round_figure(history: list[dict], experiments: list[str]) -> Figure:
    """Δ of every passing point by round, round 0 included, with the best Δ so far per experiment."""
    points = passed_points(history)
    rounds = range(max(entry["round"] for entry in history) + 1)
    figure = Figure(figsize=(7, 3.6))
    axes = figure.subplots()
    for k, name in enumerate(experiments):
        axes.scatter([entry["round"] for entry in points], [entry["evaluations"][k]["delta"] for entry in points],
                     color=f"C{k}", s=22, alpha=0.75, label=f"Δ {name}")
        best, best_so_far = np.inf, []
        for number in rounds:
            best = min([best, *(entry["evaluations"][k]["delta"] for entry in points if entry["round"] == number)])
            best_so_far.append(best)
        axes.step(list(rounds), best_so_far, where="post", color=f"C{k}", linewidth=1.5, label=f"best so far, {name}")
    delta_axis(axes)
    axes.set_xticks(list(rounds))
    axes.set_xlabel("round (0: the translation agent's point)")
    axes.set_ylabel("Δ (symlog)")
    axes.legend(fontsize="small")
    figure.tight_layout()
    return figure


def parameter_figure(history: list[dict], names: list[str], experiments: list[str]) -> Figure:
    """One small panel per plugin parameter: Δ of every passing point against that parameter's value.

    A parameter axis is logarithmic when all its values are positive and span more than a factor 10, linear otherwise.
    """
    points = passed_points(history)
    columns = min(len(names), 3)
    rows = -(-len(names) // columns)
    figure = Figure(figsize=(7, 2.6 * rows))
    panels = np.atleast_1d(figure.subplots(rows, columns, squeeze=False).ravel())
    for axes, name in zip(panels, names):
        values = np.array([entry["parameters"][name] for entry in points], dtype=float)
        for k, experiment in enumerate(experiments):
            axes.scatter(values, [entry["evaluations"][k]["delta"] for entry in points], color=f"C{k}", s=16, alpha=0.75,
                         label=f"Δ {experiment}")
        if np.all(values > 0) and values.max() > 10 * values.min():
            axes.set_xscale("log")
        delta_axis(axes)
        axes.set_xlabel(name)
        axes.set_ylabel("Δ")
    for axes in panels[len(names):]:
        axes.set_visible(False)
    if len(experiments) > 1:
        panels[0].legend(fontsize="small")
    figure.tight_layout()
    return figure
