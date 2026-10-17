import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from nuagent.experiments import registry
from nuagent.experiments.registry import Experiment
from nuagent.physics.modelcard import TEMPLATES, ModelCard, template_card
from nuagent.physics.plugin import load_coefficient_plugin, load_raw_plugin
from nuagent.rates import predicted_counts
from nuagent.stats import asimov
from nuagent.stats.likelihood import Profile, profiled


def parameter_point(assignments: list[str]) -> dict[str, float]:
    return {key: float(value) for key, value in (assignment.split("=", 1) for assignment in assignments)}


def parameter_axis(specification: str) -> tuple[str, np.ndarray]:
    """Parse NAME=min:max:n[:log] into a linear, or with :log a geometric, grid."""
    name, bounds = specification.split("=", 1)
    lower, upper, count, *spacing = bounds.split(":")
    if spacing not in ([], ["log"]):
        raise ValueError(f"grid spacing for {name!r} must be omitted or 'log', got {':'.join(spacing)!r}")
    space = np.geomspace if spacing else np.linspace
    return name, space(float(lower), float(upper), int(count))


def add_model_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--experiment", required=True, choices=registry.names())
    model = command.add_mutually_exclusive_group(required=True)
    model.add_argument("--card", type=Path)
    model.add_argument("--template", choices=TEMPLATES)
    model.add_argument("--plugin", type=Path, help="New-physics coefficient plugin.")
    model.add_argument("--raw-plugin", type=Path, help="Complete cross-section plugin, including the SM.")
    command.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")


def model_signal(args: argparse.Namespace, experiment: Experiment) -> tuple[np.ndarray, dict[str, float]]:
    """Signal counts of the model chosen by the model options, and the parameter point used."""
    parameters = parameter_point(args.set)
    if args.raw_plugin:
        plugin = load_raw_plugin(args.raw_plugin)
    elif args.plugin:
        plugin = load_coefficient_plugin(args.plugin)
    if args.plugin or args.raw_plugin:
        unknown = sorted(set(parameters) - set(plugin.parameters))
        if unknown:
            raise ValueError(f"unknown plugin parameters {unknown}; valid parameters: {', '.join(plugin.parameters)}")
        parameters = plugin.parameters | parameters

    if args.raw_plugin:
        return predicted_counts(
            experiment,
            raw_cross_section=lambda e, t, flavor, isotope: plugin.dsigma_dt(e, t, flavor, isotope, parameters),
        ), parameters
    if args.plugin:
        coefficients = lambda q2: plugin.coefficients(q2, parameters)
    else:
        card = ModelCard.model_validate_json(args.card.read_text()) if args.card else template_card(args.template, parameters)
        coefficients = card.coefficients
    return predicted_counts(experiment, coefficients), parameters


def profile_json(profile: Profile, experiment: Experiment) -> dict:
    return {
        "neg2_log_likelihood": profile.neg2_log_likelihood,
        "alpha": profile.alpha,
        "beta": {background.name: b for background, b in zip(experiment.backgrounds, profile.beta)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nuagent",
        description="Compute CEvNS event spectra for new neutrino interactions and compare them with the Standard Model.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("experiments", help="List registered experiments as JSON.")
    add_model_options(commands.add_parser("predict", help="Print predicted signal and released data as JSON."))
    add_model_options(commands.add_parser(
        "compare", help="Print Δ = −2 ln L(new physics) − (−2 ln L(SM)) on observed data and the profiled nuisances as JSON."
    ))
    scan = commands.add_parser("scan", help="Print Δ over a template parameter grid as CSV.")
    scan.add_argument("--experiment", required=True, choices=registry.names())
    scan.add_argument("--template", required=True, choices=TEMPLATES)
    scan.add_argument("--param", action="append", required=True, metavar="NAME=MIN:MAX:N[:log]")
    scan.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    scan.add_argument("--data", choices=("asimov", "observed"), default="asimov")
    args = parser.parse_args()
    if args.command == "experiments":
        print(json.dumps({"experiments": registry.names()}))
        return

    experiment = registry.get(args.experiment)
    if args.command == "scan":
        result = asimov.scan(
            experiment, args.template, dict(map(parameter_axis, args.param)), parameter_point(args.set), data=args.data
        )
        writer = csv.writer(sys.stdout, lineterminator="\n")
        writer.writerow([*result.axes, "delta"])
        grid = np.meshgrid(*result.axes.values(), indexing="ij")
        for row in zip(*(axis.ravel() for axis in grid), result.delta.ravel()):
            writer.writerow([repr(float(value)) for value in row])
        return

    signal, parameters = model_signal(args, experiment)
    if args.command == "compare":
        new_physics = profiled(experiment, signal)
        standard = profiled(experiment, predicted_counts(experiment))
        print(json.dumps({
            "experiment": experiment.name, "data_release": experiment.data_release, "parameters": parameters,
            "delta": new_physics.neg2_log_likelihood - standard.neg2_log_likelihood,
            "new_physics": profile_json(new_physics, experiment), "sm": profile_json(standard, experiment),
            "totals": {"signal": float(signal.sum()), "observed": int(experiment.observed.sum())},
        }, allow_nan=False))
        return

    bins = [
        {
            "lower": float(lower), "upper": float(upper), "signal": float(signal[i]),
            "observed": int(experiment.observed[i]),
            "backgrounds": {background.name: float(background.counts[i]) for background in experiment.backgrounds},
        }
        for i, (lower, upper) in enumerate(zip(experiment.bin_edges[:-1], experiment.bin_edges[1:]))
    ]
    print(json.dumps({
        "experiment": experiment.name, "data_release": experiment.data_release,
        "bin_unit": experiment.bin_unit, "time_window_us": experiment.time_window_us,
        "parameters": parameters, "bins": bins,
        "totals": {
            "signal": float(signal.sum()), "observed": int(experiment.observed.sum()),
            "backgrounds": {background.name: float(background.counts.sum()) for background in experiment.backgrounds},
        },
    }, allow_nan=False))


if __name__ == "__main__":
    main()
