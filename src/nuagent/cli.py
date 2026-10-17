import argparse
import json
from pathlib import Path

from nuagent.experiments import registry
from nuagent.physics.modelcard import TEMPLATES, ModelCard, template_card
from nuagent.physics.plugin import load_coefficient_plugin, load_raw_plugin
from nuagent.rates import predicted_counts


def parameter_point(assignments: list[str]) -> dict[str, float]:
    return {key: float(value) for key, value in (assignment.split("=", 1) for assignment in assignments)}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nuagent",
        description="Compute CEvNS event spectra for new neutrino interactions and compare them with the Standard Model.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("experiments", help="List registered experiments as JSON.")
    predict = commands.add_parser("predict", help="Print predicted signal and released data as JSON.")
    predict.add_argument("--experiment", required=True, choices=registry.names())
    model = predict.add_mutually_exclusive_group(required=True)
    model.add_argument("--card", type=Path)
    model.add_argument("--template", choices=TEMPLATES)
    model.add_argument("--plugin", type=Path, help="New-physics coefficient plugin.")
    model.add_argument("--raw-plugin", type=Path, help="Complete cross-section plugin, including the SM.")
    predict.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    args = parser.parse_args()
    if args.command == "experiments":
        print(json.dumps({"experiments": registry.names()}))
        return

    parameters = parameter_point(args.set)
    experiment = registry.get(args.experiment)
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
        signal = predicted_counts(
            experiment,
            raw_cross_section=lambda e, t, flavor, isotope: plugin.dsigma_dt(e, t, flavor, isotope, parameters),
        )
    else:
        if args.plugin:
            coefficients = lambda q2: plugin.coefficients(q2, parameters)
        else:
            card = (
                ModelCard.model_validate_json(args.card.read_text())
                if args.card else template_card(args.template, parameters)
            )
            coefficients = card.coefficients
        signal = predicted_counts(experiment, coefficients)

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
