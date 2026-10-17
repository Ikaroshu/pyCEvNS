import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from nuagent.experiments import registry
from nuagent.experiments.registry import Experiment
from nuagent.physics.modelcard import TEMPLATES, ModelCard, template_card
from nuagent.physics.plugin import load_coefficient_plugin, load_raw_plugin
from nuagent.rates import predicted_counts
from nuagent.stats import asimov
from nuagent.stats.likelihood import profile_record, profiled


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
    validate = commands.add_parser("validate", help="Print model validation findings as JSON; no likelihood feedback.")
    validate.add_argument("--experiments", default=",".join(registry.names()))
    validation_model = validate.add_mutually_exclusive_group()
    validation_model.add_argument("--plugin", type=Path)
    validation_model.add_argument("--raw-plugin", type=Path)
    validate.add_argument("--card", type=Path)
    for name in ("propose", "agent-eval"):
        command = commands.add_parser(name, help="Translate an interaction." if name == "propose" else "Run repository-only agent evaluations.")
        command.add_argument("--backend", choices=("claude_code", "codex"), default="claude_code")
        command.add_argument("--max-turns", type=int, default=16, help="Claude Code turn limit; Codex exec has no equivalent flag.")
        command.add_argument("--experiments", default=",".join(registry.names()))
        if name == "propose":
            command.add_argument("--description", required=True)
            command.add_argument("--out", type=Path, required=True)
            command.add_argument("--task-dir", type=Path, help="Fresh task directory; defaults to REPORT.task.")
        else:
            command.add_argument("--out-dir", type=Path, help="Fresh output directory; defaults to a new temporary directory.")
    ui = commands.add_parser("ui", help="Serve the Streamlit page on 127.0.0.1 with usage statistics disabled.")
    ui.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    if args.command == "ui":
        from nuagent.ui.launch import streamlit_command
        subprocess.run(streamlit_command(args.port), check=True)
        return
    if args.command == "experiments":
        print(json.dumps({"experiments": registry.names()}))
        return

    if args.command == "validate":
        from nuagent.agent.validate import validate_model
        if not (args.plugin or args.raw_plugin or args.card) or (args.raw_plugin and args.card):
            parser.error("validate requires --plugin [--card], --raw-plugin, or --card")
        result = validate_model([registry.get(name) for name in args.experiments.split(",")],
                                plugin=args.plugin or args.raw_plugin, card=args.card, raw=bool(args.raw_plugin))
        output = result.report.as_dict()
        if result.report.valid and args.raw_plugin:
            from nuagent.agent.report import VERIFICATION
            output.update(verification="raw-cross-section", meaning=VERIFICATION["raw-cross-section"])
        print(json.dumps(output, allow_nan=False))
        sys.exit(0 if result.report.valid else 1)
    if args.command in ("propose", "agent-eval"):
        from nuagent.agent.backend import get_backend
        backend = get_backend(args.backend)
        if args.command == "propose":
            from nuagent.agent.run import run
            run(args.description, args.experiments.split(","), backend,
                (args.task_dir or args.out.with_suffix(".task")).resolve(), args.out, args.max_turns)
            print(json.dumps({"report": str(args.out)}))
        else:
            root = Path.cwd()
            if not (root / "evals" / "agent" / "cases.py").is_file():
                parser.error("agent-eval is repository-only; run from the repository root")
            sys.path.insert(0, str(root))
            from evals.agent.run import run_evaluation
            output_root = args.out_dir or Path(tempfile.mkdtemp(prefix="nuagent-eval-")) / args.backend
            print(f"Agent evaluation outputs: {output_root.resolve()}", file=sys.stderr)
            summary = run_evaluation(output_root.resolve(), backend, args.experiments.split(","), args.max_turns)
            print(json.dumps(summary, indent=2))
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
            "new_physics": profile_record(new_physics, experiment), "sm": profile_record(standard, experiment),
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
