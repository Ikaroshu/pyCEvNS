import argparse
import csv
import json
import os
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


def propose(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Run round 1 from --description, or continue the task in --task-dir with --answer on its recorded backend."""
    from nuagent.agent.backend import get_backend
    from nuagent.agent.run import ValidationFailed, continue_run, read_conversation, run_result
    if (args.description is None) == (args.answer is None):
        parser.error("propose requires exactly one of --description and --answer")
    if args.answer is None:
        result = run_result(args.description, (args.experiments or ",".join(registry.names())).split(","),
                            get_backend(args.backend or "claude"), (args.task_dir or args.out.with_suffix(".task")).resolve(),
                            args.out, 16 if args.max_turns is None else args.max_turns)
    else:
        if args.task_dir is None:
            parser.error("--answer requires --task-dir naming the task to continue")
        given = [option for option, value in (("--backend", args.backend), ("--experiments", args.experiments),
                                              ("--max-turns", args.max_turns)) if value is not None]
        if given:
            parser.error(f"--answer continues a recorded task; {', '.join(given)} cannot be given (the recorded values apply)")
        task_dir = args.task_dir.resolve()
        answers = dict(answer.split("=", 1) for answer in args.answer)
        result = continue_run(task_dir, answers, get_backend(read_conversation(task_dir)["backend"]), args.out)
    if result.state == "invalid":
        raise ValidationFailed(result.histories)
    output = {"report": str(args.out)}
    if result.state in ("questions", "unresolved"):
        output |= {"state": result.state, "round": result.round, "questions": result.questions}
    print(json.dumps(output))


def loop(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Run the agent loop over the parameters of the valid task in --task-dir; exit 1 when it stops invalid."""
    from nuagent.agent.backend import get_backend
    from nuagent.agent.loop import deltas, pareto_set, read_history, run_loop
    from nuagent.agent.run import read_conversation
    if args.rounds < 1:
        parser.error(f"--rounds must be a positive integer, got {args.rounds}")
    task_dir = args.task_dir.resolve()
    conversation = read_conversation(task_dir)
    state = conversation["rounds"][-1]["state"]
    if state != "valid":
        parser.error(f"task {task_dir} is in state {state!r}; only a task in state 'valid' can start an agent loop")
    loop_dir = (args.loop_dir or args.out.with_suffix(".loop")).resolve()
    step = run_loop(task_dir, loop_dir, get_backend(args.backend or conversation["backend"]), args.out, args.rounds,
                    args.guidance, args.max_turns)
    pareto = [{"round": entry["round"], "index": entry["index"], "parameters": entry["parameters"],
               "delta": dict(zip(conversation["experiments"], deltas(entry)))} for entry in pareto_set(read_history(loop_dir))]
    print(json.dumps({"report": str(args.out), "state": step.loop_state, "rounds": step.round, "pareto": pareto}, allow_nan=False))
    sys.exit(1 if step.loop_state == "stopped-invalid" else 0)


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
        command = commands.add_parser(name, help="Translate an interaction: run one agent round, or continue a task with --answer." if name == "propose" else "Run repository-only agent evaluations.")
        # propose leaves these unset so that a continuation can reject them; the recorded values apply there.
        defaults = {"backend": "claude", "max_turns": 16, "experiments": ",".join(registry.names())}
        defaults = dict.fromkeys(defaults) if name == "propose" else defaults
        command.add_argument("--backend", choices=("claude", "codex"), default=defaults["backend"], help="Default claude.")
        command.add_argument("--max-turns", type=int, default=defaults["max_turns"], help="Claude Code turn limit (default 16); Codex exec has no equivalent flag.")
        command.add_argument("--experiments", default=defaults["experiments"], help="Comma-separated experiment names; default all registered.")
        if name == "propose":
            command.add_argument("--description")
            command.add_argument("--answer", action="append", metavar="ID=VALUE",
                                 help="Answer a question of the task in --task-dir and run its next round; repeat per question; free_text=... carries free text.")
            command.add_argument("--out", type=Path, required=True)
            command.add_argument("--task-dir", type=Path, help="Fresh task directory (defaults to REPORT.task), or with --answer the task to continue.")
        else:
            command.add_argument("--out-dir", type=Path, help="Fresh output directory; defaults to a new temporary directory.")
    loop_command = commands.add_parser("loop", help="Run the agent loop: an agent searches the parameters of a valid task's plugin to minimize Δ, with the interaction fixed.")
    loop_command.add_argument("--task-dir", type=Path, required=True, help="A task whose last round is valid.")
    loop_command.add_argument("--out", type=Path, required=True, help="The agent-loop report.")
    loop_command.add_argument("--rounds", type=int, default=5, help="Agent rounds, a positive integer (default 5).")
    loop_command.add_argument("--guidance", help="Your instruction for the loop agent, passed verbatim.")
    loop_command.add_argument("--backend", choices=("claude", "codex"), help="Default: the task's recorded backend.")
    loop_command.add_argument("--max-turns", type=int, default=16, help="Claude Code turn limit per round (default 16); Codex exec has no equivalent flag.")
    loop_command.add_argument("--loop-dir", type=Path, help="Fresh agent-loop directory (defaults to REPORT.loop).")
    ui = commands.add_parser("ui", help="Serve the Streamlit page on 127.0.0.1 with usage statistics disabled.")
    ui.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    if args.command == "ui":
        from nuagent.ui.launch import streamlit_command
        os.execv(sys.executable, streamlit_command(args.port))
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
    if args.command == "propose":
        propose(parser, args)
        return
    if args.command == "loop":
        loop(parser, args)
    if args.command == "agent-eval":
        from nuagent.agent.backend import get_backend
        backend = get_backend(args.backend)
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
