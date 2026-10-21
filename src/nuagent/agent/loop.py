"""Agent loop: with the interaction of a valid task frozen, a loop agent proposes parameter points round by round.

The harness alone evaluates each point with the source task's plugin; the loop agent writes only points.json and notes.md
in a fresh loop directory. The objective is the vector of per-experiment Δ and the result its Pareto set.
"""

import json
import os
import shutil
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from nuagent.agent.backend import Backend, BackendFailure
from nuagent.agent.report import LIKELIHOOD_CAPTION, VERIFICATION, likelihood_table, provenance_section, source_provenance
from nuagent.agent.run import evaluate_point, read_conversation
from nuagent.agent.task import LOOP_OUTPUT_FILES, LOOP_RECORD, copy_context
from nuagent.agent.validate import check_points
from nuagent.experiments import registry
from nuagent.experiments.registry import Experiment
from nuagent.physics.plugin import load_coefficient_plugin
from nuagent.rates import predicted_counts
from nuagent.stats.likelihood import profile_record, profiled

HISTORY = "history.json"
BATCH_LIMIT = 8
LOOP_STATES = {"evaluated": "completed", "converged": "converged", "invalid": "stopped-invalid"}


@dataclass(frozen=True)
class LoopRound:
    """One round's outcome: its state (evaluated, converged or invalid) and its history entries.

    `loop_state` (completed, converged or stopped-invalid) and `report` are set when this round ended the loop.
    """
    round: int
    state: str
    points: list[dict]
    loop_state: str | None
    report: str | None


def read_loop(loop_dir: Path) -> dict:
    """The harness's own record of the agent loop: source task, experiments, backend, limits, guidance and round states."""
    return json.loads((loop_dir / LOOP_RECORD).read_text())


def write_loop(loop_dir: Path, record: dict) -> None:
    (loop_dir / LOOP_RECORD).write_text(json.dumps(record, indent=2))


def read_history(loop_dir: Path) -> list[dict]:
    """Every point evaluated so far, in order, starting with round 0, the translation agent's point."""
    return json.loads((loop_dir / HISTORY).read_text())


def write_history(loop_dir: Path, history: list[dict]) -> None:
    (loop_dir / HISTORY).write_text(json.dumps(history, indent=2, allow_nan=False))


def experiment_data(experiment: Experiment) -> dict:
    """What the loop agent may read about one experiment: bins, observations, priors and the SM prediction and profile."""
    sm_signal = predicted_counts(experiment)
    return {"experiment": experiment.name, "data_release": experiment.data_release, "citations": list(experiment.citations),
            "bin_unit": experiment.bin_unit, "bin_edges": experiment.bin_edges.tolist(),
            "time_window_us": list(experiment.time_window_us), "observed": experiment.observed.tolist(),
            "backgrounds": [{"name": b.name, "counts": b.counts.tolist(), "normalization_prior": b.normalization_prior}
                            for b in experiment.backgrounds],
            "signal_normalization_prior": experiment.signal_normalization_prior,
            "sm_predicted_counts": sm_signal.tolist(), "sm_profile": profile_record(profiled(experiment, sm_signal), experiment)}


def evaluate_points(task_dir: Path, number: int, points: list[dict[str, float]]) -> list[dict]:
    """Check and evaluate each point with the source task's plugin; a point failing the checks keeps its findings and no Δ."""
    defaults = load_coefficient_plugin(task_dir / "plugin.py").parameters
    entries = []
    for index, point in enumerate(points, 1):
        model, evaluations = evaluate_point(task_dir, point)
        entries.append({"round": number, "index": index, "parameters": defaults | point,
                        "findings": model.report.findings, "evaluations": evaluations})
    return entries


def source_level(task_dir: Path) -> str:
    """The translation's verification level, derived as the translation does: a valid task with card.json is cross-validated."""
    return "cross-validated" if (task_dir / "card.json").is_file() else "plugin-only"


def start_loop(task_dir: Path, loop_dir: Path, backend: Backend, rounds: int, guidance: str | None = None,
              max_turns: int = 16) -> list[dict]:
    """Create the fresh loop directory, evaluate round 0 at the plugin's PARAMETERS and return the history."""
    conversation = read_conversation(task_dir)
    state = conversation["rounds"][-1]["state"]
    if state != "valid":
        raise ValueError(f"task {task_dir} is in state {state!r}; only a task in state 'valid' can start an agent loop")
    loop_dir.mkdir(parents=True)
    copy_context(files("nuagent.context"), loop_dir / "context")
    for source, target in (("plugin.py", "plugin.py"), ("card.json", "card.json"), ("notes.md", "interaction_notes.md")):
        if (task_dir / source).is_file():
            shutil.copyfile(task_dir / source, loop_dir / target)
            os.chmod(loop_dir / target, 0o444)
    (loop_dir / "data").mkdir()
    for name in conversation["experiments"]:
        (loop_dir / "data" / f"{name}.json").write_text(json.dumps(experiment_data(registry.get(name)), indent=2))
    history = evaluate_points(task_dir, 0, [{}])
    write_history(loop_dir, history)
    write_loop(loop_dir, {"source_task": str(task_dir.resolve()), "experiments": conversation["experiments"],
                        "backend": backend.name, "max_turns": max_turns, "rounds_requested": rounds,
                        "guidance": guidance, "batch_limit": BATCH_LIMIT, "state": "running", "rounds": []})
    return history


def notes_paths(loop_dir: Path, before: int) -> list[str]:
    """Paths, relative to the loop directory, of every notes.md kept from the rounds before `before`."""
    return [str(path.relative_to(loop_dir)) for number in range(1, before)
            for path in sorted((loop_dir / "history" / f"round-{number}").glob("attempt-*/notes.md"))]


def loop_prompt(loop_dir: Path, record: dict, number: int) -> str:
    agent_point = read_history(loop_dir)[0]["parameters"]
    names = list(agent_point)
    point = ", ".join(f"{name} = {value:g}" for name, value in agent_point.items())
    notes = "\n".join(f"- {path}" for path in notes_paths(loop_dir, number)) or "- none yet (this is the first round)"
    card = "- card.json: the same interaction in the fixed model-card schema, written for the translation agent's point.\n" \
        if (loop_dir / "card.json").is_file() else ""
    data = "\n".join(f"- data/{name}.json: bin edges and unit, observed counts, each background's per-bin counts and "
                     "normalization prior, the signal normalization prior, the Standard Model per-bin prediction and the "
                     "Standard Model profile (−2 ln L, α, β)." for name in record["experiments"])
    text = f"""# Agent loop: search the parameters of a fixed neutrino interaction

## Goal

The interaction defined in plugin.py is fixed; only the values of its PARAMETERS
({', '.join(names)}) vary. Propose parameter points that lower Δ on each
experiment ({', '.join(record['experiments'])}). With several experiments, look
for points that no other point beats on every experiment. Δ < 0 means the point
fits the observed counts better than the Standard Model.

## What the numbers mean

{LIKELIHOOD_CAPTION}

α near −1 means the signal prior is absorbing a prediction far above the data,
not a better fit. Δ is not monotonic in the signal size: a universal vector at
g = 1e-2 scores Δ ≈ 21.2 against 21.9 at g = 1e-4, both far worse than the
Standard Model.

## Round {number} of {record['rounds_requested']}

history.json holds every earlier point and its result, in order, starting with
round 0, the translation agent's own point ({point}). For each point it gives the
round, the index within the round, the full parameter point, the plugin-check
findings, and, for a point that passed the checks, per experiment the predicted
counts per bin and in total, Δ and both profiles.

Notes from earlier rounds:

{notes}

## Files to read

- plugin.py: the interaction. It defines the physics; do not change it.
- interaction_notes.md: the translation agent's derivation of the interaction.
{card}{data}
- history.json: every earlier point and its result.
- context/conventions.md: the conventions, units and parameter meanings.

## Output contract

Write points.json and notes.md in this directory and nothing else. On Claude
Code, read with Read and write with Write or Edit. On Codex, read files inside
this directory with read-only shell commands (cat, sed -n, ls, head) and write
with apply_patch. Run no other command, use no other tool, do not edit any
other file, and do not compute predictions or likelihoods yourself: the harness
evaluates every point you propose and adds it to history.json.

points.json has the shape {{"points": [{{"name": value, ...}}, ...]}} with at most
{record['batch_limit']} points. Each point maps parameter names of plugin.py to finite
real numbers (not booleans); a name a point omits keeps the translation agent's
value. A point that fails the plugin checks (for example a coupling above
sqrt(4 pi)) is recorded with its findings and no Δ, and still counts toward the
{record['batch_limit']} points. An empty list, {{"points": []}}, means you judge the search
converged; the agent loop then ends after this round.

notes.md: what you observed in the earlier results and why you chose these
points, in Markdown.
"""
    if record["guidance"]:
        text += f"\n## User guidance\n\nThe user's instruction for this agent loop:\n\n{record['guidance']}\n"
    return text


def clear_loop_outputs(loop_dir: Path) -> None:
    for name in LOOP_OUTPUT_FILES:
        path = loop_dir / name
        if path.exists():
            path.unlink()


def run_loop_round(loop_dir: Path, backend: Backend, out: Path) -> LoopRound:
    """Run the next round with one correction attempt, evaluate its points, and write the report if the loop ends."""
    record = read_loop(loop_dir)
    task_dir = Path(record["source_task"])
    number = len(record["rounds"]) + 1
    names = list(read_history(loop_dir)[0]["parameters"])
    (loop_dir / "TASK.md").write_text(loop_prompt(loop_dir, record, number))
    clear_loop_outputs(loop_dir)
    record["rounds"].append({"round": number, "state": "running"})
    write_loop(loop_dir, record)
    for attempt in (1, 2):
        result = backend.run(loop_dir, record["max_turns"])
        history_dir = loop_dir / "history" / f"round-{number}" / f"attempt-{attempt}"
        history_dir.mkdir(parents=True)
        (history_dir / "backend.json").write_text(json.dumps(result.as_dict(), indent=2))
        if result.exit_status or result.error:
            record["rounds"][-1]["state"] = "backend-failure"
            write_loop(loop_dir, record)
            raise BackendFailure(result)
        notes = loop_dir / "notes.md"
        if not notes.is_file() or not notes.read_text().strip():
            record["rounds"][-1]["state"] = "backend-failure"
            write_loop(loop_dir, record)
            raise RuntimeError(f"backend produced no nonempty notes.md\n{result.raw_output}\n{result.stderr}")
        for name in LOOP_OUTPUT_FILES:
            if (loop_dir / name).is_file():
                shutil.copyfile(loop_dir / name, history_dir / name)
        validation, points = check_points(loop_dir / "points.json", names, record["batch_limit"])
        (history_dir / "validation.json").write_text(json.dumps(validation.as_dict() | {"notes": notes.read_text()}, indent=2))
        if validation.valid:
            break
        if attempt == 1:
            clear_loop_outputs(loop_dir)
            with (loop_dir / "TASK.md").open("a") as prompt:
                prompt.write(f"\n## Validation correction (one attempt)\n\npoints.json did not satisfy the output contract. "
                             f"The previous outputs are preserved in history/round-{number}/attempt-1/. points.json and notes.md "
                             "have been cleared; write both again, following the output contract.\n\n```json\n"
                             + json.dumps(validation.as_dict(), indent=2) + "\n```\n")
    entries = []
    if not validation.valid:
        state = "invalid"
    elif not points:
        state = "converged"
    else:
        state = "evaluated"
        entries = evaluate_points(task_dir, number, points)
        write_history(loop_dir, read_history(loop_dir) + entries)
    record["rounds"][-1]["state"] = state
    loop_state, report = None, None
    if state != "evaluated" or number == record["rounds_requested"]:
        loop_state = LOOP_STATES[state]
        record["state"] = loop_state
        report = render_loop_report(loop_dir, record, read_history(loop_dir))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
    write_loop(loop_dir, record)
    return LoopRound(number, state, entries, loop_state, report)


def run_loop(task_dir: Path, loop_dir: Path, backend: Backend, out: Path, rounds: int = 5, guidance: str | None = None,
            max_turns: int = 16) -> LoopRound:
    """Start the agent loop and run rounds until one ends it; return that last round."""
    start_loop(task_dir, loop_dir, backend, rounds, guidance, max_turns)
    while True:
        step = run_loop_round(loop_dir, backend, out)
        if step.loop_state:
            return step


def deltas(entry: dict) -> tuple[float, ...]:
    return tuple(record["delta"] for record in entry["evaluations"])


def pareto_set(history: list[dict]) -> list[dict]:
    """The evaluated points that no other evaluated point matches or beats on every experiment and beats on one; ties kept."""
    evaluated = [entry for entry in history if entry["evaluations"]]
    def dominated(entry: dict) -> bool:
        mine = deltas(entry)
        return any(all(o <= m for o, m in zip(deltas(other), mine)) and deltas(other) != mine for other in evaluated)
    return [entry for entry in evaluated if not dominated(entry)]


def best_point(pareto: list[dict]) -> dict:
    """The Pareto point with the lowest summed Δ, for display only."""
    return min(pareto, key=lambda entry: sum(deltas(entry)))


def points_table(entries: list[dict], names: list[str], experiments: list[str], findings: bool = True) -> str:
    """One Markdown row per point: round, index, every parameter and Δ per experiment, or the check findings."""
    header = ["Round", "Point", *names, *(f"Δ {name}" for name in experiments)] + (["Plugin-check findings"] if findings else [])
    rows = [header, ["---"] * len(header)]
    for entry in entries:
        values = [f"{entry['parameters'][name]:.6g}" for name in names]
        delta_cells = [f"{delta:.6g}" for delta in deltas(entry)] if entry["evaluations"] else ["—"] * len(experiments)
        rows.append([str(entry["round"]), str(entry["index"]), *values, *delta_cells]
                    + (["; ".join(entry["findings"]) or "none"] if findings else []))
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def point_level(entry: dict, level: str) -> str:
    """Round 0 carries the translation's verification level; every other point is plugin-only."""
    if entry["round"] == 0:
        return f"Verification: **{level}** (the translation agent's point). {VERIFICATION[level]}"
    return ("Verification: **plugin-only** (a point other than the translation agent's: only the plugin was checked here; "
            "a model-card cross-check holds only at the translation agent's point).")


def round_notes(loop_dir: Path, number: int) -> str:
    """The notes of the round's last attempt."""
    return sorted((loop_dir / "history" / f"round-{number}").glob("attempt-*/notes.md"))[-1].read_text().strip()


def render_loop_report(loop_dir: Path, record: dict, history: list[dict]) -> str:
    """Outcome, best fit, source task, points by round, round notes, validation and backend records, provenance last."""
    task_dir = Path(record["source_task"])
    conversation = read_conversation(task_dir)
    experiments = record["experiments"]
    names = list(history[0]["parameters"])
    level = source_level(task_dir)
    pareto = pareto_set(history)
    run = len(record["rounds"])
    parts = [f"Outcome: **{record['state']}**\n\nAgent loop over the parameters of a fixed interaction. Rounds run: {run} of {record['rounds_requested']}. Points evaluated: "
             f"{len(history)} ({len(history) - 1} proposed by the loop agent and round 0, the translation agent's point); "
             f"{sum(1 for entry in history if entry['evaluations'])} passed the plugin checks."]
    best = ["## Best fit", "The Pareto set: the points that no other evaluated point matches or beats on every experiment "
            "and beats on one.", points_table(pareto, names, experiments, findings=False),
            "Round 0, the translation agent's point, for comparison: "
            + ", ".join(f"Δ {name} = {delta:.6g}" for name, delta in zip(experiments, deltas(history[0]))) + "."]
    for entry in pareto:
        best.append(f"### Round {entry['round']}, point {entry['index']}\n\n{point_level(entry, level)}\n\n"
                    + likelihood_table(entry["evaluations"]))
    best.append(LIKELIHOOD_CAPTION)
    parts.append("\n\n".join(best))
    parts.append(f"## Source task\n\n- Task directory: `{task_dir}`\n- Interaction: {conversation['description']}\n"
                 f"- Verification of the translation: {level}\n- Guidance: {record['guidance'] or 'none'}")
    parts.append("## Points by round\n\n" + points_table(history, names, experiments)
                 + "\n\nPer-bin predictions and full profiles of every point are in `history.json` in the loop directory.")
    parts.append("## Round notes\n\n" + "\n\n".join(f"### Round {entry['round']}\n\n{round_notes(loop_dir, entry['round'])}"
                                                    for entry in record["rounds"]))
    attempts = [(entry["round"], attempt, loop_dir / "history" / f"round-{entry['round']}" / f"attempt-{attempt}")
                for entry in record["rounds"] for attempt in (1, 2)]
    attempts = [(number, attempt, path) for number, attempt, path in attempts if path.is_dir()]
    parts.append("## Validation history")
    for number, attempt, path in attempts:
        parts.append(f"### Round {number}, attempt {attempt}\n\n```json\n{(path / 'validation.json').read_text()}\n```")
    parts.append("## Backend CLI records\n\nModel, cost and usage fields, when supplied by the CLI, are retained below verbatim. "
                 "Codex exec has no native max-turns flag; max_turns bounds Claude Code only.")
    for number, attempt, path in attempts:
        parts.append(f"### Round {number}, attempt {attempt}\n\n```json\n{(path / 'backend.json').read_text()}\n```")
    parts.append(provenance_section(source_provenance() | {
        "data_releases": {name: registry.get(name).data_release for name in experiments}, "backend": record["backend"],
        "max_turns": record["max_turns"], "rounds_requested": record["rounds_requested"], "rounds_run": run,
        "source_task": str(task_dir)}))
    return "\n\n".join(parts) + "\n"
