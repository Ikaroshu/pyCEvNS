"""Agent loop: with the interaction of a valid task frozen, a loop agent proposes parameter points round by round.

The harness alone evaluates each point with the source task's plugin; the loop agent writes only points.json (its points
and a hypothesis) and notes.md in a fresh loop directory. The objective is the vector of per-experiment Δ and the result its
Pareto set. A loop that runs its requested rounds without converging ends with one closing call for the agent's conclusion.
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
from nuagent.agent.validate import ValidationReport, check_points
from nuagent.experiments import registry
from nuagent.experiments.registry import Experiment
from nuagent.physics.plugin import load_coefficient_plugin
from nuagent.rates import predicted_counts
from nuagent.stats.likelihood import profile_record, profiled

HISTORY = "history.json"
BATCH_LIMIT = 8
LOOP_STATES = {"converged": "converged", "invalid": "stopped-invalid"}


@dataclass(frozen=True)
class LoopStep:
    """One step's outcome: a round (state evaluated, converged or invalid, with its history entries) or the closing call
    (state concluded or invalid, `closing` true, `round` the last round run).

    `loop_state` (completed, converged or stopped-invalid) and `report` are set when this step ended the loop.
    """
    round: int
    closing: bool
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
                        "guidance": guidance, "batch_limit": BATCH_LIMIT, "state": "running", "rounds": [],
                          "closing": None, "conclusion": None})
    return history


def notes_paths(loop_dir: Path, before: int) -> list[str]:
    """Paths, relative to the loop directory, of every notes.md kept from the rounds before `before`."""
    return [str(path.relative_to(loop_dir)) for number in range(1, before)
            for path in sorted((loop_dir / "history" / f"round-{number}").glob("attempt-*/notes.md"))]


def loop_prompt(loop_dir: Path, record: dict, number: int, closing: bool = False) -> str:
    """The round's prompt; with `closing`, the prompt of the closing call after the last requested round."""
    agent_point = read_history(loop_dir)[0]["parameters"]
    names = list(agent_point)
    point = ", ".join(f"{name} = {value:g}" for name, value in agent_point.items())
    notes = "\n".join(f"- {path}" for path in notes_paths(loop_dir, number)) or "- none yet (this is the first round)"
    card = "- card.json: the same interaction in the fixed model-card schema, written for the translation agent's point.\n" \
        if (loop_dir / "card.json").is_file() else ""
    data = "\n".join(f"- data/{name}.json: bin edges and unit, observed counts, each background's per-bin counts and "
                     "normalization prior, the signal normalization prior, the Standard Model per-bin prediction and the "
                     "Standard Model profile (−2 ln L, α, β)." for name in record["experiments"])
    requested = record["rounds_requested"]
    stage = (f"""## Closing call

All {requested} requested rounds have been evaluated, and no more points will be
evaluated. Read the results, then write points.json with an empty points list,
a hypothesis (it may restate the final picture) and the conclusion: a few
sentences on what the agent loop found, naming the best points and their Δ.
Write notes.md with your final reasoning.""" if closing else f"## Round {number} of {requested}")
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

{stage}

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

points.json has the shape
{{"hypothesis": "...", "points": [{{"name": value, ...}}, ...]}} with at most
{record['batch_limit']} points. The hypothesis is one or two sentences saying what you
expect this round's points to show. Each point maps parameter names of plugin.py to finite
real numbers (not booleans); a name a point omits keeps the translation agent's
value. A point that fails the plugin checks (for example a coupling above
sqrt(4 pi)) is recorded with its findings and no Δ, and still counts toward the
{record['batch_limit']} points. An empty list means you judge the search converged; the
agent loop then ends after this round, and points.json must also carry
"conclusion": a few sentences on what the agent loop found. Write the
conclusion only with an empty list, once every result is in; if the requested
rounds end without convergence, the harness asks for it in a closing call.

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


def agent_call(loop_dir: Path, record: dict, entry: dict, label: str, backend: Backend,
               closing: bool) -> tuple[ValidationReport, dict | None]:
    """Run the backend on the current TASK.md with one correction attempt, keeping every attempt under history/<label>/.

    A backend failure, or a missing or empty notes.md, marks `entry` (a round or the closing call in `record`) as
    backend-failure and raises.
    """
    names = list(read_history(loop_dir)[0]["parameters"])
    for attempt in (1, 2):
        result = backend.run(loop_dir, record["max_turns"])
        history_dir = loop_dir / "history" / label / f"attempt-{attempt}"
        history_dir.mkdir(parents=True)
        (history_dir / "backend.json").write_text(json.dumps(result.as_dict(), indent=2))
        if result.exit_status or result.error:
            entry["state"] = "backend-failure"
            write_loop(loop_dir, record)
            raise BackendFailure(result)
        notes = loop_dir / "notes.md"
        if not notes.is_file() or not notes.read_text().strip():
            entry["state"] = "backend-failure"
            write_loop(loop_dir, record)
            raise RuntimeError(f"backend produced no nonempty notes.md\n{result.raw_output}\n{result.stderr}")
        for name in LOOP_OUTPUT_FILES:
            if (loop_dir / name).is_file():
                shutil.copyfile(loop_dir / name, history_dir / name)
        validation, data = check_points(loop_dir / "points.json", names, record["batch_limit"], closing)
        (history_dir / "validation.json").write_text(json.dumps(validation.as_dict() | {"notes": notes.read_text()}, indent=2))
        if validation.valid:
            break
        if attempt == 1:
            clear_loop_outputs(loop_dir)
            with (loop_dir / "TASK.md").open("a") as prompt:
                prompt.write(f"\n## Validation correction (one attempt)\n\npoints.json did not satisfy the output contract. "
                             f"The previous outputs are preserved in history/{label}/attempt-1/. points.json and notes.md "
                             "have been cleared; write both again, following the output contract.\n\n```json\n"
                             + json.dumps(validation.as_dict(), indent=2) + "\n```\n")
    return validation, data


def finish_loop(loop_dir: Path, record: dict, state: str, out: Path) -> str:
    record["state"] = state
    write_loop(loop_dir, record)
    report = render_loop_report(loop_dir, record, read_history(loop_dir))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    return report


def run_loop_step(loop_dir: Path, backend: Backend, out: Path) -> LoopStep:
    """Run the next round, or after the last requested round the closing call; write the report if the loop ends."""
    record = read_loop(loop_dir)
    number = len(record["rounds"])
    if number == record["rounds_requested"]:
        entry = record["closing"] = {"state": "running"}
        (loop_dir / "TASK.md").write_text(loop_prompt(loop_dir, record, number + 1, closing=True))
        clear_loop_outputs(loop_dir)
        write_loop(loop_dir, record)
        validation, data = agent_call(loop_dir, record, entry, "closing", backend, closing=True)
        if validation.valid:
            entry |= {"state": "concluded", "hypothesis": data["hypothesis"], "conclusion": data["conclusion"]}
            record["conclusion"] = data["conclusion"]
        else:
            entry["state"] = "invalid"
        return LoopStep(number, True, entry["state"], [], "completed", finish_loop(loop_dir, record, "completed", out))
    number += 1
    entry = {"round": number, "state": "running"}
    record["rounds"].append(entry)
    (loop_dir / "TASK.md").write_text(loop_prompt(loop_dir, record, number))
    clear_loop_outputs(loop_dir)
    write_loop(loop_dir, record)
    validation, data = agent_call(loop_dir, record, entry, f"round-{number}", backend, closing=False)
    entries = []
    if not validation.valid:
        entry["state"] = "invalid"
    else:
        entry |= {"state": "evaluated" if data["points"] else "converged", "hypothesis": data["hypothesis"]}
        if data["points"]:
            entries = evaluate_points(Path(record["source_task"]), number, data["points"])
            write_history(loop_dir, read_history(loop_dir) + entries)
        else:
            entry["conclusion"] = record["conclusion"] = data["conclusion"]
    loop_state = LOOP_STATES.get(entry["state"])
    report = finish_loop(loop_dir, record, loop_state, out) if loop_state else None
    write_loop(loop_dir, record)
    return LoopStep(number, False, entry["state"], entries, loop_state, report)


def run_loop(task_dir: Path, loop_dir: Path, backend: Backend, out: Path, rounds: int = 5, guidance: str | None = None,
             max_turns: int = 16) -> LoopStep:
    """Start the agent loop and run steps until one ends it; return that last step."""
    start_loop(task_dir, loop_dir, backend, rounds, guidance, max_turns)
    while True:
        step = run_loop_step(loop_dir, backend, out)
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


def call_notes(loop_dir: Path, label: str) -> str:
    """The notes of the last attempt of a round (label round-N) or of the closing call (label closing)."""
    return sorted((loop_dir / "history" / label).glob("attempt-*/notes.md"))[-1].read_text().strip()


def round_summary(entry: dict, history: list[dict], experiments: list[str]) -> str:
    """The harness's line for one round: points tried and passed and the round's best Δ per experiment."""
    points = [point for point in history if point["round"] == entry["round"]]
    passed = [point for point in points if point["evaluations"]]
    line = f"Round {entry['round']} · {len(points)} points, {len(passed)} passed checks"
    if passed:
        line += " · best Δ " + ", ".join(
            f"{min(point['evaluations'][k]['delta'] for point in passed):.4g}" + (f" ({name})" if len(experiments) > 1 else "")
            for k, name in enumerate(experiments))
    return line


def missing_conclusion(record: dict) -> str:
    """Why a finished loop has no conclusion."""
    if record["state"] == "stopped-invalid":
        return "No conclusion: the loop stopped because a round's points.json was malformed twice."
    return "No conclusion: the closing call's points.json was malformed twice."


def research_log(record: dict, history: list[dict]) -> str:
    """Per round the harness's summary line and the agent's hypothesis, then the conclusion."""
    parts = []
    for entry in record["rounds"]:
        text = f"### {round_summary(entry, history, record['experiments'])}"
        if "hypothesis" in entry:
            text += f"\n\nHypothesis: {entry['hypothesis']}"
        if entry["state"] != "evaluated":
            text += f"\n\nRound state: {entry['state']}."
        parts.append(text)
    parts.append("### Conclusion\n\n" + (record["conclusion"] or missing_conclusion(record)))
    return "\n\n".join(parts)


def render_loop_report(loop_dir: Path, record: dict, history: list[dict]) -> str:
    """Outcome, research log, best fit, source task, points by round, round notes, validation and backend records, provenance last."""
    task_dir = Path(record["source_task"])
    conversation = read_conversation(task_dir)
    experiments = record["experiments"]
    names = list(history[0]["parameters"])
    level = source_level(task_dir)
    pareto = pareto_set(history)
    run = len(record["rounds"])
    labels = [(f"Round {entry['round']}", f"round-{entry['round']}") for entry in record["rounds"]]
    if record["closing"]:
        labels.append(("Closing call", "closing"))
    parts = [f"Outcome: **{record['state']}**\n\nAgent loop over the parameters of a fixed interaction. Rounds run: {run} of {record['rounds_requested']}. Points evaluated: "
             f"{len(history)} ({len(history) - 1} proposed by the loop agent and round 0, the translation agent's point); "
             f"{sum(1 for entry in history if entry['evaluations'])} passed the plugin checks."]
    parts.append("## Research log\n\n" + research_log(record, history))
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
    parts.append("## Round notes\n\n" + "\n\n".join(f"### {title}\n\n{call_notes(loop_dir, label)}" for title, label in labels))
    attempts = [(title, attempt, loop_dir / "history" / label / f"attempt-{attempt}") for title, label in labels for attempt in (1, 2)]
    attempts = [(title, attempt, path) for title, attempt, path in attempts if path.is_dir()]
    parts.append("## Validation history")
    for title, attempt, path in attempts:
        parts.append(f"### {title}, attempt {attempt}\n\n```json\n{(path / 'validation.json').read_text()}\n```")
    parts.append("## Backend CLI records\n\nModel, cost and usage fields, when supplied by the CLI, are retained below verbatim. "
                 "Codex exec has no native max-turns flag; max_turns bounds Claude Code only.")
    for title, attempt, path in attempts:
        parts.append(f"### {title}, attempt {attempt}\n\n```json\n{(path / 'backend.json').read_text()}\n```")
    parts.append(provenance_section(source_provenance() | {
        "data_releases": {name: registry.get(name).data_release for name in experiments}, "backend": record["backend"],
        "max_turns": record["max_turns"], "rounds_requested": record["rounds_requested"], "rounds_run": run,
        "closing_call": record["closing"]["state"] if record["closing"] else None, "source_task": str(task_dir)}))
    return "\n\n".join(parts) + "\n"
