"""Streamlit page: describe an interaction, converse with the agent, evaluate the interaction it wrote and run the agent loop over its parameters."""

import io
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from matplotlib.figure import Figure

from nuagent.agent.backend import get_backend
from nuagent.agent.report import LIKELIHOOD_CAPTION, answers_markdown, likelihood_table, questions_markdown
from nuagent.agent.loop import (best_point, call_notes, missing_conclusion, pareto_set, points_table, read_history, read_loop,
                                round_summary, round_zero_summary, run_loop_step, start_loop)
from nuagent.agent.run import MAX_ROUNDS, RunResult, continue_run, evaluate_point, read_conversation, run_result
from nuagent.experiments import registry
from nuagent.physics.plugin import load_coefficient_plugin
from nuagent.rates import predicted_counts
from nuagent.ui.figures import delta_by_round_figure, parameter_figure, spectrum_figure


PLUGIN_LABEL = "Plugin (plugin.py): the coefficient code the agent wrote"
CARD_LABEL = "Model card (card.json): the interaction written in the fixed schema the harness cross-checks against the plugin"
LEVEL_MEANINGS = {
    "cross-validated": "the plugin and the model card agree at the agent's values",
    "plugin-only": "no model card to cross-check; the plugin's own checks pass at the agent's values",
}


def verification_caption(level: str, at_agent_point: bool) -> str:
    """The agent's level in the report's words; at other values also the level of the point shown, plugin-only."""
    if at_agent_point:
        return f"Verification: {level} ({LEVEL_MEANINGS[level]})."
    card = "; the model-card cross-check holds at the agent's values only" if level == "cross-validated" else ""
    return (f"Verification at the agent's values: {level} ({LEVEL_MEANINGS[level]}). "
            f"At these values: plugin-only (only the plugin was checked here{card}).")


def show_figure(figure: Figure) -> None:
    # Streamlit downsamples an image to an integer display width in pixels, which blurs on a 2x display;
    # "stretch" keeps up to 1460 pixels for the roughly 700-pixel column.
    image = io.BytesIO()
    figure.savefig(image, format="png", dpi=300, bbox_inches="tight")
    st.image(image, width="stretch")


def show_spectra(signals: dict[str, np.ndarray]) -> None:
    for name, signal in signals.items():
        experiment = registry.get(name)
        show_figure(spectrum_figure(experiment, signal, predicted_counts(experiment)))


def question_form(task_dir: Path, round_number: int, questions: list[dict]) -> dict[str, str | float] | None:
    """One input per question, pre-filled with its suggestion; return the nonempty answers once submitted."""
    # Keys carry the task and round: a keyed input keeps its previous value when only its default changes.
    prefix = f"answer:{task_dir}:{round_number}"
    answers = {}
    with st.form("questions"):
        for question in questions:
            label = question["question"] + (f" ({question['unit']})" if question["unit"] else "")
            suggested = question["suggested"]
            key = f"{prefix}:{question['id']}"
            if isinstance(suggested, (int, float)):
                answers[question["id"]] = st.number_input(label, value=float(suggested), format="%g", key=key)
            else:
                answers[question["id"]] = st.text_input(label, value=suggested or "", key=key)
            if question["reason"]:
                st.caption(f"Suggested: {question['reason']}")
        answers["free_text"] = st.text_area("Anything else (free text)", key=f"{prefix}:free_text")
        submitted = st.form_submit_button("Submit answers")
    return {name: value for name, value in answers.items() if value != ""} if submitted else None


def show_notes(task_dir: Path) -> None:
    """The notes under a fixed heading, without the agent's own first level-1 title."""
    lines = (task_dir / "notes.md").read_text().splitlines()
    first = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first is not None and lines[first].startswith("# "):
        del lines[first]
    st.subheader("Note")
    st.markdown("\n".join(lines))


def provenance_markdown(provenance: dict) -> str:
    def text(value: object) -> str:
        return ", ".join(f"{key} = {item}" for key, item in value.items()) if isinstance(value, dict) else str(value)
    return "\n".join(f"- {key.replace('_', ' ')}: {text(value)}" for key, value in provenance.items())


def result_block(task_dir: Path, result: RunResult) -> None:
    """Parameter inputs with Evaluate and the agent loop, spectrum, likelihood table, loop records, notes, then the outputs.

    The figure and table show the latest agent loop's best point when there is one, else the agent's point; Evaluate shows
    the entered point.
    """
    plugin = load_coefficient_plugin(task_dir / "plugin.py")
    agent_point = {name: float(value) for name, value in plugin.parameters.items()}
    keys = {name: f"parameter:{task_dir}:{name}" for name in agent_point}
    # A keyed widget's value can change only before the widget renders: a finished loop stores its best point and reruns.
    for name, value in st.session_state.pop("loop_point", {}).items():
        st.session_state[keys[name]] = float(value)
    for name, value in agent_point.items():
        st.session_state.setdefault(keys[name], value)
    with st.form("parameters"):
        st.subheader("Evaluate the interaction")
        point = {name: st.number_input(name, format="%g", key=keys[name]) for name in agent_point}
        evaluated = st.form_submit_button("Evaluate")
    loop_form(task_dir)
    signals, records, at_agent_point = result.signals, result.evaluations, True
    loop_dir = st.session_state.get("loop_dir")
    if loop_dir:
        best = best_point(pareto_set(read_history(loop_dir)))
        signals = {record["experiment"]: np.array(record["predicted_counts"]) for record in best["evaluations"]}
        records, at_agent_point = best["evaluations"], best["round"] == 0
    if evaluated:
        model, point_records = evaluate_point(task_dir, point)
        if model.report.valid:
            signals, records, at_agent_point = model.signals, point_records, point == agent_point
        else:
            st.markdown("This point fails the plugin checks, so the figure and table stay at the point shown before:\n\n"
                        + "\n".join(f"- {finding}" for finding in model.report.findings))
    show_spectra(signals)
    st.markdown(likelihood_table(records))
    st.caption(verification_caption(result.level, at_agent_point))
    st.caption(LIKELIHOOD_CAPTION)
    if loop_dir:
        loop_display(loop_dir)
    show_notes(task_dir)
    with st.expander("Provenance"):
        st.markdown(provenance_markdown(result.provenance))
    with st.expander(PLUGIN_LABEL):
        st.code((task_dir / "plugin.py").read_text(), language="python")
    if (task_dir / "card.json").is_file():
        with st.expander(CARD_LABEL):
            st.code((task_dir / "card.json").read_text(), language="json")


def loop_form(task_dir: Path) -> None:
    """The agent-loop row; a submitted loop runs one round at a time with visible progress, then shows its best point."""
    with st.form("loop"):
        rounds_column, guidance_column, button_column = st.columns([1, 3, 1.2], vertical_alignment="bottom")
        rounds = rounds_column.number_input("Agent loop rounds", min_value=1, value=5, step=1, key=f"loop_rounds:{task_dir}")
        guidance = guidance_column.text_area("Guidance for the loop agent (optional)", height=68, key=f"loop_guidance:{task_dir}")
        submitted = button_column.form_submit_button("Run agent loop")
    if not submitted:
        return
    st.session_state.pop("loop_dir", None)
    backend = get_backend(read_conversation(task_dir)["backend"])
    loop_dir = Path(tempfile.mkdtemp(prefix="loop-", dir=task_dir.parent)) / "loop"
    start_loop(task_dir, loop_dir, backend, int(rounds), guidance or None)
    # The research log and trajectory grow in place after every step; a rerun during the loop would stop it.
    progress = st.empty()
    while True:
        done = len(read_loop(loop_dir)["rounds"])
        running = (f"Round {done + 1} of {int(rounds)} running on {backend.name}…" if done < rounds
                   else f"Closing call running on {backend.name}: the agent is writing its conclusion…")
        with progress.container():
            # The running line ends the log, above the records and charts; a running status label carries a spinner.
            loop_display(loop_dir, running=True).status(running, state="running", expanded=False)
        step = run_loop_step(loop_dir, backend, loop_dir.parent / "report.md")
        if step.loop_state:
            break
    st.session_state.loop_dir = loop_dir
    st.session_state.loop_point = best_point(pareto_set(read_history(loop_dir)))["parameters"]
    st.rerun()


def loop_display(loop_dir: Path, running: bool = False):
    """The research log (round 0, each round's summary and hypothesis or error, the conclusion, then the records collapsed),
    then the trajectory. With `running`, return an empty slot at the end of the log for the running line."""
    record, history = read_loop(loop_dir), read_history(loop_dir)
    names, experiments = list(history[0]["parameters"]), record["experiments"]
    st.subheader("Research log")
    st.markdown(f"**{round_zero_summary(history, experiments)}**")
    for entry in record["rounds"]:
        text = f"**{round_summary(entry, history, experiments)}**"
        if "hypothesis" in entry:
            text += f"  \nHypothesis: {entry['hypothesis']}"
        if "error" in entry:
            text += f"  \nError: {entry['error']}"
        elif entry["state"] not in ("evaluated", "converged"):
            text += f"  \nRound state: {entry['state']}"
        st.markdown(text)
    running_slot = st.empty() if running else None
    if record["conclusion"]:
        st.info(f"**Conclusion.** {record['conclusion']}")
    elif record["state"] != "running":
        st.info(missing_conclusion(record))
    if record["state"] != "running":
        best = best_point(pareto_set(history))
        with st.expander(f"Agent loop: {record['state']}, best fit"):
            st.markdown(f"Outcome: **{record['state']}** after {len(record['rounds'])} of {record['rounds_requested']} agent rounds; "
                        f"{len(history)} points evaluated, including the translation agent's point (round 0). The figure, table "
                        f"and inputs above show the best point, round {best['round']}, point {best['index']} (lowest summed Δ in "
                        "the Pareto set).\n\nPareto set (no other point is as good on every experiment and better on one):\n\n"
                        + points_table(pareto_set(history), names, experiments, findings=False))
    with st.expander("Agent loop: points by round"):
        st.markdown(points_table(history, names, experiments))
    labels = [(f"round {entry['round']} ({entry['state']})", f"round-{entry['round']}") for entry in record["rounds"]]
    if record["closing"]:
        labels.append((f"closing call ({record['closing']['state']})", "closing"))
    for title, label in labels:
        with st.expander(f"Agent loop: notes, {title}"):
            st.markdown(call_notes(loop_dir, label))
    with st.expander("Agent loop: directory and report"):
        st.markdown(f"- Agent-loop directory: `{loop_dir}`\n- Agent-loop report: `{loop_dir.parent / 'report.md'}`")
    st.subheader("Agent loop trajectory")
    show_figure(delta_by_round_figure(history, experiments))
    show_figure(parameter_figure(history, names, experiments))
    return running_slot


st.set_page_config(page_title="nuagent")
st.title("nuagent")
# A form sends no rerun until submit: a rerun during a run would stop the script before it stores the result.
with st.form("agent"):
    description = st.text_area("Interaction description", key="agent_description")
    backend = st.selectbox("Backend", ("claude", "codex"), key="agent_backend")
    experiments = st.multiselect("Experiments", registry.names(), default=list(registry.names()), key="agent_experiments")
    submitted = st.form_submit_button("Run agent")
if submitted and not experiments:
    st.error("Choose at least one experiment.")
elif submitted:
    st.session_state.pop("agent_result", None)
    st.session_state.pop("agent_task", None)
    st.session_state.pop("loop_dir", None)
    root = Path(tempfile.mkdtemp(prefix="nuagent-ui-"))
    with st.spinner(f"Running {backend} in {root}; this takes minutes"):
        result = run_result(description, experiments, get_backend(backend), root / "task", root / "report.md")
    st.session_state.agent_task = root / "task"
    st.session_state.agent_result = result
if "agent_result" in st.session_state:
    task_dir = st.session_state.agent_task
    result = st.session_state.agent_result
    report_path = task_dir.parent / "report.md"
    conversation = read_conversation(task_dir)
    st.caption(f"Task directory: `{task_dir}`; full report: `{report_path}`")
    with st.chat_message("user"):
        st.markdown(conversation["description"])
    units = {}
    for entry in conversation["rounds"]:
        if entry.get("questions"):
            units = {question["id"]: question["unit"] for question in entry["questions"]}
            with st.chat_message("assistant"):
                if entry is conversation["rounds"][-1] and result.state == "questions":
                    st.markdown("The description leaves the form of the interaction open. Please answer:")
                    answers = question_form(task_dir, result.round, result.questions)
                    if answers is not None:
                        st.session_state.pop("agent_result")
                        with st.spinner(f"Running {conversation['backend']} on the answers; this takes minutes"):
                            st.session_state.agent_result = continue_run(task_dir, answers, get_backend(conversation["backend"]), report_path)
                        st.rerun()
                else:
                    st.markdown("The description leaves the form of the interaction open:\n\n" + questions_markdown(entry["questions"]))
                    if result.state == "unresolved" and entry is conversation["rounds"][-1]:
                        st.markdown(f"The limit of {MAX_ROUNDS} rounds was reached; these questions remain open.")
        if "answers" in entry:
            with st.chat_message("user"):
                st.markdown(answers_markdown(entry["answers"], units))
    if result.state == "valid":
        # The result is a page section below the conversation, not an agent message.
        with st.container(key="result"):
            result_block(task_dir, result)
    elif result.state in ("invalid", "notes-only"):
        with st.chat_message("assistant"):
            if result.state == "invalid":
                st.markdown("The outputs failed validation after the one correction attempt, so there are no "
                            f"predictions. The validation findings are in the full report, `{report_path}`.")
            else:
                show_notes(task_dir)
