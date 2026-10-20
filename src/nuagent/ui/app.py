"""Streamlit page: describe an interaction, converse with the agent, evaluate the interaction it wrote, then fit its parameters."""

import io
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from nuagent.agent.backend import get_backend
from nuagent.agent.report import LIKELIHOOD_CAPTION, answers_markdown, likelihood_table, questions_markdown
from nuagent.agent.fit import best_point, pareto_set, points_table, read_fit, read_history, round_notes, run_fit_round, start_fit
from nuagent.agent.run import MAX_ROUNDS, RunResult, continue_run, evaluate_point, read_conversation, run_result
from nuagent.experiments import registry
from nuagent.physics.plugin import load_coefficient_plugin
from nuagent.rates import predicted_counts
from nuagent.ui.figures import spectrum_figure


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


def show_spectra(signals: dict[str, np.ndarray]) -> None:
    for name, signal in signals.items():
        experiment = registry.get(name)
        # Streamlit downsamples an image to an integer display width in pixels, which blurs on a 2x display;
        # "stretch" keeps up to 1460 pixels for the roughly 700-pixel column.
        image = io.BytesIO()
        spectrum_figure(experiment, signal, predicted_counts(experiment)).savefig(image, format="png", dpi=300, bbox_inches="tight")
        st.image(image, width="stretch")


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
    """Re-evaluation form, spectrum, likelihood table, notes, then the outputs collapsed; the agent's point first."""
    plugin = load_coefficient_plugin(task_dir / "plugin.py")
    agent_point = {name: float(value) for name, value in plugin.parameters.items()}
    with st.form("parameters"):
        st.subheader("Evaluate the interaction")
        point = {name: st.number_input(name, value=value, format="%g", key=f"parameter:{task_dir}:{name}")
                 for name, value in agent_point.items()}
        evaluated = st.form_submit_button("Evaluate")
    signals, records, at_agent_point = result.signals, result.evaluations, True
    if evaluated:
        model, point_records = evaluate_point(task_dir, point)
        if model.report.valid:
            signals, records, at_agent_point = model.signals, point_records, point == agent_point
        else:
            st.markdown("This point fails the plugin checks, so the figure and table stay at the agent's point:\n\n"
                        + "\n".join(f"- {finding}" for finding in model.report.findings))
    show_spectra(signals)
    st.markdown(likelihood_table(records))
    st.caption(verification_caption(result.level, at_agent_point))
    st.caption(LIKELIHOOD_CAPTION)
    show_notes(task_dir)
    with st.expander("Provenance"):
        st.markdown(provenance_markdown(result.provenance))
    with st.expander(PLUGIN_LABEL):
        st.code((task_dir / "plugin.py").read_text(), language="python")
    if (task_dir / "card.json").is_file():
        with st.expander(CARD_LABEL):
            st.code((task_dir / "card.json").read_text(), language="json")


def fit_form(task_dir: Path, result: RunResult) -> None:
    """Rounds and guidance for a fit on the task's recorded backend; run it one round at a time with visible progress."""
    with st.form("fit"):
        st.subheader("Fit parameters")
        rounds = st.number_input("Agent rounds", min_value=1, value=5, step=1, key=f"fit_rounds:{task_dir}")
        guidance = st.text_area("Guidance for the fit agent (optional)", key=f"fit_guidance:{task_dir}")
        submitted = st.form_submit_button("Fit")
    if submitted:
        st.session_state.pop("fit_dir", None)
        backend = get_backend(read_conversation(task_dir)["backend"])
        fit_dir = Path(tempfile.mkdtemp(prefix="fit-", dir=task_dir.parent)) / "fit"
        with st.status(f"Fitting with {backend.name} in {fit_dir}; each agent round takes minutes", expanded=True) as status:
            history = start_fit(task_dir, fit_dir, backend, int(rounds), guidance or None)
            names, experiments = list(history[0]["parameters"]), read_fit(fit_dir)["experiments"]
            st.markdown("The translation agent's point (round 0):\n\n" + points_table(history, names, experiments))
            while True:
                step = run_fit_round(fit_dir, backend, fit_dir.parent / "report.md")
                summary = points_table(step.points, names, experiments) if step.points else "no points"
                st.markdown(f"Agent round {step.round} of {int(rounds)}: {step.state}\n\n{summary}")
                if step.fit_state:
                    break
            status.update(label=f"Fit {step.fit_state} after {step.round} agent rounds", state="complete", expanded=False)
        st.session_state.fit_dir = fit_dir
    if "fit_dir" in st.session_state:
        with st.container(key="fit_result"):
            fit_block(st.session_state.fit_dir, result)


def fit_block(fit_dir: Path, result: RunResult) -> None:
    """Outcome, Pareto set, the best point's spectrum and table, every point, the round notes and the paths."""
    record, history = read_fit(fit_dir), read_history(fit_dir)
    names, experiments = list(history[0]["parameters"]), record["experiments"]
    pareto = pareto_set(history)
    best = best_point(pareto)
    st.subheader("Fit result")
    st.markdown(f"Outcome: **{record['state']}** after {len(record['rounds'])} of {record['rounds_requested']} agent rounds; "
                f"{len(history)} points evaluated, including the translation agent's point (round 0).")
    st.markdown("Pareto set (no other point is as good on every experiment and better on one):\n\n"
                + points_table(pareto, names, experiments, findings=False))
    st.markdown(f"Best point: round {best['round']}, point {best['index']} (lowest summed Δ in the Pareto set).")
    show_spectra({evaluation["experiment"]: np.array(evaluation["predicted_counts"]) for evaluation in best["evaluations"]})
    st.markdown(likelihood_table(best["evaluations"]))
    st.caption(verification_caption(result.level, best["round"] == 0))
    st.caption(LIKELIHOOD_CAPTION)
    st.markdown("Points by round:\n\n" + points_table(history, names, experiments))
    for entry in record["rounds"]:
        with st.expander(f"Agent notes, round {entry['round']} ({entry['state']})"):
            st.markdown(round_notes(fit_dir, entry["round"]))
    st.caption(f"Fit directory: `{fit_dir}`; fit report: `{fit_dir.parent / 'report.md'}`")


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
    st.session_state.pop("fit_dir", None)
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
        fit_form(task_dir, result)
    elif result.state in ("invalid", "notes-only"):
        with st.chat_message("assistant"):
            if result.state == "invalid":
                st.markdown("The outputs failed validation after the one correction attempt, so there are no "
                            f"predictions. The validation findings are in the full report, `{report_path}`.")
            else:
                show_notes(task_dir)
