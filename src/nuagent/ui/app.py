"""Streamlit page: describe an interaction, converse with the agent, then re-evaluate the interaction it wrote."""

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from nuagent.agent.backend import get_backend
from nuagent.agent.report import LIKELIHOOD_CAPTION, answers_markdown, likelihood_table, questions_markdown
from nuagent.agent.run import MAX_ROUNDS, RunResult, continue_run, evaluate_point, read_conversation, run_result
from nuagent.experiments import registry
from nuagent.physics.plugin import load_coefficient_plugin
from nuagent.rates import predicted_counts
from nuagent.ui.figures import spectrum_figure


def show_spectra(signals: dict[str, np.ndarray]) -> None:
    for name, signal in signals.items():
        experiment = registry.get(name)
        st.pyplot(spectrum_figure(experiment, signal, predicted_counts(experiment)), width=760)


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
    titles = [index for index, line in enumerate(lines) if line.startswith("# ")]
    if titles:
        del lines[titles[0]]
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
    st.caption(f"Verification at the agent's point: {result.level}." + ("" if at_agent_point else " Plugin-only at this point."))
    st.caption(LIKELIHOOD_CAPTION)
    show_notes(task_dir)
    with st.expander("Provenance"):
        st.markdown(provenance_markdown(result.provenance))
    with st.expander("plugin.py"):
        st.code((task_dir / "plugin.py").read_text(), language="python")
    if (task_dir / "card.json").is_file():
        with st.expander("card.json"):
            st.code((task_dir / "card.json").read_text(), language="json")


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
    if result.state in ("valid", "invalid", "notes-only"):
        with st.chat_message("assistant"):
            if result.state == "valid":
                result_block(task_dir, result)
            elif result.state == "invalid":
                st.markdown("The outputs failed validation after the one correction attempt, so there are no "
                            f"predictions. The validation findings are in the full report, `{report_path}`.")
            else:
                show_notes(task_dir)
