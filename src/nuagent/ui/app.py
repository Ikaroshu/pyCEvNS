"""Streamlit page: describe an interaction, answer the agent's questions, read the report, see the spectrum and re-evaluate the plugin."""

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from nuagent.agent.backend import get_backend
from nuagent.agent.report import VERIFICATION, conversation_markdown
from nuagent.agent.run import MAX_ROUNDS, continue_run, evaluate_point, read_conversation, run_result
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
                st.caption(f"Suggested value: {question['reason']}")
        answers["free_text"] = st.text_area("Anything else (free text)", key=f"{prefix}:free_text")
        submitted = st.form_submit_button("Submit answers")
    return {name: value for name, value in answers.items() if value != ""} if submitted else None


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
    st.write(f"Task directory: `{task_dir}`; report: `{report_path}`")
    if result.state == "questions":
        st.info(f"Round {result.round} of {MAX_ROUNDS}: the agent asks for information the description does not fix. "
                f"Submitting the answers runs the next round on {conversation['backend']}.")
        st.markdown(conversation_markdown(conversation))
        answers = question_form(task_dir, result.round, result.questions)
        with st.expander("Report"):
            st.markdown(result.report)
        if answers is not None:
            st.session_state.pop("agent_result")
            with st.spinner(f"Running round {result.round + 1} on {conversation['backend']} in {task_dir}; this takes minutes"):
                st.session_state.agent_result = continue_run(task_dir, answers, get_backend(conversation["backend"]), report_path)
            st.rerun()
    elif result.state == "unresolved":
        st.warning(f"The round limit of {MAX_ROUNDS} was reached; the last questions remain open.")
        st.markdown(result.report)
    elif result.state == "invalid":
        st.error("Outputs failed validation after the one correction attempt; the report has no predictions.")
        st.markdown(result.report)
    else:
        st.success("Outputs passed validation.")
        st.markdown(result.report)
    if result.state == "valid":
        spectra = st.container()
        plugin = load_coefficient_plugin(task_dir / "plugin.py")
        st.subheader("Re-evaluate the plugin")
        with st.form("parameters"):
            point = {name: st.number_input(name, value=float(value), format="%g", key=f"parameter:{task_dir}:{name}")
                     for name, value in plugin.parameters.items()}
            evaluated = st.form_submit_button("Evaluate")
        if evaluated:
            model, records = evaluate_point(task_dir, point)
            if model.report.valid:
                st.info(f"Plugin-only at this point: {VERIFICATION['plugin-only']} The agent's verification level "
                        "holds only at its own point, where the card's numbers were written.")
                for record in records:
                    st.markdown(f"**{record['experiment']}**: Δ = {record['delta']:.6g}, predicted total "
                                f"{record['predicted_total']:.6g} events, observed {record['observed_total']:g}")
                    st.json({"new_physics": record["new_physics"], "sm": record["sm"]})
                with spectra:
                    st.caption("Spectrum at the evaluated point (plugin-only).")
                    show_spectra(model.signals)
            else:
                for finding in model.report.findings:
                    st.error(finding)
                with spectra:
                    show_spectra(result.signals)
        else:
            with spectra:
                show_spectra(result.signals)
