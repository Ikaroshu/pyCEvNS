"""Streamlit page: describe an interaction, run the coding-agent harness, read the report and see the spectrum."""

import tempfile
from pathlib import Path

import streamlit as st

from nuagent.agent.backend import get_backend
from nuagent.agent.run import run_result
from nuagent.experiments import registry
from nuagent.rates import predicted_counts
from nuagent.ui.figures import spectrum_figure

st.set_page_config(page_title="nuagent")
st.title("nuagent")
description = st.text_area("Interaction description", key="agent_description")
backend = st.selectbox("Backend", ("claude", "codex"), key="agent_backend")
experiments = st.multiselect("Experiments", registry.names(), default=list(registry.names()), key="agent_experiments")
if st.button("Run agent"):
    st.session_state.pop("agent_result", None)
    root = Path(tempfile.mkdtemp(prefix="nuagent-ui-"))
    with st.spinner(f"Running {backend} in {root}; this takes minutes"):
        result = run_result(description, experiments, get_backend(backend), root / "task", root / "report.md")
    # Kept in session state so a rerun queued by a widget edit during the run does not clear it.
    st.session_state.agent_result = (root, result.valid, result.report, result.signals)
if "agent_result" in st.session_state:
    root, valid, report, signals = st.session_state.agent_result
    st.write(f"Task directory: `{root / 'task'}`; report: `{root / 'report.md'}`")
    if valid:
        st.success("Outputs passed validation.")
    else:
        st.error("Outputs failed validation after the one correction attempt; the report has no predictions.")
    st.markdown(report)
    for name, signal in signals.items():
        experiment = registry.get(name)
        st.pyplot(spectrum_figure(experiment, signal, predicted_counts(experiment)), width=760)
