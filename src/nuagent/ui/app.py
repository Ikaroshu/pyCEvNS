"""Streamlit page: predict and compare a model card, scan a template, and propose a model with a coding agent."""

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from nuagent.agent.backend import get_backend
from nuagent.agent.run import run_result
from nuagent.experiments import registry
from nuagent.experiments.registry import Experiment
from nuagent.physics.modelcard import TEMPLATES, ModelCard, template_card
from nuagent.rates import predicted_counts
from nuagent.stats import asimov
from nuagent.stats.likelihood import profile_record, profiled
from nuagent.ui.figures import DATA_LABELS, contour_figure, spectrum_figure

PRESETS = {
    "zprime_universal": {"mass_mev": 30.0, "g_nu": 1e-4, "g_q": 1e-4},
    "zprime_u1": {"mass_mev": 10.0, "g_prime": 1e-4, "q_nu_e": -1.0, "q_nu_mu": -1.0, "q_nu_tau": -1.0, "q_u": 1 / 3, "q_d": 1 / 3},
    "zprime_lmu_ltau": {"mass_mev": 10.0, "g_prime": 1e-4},
    "scalar_universal": {"mass_mev": 10.0, "g_nu": 1e-5, "g_q": 1e-5},
    "contact_nsi": {"eps_u_ee": 0.1, "eps_d_ee": 0.1},
}
GRID_POINTS = 8


def parameter_inputs(template: str, names: list[str], key: str) -> dict[str, float]:
    """Number inputs for template parameters, prefilled from the preset; an empty input keeps the template default."""
    columns = st.columns(3)
    values = {
        name: columns[i % 3].number_input(name, value=PRESETS[template].get(name), format="%g", key=f"{key}_{template}_{name}")
        for i, name in enumerate(names)
    }
    return {name: value for name, value in values.items() if value is not None}


def apply_preset(template: str, parameters: dict[str, float]) -> None:
    st.session_state.card = template_card(template, parameters).model_dump_json(indent=2)


@st.fragment
def model_panel(experiment: Experiment) -> None:
    st.header("Model card, spectrum and Δ")
    template = st.selectbox("Template preset", list(TEMPLATES), key="preset")
    parameters = parameter_inputs(template, list(TEMPLATES[template].parameters), "preset")
    st.button("Apply preset to card", on_click=apply_preset, args=(template, parameters))
    if "card" not in st.session_state:
        apply_preset("zprime_universal", PRESETS["zprime_universal"])
    card_text = st.text_area("Model card (JSON)", key="card", height=320)
    if st.button("Predict and compare"):
        card = ModelCard.model_validate_json(card_text)
        signal = predicted_counts(experiment, card.coefficients)
        sm_signal = predicted_counts(experiment)
        new_physics, standard = profiled(experiment, signal), profiled(experiment, sm_signal)
        st.pyplot(spectrum_figure(experiment, signal, sm_signal), width=760)
        st.metric("Δ = −2 ln L(new physics) − (−2 ln L(SM)) on observed data", repr(new_physics.neg2_log_likelihood - standard.neg2_log_likelihood))
        st.json({
            "new_physics": profile_record(new_physics, experiment), "sm": profile_record(standard, experiment),
            "totals": {"signal": float(signal.sum()), "sm_signal": float(sm_signal.sum()), "observed": int(experiment.observed.sum())},
        })


@st.fragment
def scan_panel(experiment: Experiment) -> None:
    st.header("Scan")
    template = st.selectbox("Template", list(TEMPLATES), index=list(TEMPLATES).index("zprime_lmu_ltau"), key="scan_template")
    names = list(TEMPLATES[template].parameters)
    defaults = ["mass_mev", TEMPLATES[template].couplings[0]] if "mass_mev" in names else list(TEMPLATES[template].couplings[:2])
    grid, log_axes = {}, []
    for index, (column, default) in enumerate(zip(st.columns(2), defaults)):
        name = column.selectbox(f"Parameter {index + 1}", names, index=names.index(default), key=f"scan_{template}_{index}")
        lower, upper = (1.0, 1000.0) if name == "mass_mev" else (1e-5, 1e-2)
        key = f"scan_{template}_{index}_{name}"
        lower = column.number_input("min", value=lower, format="%g", key=f"{key}_min")
        upper = column.number_input("max", value=upper, format="%g", key=f"{key}_max")
        count = column.number_input("n", value=GRID_POINTS, min_value=2, step=1, key=f"{key}_n")
        log = column.checkbox("log spacing", value=True, key=f"{key}_log")
        grid[name] = (np.geomspace if log else np.linspace)(lower, upper, count)
        log_axes.append(log)
    st.caption("Fixed parameters; an empty input keeps the template default.")
    fixed = parameter_inputs(template, [name for name in names if name not in grid], "scan")
    data = st.radio("Data", ("asimov", "observed"), horizontal=True, key="scan_data")
    columns = st.columns(3)
    thresholds = [
        columns[0].number_input("Threshold 1", value=2.71, format="%g", key="scan_threshold_1"),
        columns[1].number_input("Threshold 2", value=4.61, format="%g", key="scan_threshold_2"),
        columns[2].number_input("Threshold 3 (optional)", value=None, format="%g", key="scan_threshold_3"),
    ]
    thresholds = [threshold for threshold in thresholds if threshold is not None]
    run_column, size_column = st.columns(2)
    size_column.write(f"Grid: {' × '.join(str(len(axis)) for axis in grid.values())} = {np.prod([len(axis) for axis in grid.values()])} points")
    if run_column.button("Run scan"):
        result = asimov.scan(experiment, template, grid, fixed, data=data)
        st.pyplot(contour_figure(result, thresholds, data, tuple(log_axes)), width=640)
        st.caption(
            f"Contours of Δ at {', '.join(f'{t:g}' for t in sorted(thresholds))} on {data} data "
            f"({DATA_LABELS[data]}); Δ range {result.delta.min():.4g} to {result.delta.max():.4g}."
        )
        (x_name, x), (y_name, y) = result.axes.items()
        with st.expander("Δ grid"):
            st.dataframe({x_name: x, **{f"{y_name}={value:g}": column for value, column in zip(y, result.delta.T)}})


@st.fragment
def agent_panel() -> None:
    st.header("Propose with a coding agent")
    description = st.text_area("Interaction description", key="agent_description")
    backend = st.selectbox("Backend", ("claude_code", "codex"), key="agent_backend")
    experiments = st.multiselect("Experiments", registry.names(), default=list(registry.names()), key="agent_experiments")
    if st.button("Run agent"):
        root = Path(tempfile.mkdtemp(prefix="nuagent-ui-"))
        with st.spinner(f"Running {backend} in {root}; this takes minutes"):
            result = run_result(description, experiments, get_backend(backend), root / "task", root / "report.md")
        # Kept in session state so a rerun queued by a widget edit during the run does not clear it.
        st.session_state.agent_result = (root, result.valid, result.report)
    if "agent_result" in st.session_state:
        root, valid, report = st.session_state.agent_result
        st.write(f"Task directory: `{root / 'task'}`; report: `{root / 'report.md'}`")
        if valid:
            st.success("Outputs passed validation.")
        else:
            st.error("Outputs failed validation after the one correction attempt; the report has no predictions.")
        st.markdown(report)


st.set_page_config(page_title="nuagent", layout="wide")
st.title("nuagent")
experiment = registry.get(st.selectbox("Experiment", registry.names(), key="experiment"))
model_panel(experiment)
scan_panel(experiment)
agent_panel()
