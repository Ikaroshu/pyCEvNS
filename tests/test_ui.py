"""The ui launcher command and the Streamlit page driven headlessly, without a server."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from test_agent import FakeBackend

import nuagent.agent.backend

from nuagent.experiments.registry import get
from nuagent.physics.modelcard import template_card
from nuagent.rates import predicted_counts
from nuagent.stats import asimov
from nuagent.stats.asimov import Scan
from nuagent.stats.likelihood import profiled
from nuagent.ui.figures import contour_figure
from nuagent.ui.launch import APP_PATH, streamlit_command


def test_streamlit_command_binds_loopback_without_usage_statistics():
    command = streamlit_command(8600)
    assert command[:5] == [sys.executable, "-m", "streamlit", "run", str(APP_PATH)]
    assert Path(command[4]).is_file()
    pairs = dict(zip(command[5::2], command[6::2]))
    assert pairs == {
        "--server.address": "127.0.0.1", "--browser.gatherUsageStats": "false",
        "--server.showEmailPrompt": "false", "--server.port": "8600",
    }


def test_ui_help():
    result = subprocess.run([str(Path(sys.executable).with_name("nuagent")), "ui", "--help"], check=True, capture_output=True, text=True)
    assert "--port" in result.stdout


def page() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception
    return app


def button(app: AppTest, label: str):
    return next(widget for widget in app.button if widget.label == label)


def test_preset_spectrum_and_delta():
    app = page()
    parameters = {"mass_mev": 10.0, "g_prime": 1e-4}
    app.selectbox(key="preset").select("zprime_lmu_ltau").run()
    button(app, "Apply preset to card").click().run()
    card = template_card("zprime_lmu_ltau", parameters)
    assert app.text_area(key="card").value == card.model_dump_json(indent=2)
    button(app, "Predict and compare").click().run()
    assert not app.exception
    experiment = get("coherent_csi_2017")
    expected = (profiled(experiment, predicted_counts(experiment, card.coefficients)).neg2_log_likelihood
                - profiled(experiment, predicted_counts(experiment)).neg2_log_likelihood)
    assert float(app.metric[0].value) == pytest.approx(expected, rel=1e-9)


def small_scan(app: AppTest) -> dict[str, np.ndarray]:
    app.number_input(key="scan_zprime_lmu_ltau_0_mass_mev_n").set_value(2)
    app.number_input(key="scan_zprime_lmu_ltau_1_g_prime_n").set_value(2)
    return {"mass_mev": np.geomspace(1, 1000, 2), "g_prime": np.geomspace(1e-5, 1e-2, 2)}


def shown_delta(app: AppTest) -> np.ndarray:
    return app.dataframe[0].value.iloc[:, 1:].to_numpy()


@pytest.mark.parametrize("data, label", [("asimov", "expected"), ("observed", "observed")])
def test_scan_panel_labels_contours(data, label):
    app = page()
    grid = small_scan(app)
    app.radio(key="scan_data").set_value(data).run()
    assert "Grid: 2 × 2 = 4 points" in [markdown.value for markdown in app.markdown]
    button(app, "Run scan").click().run()
    assert not app.exception
    assert any(f"on {data} data ({label})" in caption.value for caption in app.caption)
    expected = asimov.scan(get("coherent_csi_2017"), "zprime_lmu_ltau", grid, {}, data=data).delta
    np.testing.assert_allclose(shown_delta(app), expected, rtol=1e-9)


def test_scan_fixed_parameter_and_cleared_default():
    app = page()
    grid = small_scan(app)
    experiment = get("coherent_csi_2017")
    mixing = app.number_input(key="scan_zprime_lmu_ltau_kinetic_mixing")
    assert mixing.value is None
    mixing.set_value(1e-3)
    button(app, "Run scan").click().run()
    fixed = asimov.scan(experiment, "zprime_lmu_ltau", grid, {"kinetic_mixing": 1e-3}).delta
    np.testing.assert_allclose(shown_delta(app), fixed, rtol=1e-9)
    app.number_input(key="scan_zprime_lmu_ltau_kinetic_mixing").set_value(None)
    button(app, "Run scan").click().run()
    loop = asimov.scan(experiment, "zprime_lmu_ltau", grid, {}).delta
    np.testing.assert_allclose(shown_delta(app), loop, rtol=1e-9)
    assert not np.allclose(loop, fixed)


@pytest.mark.parametrize("mode, valid", [("correct", True), ("sm-failure", False)])
def test_agent_panel_shows_report_and_validity(monkeypatch, mode, valid):
    monkeypatch.setattr(nuagent.agent.backend, "get_backend", lambda name: FakeBackend(mode))
    app = page()
    app.text_area(key="agent_description").input("a vector mediator")
    button(app, "Run agent").click().run()
    assert not app.exception
    root, shown_valid, report = app.session_state["agent_result"]
    assert shown_valid is valid
    assert report == (root / "report.md").read_text()
    message = app.success if valid else app.error
    assert len(message) == 1
    assert report.strip() in [markdown.value.strip() for markdown in app.markdown]
    app.text_area(key="agent_description").input("an edited description").run()
    assert (len(app.success) == 1) is valid and (len(app.error) == 1) is not valid
    assert report.strip() in [markdown.value.strip() for markdown in app.markdown]


def test_contour_figure_labels_and_log_axes():
    scan = Scan({"mass_mev": np.geomspace(1, 100, 3), "g_prime": np.linspace(0, 1, 3)}, np.arange(9.0).reshape(3, 3))
    axes = contour_figure(scan, [4.61, 2.71], "observed", (True, False)).axes[0]
    assert [text.get_text() for text in axes.get_legend().get_texts()] == ["Δ = 2.71 (observed)", "Δ = 4.61 (observed)"]
    assert (axes.get_xscale(), axes.get_yscale(), axes.get_xlabel(), axes.get_ylabel()) == ("log", "linear", "mass_mev", "g_prime")


def test_agent_panel_clears_result_when_a_later_run_fails(monkeypatch):
    modes = iter(["correct", "backend-failure"])
    monkeypatch.setattr(nuagent.agent.backend, "get_backend", lambda name: FakeBackend(next(modes)))
    app = page()
    app.text_area(key="agent_description").input("a vector mediator")
    button(app, "Run agent").click().run()
    report = app.session_state["agent_result"][2]
    assert len(app.success) == 1
    button(app, "Run agent").click().run()
    assert app.exception
    app.text_area(key="agent_description").input("an edited description").run()
    assert not app.exception
    assert "agent_result" not in app.session_state
    assert not app.success
    assert report.strip() not in [markdown.value.strip() for markdown in app.markdown]
