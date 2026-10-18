"""The ui launcher command and the Streamlit page driven headlessly with a fake backend, without a server."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from test_agent import FakeBackend
from test_plugins import EQUIVALENT_CARD

import nuagent.agent.backend
from nuagent.experiments.registry import get
from nuagent.physics.modelcard import ModelCard
from nuagent.rates import predicted_counts
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


def page(monkeypatch, *modes: str) -> AppTest:
    backends = iter(modes)
    monkeypatch.setattr(nuagent.agent.backend, "get_backend", lambda name: FakeBackend(next(backends)))
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception
    app.text_area(key="agent_description").input("a vector mediator")
    return app


def run_agent(app: AppTest) -> AppTest:
    return next(widget for widget in app.form_submit_button if widget.label == "Run agent").click().run()


def markdown_texts(app: AppTest) -> list[str]:
    return [markdown.value.strip() for markdown in app.markdown]


@pytest.mark.parametrize("mode, valid", [("correct", True), ("sm-failure", False)])
def test_agent_panel_shows_report_validity_and_spectrum(monkeypatch, mode, valid):
    app = run_agent(page(monkeypatch, mode))
    assert not app.exception
    root, shown_valid, report, signals = app.session_state["agent_result"]
    assert shown_valid is valid
    assert report == (root / "report.md").read_text()
    for _ in range(2):
        assert (len(app.success), len(app.error)) == ((1, 0) if valid else (0, 1))
        assert report.strip() in markdown_texts(app)
        assert len(app.get("image")) == (1 if valid else 0)
        app.text_area(key="agent_description").input("an edited description").run()
    if valid:
        experiment = get("coherent_csi_2017")
        expected = predicted_counts(experiment, ModelCard.model_validate(EQUIVALENT_CARD).coefficients)
        np.testing.assert_allclose(signals["coherent_csi_2017"], expected, rtol=1e-12)
    else:
        assert signals == {}


def test_agent_panel_clears_result_when_a_later_run_fails(monkeypatch):
    app = run_agent(page(monkeypatch, "correct", "backend-failure"))
    report = app.session_state["agent_result"][2]
    assert len(app.success) == 1
    assert run_agent(app).exception
    app.text_area(key="agent_description").input("an edited description").run()
    assert not app.exception
    assert "agent_result" not in app.session_state
    assert not app.success and not app.get("image")
    assert report.strip() not in markdown_texts(app)


def test_agent_inputs_and_run_button_are_in_one_form(monkeypatch):
    app = page(monkeypatch)
    form = app.get("form")[0]
    assert [widget.label for widget in form.button] == ["Run agent"]
    assert [widget.key for widget in (*form.text_area, *form.selectbox, *form.multiselect)] == [
        "agent_description", "agent_backend", "agent_experiments",
    ]
    assert [(widget.label, widget.form_id) for widget in app.button] == [("Run agent", "agent")]


def test_agent_run_needs_an_experiment(monkeypatch):
    calls = []
    monkeypatch.setattr(nuagent.agent.backend, "get_backend", lambda name: calls.append(name))
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    app.multiselect(key="agent_experiments").set_value([])
    app = run_agent(app)
    assert not app.exception
    assert [error.value for error in app.error] == ["Choose at least one experiment."]
    assert calls == [] and "agent_result" not in app.session_state
