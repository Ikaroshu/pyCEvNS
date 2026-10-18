"""The ui launcher command and the Streamlit page driven headlessly with a fake backend, without a server."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from test_agent import QUESTIONS, FakeBackend, snapshot
from test_plugins import EQUIVALENT_CARD

import nuagent.agent.backend
import nuagent.physics.plugin
from nuagent.agent.run import evaluate_point, read_conversation
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


def page(monkeypatch, *modes: str, calls: list | None = None) -> AppTest:
    backends = iter(modes)
    def get_backend(name):
        if calls is not None:
            calls.append(name)
        return FakeBackend(next(backends), name=name)
    monkeypatch.setattr(nuagent.agent.backend, "get_backend", get_backend)
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
    result = app.session_state["agent_result"]
    root = app.session_state["agent_task"].parent
    report, signals = result.report, result.signals
    assert result.valid is valid
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
    report = app.session_state["agent_result"].report
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


def submit(app: AppTest, label: str) -> AppTest:
    return next(widget for widget in app.button if widget.label == label).click().run()


def test_question_form_runs_the_next_round_on_the_recorded_backend(monkeypatch):
    calls = []
    app = page(monkeypatch, "questions", "questions", calls=calls)
    app.selectbox(key="agent_backend").set_value("codex")
    app = run_agent(app)
    assert not app.exception
    task = app.session_state["agent_task"]
    assert app.session_state["agent_result"].state == "questions"
    assert any("Round 1: questions" in text for text in markdown_texts(app))
    numbers = {widget.key.rsplit(":", 1)[1]: widget for widget in app.number_input}
    assert {name: widget.value for name, widget in numbers.items()} == {"mass_mev": 10.0, "g_nu": 1e-4}
    assert all(widget.proto.format == "%g" for widget in numbers.values())
    texts = {widget.key.rsplit(":", 1)[1]: widget for widget in app.text_input}
    assert texts["g_q"].value == ""
    assert all(widget.key.startswith(f"answer:{task}:1:") for widget in (*app.number_input, *app.text_input, *app.text_area[1:]))
    assert {widget.form_id for widget in (*app.number_input, *app.text_input, *app.text_area[1:])} == {"questions"}
    numbers["mass_mev"].set_value(30.0)
    texts["g_q"].input("1e-4")
    app = submit(app, "Submit answers")
    assert not app.exception
    result = app.session_state["agent_result"]
    assert (result.state, result.round) == ("valid", 2)
    assert calls == ["codex", "codex"]
    assert read_conversation(task)["rounds"][0]["answers"] == {"mass_mev": 30.0, "g_nu": 1e-4, "g_q": "1e-4"}
    assert len(app.success) == 1 and len(app.get("image")) == 1
    assert result.report.strip() in markdown_texts(app)
    assert not [widget for widget in app.number_input if widget.key.startswith("answer:")]


def test_parameter_form_evaluates_a_new_point_without_changing_the_task(monkeypatch):
    app = run_agent(page(monkeypatch, "correct"))
    task = app.session_state["agent_task"]
    plugin = nuagent.physics.plugin.load_coefficient_plugin(task / "plugin.py")
    inputs = {widget.key: widget for widget in app.number_input}
    assert {key: widget.value for key, widget in inputs.items()} == {f"parameter:{task}:{name}": float(value) for name, value in plugin.parameters.items()}
    assert all(widget.proto.format == "%g" and widget.form_id == "parameters" for widget in inputs.values())
    before = snapshot(task, task.parent / "report.md")
    agent_image = app.get("image")[0].proto.SerializeToString()
    inputs[f"parameter:{task}:mass_mev"].set_value(50.0)
    app = submit(app, "Evaluate")
    assert not app.exception
    assert any("Plugin-only at this point" in info.value for info in app.info)
    images = app.get("image")
    assert len(images) == 1 and images[0].proto.SerializeToString() != agent_image
    _, records = evaluate_point(task, plugin.parameters | {"mass_mev": 50.0})
    assert any(f"Δ = {records[0]['delta']:.6g}" in text for text in markdown_texts(app))
    assert snapshot(task, task.parent / "report.md") == before


def test_failed_point_shows_findings_and_no_new_spectrum(monkeypatch):
    app = run_agent(page(monkeypatch, "sm-failure-heavy"))
    task = app.session_state["agent_task"]
    app.number_input(key=f"parameter:{task}:mass_mev").set_value(50.0)
    app = submit(app, "Evaluate")
    assert not app.exception
    assert any("SM-limit counts differ" in error.value for error in app.error)
    assert not app.info


def test_each_run_gets_fresh_inputs_with_its_own_values(monkeypatch):
    app = run_agent(page(monkeypatch, "correct", "plugin-only-alt", "questions", "correct"))
    first = app.session_state["agent_task"]
    assert app.number_input(key=f"parameter:{first}:g_nu").value == 1e-4
    app = run_agent(app)
    second = app.session_state["agent_task"]
    assert second != first
    assert app.number_input(key=f"parameter:{second}:g_nu").value == 2e-4
    assert app.number_input(key=f"parameter:{second}:mass_mev").value == 20.0
    assert all(widget.key.startswith(f"parameter:{second}:") for widget in app.number_input)
    app = run_agent(app)
    assert app.session_state["agent_result"].state == "questions"
    app = run_agent(app)
    assert app.session_state["agent_result"].state == "valid"
    assert not [widget for widget in (*app.number_input, *app.text_input) if widget.key.startswith("answer:")]
    assert not any("Round 1: questions" in text for text in markdown_texts(app))


def test_unresolved_conversation_says_the_round_limit_was_reached(monkeypatch):
    app = run_agent(page(monkeypatch, *["questions-always"] * 3))
    for _ in range(2):
        app = submit(app, "Submit answers")
    assert not app.exception
    assert app.session_state["agent_result"].state == "unresolved"
    assert [warning.value for warning in app.warning] == ["The round limit of 3 was reached; the last questions remain open."]
    assert not [widget for widget in app.button if widget.label == "Submit answers"]
