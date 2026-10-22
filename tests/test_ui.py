"""The ui launcher command and the Streamlit page driven headlessly with a fake backend, without a server."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from test_agent import NOTES, QUESTIONS, FakeBackend, snapshot
from test_loop import LoopBackend
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


def page_text(app: AppTest) -> str:
    """Every text-bearing element on the page, for checks that some text is absent."""
    elements = (*app.markdown, *app.caption, *app.code, *app.subheader, *app.info, *app.warning, *app.error, *app.success, *app.json)
    return "\n".join(str(element.value) for element in elements)


def block_sequence(block) -> list[str]:
    """Element types in a block's order, with forms and expanders as one entry each."""
    sequence = []
    for child in block.children.values():
        kind = type(child).__name__
        if kind == "Block":
            kind = "Form"
        sequence.append(kind if kind != "Expander" else f"Expander:{child.label}")
    return sequence


def result_section(app: AppTest):
    """The result block, a keyed page section after the conversation, not a chat message."""
    sections = [child for child in app.main.children.values() if getattr(child.proto, "id", "").endswith("-result")]
    assert len(sections) == 1
    return sections[0]


CROSS_VALIDATED = "Verification: cross-validated (the plugin and the model card agree at the agent's values)."


def submit(app: AppTest, label: str) -> AppTest:
    return next(widget for widget in app.button if widget.label == label).click().run()


@pytest.mark.parametrize("mode, valid", [("correct", True), ("sm-failure", False)])
def test_result_is_a_chat_with_the_result_block(monkeypatch, mode, valid):
    app = run_agent(page(monkeypatch, mode))
    assert not app.exception
    result = app.session_state["agent_result"]
    assert result.valid is valid
    assert result.report == (app.session_state["agent_task"].parent / "report.md").read_text()
    for _ in range(2):
        assert [message.name for message in app.chat_message] == (["user"] if valid else ["user", "assistant"])
        assert app.chat_message[0].markdown[0].value == "a vector mediator"
        if valid:
            assert list(app.main.children.values())[-1] is result_section(app)
        assert len(app.get("image")) == (1 if valid else 0)
        assert not (app.success or app.warning or app.info or app.json)
        text = page_text(app)
        for absent in ("Validation and derivation history", "Backend CLI records", "total_cost_usd", "fake-model", "Round 1", '"findings"'):
            assert absent not in text
        app.text_area(key="agent_description").input("an edited description").run()
    if valid:
        experiment = get("coherent_csi_2017")
        expected = predicted_counts(experiment, ModelCard.model_validate(EQUIVALENT_CARD).coefficients)
        np.testing.assert_allclose(result.signals["coherent_csi_2017"], expected, rtol=1e-12)
    else:
        assert result.signals == {}
        assert "failed validation" in app.chat_message[1].markdown[0].value


def test_result_block_order_table_and_notes(monkeypatch):
    app = run_agent(page(monkeypatch, "correct"))
    block = result_section(app)
    assert not any(message.get("image") for message in app.chat_message)
    assert block_sequence(block) == ["Form", "Form", "Image", "Markdown", "Caption", "Caption", "Subheader", "Markdown",
                                     "Expander:Provenance", 
                                     "Expander:Plugin (plugin.py): the coefficient code the agent wrote",
                                     "Expander:Model card (card.json): the interaction written in the fixed schema the harness cross-checks against the plugin"]
    assert [subheader.value for subheader in block.subheader] == ["Evaluate the interaction", "Note"]
    table, notes = block.markdown[0].value, block.markdown[1].value
    record = app.session_state["agent_result"].evaluations[0]
    for column in ("Observed", "SM signal", "New-physics signal", "Δ", "Signal scale α (new physics / SM)",
                   "Background scale β, steady state (new physics / SM)", "Background scale β, prompt neutrons (new physics / SM)"):
        assert column in table
    assert f"{record['delta']:.6g}" in table and f"{record['new_physics']['alpha']:.4g} / {record['sm']['alpha']:.4g}" in table
    assert block.caption[0].value == CROSS_VALIDATED
    assert "α scales the signal prediction" in block.caption[1].value and "each β scales one background" in block.caption[1].value
    assert "A fake mediator" not in page_text(app)
    assert notes.strip() == NOTES.split("\n", 2)[2].strip() and "## Matching" in notes and "$C = g_\\nu" in notes and "```" not in notes
    assert all(not expander.proto.expanded for expander in app.expander)


def test_agent_panel_clears_result_when_a_later_run_fails(monkeypatch):
    app = run_agent(page(monkeypatch, "correct", "backend-failure"))
    assert len(app.get("image")) == 1
    assert run_agent(app).exception
    app.text_area(key="agent_description").input("an edited description").run()
    assert not app.exception
    assert "agent_result" not in app.session_state
    assert not app.chat_message and not app.get("image")


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


def test_question_form_runs_the_next_round_on_the_recorded_backend(monkeypatch):
    calls = []
    app = page(monkeypatch, "questions", "questions", calls=calls)
    app.selectbox(key="agent_backend").set_value("codex")
    app = run_agent(app)
    assert not app.exception
    task = app.session_state["agent_task"]
    assert app.session_state["agent_result"].state == "questions"
    assert [message.name for message in app.chat_message] == ["user", "assistant"]
    numbers = {widget.key.rsplit(":", 1)[1]: widget for widget in app.number_input}
    assert {name: widget.value for name, widget in numbers.items()} == {"mass_mev": 10.0, "g_nu": 1e-4}
    assert all(widget.proto.format == "%g" for widget in numbers.values())
    texts = {widget.key.rsplit(":", 1)[1]: widget for widget in app.text_input}
    assert texts["g_q"].value == ""
    answer_inputs = (*app.number_input, *app.text_input, *app.text_area[1:])
    assert all(widget.key.startswith(f"answer:{task}:1:") for widget in answer_inputs)
    assert {widget.form_id for widget in answer_inputs} == {"questions"}
    assert "Round" not in page_text(app)
    numbers["mass_mev"].set_value(30.0)
    texts["g_q"].input("1e-4")
    app = submit(app, "Submit answers")
    assert not app.exception
    result = app.session_state["agent_result"]
    assert (result.state, result.round) == ("valid", 2)
    assert calls == ["codex", "codex"]
    assert read_conversation(task)["rounds"][0]["answers"] == {"mass_mev": 30.0, "g_nu": 1e-4, "g_q": "1e-4"}
    assert [message.name for message in app.chat_message] == ["user", "assistant", "user"]
    assert "mass_mev: 30.0 MeV" in app.chat_message[2].markdown[0].value
    assert "What is the coupling to quarks?" in app.chat_message[1].markdown[0].value
    assert len(app.get("image")) == 1
    assert "Round" not in page_text(app)
    assert not [widget for widget in app.number_input if widget.key.startswith("answer:")]


def test_evaluate_updates_figure_and_table_in_place_without_changing_the_task(monkeypatch):
    app = run_agent(page(monkeypatch, "correct"))
    task = app.session_state["agent_task"]
    plugin = nuagent.physics.plugin.load_coefficient_plugin(task / "plugin.py")
    inputs = {widget.key: widget for widget in app.number_input if widget.key.startswith("parameter:")}
    assert {key: widget.value for key, widget in inputs.items()} == {f"parameter:{task}:{name}": float(value) for name, value in plugin.parameters.items()}
    assert all(widget.proto.format == "%g" and widget.form_id == "parameters" for widget in inputs.values())
    before = snapshot(task, task.parent / "report.md")
    agent_image = app.get("image")[0].proto.SerializeToString()
    agent_table = result_section(app).markdown[0].value
    inputs[f"parameter:{task}:mass_mev"].set_value(50.0)
    app = submit(app, "Evaluate")
    assert not app.exception
    block = result_section(app)
    assert block_sequence(block)[:6] == ["Form", "Form", "Image", "Markdown", "Caption", "Caption"]
    images = app.get("image")
    assert len(images) == 1 and images[0].proto.SerializeToString() != agent_image
    _, records = evaluate_point(task, plugin.parameters | {"mass_mev": 50.0})
    table = block.markdown[0].value
    assert table != agent_table and f"{records[0]['delta']:.6g}" in table
    assert block.caption[0].value == ("Verification at the agent's values: cross-validated (the plugin and the model card agree at the agent's values). "
                                      "At these values: plugin-only (only the plugin was checked here; the model-card cross-check holds at the agent's values only).")
    assert not (app.info or app.warning or app.json)
    assert snapshot(task, task.parent / "report.md") == before
    app.number_input(key=f"parameter:{task}:mass_mev").set_value(30.0)
    app = submit(app, "Evaluate")
    assert result_section(app).caption[0].value == CROSS_VALIDATED
    assert result_section(app).markdown[0].value == agent_table


def test_failed_point_shows_findings_and_keeps_the_agent_point(monkeypatch):
    app = run_agent(page(monkeypatch, "sm-failure-heavy"))
    task = app.session_state["agent_task"]
    agent_table = result_section(app).markdown[0].value
    app.number_input(key=f"parameter:{task}:mass_mev").set_value(50.0)
    app = submit(app, "Evaluate")
    assert not app.exception
    block = result_section(app)
    assert "SM-limit counts differ" in block.markdown[0].value
    assert block.markdown[1].value == agent_table
    assert block.caption[0].value == "Verification: plugin-only (no model card to cross-check; the plugin's own checks pass at the agent's values)."


def test_each_run_gets_fresh_inputs_with_its_own_values(monkeypatch):
    app = run_agent(page(monkeypatch, "correct", "plugin-only-alt", "questions", "correct"))
    first = app.session_state["agent_task"]
    assert app.number_input(key=f"parameter:{first}:g_nu").value == 1e-4
    app = run_agent(app)
    second = app.session_state["agent_task"]
    assert second != first
    assert app.number_input(key=f"parameter:{second}:g_nu").value == 2e-4
    assert app.number_input(key=f"parameter:{second}:mass_mev").value == 20.0
    assert all(widget.key.startswith(f"parameter:{second}:") for widget in app.number_input if not widget.key.startswith("loop_"))
    app = run_agent(app)
    assert app.session_state["agent_result"].state == "questions"
    app = run_agent(app)
    assert app.session_state["agent_result"].state == "valid"
    assert not [widget for widget in (*app.number_input, *app.text_input) if widget.key.startswith("answer:")]
    assert [message.name for message in app.chat_message] == ["user"]


def test_unresolved_conversation_says_the_round_limit_was_reached(monkeypatch):
    app = run_agent(page(monkeypatch, *["questions-always"] * 3))
    for _ in range(2):
        app = submit(app, "Submit answers")
    assert not app.exception
    assert app.session_state["agent_result"].state == "unresolved"
    assert [message.name for message in app.chat_message] == ["user", "assistant", "user", "assistant", "user", "assistant"]
    assert any("The limit of 3 rounds was reached" in markdown.value for markdown in app.chat_message[-1].markdown)
    assert not [widget for widget in app.button if widget.label == "Submit answers"]


def test_plugin_only_evaluation_at_new_values_names_no_card(monkeypatch):
    app = run_agent(page(monkeypatch, "plugin-only"))
    task = app.session_state["agent_task"]
    plugin_only = "Verification: plugin-only (no model card to cross-check; the plugin's own checks pass at the agent's values)."
    assert result_section(app).caption[0].value == plugin_only
    assert not any(expander.label.startswith("Model card") for expander in app.expander)
    app.number_input(key=f"parameter:{task}:mass_mev").set_value(50.0)
    app = submit(app, "Evaluate")
    caption = result_section(app).caption[0].value
    assert caption == ("Verification at the agent's values: plugin-only (no model card to cross-check; the plugin's own checks pass "
                       "at the agent's values). At these values: plugin-only (only the plugin was checked here).")
    assert "cross-check holds" not in caption


def test_failed_continuation_leaves_task_without_result(monkeypatch):
    app = run_agent(page(monkeypatch, "questions", "no-notes"))
    task = app.session_state["agent_task"]
    app = submit(app, "Submit answers")
    assert app.exception
    assert app.session_state["agent_task"] == task and "agent_result" not in app.session_state
    assert read_conversation(task)["rounds"][-1]["state"] == "backend-failure"
    app.text_area(key="agent_description").input("an edited description").run()
    assert not app.exception
    assert not app.chat_message and not app.get("image")


def test_notes_title_is_dropped_only_when_it_opens_the_notes(monkeypatch):
    app = run_agent(page(monkeypatch, "correct"))
    task = app.session_state["agent_task"]
    (task / "notes.md").write_text("Intro line.\n\n# Not a title\n\nBody.")
    app = submit(app, "Evaluate")
    notes = result_section(app).markdown[1].value
    assert "# Not a title" in notes and notes.startswith("Intro line.")
    (task / "notes.md").write_text("\n\n# Title\n\nBody.")
    app = submit(app, "Evaluate")
    assert result_section(app).markdown[1].value.strip() == "Body."


LOOP_LABEL = "Run agent loop"


def loop_page(monkeypatch, *backends) -> tuple[AppTest, list]:
    """A page that hands out `backends` in order, one per get_backend call, and runs the agent with the first."""
    calls = []
    def get_backend(name):
        calls.append(name)
        return backends[len(calls) - 1]
    monkeypatch.setattr(nuagent.agent.backend, "get_backend", get_backend)
    app = AppTest.from_file(APP_PATH, default_timeout=120).run()
    app.text_area(key="agent_description").input("a vector mediator")
    return run_agent(app), calls


def loop_expander_labels(app: AppTest) -> list[str]:
    return [expander.label for expander in result_section(app).expander if expander.label.startswith("Agent loop")]


@pytest.mark.parametrize("mode", ["sm-failure", "questions"])
def test_agent_loop_only_for_a_valid_result(monkeypatch, mode):
    app = run_agent(page(monkeypatch, mode))
    assert not [widget for widget in app.button if widget.label == LOOP_LABEL]
    assert not [widget for widget in app.number_input if widget.key.startswith("loop_")]


def test_agent_loop_sets_the_best_point_in_the_evaluate_section(monkeypatch):
    loop_backend = LoopBackend([{"g_nu": 2e-4}, {"g_nu": 5.0}], [{"mass_mev": 50.0}], [], name="claude")
    app, calls = loop_page(monkeypatch, FakeBackend("plugin-only", name="claude"), loop_backend, FakeBackend("correct", name="claude"))
    task = app.session_state["agent_task"]
    block = result_section(app)
    rounds = block.number_input(key=f"loop_rounds:{task}")
    assert (rounds.value, rounds.min, rounds.form_id) == (5, 1, "loop")
    assert block.text_area(key=f"loop_guidance:{task}").form_id == "loop"
    assert [button.label for button in block.button] == ["Evaluate", LOOP_LABEL]
    assert block_sequence(block)[:3] == ["Form", "Form", "Image"] and not loop_expander_labels(app)
    agent_table = block.markdown[0].value
    rounds.set_value(2)
    block.text_area(key=f"loop_guidance:{task}").input("prefer light mediators")
    app = submit(app, LOOP_LABEL)
    assert not app.exception
    assert calls == ["claude", "claude"]
    loop_dir = app.session_state["loop_dir"]
    assert loop_dir.parent.parent == task.parent and (loop_dir.parent / "report.md").read_text().startswith("Outcome: **completed**")
    assert "prefer light mediators" in (loop_dir / "TASK.md").read_text()
    history = json.loads((loop_dir / "history.json").read_text())
    best = min((entry for entry in history if entry["evaluations"]), key=lambda entry: entry["evaluations"][0]["delta"])
    assert best["round"] != 0
    block = result_section(app)
    assert {name: block.number_input(key=f"parameter:{task}:{name}").value for name in best["parameters"]} == best["parameters"]
    table = block.markdown[0].value
    assert table != agent_table and f"{best['evaluations'][0]['delta']:.6g}" in table
    assert block.caption[0].value == ("Verification at the agent's values: plugin-only (no model card to cross-check; the plugin's own checks pass "
                                      "at the agent's values). At these values: plugin-only (only the plugin was checked here).")
    # The research log, its records collapsed, then the trajectory, all after the likelihood table.
    assert block_sequence(block)[:20] == [
        "Form", "Form", "Image", "Markdown", "Caption", "Caption", "Subheader", "Markdown", "Markdown", "Markdown", "Info",
        "Expander:Agent loop: completed, best fit", "Expander:Agent loop: points by round",
        "Expander:Agent loop: notes, round 1 (evaluated)", "Expander:Agent loop: notes, round 2 (evaluated)",
        "Expander:Agent loop: notes, closing call (concluded)", "Expander:Agent loop: directory and report",
        "Subheader", "Image", "Image"]
    assert [subheader.value for subheader in block.subheader] == ["Evaluate the interaction", "Research log", "Agent loop trajectory", "Note"]
    assert block.markdown[1].value == f"**Round 0 · translation agent's point · Δ {history[0]['evaluations'][0]['delta']:.4g}**"
    first, second = block.markdown[2].value, block.markdown[3].value
    assert first.startswith("**Round 1 · 2 points, 1 passed checks · best Δ ") and first.endswith("Hypothesis: Hypothesis 1")
    assert second.startswith("**Round 2 · 1 points, 1 passed checks · best Δ ") and second.endswith("Hypothesis: Hypothesis 2")
    assert block.info[0].value == "**Conclusion.** Conclusion 3"
    outcome = block.expander[0].markdown[0].value
    assert outcome.startswith("Outcome: **completed** after 2 of 2 agent rounds; 4 points evaluated")
    assert f"round {best['round']}, point {best['index']}" in outcome and "| Round | Point |" in outcome
    assert block.expander[1].markdown[0].value.count("\n|") == 5
    assert str(loop_dir) in block.expander[5].markdown[0].value
    assert len(app.get("image")) == 3
    app = submit(app, "Evaluate")
    assert result_section(app).markdown[0].value == table
    app.number_input(key=f"parameter:{task}:mass_mev").set_value(30.0)
    app.number_input(key=f"parameter:{task}:g_nu").set_value(1e-4)
    app = submit(app, "Evaluate")
    assert result_section(app).markdown[0].value == agent_table
    app = run_agent(app)
    assert not app.exception
    assert "loop_dir" not in app.session_state and not loop_expander_labels(app)
    assert calls == ["claude"] * 3


def test_a_new_loop_clears_the_previous_one(monkeypatch):
    loop_backend = LoopBackend([{"g_nu": 2e-4}], [], "crash", name="claude")
    app, calls = loop_page(monkeypatch, FakeBackend("correct", name="claude"), loop_backend, loop_backend)
    task = app.session_state["agent_task"]
    app.number_input(key=f"loop_rounds:{task}").set_value(1)
    app = submit(app, LOOP_LABEL)
    assert loop_expander_labels(app)
    app = submit(app, LOOP_LABEL)
    assert app.exception
    assert "loop_dir" not in app.session_state
    app.text_area(key="agent_description").input("an edited description").run()
    assert not app.exception and not loop_expander_labels(app)


def test_round_zero_and_the_running_line_appear_before_round_one_ends(monkeypatch):
    # The fake backend raises inside round 1, freezing the page as it looked while round 1 was running.
    loop_backend = LoopBackend("crash", name="claude")
    app, calls = loop_page(monkeypatch, FakeBackend("correct", name="claude"), loop_backend)
    task = app.session_state["agent_task"]
    app.number_input(key=f"loop_rounds:{task}").set_value(3)
    app = submit(app, LOOP_LABEL)
    assert app.exception
    block = result_section(app)
    assert block.subheader[1].value == "Research log"
    assert block.markdown[0].value.startswith("**Round 0 · translation agent's point · Δ ")
    assert [status.label for status in block.status] == ["Round 1 of 3 running on claude…"]
    # The log grows in its own slot under the loop row: round 0, then the running line, then records and charts.
    progress = list(block.children.values())[2]
    assert block_sequence(progress) == ["Subheader", "Markdown", "Status", "Expander:Agent loop: points by round",
                                        "Expander:Agent loop: directory and report", "Subheader", "Image", "Image"]
    assert len(app.get("image")) == 2


@pytest.mark.parametrize("failure", ["failure", "nothing"])
def test_backend_failure_in_round_two_keeps_the_results(monkeypatch, failure):
    loop_backend = LoopBackend([{"g_nu": 2e-4}, {"mass_mev": 50.0}], failure, name="claude")
    app, calls = loop_page(monkeypatch, FakeBackend("correct", name="claude"), loop_backend)
    task = app.session_state["agent_task"]
    app.number_input(key=f"loop_rounds:{task}").set_value(3)
    app = submit(app, LOOP_LABEL)
    assert not app.exception
    loop_dir = app.session_state["loop_dir"]
    assert (loop_dir.parent / "report.md").read_text().startswith("Outcome: **stopped-backend-failure**")
    history = json.loads((loop_dir / "history.json").read_text())
    best = min((entry for entry in history if entry["evaluations"]), key=lambda entry: entry["evaluations"][0]["delta"])
    block = result_section(app)
    assert {name: block.number_input(key=f"parameter:{task}:{name}").value for name in best["parameters"]} == best["parameters"]
    error = "backend failed (exit 2): loop backend failure raw\nunavailable" if failure == "failure" else "backend produced no nonempty notes.md: done"
    assert block.markdown[3].value == f"**Round 2 · backend failure**  \nError: {error}"
    assert block.info[0].value == "No conclusion: the backend failed in round 2."
    assert "Agent loop: stopped-backend-failure, best fit" in loop_expander_labels(app)
    assert len(app.get("image")) == 3


@pytest.mark.parametrize("failure", ["failure", "nothing"])
def test_backend_failure_in_round_one_keeps_the_translation_point(monkeypatch, failure):
    loop_backend = LoopBackend(failure, name="claude")
    app, calls = loop_page(monkeypatch, FakeBackend("correct", name="claude"), loop_backend)
    task = app.session_state["agent_task"]
    app.number_input(key=f"loop_rounds:{task}").set_value(3)
    app = submit(app, LOOP_LABEL)
    assert not app.exception
    block = result_section(app)
    defaults = nuagent.physics.plugin.load_coefficient_plugin(task / "plugin.py").parameters
    assert {name: block.number_input(key=f"parameter:{task}:{name}").value for name in defaults} == defaults
    error = "backend failed (exit 2): loop backend failure raw\nunavailable" if failure == "failure" else "backend produced no nonempty notes.md: done"
    assert block.markdown[1].value.startswith("**Round 0 · translation agent's point · Δ ")
    assert block.markdown[2].value == f"**Round 1 · backend failure**  \nError: {error}"
    assert block.info[0].value == "No conclusion: the backend failed in round 1."
    assert block.caption[0].value == CROSS_VALIDATED


def test_the_running_line_names_the_closing_call(monkeypatch):
    # The fake backend raises inside the closing call, freezing the page as it looked while the closing call ran.
    loop_backend = LoopBackend([{"g_nu": 2e-4}], "crash", name="claude")
    app, calls = loop_page(monkeypatch, FakeBackend("correct", name="claude"), loop_backend)
    task = app.session_state["agent_task"]
    app.number_input(key=f"loop_rounds:{task}").set_value(1)
    app = submit(app, LOOP_LABEL)
    assert app.exception
    block = result_section(app)
    assert [status.label for status in block.status] == ["Closing call running on claude: the agent is writing its conclusion…"]
    progress = list(block.children.values())[2]
    assert block_sequence(progress)[:4] == ["Subheader", "Markdown", "Markdown", "Status"]
    assert progress.markdown[1].value.startswith("**Round 1 · 1 points, 1 passed checks")


def test_backend_failure_in_the_closing_call_keeps_the_results(monkeypatch):
    loop_backend = LoopBackend([{"g_nu": 2e-4}], "failure", name="claude")
    app, calls = loop_page(monkeypatch, FakeBackend("plugin-only", name="claude"), loop_backend)
    task = app.session_state["agent_task"]
    app.number_input(key=f"loop_rounds:{task}").set_value(1)
    app = submit(app, LOOP_LABEL)
    assert not app.exception
    block = result_section(app)
    assert block.info[0].value == "No conclusion: the closing call's backend failed: backend failed (exit 2): loop backend failure raw\nunavailable"
    assert "Agent loop: completed, best fit" in loop_expander_labels(app)
    assert "Agent loop: notes, closing call (backend-failure)" in loop_expander_labels(app)


def test_trajectory_figures_leave_out_failing_points_and_choose_axes():
    from nuagent.ui.figures import delta_by_round_figure, parameter_figure
    def point(number, delta, **parameters):
        return {"round": number, "index": 1, "parameters": {"mass_mev": 10.0, "g": 1e-4} | parameters, "findings": [] if delta is not None else ["x"],
                "evaluations": [] if delta is None else [{"experiment": "e", "delta": delta}]}
    history = [point(0, 7.5), point(1, 0.4, mass_mev=20.0, g=2e-3), point(1, None, g=5.0), point(2, -0.1, mass_mev=-5.0, g=2e-4)]
    rounds = delta_by_round_figure(history, ["e"])
    axes = rounds.axes[0]
    assert len(axes.collections[0].get_offsets()) == 3
    np.testing.assert_allclose(axes.lines[0].get_ydata(), [7.5, 0.4, -0.1])
    panels = parameter_figure(history, ["mass_mev", "g"], ["e"])
    visible = [axes for axes in panels.axes if axes.get_visible()]
    assert [axes.get_xlabel() for axes in visible] == ["mass_mev", "g"]
    assert [axes.get_xscale() for axes in visible] == ["linear", "log"]
    assert all(len(axes.collections[0].get_offsets()) == 3 for axes in visible)
