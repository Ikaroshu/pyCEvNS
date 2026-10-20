"""The agent parameter fit loop with a scripted fake backend: flow, points, the output contract, Pareto selection, failures,
the fit tool policy, the CLI and the report."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_agent import FakeBackend, snapshot
from test_cli import cli
from test_plugins import COEFFICIENT_PLUGIN

from nuagent.agent.backend import BackendFailure, BackendResult
from nuagent.agent.fit import pareto_set, read_fit, read_history, run_fit, run_fit_round, start_fit
from nuagent.agent.run import evaluate_point, run_result
from nuagent.agent.tool_policy import allowed_tool, deny_reason
from nuagent.physics.plugin import load_coefficient_plugin


class FitBackend:
    """Plays one scripted step per call: a points list or raw points.json text with notes, None for notes without
    points.json, "nothing" for no output at all, or "failure" for a failing CLI."""

    def __init__(self, *script, name="fake", edit_plugin_copy=False):
        self.script = list(script)
        self.name = name
        self.edit_plugin_copy = edit_plugin_copy
        self.prompts = []
        self.histories = []
        self.leftovers = []

    def run(self, task_dir, max_turns):
        self.prompts.append((task_dir / "TASK.md").read_text())
        self.histories.append(json.loads((task_dir / "history.json").read_text()))
        self.leftovers.append([name for name in ("points.json", "notes.md") if (task_dir / name).exists()])
        step = self.script.pop(0)
        if step == "failure":
            return BackendResult("failed", 2, "fit backend failure raw", "unavailable")
        if step == "nothing":
            return BackendResult("done", 0, "{}")
        if self.edit_plugin_copy:
            copy = task_dir / "plugin.py"
            copy.chmod(0o644)
            copy.write_text(COEFFICIENT_PLUGIN.replace("return c", "c.set('V', 'u', 'mm', 0.5)\n    return c"))
        (task_dir / "notes.md").write_text(f"# Fit notes\n\nObservations for call {len(self.prompts)}.")
        if step is not None:
            (task_dir / "points.json").write_text(step if isinstance(step, str) else json.dumps({"points": step}))
        return BackendResult("done", 0, '{"model":"fake-fit-model","total_cost_usd":0.01}')


def valid_task(tmp_path: Path, mode: str = "correct", name: str = "fake") -> Path:
    task = tmp_path / "task"
    assert run_result("a vector and scalar", ["coherent_csi_2017"], FakeBackend(mode, name=name), task,
                      tmp_path / "report.md").state == "valid"
    return task


def defaults(task: Path) -> dict:
    return load_coefficient_plugin(task / "plugin.py").parameters


def test_three_round_flow_records_and_leaves_the_source_task_unchanged(tmp_path):
    task = valid_task(tmp_path)
    before = snapshot(task)
    fit_dir, out = tmp_path / "F.fit", tmp_path / "F.md"
    backend = FitBackend([{"g_nu": 2e-4}], [{"mass_mev": 50.0}, {"g_nu": 5.0}], [{"g_u": -4e-5, "g_d": 8e-5}], name="other")
    step = run_fit(task, fit_dir, backend, out, rounds=3)
    assert (step.round, step.state, step.fit_state) == (3, "evaluated", "completed")
    assert snapshot(task) == before
    record = read_fit(fit_dir)
    assert record["backend"] == "other" and record["source_task"] == str(task.resolve())
    assert (record["experiments"], record["rounds_requested"], record["batch_limit"], record["guidance"]) == (["coherent_csi_2017"], 3, 8, None)
    assert record["rounds"] == [{"round": n, "state": "evaluated"} for n in (1, 2, 3)] and record["state"] == "completed"
    history = read_history(fit_dir)
    assert [(entry["round"], entry["index"]) for entry in history] == [(0, 1), (1, 1), (2, 1), (2, 2), (3, 1)]
    assert history[0]["parameters"] == defaults(task)
    for number in (1, 2, 3):
        assert {path.name for path in (fit_dir / f"history/round-{number}/attempt-1").iterdir()} == {
            "backend.json", "validation.json", "points.json", "notes.md"}
    assert {path.name for path in fit_dir.iterdir()} >= {"TASK.md", "plugin.py", "card.json", "interaction_notes.md", "context",
                                                         "data", "history.json", "fit.json", "history"}
    assert (fit_dir / "interaction_notes.md").read_text() == (task / "notes.md").read_text()
    assert not os.access(fit_dir / "plugin.py", os.W_OK)
    data = json.loads((fit_dir / "data/coherent_csi_2017.json").read_text())
    assert sum(data["observed"]) == 855 and len(data["bin_edges"]) == 13 and data["sm_profile"]["beta"].keys() == {"steady_state", "prompt_neutrons"}
    assert "Round 1 of 3" in backend.prompts[0] and "none yet" in backend.prompts[0]
    assert "Round 2 of 3" in backend.prompts[1] and "- history/round-1/attempt-1/notes.md" in backend.prompts[1]
    assert "history/round-2/attempt-1/notes.md" in backend.prompts[2]
    assert [(entry["round"], bool(entry["evaluations"])) for entry in backend.histories[1]] == [(0, True), (1, True)]
    assert backend.leftovers == [[], [], []]
    for prompt in backend.prompts:
        assert "α near −1" in prompt and "points.json" in prompt and "file reads and apply_patch\non Codex" in prompt and "User guidance" not in prompt


def test_points_take_defaults_match_evaluate_point_and_ignore_the_plugin_copy(tmp_path):
    task = valid_task(tmp_path)
    backend = FitBackend([{"mass_mev": 50.0}, {"g_nu": 5.0}, {"g_nu": 2e-4, "mass_mev": 20}], edit_plugin_copy=True)
    run_fit(task, tmp_path / "F.fit", backend, tmp_path / "F.md", rounds=1)
    history = read_history(tmp_path / "F.fit")
    first, failed, third = history[1:]
    assert first["parameters"] == defaults(task) | {"mass_mev": 50.0}
    model, records = evaluate_point(task, {"mass_mev": 50.0})
    assert first["evaluations"][0]["delta"] == records[0]["delta"]
    np.testing.assert_array_equal(first["evaluations"][0]["predicted_counts"], model.signals["coherent_csi_2017"])
    assert failed["evaluations"] == [] and any("non-perturbative coupling g_nu" in finding for finding in failed["findings"])
    assert third["evaluations"][0]["delta"] == evaluate_point(task, {"g_nu": 2e-4, "mass_mev": 20})[1][0]["delta"]


NINE = json.dumps({"points": [{"g_nu": 1e-4 * n} for n in range(1, 10)]})


@pytest.mark.parametrize("bad,finding", [
    (json.dumps({"points": [{"width": 1.0}]}), "unknown parameters width"),
    (NINE, "has 9 points; at most 8"),
    (json.dumps({"points": [{"g_nu": True}]}), "g_nu must be a finite real number (not a boolean)"),
    ('{"points": [{"g_nu": NaN}]}', "g_nu must be a finite real number"),
    ('{"points": [', "not valid JSON"),
    (json.dumps([{"g_nu": 1e-4}]), "points.json must be"),
    (None, "points.json is missing"),
])
def test_malformed_points_get_one_correction_then_stop_the_fit(tmp_path, bad, finding):
    task = valid_task(tmp_path)
    backend = FitBackend(bad, [{"g_nu": 2e-4}])
    step = run_fit(task, tmp_path / "A.fit", backend, tmp_path / "A.md", rounds=1)
    assert (step.state, len(backend.prompts)) == ("evaluated", 2)
    assert finding in backend.prompts[1] and "## Validation correction (one attempt)" in backend.prompts[1]
    assert backend.leftovers[1] == []
    validation = json.loads((tmp_path / "A.fit/history/round-1/attempt-1/validation.json").read_text())
    assert not validation["valid"] and any(finding in text for text in validation["findings"])
    backend = FitBackend([{"g_nu": 2e-4}], bad, bad)
    step = run_fit(task, tmp_path / "B.fit", backend, tmp_path / "B.md", rounds=3)
    assert (step.round, step.state, step.fit_state) == (2, "invalid", "stopped-invalid")
    report = (tmp_path / "B.md").read_text()
    assert report == step.report and report.startswith("Outcome: **stopped-invalid**")
    assert "| 1 | 1 |" in report and "### Round 2, attempt 2" in report
    assert read_fit(tmp_path / "B.fit")["rounds"][-1] == {"round": 2, "state": "invalid"}


def test_round_two_outputs_start_cleared(tmp_path):
    task = valid_task(tmp_path)
    backend = FitBackend([{"g_nu": 2e-4}], None, [{"g_nu": 3e-4}])
    step = run_fit(task, tmp_path / "A.fit", backend, tmp_path / "A.md", rounds=2)
    assert (step.state, len(backend.prompts)) == ("evaluated", 3)
    assert backend.leftovers == [[], [], []]
    assert "points.json is missing" in backend.prompts[2]
    backend = FitBackend([{"g_nu": 2e-4}], "nothing")
    with pytest.raises(RuntimeError, match="no nonempty notes.md"):
        run_fit(task, tmp_path / "B.fit", backend, tmp_path / "B.md", rounds=3)
    assert read_fit(tmp_path / "B.fit")["rounds"][-1] == {"round": 2, "state": "backend-failure"}
    assert not (tmp_path / "B.md").exists()


def test_empty_points_converge_early(tmp_path):
    task = valid_task(tmp_path)
    backend = FitBackend([{"g_nu": 2e-4}], [])
    step = run_fit(task, tmp_path / "F.fit", backend, tmp_path / "F.md", rounds=5)
    assert (step.round, step.state, step.fit_state, step.points) == (2, "converged", "converged", [])
    assert len(backend.prompts) == 2
    assert step.report.startswith("Outcome: **converged**") and "Rounds run: 2 of 5" in step.report
    assert [entry["state"] for entry in read_fit(tmp_path / "F.fit")["rounds"]] == ["evaluated", "converged"]


def entry(number: int, index: int, *deltas: float) -> dict:
    return {"round": number, "index": index, "parameters": {}, "findings": [] if deltas else ["failed"],
            "evaluations": [{"experiment": f"e{k}", "delta": delta} for k, delta in enumerate(deltas)]}


def test_pareto_keeps_exactly_the_nondominated_points():
    history = [entry(0, 1, 1.0, 5.0), entry(1, 1, 2.0, 2.0), entry(1, 2, 2.0, 2.0), entry(1, 3, 3.0, 3.0),
               entry(1, 4), entry(2, 1, 5.0, 0.5), entry(2, 2, 2.0, 6.0), entry(2, 3, 1.0, 5.0)]
    selected = [(e["round"], e["index"]) for e in pareto_set(history)]
    assert selected == [(0, 1), (1, 1), (1, 2), (2, 1), (2, 3)]
    assert [(e["round"], e["index"]) for e in pareto_set([entry(0, 1, 0.5), entry(1, 1, 0.2), entry(1, 2, 0.3), entry(2, 1)])] == [(1, 1)]
    assert [(e["round"], e["index"]) for e in pareto_set([entry(0, 1, 0.2), entry(1, 1, 0.2), entry(1, 2, 0.3)])] == [(0, 1), (1, 1)]


def test_failures_name_the_state_and_record_backend_failure(tmp_path):
    questions = tmp_path / "questions"
    run_result("a light vector boson", ["coherent_csi_2017"], FakeBackend("questions"), questions, tmp_path / "q.md")
    with pytest.raises(ValueError, match="state 'questions'"):
        start_fit(questions, tmp_path / "Q.fit", FitBackend(), 3)
    assert not (tmp_path / "Q.fit").exists()
    task = valid_task(tmp_path)
    fit_dir = tmp_path / "F.fit"
    backend = FitBackend([{"g_nu": 2e-4}], "failure")
    start_fit(task, fit_dir, backend, 3)
    assert run_fit_round(fit_dir, backend, tmp_path / "F.md").fit_state is None
    with pytest.raises(BackendFailure, match="fit backend failure raw"):
        run_fit_round(fit_dir, backend, tmp_path / "F.md")
    assert read_fit(fit_dir)["rounds"] == [{"round": 1, "state": "evaluated"}, {"round": 2, "state": "backend-failure"}]
    assert "unavailable" in (fit_dir / "history/round-2/attempt-1/backend.json").read_text()
    assert not (tmp_path / "F.md").exists()


def test_fit_tool_policy(tmp_path):
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    (fit_dir / "fit.json").write_text("{}")
    assert allowed_tool("Read", {"file_path": str(fit_dir / "history.json")}, fit_dir)
    assert allowed_tool("Read", {"file_path": "data/coherent_csi_2017.json"}, fit_dir)
    assert not allowed_tool("Read", {"file_path": str(tmp_path / "task/plugin.py")}, fit_dir)
    for name in ("points.json", "notes.md"):
        assert allowed_tool("Write", {"file_path": str(fit_dir / name)}, fit_dir)
        assert allowed_tool("Edit", {"file_path": str(fit_dir / name)}, fit_dir)
    for name in ("plugin.py", "card.json", "fit.json", "history.json", "TASK.md", "questions.json", "../task/plugin.py"):
        assert not allowed_tool("Write", {"file_path": str(fit_dir / name)}, fit_dir)
        assert not allowed_tool("Edit", {"file_path": str(fit_dir / name)}, fit_dir)
    assert not allowed_tool("Write", {"file_path": "/tmp/points.json"}, fit_dir)
    for command in ("nuagent validate --plugin plugin.py", "ls", "python -c 'print(1)'"):
        assert not allowed_tool("Bash", {"command": command}, fit_dir)
    assert "points.json/notes.md" in deny_reason(fit_dir) and "shell" in deny_reason(fit_dir)
    task = tmp_path / "task"
    task.mkdir()
    assert allowed_tool("Write", {"file_path": str(task / "plugin.py")}, task)
    assert not allowed_tool("Write", {"file_path": str(task / "points.json")}, task)
    assert allowed_tool("Bash", {"command": "nuagent validate --plugin plugin.py"}, task)
    assert "nuagent validate" in deny_reason(task)
    hook = subprocess.run([sys.executable, "-m", "nuagent.agent.tool_policy", str(fit_dir)], capture_output=True, text=True, check=True,
                          input=json.dumps({"tool_name": "Write", "tool_input": {"file_path": str(fit_dir / "plugin.py")}}))
    decision = json.loads(hook.stdout)["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny" and "points.json/notes.md" in decision["permissionDecisionReason"]


def fake_fit_cli(directory: Path, points: str) -> None:
    directory.mkdir()
    executable = directory / "claude"
    executable.write_text(f"#!{sys.executable}\nfrom pathlib import Path\nimport json\n"
                          f"Path('points.json').write_text({points!r})\n"
                          "Path('notes.md').write_text('Raised g_nu to probe a larger signal.')\n"
                          "print(json.dumps({'result':'done','model':'fake','total_cost_usd':0}))\n")
    executable.chmod(0o755)


def nuagent(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(Path(sys.executable).with_name("nuagent")), *arguments], capture_output=True, text=True)


def test_fit_cli(tmp_path, monkeypatch):
    task = valid_task(tmp_path, name="claude")
    fake_fit_cli(tmp_path / "good", json.dumps({"points": [{"g_nu": 2e-4}, {"g_nu": 5e-5}]}))
    monkeypatch.setenv("PATH", str(tmp_path / "good") + os.pathsep + os.environ["PATH"])
    out = tmp_path / "F.md"
    output = cli("fit", "--task-dir", str(task), "--out", str(out), "--rounds", "2", "--guidance", "keep mass_mev at 30")
    assert (output["report"], output["state"], output["rounds"]) == (str(out), "completed", 2)
    history = read_history(tmp_path / "F.fit")
    lowest = min(e["evaluations"][0]["delta"] for e in history if e["evaluations"])
    # Both rounds propose the same points, so the lowest-Δ point is an exact tie kept twice.
    assert output["pareto"] == [{"round": e["round"], "index": e["index"], "parameters": e["parameters"],
                                 "delta": {"coherent_csi_2017": lowest}}
                                for e in history if e["evaluations"] and e["evaluations"][0]["delta"] == lowest]
    assert len(output["pareto"]) == 2
    assert read_fit(tmp_path / "F.fit")["backend"] == "claude"
    prompt = (tmp_path / "F.fit/TASK.md").read_text()
    assert "## User guidance\n\nThe user's instruction for this fit:\n\nkeep mass_mev at 30\n" in prompt
    assert "- Guidance: keep mass_mev at 30" in out.read_text()
    rejected = nuagent("fit", "--task-dir", str(task), "--out", str(tmp_path / "Z.md"), "--rounds", "0")
    assert rejected.returncode == 2 and "--rounds must be a positive integer" in rejected.stderr
    questions = tmp_path / "questions"
    run_result("a light vector boson", ["coherent_csi_2017"], FakeBackend("questions"), questions, tmp_path / "q.md")
    rejected = nuagent("fit", "--task-dir", str(questions), "--out", str(tmp_path / "Z.md"))
    assert rejected.returncode == 2 and "state 'questions'" in rejected.stderr
    assert not (tmp_path / "Z.fit").exists()
    fake_fit_cli(tmp_path / "bad", '{"points": [')
    monkeypatch.setenv("PATH", str(tmp_path / "bad") + os.pathsep + os.environ["PATH"])
    failed = nuagent("fit", "--task-dir", str(task), "--out", str(tmp_path / "I.md"), "--fit-dir", str(tmp_path / "custom"))
    assert failed.returncode == 1
    assert json.loads(failed.stdout) | {"pareto": None} == {"report": str(tmp_path / "I.md"), "state": "stopped-invalid", "rounds": 1, "pareto": None}
    assert json.loads(failed.stdout)["pareto"][0]["round"] == 0
    assert read_fit(tmp_path / "custom")["state"] == "stopped-invalid"


def test_report_sections_points_and_guidance(tmp_path):
    task = valid_task(tmp_path, mode="plugin-only")
    backend = FitBackend([{"g_nu": 2e-4}, {"g_nu": 5.0}], [{"mass_mev": 50.0}])
    step = run_fit(task, tmp_path / "F.fit", backend, tmp_path / "F.md", rounds=2, guidance="prefer light mediators")
    report = step.report
    assert report.splitlines()[0] == "Outcome: **completed**"
    headings = [line for line in report.splitlines() if line.startswith("## ")]
    assert headings == ["## Best fit", "## Source task", "## Points by round", "## Round notes", "## Validation history",
                        "## Backend CLI records", "## Provenance"]
    assert "Rounds run: 2 of 2. Points evaluated: 4" in report
    points = report.split("## Points by round\n\n")[1].split("\n\n")[0].splitlines()[2:]
    assert [row.split(" | ")[:2] for row in points] == [["| 0", "1"], ["| 1", "1"], ["| 1", "2"], ["| 2", "1"]]
    assert "non-perturbative coupling g_nu" in points[2]
    assert "- Guidance: prefer light mediators" in report and "- Verification of the translation: plugin-only" in report
    assert "Round 0, the translation agent's point, for comparison: Δ coherent_csi_2017 = " in report
    assert "### Round 1\n\n# Fit notes" in report and "fake-fit-model" in report
    provenance = json.loads(report.split("## Provenance\n\n```json\n")[1].split("\n```")[0])
    assert (provenance["rounds_requested"], provenance["rounds_run"], provenance["backend"]) == (2, 2, "fake")
    assert provenance["source_task"] == str(task.resolve()) and "git_revision" in provenance
    best = report.split("## Best fit")[1].split("## Source task")[0]
    assert "Verification: **plugin-only**" in best and "Background scale β" in best
    assert "prefer light mediators" in backend.prompts[0]
