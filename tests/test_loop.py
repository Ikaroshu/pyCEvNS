"""The agent loop with a scripted fake backend: flow, points, the output contract, Pareto selection, failures,
the loop tool policy, the CLI and the report."""

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

from nuagent.agent.backend import BackendResult
from nuagent.agent.loop import pareto_set, read_loop, read_history, run_loop, run_loop_step, start_loop
from nuagent.agent.run import evaluate_point, run_result
from nuagent.agent.tool_policy import allowed_tool, deny_reason
from nuagent.physics.plugin import load_coefficient_plugin


class LoopBackend:
    """Plays one scripted step per call, writing notes: a points list (with the hypothesis "Hypothesis <call>", and for an
    empty list the conclusion "Conclusion <call>"), raw points.json text, None for no points.json, "nothing" for no output
    at all, "failure" for a failing CLI, "crash" to raise, or "capacity" for a CLI reporting an error."""

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
        if step == "crash":
            raise RuntimeError("the fake backend crashed")
        if step == "failure":
            return BackendResult("", 2, "loop backend failure raw", "unavailable")
        if step == "capacity":
            return BackendResult("Selected model is at capacity.", 0, '{"is_error": true}', error=True)
        if step == "nothing":
            return BackendResult("done", 0, "{}")
        if self.edit_plugin_copy:
            copy = task_dir / "plugin.py"
            copy.chmod(0o644)
            copy.write_text(COEFFICIENT_PLUGIN.replace("return c", "c.set('V', 'u', 'mm', 0.5)\n    return c"))
        (task_dir / "notes.md").write_text(f"# Loop notes\n\nObservations for call {len(self.prompts)}.")
        if step is not None:
            call = len(self.prompts)
            output = {"hypothesis": f"Hypothesis {call}", "points": step} | ({} if step else {"conclusion": f"Conclusion {call}"})
            (task_dir / "points.json").write_text(step if isinstance(step, str) else json.dumps(output))
        return BackendResult("done", 0, '{"model":"fake-loop-model","total_cost_usd":0.01}')


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
    loop_dir, out = tmp_path / "F.loop", tmp_path / "F.md"
    backend = LoopBackend([{"g_nu": 2e-4}], [{"mass_mev": 50.0}, {"g_nu": 5.0}], [{"g_u": -4e-5, "g_d": 8e-5}], [], name="other")
    step = run_loop(task, loop_dir, backend, out, rounds=3)
    assert (step.round, step.closing, step.state, step.loop_state) == (3, True, "concluded", "completed")
    assert snapshot(task) == before
    record = read_loop(loop_dir)
    assert record["backend"] == "other" and record["source_task"] == str(task.resolve())
    assert (record["experiments"], record["rounds_requested"], record["batch_limit"], record["guidance"]) == (["coherent_csi_2017"], 3, 8, None)
    assert record["rounds"] == [{"round": n, "state": "evaluated", "hypothesis": f"Hypothesis {n}"} for n in (1, 2, 3)]
    assert record["closing"] == {"state": "concluded", "hypothesis": "Hypothesis 4", "conclusion": "Conclusion 4"}
    assert (record["conclusion"], record["state"]) == ("Conclusion 4", "completed")
    assert {path.name for path in (loop_dir / "history/closing/attempt-1").iterdir()} == {
        "backend.json", "validation.json", "points.json", "notes.md"}
    assert "## Closing call" in backend.prompts[3] and "no more points will be\nevaluated" in backend.prompts[3]
    assert "- history/round-3/attempt-1/notes.md" in backend.prompts[3] and "## Closing call" not in backend.prompts[2]
    assert "Conclusion 4" in step.report
    history = read_history(loop_dir)
    assert [(entry["round"], entry["index"]) for entry in history] == [(0, 1), (1, 1), (2, 1), (2, 2), (3, 1)]
    assert history[0]["parameters"] == defaults(task)
    for number in (1, 2, 3):
        assert {path.name for path in (loop_dir / f"history/round-{number}/attempt-1").iterdir()} == {
            "backend.json", "validation.json", "points.json", "notes.md"}
    assert {path.name for path in loop_dir.iterdir()} >= {"TASK.md", "plugin.py", "card.json", "interaction_notes.md", "context",
                                                         "data", "history.json", "loop.json", "history"}
    assert (loop_dir / "interaction_notes.md").read_text() == (task / "notes.md").read_text()
    assert not os.access(loop_dir / "plugin.py", os.W_OK)
    data = json.loads((loop_dir / "data/coherent_csi_2017.json").read_text())
    assert sum(data["observed"]) == 855 and len(data["bin_edges"]) == 13 and data["sm_profile"]["beta"].keys() == {"steady_state", "prompt_neutrons"}
    assert "Round 1 of 3" in backend.prompts[0] and "none yet" in backend.prompts[0]
    assert "Round 2 of 3" in backend.prompts[1] and "- history/round-1/attempt-1/notes.md" in backend.prompts[1]
    assert "history/round-2/attempt-1/notes.md" in backend.prompts[2]
    assert [(entry["round"], bool(entry["evaluations"])) for entry in backend.histories[1]] == [(0, True), (1, True)]
    assert backend.leftovers == [[], [], [], []]
    for prompt in backend.prompts:
        assert "α near −1" in prompt and "points.json" in prompt and "User guidance" not in prompt
        assert ("On Codex, read files inside\nthis directory with read-only shell commands (cat, sed -n, ls, head) and write\nwith apply_patch. "
                "Run no other command, use no other tool" in prompt)
        assert "On Claude\nCode, read with Read and write with Write or Edit." in prompt
        assert "run no shell commands" not in prompt.lower()


def test_points_take_defaults_match_evaluate_point_and_ignore_the_plugin_copy(tmp_path):
    task = valid_task(tmp_path)
    backend = LoopBackend([{"mass_mev": 50.0}, {"g_nu": 5.0}, {"g_nu": 2e-4, "mass_mev": 20}], [], edit_plugin_copy=True)
    run_loop(task, tmp_path / "F.loop", backend, tmp_path / "F.md", rounds=1)
    history = read_history(tmp_path / "F.loop")
    first, failed, third = history[1:]
    assert first["parameters"] == defaults(task) | {"mass_mev": 50.0}
    model, records = evaluate_point(task, {"mass_mev": 50.0})
    assert first["evaluations"][0]["delta"] == records[0]["delta"]
    np.testing.assert_array_equal(first["evaluations"][0]["predicted_counts"], model.signals["coherent_csi_2017"])
    assert failed["evaluations"] == [] and any("non-perturbative coupling g_nu" in finding for finding in failed["findings"])
    assert third["evaluations"][0]["delta"] == evaluate_point(task, {"g_nu": 2e-4, "mass_mev": 20})[1][0]["delta"]


def points_text(points, **fields) -> str:
    return json.dumps({"hypothesis": "A larger coupling raises the signal.", "points": points} | fields)


NINE = points_text([{"g_nu": 1e-4 * n} for n in range(1, 10)])


@pytest.mark.parametrize("bad,finding", [
    (points_text([{"width": 1.0}]), "unknown parameters width"),
    (NINE, "has 9 points; at most 8"),
    (points_text([{"g_nu": True}]), "g_nu must be a finite real number (not a boolean)"),
    ('{"hypothesis": "h", "points": [{"g_nu": NaN}]}', "g_nu must be a finite real number"),
    (json.dumps({"points": [{"g_nu": 1e-4}]}), "points.json must be"),
    (points_text([{"g_nu": 1e-4}], hypothesis=""), "hypothesis must be a nonempty string"),
    (points_text([{"g_nu": 1e-4}], hypothesis=3), "hypothesis must be a nonempty string"),
    (points_text([{"g_nu": 1e-4}], conclusion="Done."), "conclusion is written only with an empty points list"),
    (points_text([]), "must carry the conclusion"),
    (points_text([], conclusion=" "), "conclusion must be a nonempty string"),
    ('{"points": [', "not valid JSON"),
    (json.dumps([{"g_nu": 1e-4}]), "points.json must be"),
    (None, "points.json is missing"),
])
def test_malformed_points_get_one_correction_then_stop_the_loop(tmp_path, bad, finding):
    task = valid_task(tmp_path)
    backend = LoopBackend(bad, [{"g_nu": 2e-4}], [])
    run_loop(task, tmp_path / "A.loop", backend, tmp_path / "A.md", rounds=1)
    assert (read_loop(tmp_path / "A.loop")["rounds"][0]["state"], len(backend.prompts)) == ("evaluated", 3)
    assert finding in backend.prompts[1] and "## Validation correction (one attempt)" in backend.prompts[1]
    assert backend.leftovers[1] == []
    validation = json.loads((tmp_path / "A.loop/history/round-1/attempt-1/validation.json").read_text())
    assert not validation["valid"] and any(finding in text for text in validation["findings"])
    backend = LoopBackend([{"g_nu": 2e-4}], bad, bad)
    step = run_loop(task, tmp_path / "B.loop", backend, tmp_path / "B.md", rounds=3)
    assert (step.round, step.state, step.loop_state) == (2, "invalid", "stopped-invalid")
    report = (tmp_path / "B.md").read_text()
    assert report == step.report and report.startswith("Outcome: **stopped-invalid**")
    assert "| 1 | 1 |" in report and "### Round 2, attempt 2" in report
    record = read_loop(tmp_path / "B.loop")
    assert record["rounds"][-1] == {"round": 2, "state": "invalid"} and (record["closing"], record["conclusion"]) == (None, None)
    assert len(backend.prompts) == 3 and "No conclusion: the loop stopped" in report


def test_round_two_outputs_start_cleared(tmp_path):
    task = valid_task(tmp_path)
    backend = LoopBackend([{"g_nu": 2e-4}], None, [{"g_nu": 3e-4}], [])
    run_loop(task, tmp_path / "A.loop", backend, tmp_path / "A.md", rounds=2)
    assert (read_loop(tmp_path / "A.loop")["rounds"][1]["state"], len(backend.prompts)) == ("evaluated", 4)
    assert backend.leftovers == [[], [], [], []]
    assert "points.json is missing" in backend.prompts[2]
    backend = LoopBackend([{"g_nu": 2e-4}], "nothing")
    step = run_loop(task, tmp_path / "B.loop", backend, tmp_path / "B.md", rounds=3)
    assert (step.round, step.state, step.loop_state) == (2, "backend-failure", "stopped-backend-failure")
    assert read_loop(tmp_path / "B.loop")["rounds"][-1] == {"round": 2, "state": "backend-failure", "error": "backend produced no nonempty notes.md: done"}
    assert step.report.startswith("Outcome: **stopped-backend-failure**") and "No conclusion: the backend failed in round 2." in step.report


def test_empty_points_converge_early(tmp_path):
    task = valid_task(tmp_path)
    backend = LoopBackend([{"g_nu": 2e-4}], [])
    step = run_loop(task, tmp_path / "F.loop", backend, tmp_path / "F.md", rounds=5)
    assert (step.round, step.state, step.loop_state, step.points) == (2, "converged", "converged", [])
    assert len(backend.prompts) == 2
    assert step.report.startswith("Outcome: **converged**") and "Rounds run: 2 of 5" in step.report
    record = read_loop(tmp_path / "F.loop")
    assert [entry["state"] for entry in record["rounds"]] == ["evaluated", "converged"]
    assert (record["rounds"][1]["conclusion"], record["conclusion"], record["closing"]) == ("Conclusion 2", "Conclusion 2", None)
    assert "### Conclusion\n\nConclusion 2" in step.report


def test_closing_call_malformed_twice_or_failing(tmp_path):
    task = valid_task(tmp_path)
    backend = LoopBackend([{"g_nu": 2e-4}], points_text([{"g_nu": 3e-4}]), None)
    step = run_loop(task, tmp_path / "A.loop", backend, tmp_path / "A.md", rounds=1)
    assert (step.closing, step.state, step.loop_state) == (True, "invalid", "completed")
    assert "points must be an empty list in the closing call" in backend.prompts[2] and "## Closing call" in backend.prompts[2]
    record = read_loop(tmp_path / "A.loop")
    assert (record["state"], record["closing"], record["conclusion"]) == ("completed", {"state": "invalid"}, None)
    assert "No conclusion: the closing call's points.json was malformed twice." in step.report
    assert "### Closing call, attempt 2" in step.report
    backend = LoopBackend([{"g_nu": 2e-4}], "failure")
    step = run_loop(task, tmp_path / "B.loop", backend, tmp_path / "B.md", rounds=1)
    assert (step.closing, step.state, step.loop_state) == (True, "backend-failure", "completed")
    error = "backend failed (exit 2): loop backend failure raw\nunavailable"
    record = read_loop(tmp_path / "B.loop")
    assert (record["state"], record["closing"], record["conclusion"]) == ("completed", {"state": "backend-failure", "error": error}, None)
    assert step.report == (tmp_path / "B.md").read_text()
    assert f"No conclusion: the closing call's backend failed: {error}" in step.report
    assert "### Closing call\n\nNo notes: the backend failed" in step.report


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
        start_loop(questions, tmp_path / "Q.loop", LoopBackend(), 3)
    assert not (tmp_path / "Q.loop").exists()
    task = valid_task(tmp_path)
    loop_dir = tmp_path / "F.loop"
    backend = LoopBackend([{"g_nu": 2e-4}], "capacity")
    start_loop(task, loop_dir, backend, 3)
    assert run_loop_step(loop_dir, backend, tmp_path / "F.md").loop_state is None
    step = run_loop_step(loop_dir, backend, tmp_path / "F.md")
    assert (step.round, step.state, step.loop_state) == (2, "backend-failure", "stopped-backend-failure")
    error = "backend failed (exit 0): Selected model is at capacity."
    assert read_loop(loop_dir)["rounds"] == [{"round": 1, "state": "evaluated", "hypothesis": "Hypothesis 1"},
                                             {"round": 2, "state": "backend-failure", "error": error}]
    assert "is_error" in (loop_dir / "history/round-2/attempt-1/backend.json").read_text()
    report = (tmp_path / "F.md").read_text()
    log = report.split("## Research log\n\n")[1].split("\n## Best fit")[0]
    assert log.endswith(f"### Round 2 · backend failure\n\nError: {error}\n\n### Conclusion\n\nNo conclusion: the backend failed in round 2.\n")
    assert "| 1 | 1 |" in report


def test_loop_tool_policy(tmp_path):
    loop_dir = tmp_path / "loop"
    loop_dir.mkdir()
    (loop_dir / "loop.json").write_text("{}")
    assert allowed_tool("Read", {"file_path": str(loop_dir / "history.json")}, loop_dir)
    assert allowed_tool("Read", {"file_path": "data/coherent_csi_2017.json"}, loop_dir)
    assert not allowed_tool("Read", {"file_path": str(tmp_path / "task/plugin.py")}, loop_dir)
    for name in ("points.json", "notes.md"):
        assert allowed_tool("Write", {"file_path": str(loop_dir / name)}, loop_dir)
        assert allowed_tool("Edit", {"file_path": str(loop_dir / name)}, loop_dir)
    for name in ("plugin.py", "card.json", "loop.json", "history.json", "TASK.md", "questions.json", "../task/plugin.py"):
        assert not allowed_tool("Write", {"file_path": str(loop_dir / name)}, loop_dir)
        assert not allowed_tool("Edit", {"file_path": str(loop_dir / name)}, loop_dir)
    assert not allowed_tool("Write", {"file_path": "/tmp/points.json"}, loop_dir)
    for command in ("nuagent validate --plugin plugin.py", "ls", "python -c 'print(1)'"):
        assert not allowed_tool("Bash", {"command": command}, loop_dir)
    assert "points.json/notes.md" in deny_reason(loop_dir) and "shell" in deny_reason(loop_dir)
    task = tmp_path / "task"
    task.mkdir()
    assert allowed_tool("Write", {"file_path": str(task / "plugin.py")}, task)
    assert not allowed_tool("Write", {"file_path": str(task / "points.json")}, task)
    assert allowed_tool("Bash", {"command": "nuagent validate --plugin plugin.py"}, task)
    assert "nuagent validate" in deny_reason(task)
    hook = subprocess.run([sys.executable, "-m", "nuagent.agent.tool_policy", str(loop_dir)], capture_output=True, text=True, check=True,
                          input=json.dumps({"tool_name": "Write", "tool_input": {"file_path": str(loop_dir / "plugin.py")}}))
    decision = json.loads(hook.stdout)["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny" and "points.json/notes.md" in decision["permissionDecisionReason"]


CLOSING = json.dumps({"hypothesis": "The best point stands.", "points": [], "conclusion": "A larger g_nu fits slightly better."})


def fake_loop_cli(directory: Path, points: str, fail_on: str | None = None) -> None:
    """A fake claude executable writing `points` in rounds and a valid conclusion in the closing call; it exits 1 with a
    capacity error, writing nothing, when TASK.md contains `fail_on`."""
    directory.mkdir()
    executable = directory / "claude"
    executable.write_text(f"#!{sys.executable}\nfrom pathlib import Path\nimport json, sys\n"
                          f"if {fail_on!r} and {fail_on!r} in Path('TASK.md').read_text():\n"
                          "    print('Selected model is at capacity.', file=sys.stderr)\n    sys.exit(1)\n"
                          "closing = '## Closing call' in Path('TASK.md').read_text()\n"
                          f"Path('points.json').write_text({CLOSING!r} if closing else {points!r})\n"
                          "Path('notes.md').write_text('Raised g_nu to probe a larger signal.')\n"
                          "print(json.dumps({'result':'done','model':'fake','total_cost_usd':0}))\n")
    executable.chmod(0o755)


def nuagent(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(Path(sys.executable).with_name("nuagent")), *arguments], capture_output=True, text=True)


def test_loop_cli(tmp_path, monkeypatch):
    task = valid_task(tmp_path, name="claude")
    fake_loop_cli(tmp_path / "good", points_text([{"g_nu": 2e-4}, {"g_nu": 5e-5}]))
    monkeypatch.setenv("PATH", str(tmp_path / "good") + os.pathsep + os.environ["PATH"])
    out = tmp_path / "F.md"
    output = cli("loop", "--task-dir", str(task), "--out", str(out), "--rounds", "2", "--guidance", "keep mass_mev at 30")
    assert (output["report"], output["state"], output["rounds"]) == (str(out), "completed", 2)
    history = read_history(tmp_path / "F.loop")
    lowest = min(e["evaluations"][0]["delta"] for e in history if e["evaluations"])
    # Both rounds propose the same points, so the lowest-Δ point is an exact tie kept twice.
    assert output["pareto"] == [{"round": e["round"], "index": e["index"], "parameters": e["parameters"],
                                 "delta": {"coherent_csi_2017": lowest}}
                                for e in history if e["evaluations"] and e["evaluations"][0]["delta"] == lowest]
    assert len(output["pareto"]) == 2
    assert read_loop(tmp_path / "F.loop")["backend"] == "claude"
    prompt = (tmp_path / "F.loop/TASK.md").read_text()
    assert "## User guidance\n\nThe user's instruction for this agent loop:\n\nkeep mass_mev at 30\n" in prompt
    assert "- Guidance: keep mass_mev at 30" in out.read_text() and "A larger g_nu fits slightly better." in out.read_text()
    rejected = nuagent("loop", "--task-dir", str(task), "--out", str(tmp_path / "Z.md"), "--rounds", "0")
    assert rejected.returncode == 2 and "--rounds must be a positive integer" in rejected.stderr
    questions = tmp_path / "questions"
    run_result("a light vector boson", ["coherent_csi_2017"], FakeBackend("questions"), questions, tmp_path / "q.md")
    rejected = nuagent("loop", "--task-dir", str(questions), "--out", str(tmp_path / "Z.md"))
    assert rejected.returncode == 2 and "state 'questions'" in rejected.stderr
    assert not (tmp_path / "Z.loop").exists()
    fake_loop_cli(tmp_path / "bad", '{"points": [')
    monkeypatch.setenv("PATH", str(tmp_path / "bad") + os.pathsep + os.environ["PATH"])
    failed = nuagent("loop", "--task-dir", str(task), "--out", str(tmp_path / "I.md"), "--loop-dir", str(tmp_path / "custom"))
    assert failed.returncode == 1
    assert json.loads(failed.stdout) | {"pareto": None} == {"report": str(tmp_path / "I.md"), "state": "stopped-invalid", "rounds": 1, "pareto": None}
    assert json.loads(failed.stdout)["pareto"][0]["round"] == 0
    assert read_loop(tmp_path / "custom")["state"] == "stopped-invalid"


def test_report_sections_points_and_guidance(tmp_path):
    task = valid_task(tmp_path, mode="plugin-only")
    backend = LoopBackend([{"g_nu": 2e-4}, {"g_nu": 5.0}], [{"mass_mev": 50.0}], [])
    step = run_loop(task, tmp_path / "F.loop", backend, tmp_path / "F.md", rounds=2, guidance="prefer light mediators")
    report = step.report
    assert report.splitlines()[0] == "Outcome: **completed**"
    headings = [line for line in report.splitlines() if line.startswith("## ")]
    assert headings == ["## Research log", "## Best fit", "## Source task", "## Points by round", "## Round notes", "## Validation history",
                        "## Backend CLI records", "## Provenance"]
    assert "Rounds run: 2 of 2. Points evaluated: 4" in report
    points = report.split("## Points by round\n\n")[1].split("\n\n")[0].splitlines()[2:]
    assert [row.split(" | ")[:2] for row in points] == [["| 0", "1"], ["| 1", "1"], ["| 1", "2"], ["| 2", "1"]]
    assert "non-perturbative coupling g_nu" in points[2]
    assert "- Guidance: prefer light mediators" in report and "- Verification of the translation: plugin-only" in report
    assert "Round 0, the translation agent's point, for comparison: Δ coherent_csi_2017 = " in report
    assert "### Round 1\n\n# Loop notes" in report and "### Closing call\n\n# Loop notes" in report and "fake-loop-model" in report
    log = report.split("## Research log\n\n")[1].split("\n## Best fit")[0].strip()
    assert log.split("\n\n")[0] == f"### Round 0 · translation agent's point · Δ {read_history(tmp_path / 'F.loop')[0]['evaluations'][0]['delta']:.4g}"
    log = log.split("\n\n", 1)[1]
    assert log.split("\n\n") == ["### Round 1 · 2 points, 1 passed checks · best Δ " + log.split("best Δ ")[1].split("\n")[0],
                                   "Hypothesis: Hypothesis 1", log.split("\n\n")[2], "Hypothesis: Hypothesis 2",
                                   "### Conclusion", "Conclusion 3"]
    assert log.split("\n\n")[2].startswith("### Round 2 · 1 points, 1 passed checks · best Δ ")
    assert '"closing_call": "concluded"' in report
    provenance = json.loads(report.split("## Provenance\n\n```json\n")[1].split("\n```")[0])
    assert (provenance["rounds_requested"], provenance["rounds_run"], provenance["backend"]) == (2, 2, "fake")
    assert provenance["source_task"] == str(task.resolve()) and "git_revision" in provenance
    best = report.split("## Best fit")[1].split("## Source task")[0]
    assert "Verification: **plugin-only**" in best and "Background scale β" in best
    assert "prefer light mediators" in backend.prompts[0]


@pytest.mark.parametrize("fail_on,state,exit_status", [("## Round 2 of 3", "stopped-backend-failure", 1),
                                                       ("## Closing call", "completed", 0)])
def test_loop_cli_after_a_backend_failure(tmp_path, monkeypatch, fail_on, state, exit_status):
    task = valid_task(tmp_path, name="claude")
    fake_loop_cli(tmp_path / "bin", points_text([{"g_nu": 2e-4}]), fail_on)
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ["PATH"])
    result = nuagent("loop", "--task-dir", str(task), "--out", str(tmp_path / "F.md"), "--rounds", "3")
    assert result.returncode == exit_status
    output = json.loads(result.stdout)
    assert (output["state"], output["rounds"]) == (state, 2 if exit_status else 3)
    report = (tmp_path / "F.md").read_text()
    assert report.startswith(f"Outcome: **{state}**") and "Selected model is at capacity." in report
    assert ("No conclusion: the backend failed in round 2." if exit_status else "No conclusion: the closing call's backend failed") in report
