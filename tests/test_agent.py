"""Validation findings, one correction, notes-only outcomes and subprocess isolation."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_cli import cli
from test_plugins import COEFFICIENT_PLUGIN, EQUIVALENT_CARD, RAW_SM_PLUGIN

from nuagent.agent.backend import BackendFailure, BackendResult, SubprocessBackend
from nuagent.agent.run import ValidationFailed, run
from nuagent.agent.task import build_task
from nuagent.agent.tool_policy import allowed_tool
from nuagent.agent.validate import check_card, check_plugin, cross_check, reference_q2
from nuagent.experiments.registry import get
from nuagent.physics.modelcard import ModelCard
from nuagent.physics.plugin import load_coefficient_plugin

EXPERIMENTS = [get("coherent_csi_2017")]


class FakeBackend:
    name = "fake"

    def __init__(self, mode="correct"):
        self.mode = mode
        self.prompts = []

    def run(self, task_dir, max_turns):
        self.prompts.append((task_dir / "TASK.md").read_text())
        (task_dir / "notes.md").write_text("Derivation: mediator matching with sqrt2 GF. No card for general running mixing.")
        if self.mode == "notes":
            (task_dir / "notes.md").write_text("Missing mass, neutrino coupling g_nu and quark coupling g_q; please specify them.")
        elif self.mode == "backend-failure":
            return BackendResult("failed", 2, "backend failure raw", "unavailable")
        else:
            source = COEFFICIENT_PLUGIN
            if self.mode == "sm-failure":
                source = source.replace('return c', 'c.set("V", "u", "mm", 0.2)\n    return c')
            (task_dir / "plugin.py").write_text(source)
            if self.mode != "plugin-only":
                card = json.loads(json.dumps(EQUIVALENT_CARD))
                if self.mode == "correction" and len(self.prompts) == 1:
                    card["mediators"][0]["quark_couplings"]["u"] *= 2
                (task_dir / "card.json").write_text(json.dumps(card))
        return BackendResult("done", 0, '{"model":"fake-model","total_cost_usd":0.1}')


def test_task_inputs_only_context_and_description(tmp_path):
    task = tmp_path / "task"
    build_task(task, "description seed", ["coherent_csi_2017"])
    assert {p.name for p in task.iterdir()} == {"TASK.md", "context"}
    assert "description seed" in (task / "TASK.md").read_text()
    assert "notes-only" in (task / "TASK.md").read_text()
    assert len([p for p in (task / "context").rglob("*") if p.is_file()]) == 12
    assert not list(task.rglob("cases.py"))


def test_one_correction_and_report_provenance(tmp_path):
    backend = FakeBackend("correction")
    report = run("a vector and scalar", ["coherent_csi_2017"], backend, tmp_path / "task", tmp_path / "report.md")
    assert len(backend.prompts) == 2
    assert "plugin/card mismatch" in backend.prompts[1]
    assert "neg2_log_likelihood" not in backend.prompts[1]
    assert "predicted_total" not in backend.prompts[1]
    assert report.startswith("Verification: **cross-validated**")
    for field in ("git_revision", "git_tree", "source_dirty", "package_version", "10.5281/zenodo.1228631", "delta", "alpha", "beta", "fake-model", "total_cost_usd", "Attempt 1", "Attempt 2"):
        assert field in report
    assert (tmp_path / "task/history/attempt-1/card.json").read_text() != (tmp_path / "task/card.json").read_text()


@pytest.mark.parametrize("mode,label", [("plugin-only", "Verification: **plugin-only**"), ("notes", "Outcome: **notes-only**")])
def test_legitimate_outputs(tmp_path, mode, label):
    backend = FakeBackend(mode)
    report = run("description", ["coherent_csi_2017"], backend, tmp_path / "task", tmp_path / "report.md")
    assert report.startswith(label)
    assert len(backend.prompts) == 1
    if mode == "notes":
        assert "predicted_total" not in report
        assert "Verification:" not in report


def test_sm_limit_failure_stops_with_both_reports(tmp_path):
    backend = FakeBackend("sm-failure")
    with pytest.raises(ValidationFailed) as failure:
        run("description", ["coherent_csi_2017"], backend, tmp_path / "task", tmp_path / "report.md")
    assert len(failure.value.histories) == 2
    assert all(any("SM-limit" in finding for finding in history["findings"]) for history in failure.value.histories)
    assert (tmp_path / "report.md").read_text().startswith("Outcome: **validation-failed**")
    assert (tmp_path / "task/history/attempt-2/validation.json").is_file()


@pytest.mark.parametrize("outcome", ["plugin-only", "notes-only"])
def test_correction_can_omit_outputs_under_claude_policy(tmp_path, outcome):
    class RestrictedBackend:
        name = "restricted-fake"
        calls = 0

        def run(self, task_dir, max_turns):
            self.calls += 1
            assert not allowed_tool("Bash", {"command": "rm card.json"}, task_dir)
            def write(name, text):
                path = task_dir / name
                assert allowed_tool("Write", {"file_path": str(path)}, task_dir)
                path.write_text(text)
            if self.calls == 1:
                card = json.loads(json.dumps(EQUIVALENT_CARD))
                card["mediators"][0]["quark_couplings"]["u"] *= 2
                write("plugin.py", COEFFICIENT_PLUGIN)
                write("card.json", json.dumps(card))
                write("notes.md", "Initial derivation with an inconsistent card.")
            else:
                assert not any((task_dir / name).exists() for name in ("plugin.py", "card.json", "notes.md"))
                assert "plugin/card mismatch" in (task_dir / "TASK.md").read_text()
                if outcome == "plugin-only":
                    write("plugin.py", COEFFICIENT_PLUGIN)
                    write("notes.md", "Corrected derivation; no card can express the momentum dependence.")
                else:
                    write("notes.md", "Missing mass and couplings; please specify them.")
            return BackendResult("done", 0, "{}")
    task = tmp_path / "task"
    backend = RestrictedBackend()
    report = run("description", ["coherent_csi_2017"], backend, task, tmp_path / "report.md")
    assert backend.calls == 2
    assert not (task / "card.json").exists()
    assert (task / "plugin.py").exists() == (outcome == "plugin-only")
    first = task / "history/attempt-1"
    assert (first / "plugin.py").read_text() == COEFFICIENT_PLUGIN
    assert (first / "card.json").is_file()
    assert (first / "notes.md").read_text() == "Initial derivation with an inconsistent card."
    assert report.startswith("Verification: **plugin-only**" if outcome == "plugin-only" else "Outcome: **notes-only**")
    if outcome == "notes-only":
        assert "predicted_total" not in report


@pytest.mark.parametrize("invalid_output", ["card-only", "missing-coefficients"])
def test_evaluation_continues_after_exhausted_validation(tmp_path, invalid_output):
    from evals.agent.run import run_evaluation
    class ExhaustedFirstBackend(FakeBackend):
        def run(self, task_dir, max_turns):
            if task_dir.name == "01_universal_vector":
                self.prompts.append((task_dir / "TASK.md").read_text())
                (task_dir / "notes.md").write_text("An invalid translation.")
                if invalid_output == "card-only":
                    (task_dir / "card.json").write_text("{}")
                else:
                    (task_dir / "plugin.py").write_text("PARAMETERS = {}\nCOUPLINGS = ()\n")
                return BackendResult("done", 0, "{}")
            return super().run(task_dir, max_turns)
    backend = ExhaustedFirstBackend("notes")
    root = tmp_path / "eval"
    summary = run_evaluation(root, backend, ["coherent_csi_2017"])
    assert len(summary["cases"]) == 15
    assert len(backend.prompts) == 16
    assert summary["physics_accuracy"] == pytest.approx(1 / 15)
    assert summary["expressibility_recognition"] == pytest.approx((5 if invalid_output == "card-only" else 4) / 15)
    first = summary["cases"][0]
    assert not first["physics_match"]
    assert not first["validation_valid"]
    assert first["expressibility_match"] == (invalid_output == "card-only")
    assert any("attempt 1" in finding for finding in first["physics_failures"])
    assert any("attempt 2" in finding for finding in first["physics_failures"])
    for attempt in (1, 2):
        assert (root / f"01_universal_vector/history/attempt-{attempt}/validation.json").is_file()
    failed_report = (root / "01_universal_vector.md").read_text()
    assert failed_report.startswith("Outcome: **validation-failed**")
    assert "predicted_total" not in failed_report
    assert (root / "15_magnetic_moment.score.json").is_file()
    assert json.loads((root / "summary.json").read_text()) == json.loads(json.dumps(summary))


@pytest.mark.parametrize("failure", ["backend", "import", "execution"])
def test_evaluation_propagates_unexpected_failures(tmp_path, failure):
    from evals.agent.run import run_evaluation
    class FailingBackend(FakeBackend):
        def run(self, task_dir, max_turns):
            result = super().run(task_dir, max_turns)
            if failure == "backend":
                return BackendResult("failed", 2, "unexpected backend failure")
            source = ('raise RuntimeError("unexpected import failure")' if failure == "import" else
                      'PARAMETERS = {}\nCOUPLINGS = ()\ndef coefficients(q2, params):\n    raise RuntimeError("unexpected execution failure")\n')
            (task_dir / "plugin.py").write_text(source)
            return result
    backend = FailingBackend("notes")
    root = tmp_path / "eval"
    with pytest.raises(RuntimeError, match=f"unexpected {failure} failure"):
        run_evaluation(root, backend, ["coherent_csi_2017"])
    assert len(backend.prompts) == 1
    assert not (root / "02_alpha_conversion").exists()
    assert not (root / "summary.json").exists()
    assert (root / "01_universal_vector/history/attempt-1/backend.json").is_file()


def test_backend_failure_retained_and_not_retried(tmp_path):
    backend = FakeBackend("backend-failure")
    with pytest.raises(BackendFailure, match="backend failure raw"):
        run("description", ["coherent_csi_2017"], backend, tmp_path / "task", tmp_path / "report.md")
    assert len(backend.prompts) == 1
    assert "unavailable" in (tmp_path / "task/history/attempt-1/backend.json").read_text()


def test_valid_plugin_card_and_raw(tmp_path):
    plugin = tmp_path / "plugin.py"
    plugin.write_text(COEFFICIENT_PLUGIN)
    card = tmp_path / "card.json"
    card.write_text(json.dumps(EQUIVALENT_CARD))
    assert check_plugin(plugin, EXPERIMENTS).report.valid
    assert check_card(card, EXPERIMENTS).report.valid
    assert cross_check(load_coefficient_plugin(plugin), ModelCard.model_validate(EQUIVALENT_CARD), EXPERIMENTS).valid
    plugin.write_text(RAW_SM_PLUGIN)
    report = check_plugin(plugin, EXPERIMENTS, raw=True).report
    assert report.valid
    assert any("skipped" in text for text in report.checks)


@pytest.mark.parametrize("text", ['{', '{"mediators":[{"structure":"T","mass_mev":10}]}', '{"mediators":[{"structure":"V","mass_mev":0}]}', '{"contacts":[{"fermion":"u","epsilon":{"ee":[0,1]}}]}'])
def test_invalid_cards_are_findings(tmp_path, text):
    path = tmp_path / "card.json"
    path.write_text(text)
    assert not check_card(path, EXPERIMENTS).report.valid


@pytest.mark.parametrize("source,finding", [
    (COEFFICIENT_PLUGIN.replace('"g_nu": 1e-4', '"g_nu": 4.0'), "non-perturbative"),
    (COEFFICIENT_PLUGIN.replace('COUPLINGS = ("g_nu",', 'COUPLINGS = ("missing",'), "COUPLINGS"),
    (COEFFICIENT_PLUGIN.replace('return c', 'c.values["V", "u"][0, 1] = 1\n    return c'), "non-Hermitian"),
    (COEFFICIENT_PLUGIN.replace('return c', 'c.values["V", "u"] = np.zeros((3, 3, 1))\n    return c'), "shape"),
    (COEFFICIENT_PLUGIN.replace('return c', 'c.values["V", "u"][0, 0] = np.nan\n    return c'), "non-finite"),
    ("PARAMETERS = {}\nCOUPLINGS = ()", "missing coefficients"),
])
def test_plugin_boundary_findings(tmp_path, source, finding):
    path = tmp_path / "plugin.py"
    path.write_text(source)
    assert finding in " ".join(check_plugin(path, EXPERIMENTS).report.findings)


def test_unexpected_plugin_failure_propagates(tmp_path):
    plugin = tmp_path / "plugin.py"
    plugin.write_text('raise RuntimeError("import failure")')
    with pytest.raises(RuntimeError, match="import failure"):
        check_plugin(plugin, EXPERIMENTS)


def test_correction_reads_source_even_with_same_size_and_timestamp(tmp_path):
    import os
    plugin = tmp_path / "plugin.py"
    plugin.write_text(COEFFICIENT_PLUGIN)
    first = load_coefficient_plugin(plugin)
    timestamp = plugin.stat().st_mtime_ns
    plugin.write_text(COEFFICIENT_PLUGIN.replace('"g_nu": 1e-4', '"g_nu": 2e-4'))
    os.utime(plugin, ns=(timestamp, timestamp))
    second = load_coefficient_plugin(plugin)
    assert first.parameters["g_nu"] == 1e-4
    assert second.parameters["g_nu"] == 2e-4


def test_negative_raw_counts_are_findings(tmp_path):
    path = tmp_path / "raw.py"
    path.write_text(RAW_SM_PLUGIN.replace("return framework_dsigma_dt", "return -framework_dsigma_dt"))
    assert any("negative count" in f for f in check_plugin(path, EXPERIMENTS, raw=True).report.findings)


def test_crosscheck_all_entries_and_zero_absolute_floor(tmp_path):
    path = tmp_path / "plugin.py"
    path.write_text(COEFFICIENT_PLUGIN.replace('return c', 'c.set("S", "d", "mt", 1e-30)\n    return c'))
    report = cross_check(load_coefficient_plugin(path), ModelCard.model_validate(EQUIVALENT_CARD), EXPERIMENTS)
    assert not report.valid
    q2 = reference_q2(EXPERIMENTS)
    for isotope in EXPERIMENTS[0].detector.isotopes:
        assert np.min(q2) <= isotope.q2(EXPERIMENTS[0].recoil_edges_mev()[0])
        assert np.max(q2) >= isotope.q2(EXPERIMENTS[0].recoil_edges_mev()[-1])


def test_validate_cli(tmp_path):
    path = tmp_path / "plugin.py"
    path.write_text(COEFFICIENT_PLUGIN)
    card = tmp_path / "card.json"
    card.write_text(json.dumps(EQUIVALENT_CARD))
    assert cli("validate", "--plugin", str(path), "--card", str(card))["valid"]
    assert cli("validate", "--card", str(card))["valid"]
    path.write_text(RAW_SM_PLUGIN)
    assert cli("validate", "--raw-plugin", str(path))["valid"]
    card.write_text('{')
    process = subprocess.run([str(Path(sys.executable).with_name("nuagent")), "validate", "--card", str(card)], capture_output=True, text=True)
    assert process.returncode == 1
    assert not json.loads(process.stdout)["valid"]
    assert not process.stderr


def test_claude_tool_policy(tmp_path):
    assert allowed_tool("Read", {"file_path": str(tmp_path / "context/conventions.md")}, tmp_path)
    assert not allowed_tool("Read", {"file_path": "/etc/passwd"}, tmp_path)
    assert allowed_tool("Write", {"file_path": str(tmp_path / "plugin.py")}, tmp_path)
    assert not allowed_tool("Edit", {"file_path": str(tmp_path / "TASK.md")}, tmp_path)
    assert allowed_tool("Bash", {"command": "nuagent validate --plugin plugin.py --card card.json"}, tmp_path)
    for command in ("cat /etc/passwd", "nuagent validate --plugin ../plugin.py", "nuagent validate --plugin plugin.py; pwd"):
        assert not allowed_tool("Bash", {"command": command}, tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "-outside.py")
    (tmp_path / "plugin.py").symlink_to(outside)
    assert not allowed_tool("Write", {"file_path": str(tmp_path / "plugin.py")}, tmp_path)


@pytest.mark.parametrize("name", ["claude", "codex"])
def test_subprocess_backend_configuration(tmp_path, monkeypatch, name):
    (tmp_path / "TASK.md").write_text("task prompt")
    calls = []
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        raw = '{"result":"done","model":"configured"}' if name == "claude" else '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}\n'
        return subprocess.CompletedProcess(command, 0, raw, "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = SubprocessBackend(name).run(tmp_path, 9)
    command, kwargs = calls[0]
    assert result.final_text == "done"
    assert kwargs["cwd"] == tmp_path
    assert kwargs["input"] == "task prompt"
    if name == "claude":
        assert command[command.index("--max-turns") + 1] == "9"
        assert "--restricted" in command
        assert "PreToolUse" in command[command.index("--settings") + 1]
        assert "--strict-mcp-config" in command
    else:
        assert command[command.index("--sandbox") + 1] == "workspace-write"
        assert "sandbox_workspace_write.writable_roots=[]" in command
        assert "--dangerously-bypass-approvals-and-sandbox" not in command


def test_propose_cli_through_fake_executable(tmp_path, monkeypatch):
    import os
    executable = tmp_path / "claude"
    executable.write_text(f"#!{sys.executable}\nfrom pathlib import Path\nimport json\n"
                          f"Path('plugin.py').write_text({COEFFICIENT_PLUGIN!r})\n"
                          f"Path('card.json').write_text({json.dumps(EQUIVALENT_CARD)!r})\n"
                          "Path('notes.md').write_text('Derivation from the stated interaction.')\n"
                          "print(json.dumps({'result':'done','model':'fake','total_cost_usd':0}))\n")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    report = tmp_path / "report.md"
    assert cli("propose", "--description", "description", "--backend", "claude", "--out", str(report)) == {"report": str(report)}
    assert report.read_text().startswith("Verification: **cross-validated**")


def test_repository_evaluation_keeps_answers_outside_tasks(tmp_path):
    from evals.agent.cases import CASES
    from evals.agent.run import run_evaluation
    backend = FakeBackend("notes")
    summary = run_evaluation(tmp_path / "eval", backend, ["coherent_csi_2017"])
    assert len(summary["cases"]) == 15
    assert summary["physics_accuracy"] == pytest.approx(1 / 15)
    assert summary["expressibility_recognition"] == pytest.approx(4 / 15)
    assert summary["card_expressible"]["count"] == 11
    assert summary["plugin_only"]["count"] == summary["notes_only"]["count"] == 2
    for case, prompt in zip(CASES, backend.prompts):
        assert case.description in prompt
        assert "expected_coefficients" not in prompt
        assert not list((tmp_path / "eval" / case.case_id).rglob("cases.py"))
    assert (tmp_path / "eval/summary.json").is_file()


def test_agent_eval_cli_is_repository_only(tmp_path):
    result = subprocess.run([str(Path(sys.executable).with_name("nuagent")), "agent-eval"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2
    assert "repository-only" in result.stderr


def test_agent_eval_cli_default_output(tmp_path, monkeypatch):
    import os
    executable = tmp_path / "claude"
    executable.write_text(f"#!{sys.executable}\nfrom pathlib import Path\nimport json\n"
                          "Path('notes.md').write_text('Missing mass, neutrino coupling and quark coupling; please specify them.')\n"
                          "print(json.dumps({'result':'notes only'}))\n")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    output = cli("agent-eval", "--backend", "claude")
    root = Path(output["output_root"])
    assert root.is_relative_to(tmp_path)
    assert len(output["cases"]) == 15
    assert json.loads((root / "summary.json").read_text()) == output
