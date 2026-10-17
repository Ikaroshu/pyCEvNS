"""Reproduce a seeded invalid attempt followed by one restricted live correction."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from evals.agent.cases import CASES
from evals.agent.scorer import score_case
from nuagent.agent.backend import BackendResult, get_backend
from nuagent.agent.run import run


class SeededCorrectionBackend:
    name = "seeded_then_claude_code"

    def __init__(self, source: Path):
        self.source = source
        self.calls = 0
        self.fresh_outputs_before_live = False
        self.native = get_backend("claude_code")

    def run(self, task_dir: Path, max_turns: int) -> BackendResult:
        self.calls += 1
        if self.calls == 1:
            (task_dir / "plugin.py").write_bytes(self.source.read_bytes())
            (task_dir / "card.json").write_text("{}")
            (task_dir / "notes.md").write_text("Seeded inconsistent card for correction-path verification; this first attempt is a test fixture, not model output.")
            return BackendResult("seeded first attempt", 0, json.dumps({"fixture": True, "source": str(self.source)}))
        assert self.calls == 2
        self.fresh_outputs_before_live = not any((task_dir / name).exists() for name in ("plugin.py", "card.json", "notes.md"))
        assert self.fresh_outputs_before_live
        return self.native.run(task_dir, max_turns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outcome", choices=("plugin-only", "notes-only"))
    parser.add_argument("--out-root", required=True, type=Path)
    args = parser.parse_args()
    case_id, seed_id = (("12_running_mixing", "12_running_mixing") if args.outcome == "plugin-only" else
                        ("15_magnetic_moment", "01_universal_vector"))
    case = next(case for case in CASES if case.case_id == case_id)
    backend = SeededCorrectionBackend(ROOT / "docs/evidence/issue-17/eval-claude_code" / seed_id / "plugin.py")
    root = args.out_root.resolve()
    task = root / "report.task"
    report = run(case.description, ["coherent_csi_2017"], backend, task, root / "report.md")
    assert backend.calls == 2
    assert report.startswith("Verification: **plugin-only**" if args.outcome == "plugin-only" else "Outcome: **notes-only**")
    assert not (task / "card.json").exists()
    assert (task / "plugin.py").exists() == (args.outcome == "plugin-only")
    score = score_case(case, task)
    assert score.physics_match and score.expressibility_match
    result = {"case_id": case_id, "first_attempt": "seeded invalid card, not model-generated",
              "live_backend_attempts": backend.calls - 1, "fresh_outputs_before_live": backend.fresh_outputs_before_live,
              "score": asdict(score), "report": str(root / "report.md")}
    (root / "repair-check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
