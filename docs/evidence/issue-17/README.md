# M5b implementation and live acceptance

All acceptance runs below completed on 2026-10-17 without human edits to agent
outputs. The live source commit was
`934e7113687b7d2927d34462b77925c3fef56c73`, tree
`8d29d0e53eca3f2cb3350458bc84362237f6121c`; every report records
`source_dirty: false`. The later commit
`875ca43ea48769e329c2b4ae72773ee89d7500cf` (tree
`bb25225049685f8a14e7efeae665e38abbf29653`) makes `agent-eval`'s output
directory optional and adds output-location metadata. It changes no backend,
task, validation, likelihood or report behavior exercised by these live commands.
The focused default-output check and full suite were rerun for that change.

## Live proposals

The installed executables were Claude Code 2.1.286 and codex-cli 0.160.0. Their
models were left to their own configuration. Both commands exited 0, produced a
coefficient plugin and card, passed on the first harness attempt, and reported
`cross-validated`. Independently applying evaluation case
`04_lmu_ltau_explicit` to each output passed both metrics. Each gives
55.64316965812137 signal events and Δ = 7.507566092489984 on
`coherent_csi_2017`, with release `10.5281/zenodo.1228631`.

```bash
uv run nuagent propose --backend claude_code --experiments coherent_csi_2017 \
  --description 'a gauged L_mu minus L_tau Z prime with mass 10 MeV, gauge coupling 1e-4 and kinetic mixing 1e-3' \
  --out /private/tmp/nuagent-issue17-live-claude/report.md
uv run nuagent propose --backend codex --experiments coherent_csi_2017 \
  --description 'a gauged L_mu minus L_tau Z prime with mass 10 MeV, gauge coupling 1e-4 and kinetic mixing 1e-3' \
  --out /private/tmp/nuagent-issue17-live-codex/report.md
```

Retained reports: [Claude Code](propose-claude_code/report.md) and
[Codex](propose-codex/report.md). Adjacent task directories retain the original
prompt, outputs, validation report and raw backend record. Codex emitted local
configuration/state warnings on stderr; its exit status, successful validation
and unmodified stderr are retained, rather than suppressing those warnings.

## Live evaluation

```bash
uv run nuagent agent-eval --backend claude_code --out-dir /private/tmp/nuagent-issue17-eval-claude
```

The command exited 0. All fifteen cases completed, each on one harness attempt:

| Group | Cases | Physics accuracy | Expressibility recognition |
| --- | ---: | ---: | ---: |
| All | 15 | 15/15 (100%) | 15/15 (100%) |
| Card-expressible | 11 | 11/11 (100%) | 11/11 (100%) |
| Plugin-only | 2 | 2/2 (100%) | 2/2 (100%) |
| Notes-only | 2 | 2/2 (100%) | 2/2 (100%) |

See [the unmodified summary](eval-claude_code/summary.json), the fifteen sibling
Markdown reports and `.score.json` files, and each case's output directory.
The two plugin-only cases have no minimum acceptance threshold. The two
notes-only cases contain neither plugin nor card and have no predictions or
physics verification label. No expected evaluation answer was copied into a
task or supplied to a backend; scoring happened outside each task afterward.

The original scratch roots remain as named in the commands. Retained copies
omit repeated `context/` directories: every task used the twelve installed
context files at the live source commit above. Agent outputs, prompts, check
histories, raw CLI records and reports are otherwise unmodified. The retained
outputs can be scored again without a live backend:

```bash
uv run python - <<'PY'
from pathlib import Path
from evals.agent.scorer import score_outputs
print(score_outputs(Path('docs/evidence/issue-17/eval-claude_code')))
PY
```

## Offline and packaging verification

- `uv run pytest tests/test_agent.py -q`: 28 passed before the default-output
  addition. Fake backends exercise an initial factor-two mismatch corrected on
  the second attempt, two SM-limit failures with both reports retained,
  plugin-only and notes-only outcomes, backend failures, plugin execution
  failures, validation boundaries and backend tool restrictions.
- `uv run pytest tests/test_agent.py::test_agent_eval_cli_default_output tests/test_agent.py::test_agent_eval_cli_is_repository_only -q`:
  2 passed after the default-output addition.
- `uv run pytest`: 315 passed after the default-output addition. The only warning
  is the frozen legacy package's existing `pkg_resources` deprecation warning.
- `uv build --wheel --out-dir /private/tmp/nuagent-issue17-wheel`: succeeded.
  Installing the wheel into `/private/tmp/nuagent-wheel-smoke.3tMBqG` and running
  its `nuagent validate --plugin` on the packaged example outside the repository
  succeeded. Its installed `nuagent propose` also produced a valid notes-only
  report through a fake CLI outside the repository, with all twelve context
  files present and no evaluation answers. These packaging checks used the live
  source commit; the subsequent output-directory change affects repository-only
  evaluation and is covered separately above.

CLI choices were grounded in installed help and primary documentation:
[Codex exec](https://learn.chatgpt.com/docs/developer-commands#codex-exec),
[Claude Code CLI](https://code.claude.com/docs/en/cli-reference) and
[Claude Code permissions](https://code.claude.com/docs/en/permissions).
Claude tool restrictions use a PreToolUse hook because permission allow rules
alone do not restrict built-in read-only shell commands. Codex uses the requested
workspace-write sandbox. Neither backend's isolation was weakened for acceptance.
Codex exec has no native turn-limit flag; `max_turns` is enforced by Claude Code
only. Plugins still execute in-process, and finite-grid representation agreement
does not prove a shared physical interpretation correct; the reports state
these limits explicitly.
