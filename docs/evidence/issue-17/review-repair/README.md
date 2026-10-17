# Review pass 1 repairs

Both reported defects were reproduced before repair: an exhausted first case
raised before scoring and stopped evaluation, and obsolete output files remained
present when a restricted backend attempted a plugin-only or notes-only correction.
The four new parameterized regression cases failed against the reviewed candidate.

The repaired source commit is `543b84d30985ca4f3e75a0055f5664f9fc111921`, tree
`42dce7f80339b1a6a18d18be3fb96de9d1b6dfca`. Both targeted live reports below
record that source identity with `source_dirty: false`.

## Shared result and correction behavior

`run_result` returns exhausted validation as an ordinary result and writes a
`validation-failed` report without physics verification or predictions. Evaluation
records a failed physics score with both attempts' findings and continues. The
existing scorer still measures expressibility from card presence; it does not
execute a plugin already known to have exhausted validation. The `run` entry point
used by `propose` still raises `ValidationFailed` with both reports after exhaustion.
Unexpected backend, import and execution failures propagate without interception.

Before correction the harness preserves all first-attempt outputs and reports,
then clears only `plugin.py`, `card.json` and `notes.md`. The second backend writes
the complete corrected outcome, so omitted files cannot leave a stale card or
plugin behind. Claude's tool policy is unchanged; removal commands remain denied.

## Targeted live correction checks

These checks deliberately **seed the first attempt** with a previously retained
agent coefficient plugin, an incompatible empty card and notes identifying the
fixture. That first attempt is not new model output. Only the correction invokes
the real restricted Claude Code backend. No human edits the live correction's
outputs, and the correction receives validation findings without likelihood or
expected-answer feedback.

```bash
uv run python docs/evidence/issue-17/review-repair/check_corrections.py plugin-only \
  --out-root /private/tmp/nuagent-issue17-repair-plugin-only
uv run python docs/evidence/issue-17/review-repair/check_corrections.py notes-only \
  --out-root /private/tmp/nuagent-issue17-repair-notes-only
```

Both commands exited 0. Each confirms that all three output paths were absent
before the live correction, the first validation failed, the second passed, and
the first plugin/card/notes remain in the first attempt's history.

| Target | Live correction calls | Final outcome | Physics | Expressibility |
| --- | ---: | --- | --- | --- |
| Full running mixing | 1 | plugin-only; no card | pass | pass |
| Magnetic moment | 1 | notes-only; no plugin/card | pass | pass |

See the [plugin-only report](plugin-only/report.md) and
[notes-only report](notes-only/report.md), their `repair-check.json` records,
and both attempts' original outputs, validation reports and raw backend records.
Copies omit repeated context directories, which are the installed context at the
source commit above. The script is retained to reproduce the checks. The original
two-backend proposals and fifteen-case live evaluation were preserved rather than
rerun; these targeted calls exercise the newly repaired correction transitions.

## Offline verification

```bash
uv run pytest tests/test_agent.py tests/test_agent_evals.py tests/test_plugins.py tests/test_golden_regression.py -q
uv run pytest
```

The focused command passed **218 tests**; the full suite passed **322 tests**.
The only warning is the frozen legacy package's existing `pkg_resources`
deprecation warning. Batch regressions cover both a card without its plugin and
a plugin missing `coefficients`: after two failed attempts, every later case runs,
all fifteen cases receive scores and the summary includes the failed case in its
denominator. Both invalid attempts are retained. Three additional regressions
confirm that unexpected backend, import and execution failures still stop on the
first case. The two output-transition regressions enforce the actual Claude
policy function, including its denial of deletion commands.

Re-scoring the original fifteen retained outputs still returns 15/15 physics and
15/15 expressibility. The changes add no exception-catching constructs and do
not alter physics, rate integration or likelihood calculations.
