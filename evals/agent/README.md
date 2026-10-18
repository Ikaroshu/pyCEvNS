# Offline agent evaluation

`cases.py` turns the fifteen original seeds into executable `EvalCase` records. Eleven cases expect a card, two require a coefficient plugin without a card, and two expect notes only. Cases cover universal vectors, alpha-to-coupling conversion, B−L charges, explicit and loop-induced L_mu−L_tau mixing including a negative sign, a down-quark scalar, complex flavor-changing contact NSI, antineutrino wording, the left-handed-current normalization, nucleon-to-quark conversion, full running mixing and a dipole quark coupling.

This directory is separate from `src/nuagent/context/`, is not installed in the wheel, and must never be copied into an agent task directory. The context's worked examples use different parameters from the evaluation cases. Cases contain the expected answers: pass only `case.description` to the agent, not the record or this directory.

## Interface for the harness in issue #17

Run from the repository root, with already-produced output directories arranged as `OUTPUT_ROOT/<case.case_id>/plugin.py`, `card.json` and `notes.md`. Each `EvalCase` provides:

- `case_id`: stable output-directory name.
- `description`: the complete agent input.
- `card_expected`: whether `card.json` should exist.
- `expected_coefficients`: a `Coefficients` object on `Q2_MEV2 = [1238, 3714, 6190]` MeV², or `None` for a notes-only outcome.
- `notes_requirements`: labeled groups of regular expressions for the notes-only rubric.

Expectations are generated at full precision from the exported physics constants and direct matching formulas, independently of model cards. They include every structure, quark and flavor entry, including exact zeros and Hermitian partners. The running-mixing answer uses adaptive quadrature of the one-loop integral; tests compare it with the worked example's independent Gauss–Legendre implementation. No rounded coefficient tables are used as answers.

```python
from pathlib import Path
from evals.agent.cases import CASES
from evals.agent.scorer import score_case, score_outputs

result = score_case(CASES[0], Path("outputs/01_universal_vector"))
print(result.physics_match, result.expressibility_match, result.physics_failures)

summary = score_outputs(Path("outputs"))
for result in summary.cases:
    print(result)
print(summary.physics_accuracy, summary.expressibility_recognition)
```

`score_outputs(output_root, cases=CASES)` accepts a nonempty sequence of cases, scores each corresponding directory and returns an `EvaluationScore`. Its `cases` tuple retains per-case results in input order; the two rates are passing cases divided by all supplied cases. The scorer creates no task inputs, runs no backend and makes no network calls. It executes coefficient plugins through the existing in-process loader, which may create Python bytecode caches; only score plugins from a trusted task directory.

## Metrics

**Physics accuracy**: evaluate `plugin.py` at its `PARAMETERS` defaults on the three reference Q² values. All four structure/quark matrices must have shape `(3, 3, 3)` and every complex entry must match with `rtol=1e-6`, `atol=0`. Thus any nonzero entry where zero is expected fails, and a missing matrix cannot pass by broadcasting. Missing required plugins, non-finite coefficients and numerical mismatches are failed scores with reasons. Plugin import or execution errors propagate directly to the caller.

For notes-only outcomes, physics passes only if `plugin.py` is absent and `notes.md` meets the case's rubric:

- Under-specified: identify the missing mass, neutrino coupling and quark coupling. Each item must share a sentence or clause with a missing-information statement or request, such as “Missing mass, g_nu and g_q” or “What is the mass? Please provide the neutrino coupling and quark coupling.” A Markdown list may put the request in its heading or a colon-ended introduction, such as “Missing information:” followed by bullets naming the three inputs. Merely mentioning the inputs does not pass.
- Out of basis: explicitly identify physics outside the V/S (or vector/scalar) basis and name the user-authored raw cross-section route. A generic refusal or a reference to “raw” alone does not pass.

The rubric is an English text check, with aliases listed in `cases.py`; it is not semantic review of the derivation. Soft line breaks are joined within paragraphs and list items. Each list item inherits its preceding Markdown heading or colon-ended introduction. Periods, question marks, exclamation marks, semicolons, paragraph breaks and separate list items retain their clause boundaries, so a missing-information statement in unrelated prose does not qualify a known input. Equivalent wording outside the listed aliases may require human review or a rubric update.

**Expressibility recognition**: `card.json` is present exactly when `card_expected` is true. This is independent of physics accuracy, including for notes-only outcomes. Presence alone does not establish valid or correct card contents. Card validation, plugin/card cross-checking, sanity checks and verification labels belong to the harness in #17.

Issue #16 delivers these offline cases and the scorer. Issue #17 owns the backends, harness, `nuagent validate`, `nuagent agent-eval --backend NAME`, and live acceptance evidence. No live backend performance is established by the offline tests.

## Live runner

From the repository root, run `uv run nuagent agent-eval --backend claude --out-dir /private/tmp/nuagent-eval` (or `--backend codex`). The output directory must be new. Without `--out-dir`, the command selects a fresh system temporary directory and prints its location; `output_root` in the JSON summary names that directory. Every case retains its task directory, report and `.score.json`; `summary.json` records all per-case results, both overall rates and separate card-expressible, plugin-only and notes-only rates. Only the case description enters a task. Exhausted validation produces a `validation-failed` report and a failed physics score containing both attempts' findings, marks `validation_valid` false and continues to the next case. Expressibility still measures card presence. Unexpected backend, import or execution failures stop the run with retained evidence.
