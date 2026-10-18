# Offline agent evaluation

`cases.py` turns the fifteen original seeds, one answered conversation and one defaults case into seventeen executable `EvalCase` records. Thirteen cases expect a card, two require a coefficient plugin without a card, one expects questions and one expects notes only. Cases cover universal vectors, alpha-to-coupling conversion, B−L charges, explicit and loop-induced L_mu−L_tau mixing including a negative sign, a down-quark scalar, complex flavor-changing contact NSI, antineutrino wording, the left-handed-current normalization, nucleon-to-quark conversion, full running mixing, a dipole quark coupling, an under-specified description that must end in questions, the same description answered in a second round, a description with no values that must use the assumed defaults, and out-of-basis physics.

This directory is separate from `src/nuagent/context/`, is not installed in the wheel, and must never be copied into an agent task directory. The context's worked examples use different parameters from the evaluation cases. Cases contain the expected answers: pass only `case.description`, and for a case with `answers` those answers through `continue_run`, to the agent, not the record or this directory. Case answers are task inputs like descriptions.

## Interface for the harness in issue #17

Run from the repository root, with already-produced output directories arranged as `OUTPUT_ROOT/<case.case_id>/plugin.py`, `card.json`, `notes.md` and `questions.json`. Each `EvalCase` provides:

- `case_id`: stable output-directory name.
- `description`: the complete agent input.
- `card_expected`: whether `card.json` should exist.
- `expected_coefficients`: a `Coefficients` object on `Q2_MEV2 = [1238, 3714, 6190]` MeV², or `None` for a questions or notes-only outcome.
- `notes_requirements`: labeled groups of regular expressions for the notes-only rubric.
- `question_requirements`: labeled groups of regular expressions for the questions rubric.
- `forbidden_questions`: labeled groups of regular expressions that no question may match.
- `answers`: answers supplied through `continue_run` when the first round ends in questions, or `None`.

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

For the questions outcome (case 14), physics passes only if `plugin.py` is absent and `questions.json` asks about the form of the interaction and never for a number. Each question forms one clause, its `id` and `question` joined, casefolded and with ν written nu. Some clause must match the quark pattern (`g_q`, `g_u`, `g_d`, `g_p`, `g_n` or `quark`) or `structure`, `vector`, `scalar` or `couple`; no clause may match `mass` or a value word (`value`, `numerical`, `numeric`, `magnitude`, `how large`, `how big`, `how strong`). Notes alone, without `questions.json`, fail.

For notes-only outcomes (case 15), physics passes only if `plugin.py` and `questions.json` are absent and `notes.md` meets the case's rubric:

- Out of basis: explicitly identify physics outside the V/S (or vector/scalar) basis and name the user-authored raw cross-section route. A generic refusal or a reference to “raw” alone does not pass.

The answered case (16) repeats case 14's description; the runner answers its first-round questions with the free text “a universal vector coupling to u and d quarks and all three neutrino flavors, mass 30 MeV, coupling 1e-4 to neutrinos and to quarks” and scores the final round against the universal-vector coefficients at that point. The defaults case (17), “A universal vector Z′ coupling equally to all three neutrino flavors and to u and d quarks.”, gives no values and expects a card and the universal-vector coefficients at the assumed defaults of `conventions.md` section 12 (10 MeV, 1e-4, 1e-4, 1e-4), which checks that the defaults are reproducible.

The rubrics are English text checks, with aliases listed in `cases.py`; they are not semantic review of the derivation. In notes, soft line breaks are joined within paragraphs and list items. Each list item inherits its preceding Markdown heading or colon-ended introduction. Periods, question marks, exclamation marks, semicolons, paragraph breaks and separate list items retain their clause boundaries, so a statement in unrelated prose does not satisfy a requirement. Equivalent wording outside the listed aliases may require human review or a rubric update.

**Expressibility recognition**: `card.json` is present exactly when `card_expected` is true. This is independent of physics accuracy, including for notes-only outcomes. Presence alone does not establish valid or correct card contents. Card validation, plugin/card cross-checking, sanity checks and verification labels belong to the harness in #17.

Issue #16 delivers these offline cases and the scorer. Issue #17 owns the backends, harness, `nuagent validate`, `nuagent agent-eval --backend NAME`, and live acceptance evidence. No live backend performance is established by the offline tests.

## Live runner

From the repository root, run `uv run nuagent agent-eval --backend claude --out-dir /private/tmp/nuagent-eval` (or `--backend codex`). The output directory must be new. Without `--out-dir`, the command selects a fresh system temporary directory and prints its location; `output_root` in the JSON summary names that directory. Every case retains its task directory, report and `.score.json`, which also records the final `state` and the number of `rounds`; `summary.json` records all per-case results, both overall rates and separate card-expressible, plugin-only and no-plugin (questions and notes-only) rates. Only the case description, and a case's answers when its first round ends in questions, enter a task. Exhausted validation produces a `validation-failed` report and a failed physics score containing both attempts' findings, marks `validation_valid` false and continues to the next case. Expressibility still measures card presence. Unexpected backend, import or execution failures stop the run with retained evidence.
