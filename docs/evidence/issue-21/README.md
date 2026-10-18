# M7 conversational agent rounds and parameter re-evaluation

All live runs below ran on 2026-10-18 at commit
`9a0bb9100a177d1959606dfd94eeb4b92afeed81` (tree
`79c38e2695fcbf26a3535cbcb843611ccf137fd2`), with Claude Code 2.1.286,
codex-cli 0.160.0 and Streamlit 1.66.0. Models were left to each CLI's own
configuration; Claude Code reported `claude-fable-5-1`. Each run was made
once, and no agent output was edited.

Some reports record `source_dirty: true`. In each such case the only
difference from the commit was this directory's untracked screenshots,
which were being saved while the run was in progress: the two page round-2
reports and the codex CLI round-2 report. All other reports, including all
sixteen evaluation reports, record `source_dirty: false`.

Retained copies omit each task's `context/` directory, which is the
installed context pack of the commit above (13 files, including the new
worked example 07). Everything else is unmodified: `TASK.md`,
`conversation.json`, outputs, `history/round-N/attempt-M/` and reports.

## Conversation from the page

`uv run nuagent ui --port 8611 </dev/null` served the page. Two browser
tabs ran the description "A new light vector boson that couples to
neutrinos." at the same time, one with backend `claude` and one with
`codex`, on `coherent_csi_2017`. The server was stopped afterwards; no
listener remained on 8611 and no `.agents/` or `.claude/` skill links were
created (the Streamlit skill toast was closed with its × control).

| | claude | codex |
|---|---|---|
| Round 1 | questions: `mass_mev` (suggested 10 MeV), `g_nu` (suggested 1e-4), `nu_flavor_structure`, `g_u`, `g_d`, `kinetic_mixing` | questions: `mass_mev` (suggested 10 MeV), `nu_couplings`, `g_u`, `g_d` (no coupling suggestions) |
| Answers entered | mass 30, g_nu 0.0001 (kept), flavors ee, mm, tt equal and no off-diagonal, g_u 1e-4, g_d 1e-4, no kinetic mixing | mass 30, ν couplings 1e-4 for ee, mm, tt and no off-diagonal, g_u 1e-4, g_d 1e-4 |
| Round 2 | `cross-validated`, one attempt | `cross-validated`, one attempt |
| Δ, predicted total | 14.196339128054618, 102.54128867480074 | 14.196339128054575, 102.54128867480074 |
| Backend records | round 1: 38 turns, $0.97; round 2: 24 turns, $0.81 | round 1: 225461 input / 1219 output tokens; round 2: 243696 / 1930 |

- Question forms: [claude](page-claude-questions.jpg) (number inputs show
  `10` and `0.0001` in `%g`; text inputs empty where no value was
  suggested, with the reason under each suggested value),
  [codex](page-codex-questions.jpg) (with the conversation above the form).
  Filled forms: [claude](page-claude-answers.jpg),
  [codex](page-codex-answers.jpg).
- Results: [claude report](page-claude-report.jpg) and
  [codex report](page-codex-report.jpg) begin with `cross-validated` and list
  round 1's questions and answers in the Conversation section.
  [claude spectrum and parameter form](page-claude-spectrum-and-parameters.jpg),
  [codex](page-codex-spectrum-and-parameters.jpg): the form holds the
  plugin's `PARAMETERS` with the agent's values (0.0001, 30).
- Both answered rounds recorded the backend of round 1: the spinner said
  "Running round 2 on claude" and "Running round 2 on codex", and each
  `conversation.json` names that backend.
- Re-evaluation: in each tab, `mass_mev` was changed to 50 and "Evaluate"
  pressed. The page showed "Plugin-only at this point: …", Δ = 16.221,
  predicted total 11.5218 events, the profiled nuisances, and a redrawn
  spectrum ([claude statement](page-claude-reevaluated-statement.jpg),
  [claude Δ](page-claude-reevaluated-delta.jpg),
  [claude spectrum](page-claude-reevaluated-spectrum.jpg),
  [codex Δ](page-codex-reevaluated-delta.jpg),
  [codex spectrum](page-codex-reevaluated-spectrum.jpg)). The result
  appeared within seconds, without a backend call.
- `uv run nuagent compare --experiment coherent_csi_2017 --plugin
  <claude task>/plugin.py --set mass_mev=50` gives Δ 16.220961829251966,
  signal 11.521798691439773 and new-physics α 0.06161559029124771, equal to
  the page.
- SHA-1 hashes of every file in both task directories and reports, taken
  after round 2 and again after re-evaluation, were identical.

Retained: [page-claude/](page-claude/) and [page-codex/](page-codex/).

## Continuation from the command line

For each backend B:

```bash
uv run nuagent propose --backend B --description "A new light vector boson that couples to neutrinos." \
  --out cli-B/round1.md --task-dir cli-B/task
uv run nuagent propose --task-dir cli-B/task --out cli-B/report.md \
  --answer free_text="mass 30 MeV, coupling 1e-4 to all three left-handed neutrino flavors and the same coupling 1e-4 to both u and d quarks, nothing else"
```

All four commands exited 0. Round 1 printed `"state": "questions"`, the
round and the questions ([claude](cli-claude/round1.stdout),
[codex](cli-codex/round1.stdout)); round 2 printed only the report path.
The second command has no `--backend`; round 2 ran on the backend recorded
at round 1: `conversation.json` says `"backend": "claude"` and `"codex"`,
the round-2 backend records show the `claude -p` and `codex exec` command
lines, and both reports record `"rounds": 2`.

| | claude | codex |
|---|---|---|
| Round 1 questions | `mass_mev` (10 MeV), `nu_flavors`, `g_nu` (1e-4), `g_u`, `g_d` | `mass_mev` (10 MeV), `nu_couplings`, `g_u`, `g_d` |
| Round 2 | `cross-validated`, one attempt | `cross-validated`, one attempt |
| Δ, predicted total | 14.196339128054618, 102.54128867480074 | 14.196339128054575, 102.54128867480074 |
| Backend records | 17 turns, $0.82; 24 turns, $0.82 | 190833 / 1199 tokens; 311399 / 1526 |

`uv run nuagent compare --experiment coherent_csi_2017 --template
zprime_universal --set mass_mev=30 --set g_nu=1e-4 --set g_q=1e-4` gives
Δ 14.196339128054618 and signal 102.54128867480074. The codex plugin's Δ
differs in the fourteenth digit, from its order of floating-point
operations.

Retained: [cli-claude/](cli-claude/) and [cli-codex/](cli-codex/).

## Parent acceptance description

```bash
uv run nuagent propose --backend B --experiments coherent_csi_2017 \
  --description "a gauged L_mu minus L_tau Z prime with mass 10 MeV, gauge coupling 1e-4 and kinetic mixing 1e-3" \
  --out propose-B/report.md
```

Both backends completed in one round and one attempt, `cross-validated`,
with Δ = 7.507566092489984 and predicted total 55.64316965812137, the same
values as in `docs/evidence/issue-17/` and `issue-18/`. Claude Code: 20
turns, $1.07, 78 s; codex: 201343 input and 1497 output tokens, 52 s.
The new backend-neutral task prompt did not cause a refusal on either.
Retained: [propose-claude/](propose-claude/), [propose-codex/](propose-codex/).

## Live evaluation

```bash
uv run nuagent agent-eval --backend claude --out-dir <scratch>/live/eval-claude
```

The command exited 0 after 27 min 11 s. Claude Code reported $15.04 in
total over 17 backend calls (16 cases, case 16 in two rounds, no
correction attempts).

| Group | Cases | Physics accuracy | Expressibility recognition |
| --- | ---: | ---: | ---: |
| All | 16 | 16/16 | 16/16 |
| Card-expressible | 12 | 12/12 | 12/12 |
| Plugin-only | 2 | 2/2 | 2/2 |
| No plugin (questions, notes-only) | 2 | 2/2 | 2/2 |

| Case | State | Rounds |
| --- | --- | ---: |
| 01–13 | valid | 1 |
| 14_under_specified | questions | 1 |
| 15_magnetic_moment | notes-only | 1 |
| 16_answered_light_vector | valid | 2 |

Case 14 asked eight questions: `mass_mev`, `g_nu_ee`, `g_nu_mm`, `g_nu_tt`,
`g_nu_offdiagonal`, `g_u`, `g_d` and `kinetic_mixing`; the per-quark
`g_u`/`g_d` questions satisfy the quark requirement. Case 16 received the
case's free-text answer through `continue_run` and produced the universal
vector at (30 MeV, 1e-4, 1e-4, 1e-4). Every case that passed in
`docs/evidence/issue-17/` passes again.

[eval-claude/](eval-claude/) keeps `summary.json`, the command's stdout,
each report, `.score.json` (with `state` and `rounds`) and each task
directory. The retained outputs score again offline:

```bash
uv run python -c "from pathlib import Path; from evals.agent.scorer import score_outputs; s = score_outputs(Path('docs/evidence/issue-21/eval-claude')); print(s.physics_accuracy, s.expressibility_recognition)"
```

which prints `1.0 1.0`.

## Offline verification

At the commit above:

- `uv run pytest`: 367 passed (329 before this milestone). The only
  warning is the frozen legacy package's `pkg_resources` deprecation.
- `uv run nuagent ui --help`: exits 0 and lists `--port`.
- `uv build --out-dir <scratch>`: the wheel's `nuagent/context/` holds the
  thirteen context files, including `examples/07_under_specified_scalar.md`
  and the updated `README.md`, `conventions.md`, `framework.md` and
  `plugin_api.md`.

## Not verified

- The page's unresolved state (three question rounds) and the
  re-evaluation of a point that fails the plugin checks were exercised by
  the offline page tests only, not live.
- `agent-eval` ran on Claude Code only; codex covered the page, the CLI
  continuation and the acceptance description.
