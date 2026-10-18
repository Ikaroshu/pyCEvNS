# nuagent

A research prototype for neutrino new physics in coherent elastic neutrino-nucleus scattering (CEvNS). You describe a new interaction; `nuagent` computes the predicted event spectrum for an experiment and reports a log-likelihood or Asimov sensitivity relative to the Standard Model. A coding-agent CLI can translate a plain-language description of the interaction into the coefficient plugin that `nuagent` evaluates.

Status: milestone 7 of [ROADMAP.md](ROADMAP.md). Physics, prediction, likelihoods, scans, the context pack, the coding-agent harness with conversation rounds, and a local Streamlit page with parameter re-evaluation are implemented.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is installed by uv if needed.

```bash
uv sync
uv run pytest
uv run nuagent --help
```

## Predict a spectrum

```bash
uv run nuagent experiments
uv run nuagent predict --experiment coherent_csi_2017 --template zprime_universal --set mass_mev=30 --set g_nu=0 --set g_q=0
uv run nuagent predict --experiment coherent_csi_2017 --template zprime_universal --set mass_mev=30 --set g_nu=1e-4 --set g_q=1e-4
uv run nuagent predict --experiment coherent_csi_2017 --card card.json
uv run nuagent predict --experiment coherent_csi_2017 --plugin plugin.py
uv run nuagent predict --experiment coherent_csi_2017 --raw-plugin raw.py
```

Commands print JSON. `predict` includes signal, observed counts and both background
components in each bin, plus their totals. Cards and coefficient plugins specify
new physics; the framework adds the SM once. `--raw-plugin` takes the complete
cross section, including the SM. Plugin defaults can be overridden with `--set`.
See [the model examples](docs/agent/examples/01_zprime_universal.md) and
[plugin API](docs/agent/plugin_api.md).

The CsI record uses 4466 kg day, Helm radius 5.5 fm, constant quenching 0.0878,
and light yield 13.348 PE/keVee. Its 12 bins span 6 <= PE < 30. The zero-coupling
SM prediction is about **133.9555 events**; the observed total is 855, the
steady-state background is 703, and the prompt neutron expectation is 5.41934.
The observed and steady-state counts sum the full released **0–12 μs** window.
The paper's likelihood uses time <6 μs; this prototype integrates timing out.

Data come from [Zenodo 10.5281/zenodo.1228631](https://doi.org/10.5281/zenodo.1228631),
with the inputs and offline conversion script checked in
[beside the converted data](src/nuagent/experiments/data/README.md).
Signal folding preserves the legacy Michel tabulation and exact 29 MeV line,
resolving efficiency steps and each isotope's line endpoint. It uses a continuous
recoil response without photoelectron smearing and the coherent spin-0 vector
and scalar formulas. The separate legacy README recoil window (0.00425–0.026
MeV) gives about 135.933 events. Tests preserve its original single-window result
and compare production folding with a converged reference.

## Compare with the Standard Model

```bash
uv run nuagent compare --experiment coherent_csi_2017 --card card.json
uv run nuagent scan --experiment coherent_csi_2017 --template zprime_lmu_ltau \
  --param mass_mev=1:10000:25:log --param g_prime=1e-4:1:65:log --data observed
uv run nuagent scan --experiment coherent_csi_2017 --template zprime_lmu_ltau \
  --param g_prime=1e-4:1e-2:30:log --set mass_mev=10
```

`compare` takes the same model options as `predict` and prints, as JSON,
Δ = −2 ln L(new physics) − (−2 ln L(SM)) on the observed counts, with both
profiled likelihoods and their nuisances: `alpha` scales the signal (28 percent
prior) and `beta` each background (steady state 1/√703, prompt neutrons 25
percent). Positive Δ means the data prefer the SM. The likelihood is binned
Poisson with Gaussian priors on the normalizations.

`scan` prints CSV: one column per `--param`, then `delta`. Each `--param` is
`NAME=min:max:n`, linearly spaced, or geometrically spaced with a `:log` suffix;
`--set` fixes the other template parameters. `--data asimov` (the default) scans
against SM Asimov data, the expected counts with nuisances at zero, for an
expected sensitivity; `--data observed` uses the released counts. Under the
asymptotic approximation Δ = 2.71 and 4.61 are the 90 percent confidence-level
thresholds for one and two parameters, and Δ = 4 is the one-parameter 2σ limit.

## Use the local page

```bash
uv run nuagent ui
uv run nuagent ui --port 8600
```

`ui` serves a Streamlit page at http://127.0.0.1:8501 (or `--port`), bound to
127.0.0.1 only, with Streamlit's usage-statistics gathering and first-run email
prompt disabled. Describe an interaction, pick a backend (`claude` or
`codex`, installed as described below) and the experiments, and press
"Run agent". The page runs the `propose` harness in a fresh temporary
directory and shows the exchange as a chat: your description, the agent's
questions with an answer form inside, and your answers. A validated result
follows as a page section below the chat; a notes-only or failed outcome is
a short message from the agent. A run takes minutes and uses the backend's
paid model.

The agent asks only about the form of the interaction (which particles
couple, vector or scalar, the flavor structure, or a physics-changing sign
or convention), never for a number. Missing masses, couplings and mixings
take the benchmark defaults of `conventions.md` section 12 (10 MeV, vector
couplings 1e-4, scalar couplings 1e-5, contact ε 0.1, kinetic mixing 0
unless the model fixes it), declared in the plugin's `PARAMETERS`. The
answer form is pre-filled with the agent's suggestion where it gave one
(numbers in a general `%g` format), with the reason beside it, plus a
free-text box. "Submit answers" runs the next round in the same task with
the backend and experiments of the first round; after three rounds that
still end in questions the page says the round limit was reached.

For a validated model the result section has, in this order:

1. "Evaluate the interaction": one input per plugin parameter, pre-filled
   with the agent's values, and an Evaluate button.
2. The predicted spectrum per experiment, of the model and of the SM, each
   plus the summed backgrounds, against the observed counts with √N errors.
3. A table per experiment of observed counts, SM and new-physics signal, Δ,
   and the profiled signal scale α and background scales β of both models,
   with a caption saying what α and β mean and the verification level.
4. The agent's notes under the heading "Note" (without their own title),
   rendered as Markdown with LaTeX math.
5. Collapsed expanders: "Provenance", "Plugin (plugin.py): the coefficient
   code the agent wrote" and "Model card (card.json): the interaction written
   in the fixed schema the harness cross-checks against the plugin".

The figure and table show the agent's point first. Evaluate checks the
plugin at the entered point (the SM limit with `COUPLINGS` zeroed,
perturbativity and counts) and redraws both in place. The caption names the
verification level as the report does, for example "Verification:
cross-validated (the plugin and the model card agree at the agent's
values).", and at other values adds "Evaluated by the plugin alone at these
values; the model-card cross-check holds at the agent's values.", because
the card's numbers were written for the agent's point only. A point that fails the checks lists the
findings and keeps the agent's point. Re-evaluation does not call the agent
and leaves the task directory and report unchanged. On the command line the
same evaluation is `predict` or `compare` with
`--plugin TASK/plugin.py --set NAME=VALUE`. The page shows no validation
history or backend records; the report file has them.

## Translate an interaction with an installed coding agent

Install and authenticate `claude` (Claude Code) or `codex` separately. The package
uses their configured models through subprocesses and has no LLM SDK dependency.

```bash
uv run nuagent validate --plugin src/nuagent/context/example_plugin.py
uv run nuagent validate --card card.json
uv run nuagent validate --plugin plugin.py --card card.json
uv run nuagent validate --raw-plugin raw.py
uv run nuagent propose --backend claude --experiments coherent_csi_2017 \
  --description "a gauged L_mu minus L_tau Z prime with mass 10 MeV, gauge coupling 1e-4 and kinetic mixing 1e-3" \
  --out report.md
uv run nuagent propose --backend codex --description "A new light vector boson that couples to neutrinos." \
  --out questions.md --task-dir light-vector.task
uv run nuagent propose --task-dir light-vector.task --out report.md \
  --answer free_text="a universal vector coupling to u and d quarks and all three neutrino flavors"
uv run nuagent agent-eval --backend claude --out-dir /private/tmp/nuagent-eval
```

`propose` creates a fresh `REPORT.task/` directory (or `--task-dir`) containing
`TASK.md`, the installed context pack and `conversation.json`, the harness's
record of the description, experiments, backend, `--max-turns` and rounds.
Complete in-basis descriptions produce `plugin.py`, `notes.md` and `card.json`
whenever expressible, with missing numeric values assumed from the defaults
table and listed in `notes.md`. An open form of the interaction produces
`questions.json` and `notes.md`; out-of-basis physics produces notes only. Neither has predictions
or a physics verification label. Agents never write raw cross-section plugins.

A questions outcome prints the report path, `"state": "questions"`, the round
and the questions as JSON, and exits 0. `--task-dir T --answer ID=VALUE`
(repeated, with `free_text=...` for free text) continues that task: the
harness checks that every ID was asked, records the answers, appends them to
`TASK.md` under `## Answers (round N)`, clears the outputs and runs the next
round with the backend, experiments and turn limit recorded at round 1, so
`--description`, `--backend`, `--experiments` and `--max-turns` are rejected
there. A conversation has at most three rounds; a third round that still asks
ends `unresolved`, printed like a questions outcome. Only a task whose last
round ended in questions can be continued. The report's first line names the
verification level, or the outcome (`questions`, `unresolved`, `notes-only`,
`validation-failed`). Then follow the description and the conversation
(every round's questions and answers), the predictions and likelihood table,
the notes inline as Markdown, `plugin.py`, `card.json`, the validation
history, the backend records, and the provenance, which records the number
of rounds, last.

Validation prints JSON findings and exits 1 for invalid output. Schema,
perturbativity, coefficient shape/Hermiticity/finiteness, finite non-negative
counts and the coefficient plugin's SM count limit are checked. Plugin/card
coefficients must agree at every entry to relative 1e-6 (no absolute floor) on
33-point isotope grids over the selected experiments' recoil ranges. A raw
plugin receives `raw-cross-section`, explicitly skipping the SM-limit check.
Plugin import/execution errors and backend failures propagate directly.

The harness allows one correction per round with validation findings only.
Before that correction it preserves the first attempt and clears the four
output files (`plugin.py`, `card.json`, `notes.md`, `questions.json`), so the
agent can change to a plugin-only, questions or notes-only outcome using the
same restricted tools. `questions.json` beside a plugin or card, or one that
does not match the shape in the [plugin API](docs/agent/plugin_api.md), is a
validation finding. It evaluates
likelihoods after validation succeeds, so the agent never sees likelihood
feedback. The report begins with `cross-validated` or `plugin-only` and its
limitations, or a notes-only outcome. It retains derivations, both validation
histories, source/package provenance, experiment data releases and unmodified
backend CLI records including model/cost fields when supplied. Every attempt's
outputs and backend record remain in `REPORT.task/history/round-N/attempt-M/`
on failure too. A backend failure ends the conversation: `conversation.json`
records that round as `backend-failure`, and the task cannot be continued.
After two invalid attempts, `propose` leaves a `validation-failed` report with
both histories and raises; the report has no physics verification or predictions.

Claude Code runs with only task-file reads, writes to the four named output
files and exact `nuagent validate` commands, enforced by a PreToolUse hook and
restricted mode. Codex uses its workspace-write sandbox rooted at the task
directory, with extra writable roots and command network access disabled.
`--max-turns` (default 16) is enforced by Claude Code; Codex exec exposes no
corresponding limit. CLI authentication and model selection remain the user's
configuration. Coefficient plugins execute in-process without a sandbox, as
specified by the [plugin API](docs/agent/plugin_api.md).

`agent-eval` runs from the repository root and writes per-case task directories,
reports, scores and a summary with separate physics and expressibility rates.
Omitting `--out-dir` selects a fresh system temporary directory; the command
prints the chosen location and includes `output_root` in its JSON summary.
Exhausted validation produces a failed case with both check histories and the
batch continues; unexpected backend, import and execution failures propagate.
Only descriptions, and for the answered case its free-text answer, enter tasks;
expected answers remain in `evals/agent/`.
Installed packages support `propose` and `validate` without this eval directory.

## Layout

- `src/nuagent/`: the new package, including the installed agent context resources in `context/`.
- `tests/`: tests, including regression comparisons against the legacy package.
- `docs/agent/`: generated human mirror of `src/nuagent/context/` (conventions, plugin API, framework responsibilities, worked examples). Its `conventions.md` is normative for code as well as for agents.
- `evals/agent/`: executable agent eval cases and the offline scorer, kept outside the context pack. See its [interface and metric definitions](evals/agent/README.md). Live execution remains repository-only.
- `AGENTS.md`: branching, layout, verification and tracking conventions for this repository.
- `legacy/pyCEvNS/`: the original pyCEvNS calculator, frozen. The new package never imports it; tests put `legacy/` on `sys.path`. Its usage notes are in [legacy/README.md](legacy/README.md).

## License

MIT, see [LICENSE](LICENSE).
