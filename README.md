# nuagent

A research prototype for neutrino new physics in coherent elastic neutrino-nucleus scattering (CEvNS). You describe a new interaction; `nuagent` computes the predicted event spectrum for an experiment and reports a log-likelihood or Asimov sensitivity relative to the Standard Model. A coding-agent CLI can translate a plain-language description of the interaction into the coefficient plugin that `nuagent` evaluates.

Status: milestone 5b of [ROADMAP.md](ROADMAP.md). Physics, prediction, likelihoods, scans, the context pack and coding-agent harness are implemented. The GUI remains milestone 6.

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

## Translate an interaction with an installed coding agent

Install and authenticate `claude` (Claude Code) or `codex` separately. The package
uses their configured models through subprocesses and has no LLM SDK dependency.

```bash
uv run nuagent validate --plugin src/nuagent/context/example_plugin.py
uv run nuagent validate --card card.json
uv run nuagent validate --plugin plugin.py --card card.json
uv run nuagent validate --raw-plugin raw.py
uv run nuagent propose --backend claude_code --experiments coherent_csi_2017 \
  --description "a gauged L_mu minus L_tau Z prime with mass 10 MeV, gauge coupling 1e-4 and kinetic mixing 1e-3" \
  --out report.md
uv run nuagent propose --backend codex --description "a new light vector boson that couples to neutrinos" --out questions.md
uv run nuagent agent-eval --backend claude_code --out-dir /private/tmp/nuagent-eval
```

`propose` creates a fresh `REPORT.task/` directory (or `--task-dir`) containing
`TASK.md` and the installed context pack. Complete in-basis descriptions produce
`plugin.py`, `notes.md` and `card.json` whenever expressible. Missing information
or out-of-basis physics produces notes only, with no predictions or physics
verification label. Agents never write raw cross-section plugins.

Validation prints JSON findings and exits 1 for invalid output. Schema,
perturbativity, coefficient shape/Hermiticity/finiteness, finite non-negative
counts and the coefficient plugin's SM count limit are checked. Plugin/card
coefficients must agree at every entry to relative 1e-6 (no absolute floor) on
33-point isotope grids over the selected experiments' recoil ranges. A raw
plugin receives `raw-cross-section`, explicitly skipping the SM-limit check.
Plugin import/execution errors and backend failures propagate directly.

The harness allows one correction with validation findings only. Before that
correction it preserves the first attempt and clears the three output files,
so the agent can change to a plugin-only or notes-only outcome using the same
restricted tools. It evaluates
likelihoods after validation succeeds, so the agent never sees likelihood
feedback. The report begins with `cross-validated` or `plugin-only` and its
limitations, or a notes-only outcome. It retains derivations, both validation
histories, source/package provenance, experiment data releases and unmodified
backend CLI records including model/cost fields when supplied. Every attempt's
outputs and backend record remain in `REPORT.task/history/` on failure too.

Claude Code runs with only task-file reads, writes to the three named output
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
Only descriptions enter tasks; expected answers remain in `evals/agent/`.
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
