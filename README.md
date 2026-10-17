# nuagent

A research prototype for neutrino new physics in coherent elastic neutrino-nucleus scattering (CEvNS). You describe a new interaction; `nuagent` computes the predicted event spectrum for an experiment and reports a log-likelihood or Asimov sensitivity relative to the Standard Model. A coding-agent CLI can translate a plain-language description of the interaction into the coefficient plugin that `nuagent` evaluates.

Status: milestone 4 of [ROADMAP.md](ROADMAP.md). The physics, COHERENT CsI 2017 experiment registry, flux folding, the likelihood with `compare` and `scan`, and the `predict` CLI are implemented. The agent harness and the GUI arrive in later milestones.

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

## Layout

- `src/nuagent/`: the new package.
- `tests/`: tests, including regression comparisons against the legacy package.
- `docs/agent/`: the agent context pack (conventions, plugin API, worked examples). Its `conventions.md` is normative for code as well as for agents.
- `evals/agent/`: agent eval cases, kept outside the context pack.
- `AGENTS.md`: branching, layout, verification and tracking conventions for this repository.
- `legacy/pyCEvNS/`: the original pyCEvNS calculator, frozen. The new package never imports it; tests put `legacy/` on `sys.path`. Its usage notes are in [legacy/README.md](legacy/README.md).

## License

MIT, see [LICENSE](LICENSE).
