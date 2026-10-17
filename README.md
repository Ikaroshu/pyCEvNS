# nuagent

A research prototype for neutrino new physics in coherent elastic neutrino-nucleus scattering (CEvNS). You describe a new interaction; `nuagent` computes the predicted event spectrum for an experiment and reports a log-likelihood or Asimov sensitivity relative to the Standard Model. A coding-agent CLI can translate a plain-language description of the interaction into the coefficient plugin that `nuagent` evaluates.

Status: milestone 3 of [ROADMAP.md](ROADMAP.md). The physics, COHERENT CsI 2017 experiment registry, flux folding and `predict` CLI are implemented. Statistics, the agent harness and the GUI arrive in later milestones.

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

## Layout

- `src/nuagent/`: the new package, including the installed agent context resources in `context/`.
- `tests/`: tests, including regression comparisons against the legacy package.
- `docs/agent/`: generated human mirror of `src/nuagent/context/` (conventions, plugin API, framework responsibilities, worked examples). Its `conventions.md` is normative for code as well as for agents.
- `evals/agent/`: executable agent eval cases and the offline scorer, kept outside the context pack. See its [interface and metric definitions](evals/agent/README.md); backend execution and live evaluation arrive with the harness.
- `AGENTS.md`: branching, layout, verification and tracking conventions for this repository.
- `legacy/pyCEvNS/`: the original pyCEvNS calculator, frozen. The new package never imports it; tests put `legacy/` on `sys.path`. Its usage notes are in [legacy/README.md](legacy/README.md).

## License

MIT, see [LICENSE](LICENSE).
