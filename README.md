# nuagent

A research prototype for neutrino new physics in coherent elastic neutrino-nucleus scattering (CEvNS). You describe a new interaction; `nuagent` computes the predicted event spectrum for an experiment and reports a log-likelihood or Asimov sensitivity relative to the Standard Model. A coding-agent CLI can translate a plain-language description of the interaction into the coefficient plugin that `nuagent` evaluates.

Status: milestone 1 of [ROADMAP.md](ROADMAP.md). The repository layout, packaging, `nuagent` entry point and the legacy golden regression are in place; the physics, experiments, statistics and agent arrive in later milestones.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is installed by uv if needed.

```bash
uv sync
uv run pytest
uv run nuagent --help
```

## Layout

- `src/nuagent/`: the new package.
- `tests/`: tests, including regression comparisons against the legacy package.
- `docs/agent/`: the agent context pack (conventions, plugin API, worked examples). Its `conventions.md` is normative for code as well as for agents.
- `evals/agent/`: agent eval cases, kept outside the context pack.
- `AGENTS.md`: branching, layout, verification and tracking conventions for this repository.
- `legacy/pyCEvNS/`: the original pyCEvNS calculator, frozen. The new package never imports it; tests put `legacy/` on `sys.path`. Its usage notes are in [legacy/README.md](legacy/README.md).

## License

MIT, see [LICENSE](LICENSE).
