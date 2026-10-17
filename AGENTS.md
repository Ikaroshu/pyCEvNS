# Working in this repository

pyCEvNS is being repurposed from a CEvNS event calculator into `nuagent`, an agentic neutrino new-physics research tool. These conventions apply to humans and coding agents alike.

## Layout

- `src/nuagent/` is the new package (uv-managed, Python 3.12, src layout).
- `legacy/pyCEvNS/` is the original package, frozen. Only tests import it, as a numerical oracle; pytest puts `legacy/` on `sys.path`, so the new package must never import it.
- `docs/agent/` is the agent context pack. Its `conventions.md` is normative for code as well as for agents: code that disagrees with it is wrong, or the document is; fix both together.
- `evals/agent/` holds the agent eval cases. Nothing in it is ever copied into an agent task directory.
- `.plans/` holds design specs, uncommitted and ignored by git.

## Legacy gotchas

- Do not fix the legacy package's known bugs; tests that compare against it account for them:
  - `NSIparameters.ed()` divides `gu` instead of `gd`, so the light-mediator d-quark path is wrong.
  - The light-mediator mapping is a factor 2 below the physical matching: compare with legacy `g = 2 g_ν g_f`.
- The legacy modules import `pkg_resources` and `pandas`, so tests need `setuptools<82` and `pandas` (the `test` dependency group).
- Legacy `events()` integrates recoil energy with one `quad` call that under-resolves the efficiency steps at 5 and 6 PE. Compare rates pointwise, or sum `events()` over sub-windows with edges at the steps.
- The README configuration (CsI, COHERENT flux, 4466 kg day, Helm radius 5.5 fm) gives `135.660528845678` over the single window and `135.9330188425705` converged. The old README value `138.465...` predates the flux-integration fix in d379733.

## Physics and data

- Every experiment data file begins with a header naming the source record (for example the Zenodo DOI), the release file names it was converted from, and the conversion date. The conversion script lives next to the data.
- A change to a cross-section formula comes with the matching change to the legacy-equivalence tests.

## Verification

- `uv run pytest` passes before a merge. Tests need no network.
- The golden regression in `tests/test_golden_regression.py` is the first thing to check when any rate changes.

## Git

- `main` is the only long-lived branch. Work on a branch in a worktree under `.worktrees/` and merge it locally into `main` after review; no pull requests.
- Commit messages: imperative subject under 72 characters and a body that says what changed and why. No attribution lines.

## Style

- `snake_case` modules and functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants with units in the name where ambiguous (`mass_mev`).
- Type public functions. Namespace packages; add `__init__.py` only for a genuine public surface. No underscore-prefixed modules; no cross-module imports of private names.
- Prefer the simplest design that meets the contract. No error-catching constructs (`try`/`except` and equivalents) without a stated reason. Validate only at real external boundaries: the agent's output files and downloaded data releases are such boundaries; internal calls are not.
