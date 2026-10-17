# Working in this repository

pyCEvNS is being repurposed from a CEvNS event calculator into an agentic neutrino new-physics research tool. The design spec is mirrored in GitHub issue #5; the roadmap is the GitHub project "pyCEvNS: Agentic CEvNS prototype" (milestone issues #12 to #18, deferred items #6 to #11). These conventions apply to humans and coding agents alike.

## Branching and integration

- `main` is the integration branch. Documents with no code may land on `main` directly; everything else goes through a pull request.
- One branch per milestone issue, named after it (`m1-repurposing`, `m2-physics-core`, ...), checked out as a worktree under `.worktrees/`. One pull request per milestone, referencing its issue, merged after code review.
- Commit messages: imperative subject under 72 characters, a body that says what changed and why, and `Closes #N` when the commit resolves an issue. No attribution lines.

## Layout and ownership

- `legacy/pyCEvNS/` is the original package, frozen. Only tests import it, as a numerical oracle for regression comparisons. Do not fix its known issues (the `ed()` quark bug, the factor-2 light-mediator mapping); they are documented in the tests that compare against it.
- `src/nuagent/` is the new package (uv-managed, Python 3.12, src layout).
- `docs/agent/` is the single source of the agent context pack until milestone M2 ships it as package data. Its `conventions.md` is normative for code as well as for agents: a physics convention changes there and in the code in the same pull request.
- `evals/agent/` holds the agent eval cases. Nothing in it is ever copied into an agent task directory.
- `.plans/` holds uncommitted design specs and is ignored by git. The committed record of a spec is its mirror issue.

## Physics and data

- Coefficient conventions, matching formulas and sign rules are in `docs/agent/conventions.md`. Code that disagrees with it is wrong, or the document is; fix both together.
- Every experiment data file begins with a header naming the source record (for example the Zenodo DOI), the release file names it was converted from, and the conversion date. The conversion script lives next to the data.
- A change to a cross-section formula comes with the matching change to the legacy-equivalence tests and a sentence in the pull request saying what moved and why.

## Verification

- `uv run pytest` passes before a pull request is opened. Tests need no network.
- The golden regression (legacy CsI configuration, `135.660528845678` single-window, 1e-4 against the converged reference) is the first thing to check when any rate changes.
- Manual checks named in the spec (live agent runs, UI screenshots, the arXiv:1803.01224 magnitude comparison) are attached to the milestone issue as evidence.

## Tracking

- Tick the checklist item in the milestone issue when it is done; move the project card to In Progress when work starts and to Done when the pull request merges.
- Close an issue only from the pull request that resolves it, with the evidence linked.
- Decisions that change the spec are made in the design session and reflected in issue #5 before implementation follows them.

## Style

- `uv` for environments and running; `snake_case` modules and functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants with units in the name where ambiguous (`mass_mev`).
- Type public functions. Namespace packages; add `__init__.py` only for a genuine public surface. No underscore-prefixed modules; no cross-module imports of private names.
- Prefer the simplest design that meets the contract. No error-catching constructs (`try`/`except` and equivalents) without a stated reason in the pull request. Validate at real external boundaries only: the agent's output files and downloaded data releases are such boundaries; internal calls are not.
