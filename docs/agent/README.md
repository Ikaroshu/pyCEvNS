# Agent context pack (draft)

Files the agent harness copies into every task directory. They are also the human reference for the package's physics conventions. Status: drafted 2026-10-17 in the design session; milestone M2 implemented the `Coefficients` API, the model card and the cross section they describe, and `tests/test_modelcard.py` checks that every example card and plugin agree and that the examples' coefficient tables and derived charges, couplings and ratios, and the numerical values in `conventions.md`, match the code (quadrature-accuracy remarks in example 06 are not tested).

- `conventions.md`: normative conventions. Read first.
- `plugin_api.md`: the plugin contract.
- `card_schema.json`: JSON schema of a model card, generated from the pydantic model `nuagent.physics.modelcard.ModelCard` (a test keeps it current). A field that is absent means zero.
- Section 9 of `conventions.md` plays the role of the spec's `framework.md`.
- `examples/`: worked description-to-plugin translations with derivations, expected coefficient values at reference momentum transfers, and the card where one exists.
  - `01_zprime_universal.md`: flavor-universal vector mediator (card-expressible)
  - `02_zprime_b_minus_l.md`: B−L gauge boson from charges (card-expressible)
  - `03_zprime_lmu_ltau.md`: L_μ−L_τ with kinetic mixing, given and loop-induced (card-expressible)
  - `04_light_scalar.md`: scalar mediator (card-expressible)
  - `05_contact_nsi_offdiagonal.md`: contact ε with a flavor-changing entry (card-expressible)
  - `06_running_kinetic_mixing.md`: Q²-dependent loop mixing (plugin-only; reduces to example 03 B at Q² → 0)
- Eval cases live outside this pack, under `evals/agent/`, so they are never copied into a task directory.

Output contract for a task (repeated in every `TASK.md`): write `plugin.py` always; write `card.json` whenever the interaction is expressible as a card and say in `notes.md` why not otherwise; write `notes.md` with the derivation from the description to the numbers, the convention conversions you made, and any assumption. Run `nuagent validate --plugin plugin.py [--card card.json]` and fix what it reports. If the description is under-specified, write only `notes.md` naming what is missing. If the physics is outside the V and S basis (magnetic moments, inelastic channels, oscillation effects), write only `notes.md` saying so and naming the user-authored raw cross-section route.
