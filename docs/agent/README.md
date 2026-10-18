# Agent context pack

Read `conventions.md` first, then `plugin_api.md` and `framework.md`. This pack supplies the conventions and worked translations for an agent producing new-physics coefficients. It is available in installed distributions through `importlib.resources.files("nuagent.context")`.

- `conventions.md`: normative matching, units, flavor and sign conventions.
- `plugin_api.md`: the coefficient plugin contract and user-authored raw cross-section route.
- `example_plugin.py`: a standalone, runnable universal-vector plugin. Load it with `nuagent.physics.plugin.load_coefficient_plugin`, or pass it to `nuagent predict --plugin`.
- `card_schema.json`: generated from `nuagent.physics.modelcard.ModelCard`, including field descriptions. Absent coupling entries mean zero.
- `framework.md`: responsibilities and the meaning and limits of the three verification levels.
- `examples/`: worked translations with plugins, derivations, reference numbers and cards where expressible:
  - `01_zprime_universal.md`: universal vector mediator.
  - `02_zprime_b_minus_l.md`: B−L from charges.
  - `03_zprime_lmu_ltau.md`: L_μ−L_τ with given and loop-induced mixing.
  - `04_light_scalar.md`: scalar mediator.
  - `05_contact_nsi_offdiagonal.md`: flavor-changing contact NSI.
  - `06_running_kinetic_mixing.md`: momentum-dependent mixing, which requires a plugin.
  - `07_under_specified_scalar.md`: a description whose form is open, its `questions.json`, and the translation after the answer with assumed values.

For a complete description within the V and S basis, write `plugin.py`, `notes.md` with the derivation and convention conversions, and `card.json` exactly when a card can express the interaction. Explain the absence of a card in `notes.md`. If the form of the interaction is open, write `questions.json` asking about the form and `notes.md`, and no plugin or card; if only numeric values are missing, assume the defaults of `conventions.md` section 12 in `PARAMETERS`. Write `notes.md` in Markdown with LaTeX math in `$...$` and `$$...$$`. For physics outside the basis, write only `notes.md` explaining the limitation and naming the user-authored raw cross-section route.

The context pack and offline evaluation scorer are implemented. Backend execution, `nuagent validate`, the translation harness, verification reports and live evaluation are tracked in issue #17. The verification labels described here are the harness contract; copying this pack or using a plugin does not itself assign a verification level.

`src/nuagent/context/` is the canonical source. `docs/agent/` is its generated human-readable mirror; edit the source and refresh the mirror with `cp -R src/nuagent/context/. docs/agent/`. Tests compare both representations and check the generated schema and worked examples. Evaluation cases and answers live separately under `evals/agent/` and must never enter the copied context pack.
