# Example 07: under-specified scalar mediator, a questions outcome

Description given to the agent: "A light scalar that couples to muon neutrinos and to down quarks."

## What the description fixes and what it leaves open

1. Lorentz structure: scalar exchange, so structure `S` (conventions section 4). Fixed.
2. Flavors: only the μμ neutrino entry and only the d quark. Fixed; every other coupling is zero.
3. Mediator mass: not given. "Light" is not a number.
4. Coupling values: neither the neutrino coupling nor the down-quark coupling is given.

The coefficient C^{dS}_{μμ} = g_ν g_d / (√2 G_F (Q² + m²)) needs all three numbers, so no plugin or card can be written without guessing. This is a questions outcome: write `questions.json` and `notes.md`, and no `plugin.py` or `card.json`.

## questions.json

```json
{
  "questions": [
    {
      "id": "mass_mev",
      "question": "What is the mass of the scalar mediator?",
      "unit": "MeV",
      "suggested": 10,
      "reason": "10 MeV is a common light-mediator benchmark in CEvNS studies; the description gives no mass."
    },
    {
      "id": "g_nu",
      "question": "What is the scalar coupling g_nu in g_nu phi nu-bar_mu nu_mu to muon neutrinos?",
      "unit": null,
      "suggested": 1e-4,
      "reason": "1e-4 is a common order-of-magnitude benchmark for light-mediator couplings; the description gives no value."
    },
    {
      "id": "g_d",
      "question": "What is the scalar coupling g_d in g_d phi d-bar d to down quarks?",
      "unit": null,
      "suggested": null,
      "reason": null
    }
  ]
}
```

Each `id` is the parameter name the plugin will use for that quantity. A suggestion is offered only where a conventional benchmark exists, its `reason` says where it comes from, and it is never treated as part of the description. The third question shows a question without a suggestion.

## notes.md (what the agent should record)

Structure S with only the μμ neutrino entry and the d quark, as stated. Missing: the mediator mass and the values of the neutrino and down-quark couplings, which the scalar matching of conventions section 4 needs. No plugin or card is written until they are given; questions.json asks for them.

## After the answers

The harness appends the answers to `TASK.md` under `## Answers (round 2)`; they are then part of the description. With mass_mev = 10, g_nu = 1e-4 and g_d = 1e-4, the translation is a plugin with `PARAMETERS = {"mass_mev": 10.0, "g_nu": 1e-4, "g_d": 1e-4}`, `COUPLINGS = ("g_nu", "g_d")` and the single entry C^{dS}_{μμ} = g_nu g_d / (√2 G_F (Q² + m²)), plus a card with one `S` mediator (ν coupling `mm` only, quark coupling `d` only), as in example 04. Write `questions.json` again only if the answers still leave the interaction undetermined.
