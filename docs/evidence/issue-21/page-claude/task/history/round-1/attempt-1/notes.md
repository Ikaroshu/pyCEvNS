# Notes: light vector boson coupled to neutrinos

## Outcome

Questions outcome. The description is within the vector/scalar basis but is
under-specified, so no `plugin.py` and no `card.json` are written. The missing
inputs are listed in `questions.json`.

## What the description fixes

- Lorentz structure: a vector boson, so the structure is `V` and the matching
  formula is the one of conventions section 3,

      C^{fV}_{αβ}(Q²) = g_ν^{αβ} g_f / ( √2 G_F (Q² + m²) ),

  with the left-handed neutrino current and the vector (chirality-averaged)
  quark couplings g_u and g_d.
- "Light": the mediator is light enough that the Q² dependence of the
  propagator matters, so the mediator form (not a contact term) is the right
  representation once the numbers are known.

## What is missing

1. Mass m of the boson. "Light" is not a number. A 10 MeV benchmark is
   suggested in `questions.json` as a conventional choice; it is not part of
   the description.
2. Neutrino coupling g_ν. No value is given. The question states the
   normalization (g_ν ν̄ γ^μ P_L ν) so that an answer written as
   (g′/2) ν̄ γ^μ (1 − γ5) ν maps to g_ν = g′, and one written without the 1/2
   maps to g_ν = 2g′ (conventions section 3). The 1e-4 suggestion is the
   conventional benchmark of conventions section 11, not a described value.
3. Flavor structure of g_ν. "Couples to neutrinos" does not say whether the
   coupling is universal across e, μ, τ, restricted to one flavor (as in
   L_μ − L_τ or B − 3L_e models), or flavor off-diagonal. The COHERENT CsI
   flux contains ν_μ, ν̄_μ and ν_e, so the flavor pattern changes the
   prediction directly.
4. Quark couplings g_u and g_d. The description mentions only neutrinos. A
   boson with g_u = g_d = 0 and no kinetic mixing gives C^{fV} = 0 and no
   tree-level contribution to coherent elastic neutrino-nucleus scattering at
   all. The couplings are asked for per quark, as conventions section 11
   requires, because the description does not fix their relation. If the
   model specifies nucleon couplings instead, they convert through
   g_u = (2 g_p − g_n)/3 and g_d = (2 g_n − g_p)/3.
5. Kinetic mixing. If the boson reaches quarks only through mixing with the
   photon, the mixing parameter ε and its sign relative to g_ν are needed,
   since g_q → g_q + ε e Q_q^em (conventions section 6). This question is
   an alternative to item 4; a zero answer is acceptable when g_u and g_d are
   given directly.

## What was not asked

- The Lorentz structure, which the description fixes as vector.
- The sign of interference with the Standard Model: it follows from the sign
  of g_ν g_q once the couplings are given and is not an independent input.

## Next step

When the answers arrive under `## Answers (round N)` in `TASK.md`, write the
plugin with `PARAMETERS = {"g_nu", "g_u", "g_d", "mass_mev", ...}` (plus a
kinetic-mixing parameter if used), `COUPLINGS` listing the coupling names, and
a matching single-mediator `card.json`, then run
`nuagent validate --plugin plugin.py --card card.json`.
