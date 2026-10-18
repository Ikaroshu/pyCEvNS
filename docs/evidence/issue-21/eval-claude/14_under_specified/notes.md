# Notes: light vector boson coupling to neutrinos (under-specified)

Description given: "A new light vector boson that couples to neutrinos."

## Outcome

Questions outcome. No `plugin.py` and no `card.json` are produced, because the
description does not fix enough of the interaction to write numbers. Only
`questions.json` and this file are written.

## What the description does fix

- Lorentz structure: a vector boson exchanged at tree level gives the vector
  structure `V` of conventions section 3. This is inside the V/S basis, so the
  interaction is translatable once the missing inputs are supplied.
- Chirality: SM-produced neutrinos are left-handed, so only the left-handed
  neutrino current enters; a possible right-handed coupling of a Dirac neutrino
  would be dropped (conventions section 3). The matching will be

      C^{fV}_{αβ}(Q²) = g_ν^{αβ} g_f / ( √2 G_F (Q² + m²) ),   f ∈ {u, d}.

## What is missing

1. **Mass.** "Light" is qualitative. The propagator 1/(Q² + m²) needs a number
   in MeV; the Q² dependence over the COHERENT recoil range depends strongly on
   it for a light mediator. Suggested benchmark in `questions.json`: 10 MeV,
   a common light-mediator choice. This is a suggestion, not part of the
   description.
2. **Neutrino couplings.** No value and no flavor structure are given. The
   nuclear charge depends on g_ν^{ee}, g_ν^{mm}, g_ν^{tt} separately because the
   COHERENT flux has distinct ν_μ, ν̄_μ and ν_e components, and off-diagonal
   entries enter incoherently. Asked per diagonal flavor, with 1e-4 offered as a
   conventional benchmark only, plus one question on off-diagonal entries.
3. **Quark couplings.** The description mentions no coupling to quarks at all.
   Without g_u or g_d (or an equivalent nucleon-level or kinetic-mixing route)
   the tree-level coherent elastic coefficient is identically zero, so there is
   nothing to translate. Asked separately for g_u and g_d as conventions
   section 11 requires, since no relation between them is given. If the user
   supplies nucleon couplings instead, they convert as g_u = (2 g_p − g_n)/3 and
   g_d = (2 g_n − g_p)/3. A separate question asks whether quark coupling arises
   only through kinetic mixing with the photon, which needs the signed ε.

No values were invented and no parameter was tuned. Once answers arrive in
`TASK.md` under `## Answers (round N)`, the translation is a single vector
mediator, expressible in the card schema (one `mediators` entry, structure `V`),
with a plugin that reproduces the same coefficients from the exported constants.
