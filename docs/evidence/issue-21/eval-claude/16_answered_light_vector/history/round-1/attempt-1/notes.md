# Notes: light vector boson coupled to neutrinos (questions outcome)

Description: "A new light vector boson that couples to neutrinos." TASK.md contains no
`## Answers` section, so the description above is all that is available.

## What the description fixes

1. Lorentz structure: a vector boson, so structure `V`. The matching of conventions
   section 3 applies,

       C^{qV}_{αβ}(Q²) = g_ν^{αβ} g_q / (√2 G_F (Q² + m²)),   q = u, d,

   with g_ν the coupling to the left-handed neutrino current. This is inside the V/S
   basis, so no basis limitation arises and no raw cross-section route is needed.

## What is missing

2. Mass m: "light" is not a number. No propagator can be evaluated without it.
3. Neutrino couplings: neither the value nor the flavor structure is given. The
   description does not say whether the boson couples to ν_e, ν_μ, ν_τ or all three, nor
   with what strength. For coherent_csi_2017 the flux contains ν_μ, ν̄_μ and ν_e, so the
   flavor assignment changes the prediction materially. Off-diagonal (flavor-changing)
   couplings are assumed absent unless the answer states otherwise.
4. Quark couplings g_u and g_d: the description says the boson couples to neutrinos and
   says nothing about quarks. A CEvNS coefficient is the product g_ν g_q, so without a
   quark coupling the coefficient is undetermined (and if the boson truly couples only to
   neutrinos at tree level, every C^{qV} is zero and there is no CEvNS signal to translate).
   Following conventions section 11, the quark couplings are requested per quark.

Because C^{qV}_{αα} needs m, g_ν^{αα} and g_q all at once, no `plugin.py` or `card.json`
can be written without inventing numbers. `questions.json` asks for each missing input.

## Suggestions in questions.json

- `mass_mev`: 10 MeV is offered only as a conventional light-mediator benchmark.
- `g_nu_ee`, `g_nu_mm`, `g_nu_tt`: 1e-4 is offered only as a conventional
  order-of-magnitude benchmark; 0 is the answer for a flavor the boson does not couple to.
- `g_u`, `g_d`: no suggestion, since the description gives no hint of the quark sector.

None of these suggestions is part of the description.

## Conversions available once answers arrive

- If the quark coupling is given through a U(1)′ charge assignment, g_q = g′ Q′_q and
  g_ν = g′ Q′_ν (conventions section 7), keeping the relative sign.
- If it is given at nucleon level, g_u = (2 g_p − g_n)/3 and g_d = (2 g_n − g_p)/3.
- If it arises from kinetic mixing ε with the photon, g_u = +(2/3) ε e and
  g_d = −(1/3) ε e with e = 0.303, and the sign of ε relative to g_ν must be stated.
- If the neutrino coupling is written as g ν̄ γ^μ (1 − γ5) ν without a 1/2, g_ν = 2g;
  a Dirac coupling to both chiralities contributes only its left-handed part.

With the answers, the translation will be one `V` mediator: a plugin with
`PARAMETERS = {"mass_mev", "g_nu_ee", "g_nu_mm", "g_nu_tt", "g_u", "g_d"}`,
`COUPLINGS = ("g_nu_ee", "g_nu_mm", "g_nu_tt", "g_u", "g_d")`, and
C^{qV}_{αα} = g_nu_αα g_q / (√2 G_F (Q² + m²)), plus a card with a single mediator entry
carrying the same signed couplings, as in examples 01 and 02.
