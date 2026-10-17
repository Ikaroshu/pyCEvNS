# Notes: "A new light vector boson that couples to neutrinos"

## Outcome: under-specified, notes only

No `plugin.py` and no `card.json` are produced. The description does not fix
enough of the interaction to write numbers, and conventions.md section 11 says
not to guess in that case.

## What the description does fix

- Lorentz structure: a vector boson Z′. This is inside the V/S basis, so once
  the missing inputs are supplied the model can be expressed as a `V` mediator
  in both a coefficient plugin and a card. No raw-cross-section route is
  needed.
- The matching formula that would apply (conventions.md section 3):

      C^{fV}_{αβ}(Q²) = g_ν^{αβ} g_f / ( √2 G_F (Q² + m²) ),   f ∈ {u, d}

  with g_ν the coupling to the left-handed neutrino current and g_f the vector
  (chirality-averaged) quark coupling.

## What is missing

1. **Mediator mass m.** "Light" is not a number. The Q² dependence of the
   coefficient depends on m through the propagator 1/(Q² + m²), and for a
   light mediator (m comparable to or below the COHERENT momentum transfer,
   roughly tens of MeV) the shape of the recoil spectrum is set by m. A value
   in MeV is required.

2. **Neutrino coupling g_ν.** No value and no normalization are given. Please
   state the coupling and the form of the Lagrangian term so that the
   left-handed coupling can be read off, for example whether it is written as
   g ν̄ γ^μ P_L ν, (g/2) ν̄ γ^μ (1 − γ5) ν, or g ν̄ γ^μ (1 − γ5) ν, since these
   differ by factors of 2 (conventions.md section 3).

3. **Flavor structure of g_ν.** Does the Z′ couple to all three flavors
   equally, to one flavor only, or with a flavor off-diagonal matrix? The
   COHERENT CsI flux contains ν_μ, ν̄_μ and ν_e, so the flavor assignment
   changes the prediction qualitatively. A U(1)′ charge table or an explicit
   3×3 Hermitian matrix in the e, μ, τ basis would settle this.

4. **Quark couplings g_u and g_d (or g_p and g_n).** A vector boson that
   couples only to neutrinos and not to quarks gives no CEvNS coefficient at
   tree level: the coefficient is proportional to g_ν g_f. The description
   does not say whether the boson couples to quarks directly, through a U(1)′
   charge assignment (for example B − L or a universal charge), or only
   through kinetic mixing with the photon. If it is kinetic mixing, the
   mixing parameter ε, its sign convention and whether it is a tree-level or
   loop-induced quantity (as in L_μ − L_τ) are needed.

5. **Sign of the product g_ν g_f.** This decides whether the new physics
   interferes constructively or destructively with the Standard Model charge.
   It follows from items 2 and 4 once the charges are given, but it must be
   determinable, not assumed.

## What would be enough

Any one of the following would make the task complete:

- A U(1)′ name from the table in conventions.md section 7 (for example B − L),
  together with the gauge coupling g′ and the mass m in MeV.
- Explicit values of m, g_ν (per flavor or as a matrix), and g_u, g_d or
  g_p, g_n.
- Explicit values of m, g_ν and a kinetic-mixing ε with its sign convention.

With these the plugin would compute the vector coefficient above for u and d
and the specified flavor entries, and the card would list one `V` mediator
with `mass_mev`, `nu_couplings`, `quark_couplings` and, if applicable,
`kinetic_mixing`.

## Why no default values were used

Inventing a mass or coupling would silently change the physics point being
tested. The task requires preserving the described point exactly, and the
described point is not yet a point.
