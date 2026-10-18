# Notes: gauged L_μ − L_τ Z′, m = 10 MeV, g′ = 1e-4, kinetic mixing ε = 1e-3

## Description as read

"A gauged L_μ − L_τ Z′ with mass 10 MeV, gauge coupling 1e-4 and kinetic mixing 1e-3."

All three inputs needed for the V basis are fixed: Lorentz structure (gauge boson,
hence vector exchange), mass, and couplings (gauge coupling plus an explicit
kinetic-mixing parameter). Nothing is missing, so a plugin and a card are produced.

## Derivation

1. **Lorentz structure.** A U(1)′ gauge boson couples to vector currents, so the
   structure is `V`. The lepton-doublet coupling acts on the left-handed neutrino
   current, exactly the operator ν̄_α γ^μ P_L ν_β of conventions section 2.

2. **Charges (conventions section 7, L_μ − L_τ row).**
   Q′_{ν_e} = 0, Q′_{ν_μ} = +1, Q′_{ν_τ} = −1, Q′_u = Q′_d = 0.
   With g′ = 1e-4:

       g_ν^{ee} = 0,   g_ν^{mm} = +1e-4,   g_ν^{tt} = −1e-4,   off-diagonal = 0.

   The charges are flavor-diagonal in the mass basis, so there are no `em`, `et`,
   `mt` entries. The assignment is anomaly-free (L_μ − L_τ), nothing to flag.

3. **Quark couplings.** Quarks carry no L_μ − L_τ charge, so the only quark coupling
   is the one induced by kinetic mixing with the photon (conventions section 6):

       g_q = g_q^{direct} + ε e Q_q^em = 0 + ε e Q_q^em,
       g_u = ε e (2/3) = 1e-3 × 0.303 × 2/3 = +2.02e-4,
       g_d = ε e (−1/3) = 1e-3 × 0.303 × (−1/3) = −1.01e-4.

   The description gives ε = 1e-3 explicitly. This replaces, rather than adds to,
   the loop-induced value κ g′ = 1.44e-6 of section 6: the stated ε is the total
   low-energy photon mixing at the given point (same reading as worked example 03 A).
   The loop piece is three orders of magnitude smaller anyway. The mixing is taken
   as a Q²-independent constant, which is the meaning of the card field.

4. **Sign of ε.** The description gives a positive number with no Lagrangian
   convention, so ε = +1e-3 is taken in the framework convention of section 6,
   i.e. the Z′ acquires +ε e Q_f f̄ γ^μ f. With this sign ε g′ > 0, the same sign as
   the loop-induced mixing, so the ν_μ charge shift is positive and interferes
   destructively with the negative SM nuclear charge; ν_τ is the opposite sign.
   If the user's Lagrangian uses the D_μ = ∂_μ − i e Q A_μ convention, the sign of
   ε flips and the ν_μ interference becomes constructive; that would require
   changing `kinetic_mixing` to −1e-3 in both files.

5. **Matching (conventions section 3).**

       C^{qV}_{αβ}(Q²) = g_ν^{αβ} g_q / ( √2 G_F (Q² + m²) ),   m = 10 MeV.

   Explicitly:

       C^{uV}_{mm} = + g′ ε e (2/3) / (√2 G_F (Q² + m²)),
       C^{dV}_{mm} = − g′ ε e (1/3) / (√2 G_F (Q² + m²)),
       C^{qV}_{tt} = − C^{qV}_{mm},
       C^{qV}_{ee} = 0,  all off-diagonal and all scalar entries = 0.

   Numerical check at the first reference point Q² = 1238 MeV² (Cs-133, T = 5 keV):
   g′ g_u = 1e-4 × 2.02e-4 = 2.02e-8; √2 G_F (Q² + m²) = 1.41421 × 1.16638e-11 × 1338
   = 2.2071e-8; so C^{uV}_{mm} ≈ 0.915 and C^{dV}_{mm} ≈ −0.458. Scaling worked
   example 03 A (ε g′ = 1e-7, m = 20 MeV, C^{uV}_mm = 0.7476 at Q² = 1238) by
   (1e-7/1e-7) × (1638/1338) gives 0.915, consistent.

6. **Physics check.** Proton-only coupling: 2 C^u + C^d ∝ 2(2/3) − 1/3 = 1 while
   C^u + 2 C^d ∝ 2/3 − 2/3 = 0, as expected for photon-like mixing. The ν_μ nuclear
   charge shift is ΔQ_{mm} = Z g′ ε e / (√2 G_F (Q² + m²)) ≈ 55 × 1.37 ≈ 75 at
   Q² = 1238, about twice |Q_SM| ≈ 37 and of opposite sign, so the ν_μ rate is
   strongly modified (total charge has flipped sign) over the COHERENT recoil
   range. ν_e sees no new physics; ν_μ and ν̄_μ (prompt and delayed COHERENT
   fluxes) do. No parameter has been tuned; the described point is preserved.

7. **Perturbativity.** |g′| = 1e-4 and |ε| = 1e-3 are far below √(4π).

## Card

Card-expressible: yes. One vector mediator with m = 10 MeV, ν couplings
(ee, mm, tt) = (0, +1e-4, −1e-4), zero direct quark couplings, and
`kinetic_mixing` = 1e-3 written explicitly (a card has no default mixing).

## Plugin

`plugin.py` computes the same coefficients independently of the card from the
exported constants `GF`, `SQRT2`, `E_CHARGE` and `Q_EM`. Parameters: `g_prime`
(1e-4), `mass_mev` (10.0), `kinetic_mixing` (1e-3); couplings are `g_prime` and
`kinetic_mixing`, so zeroing them removes all new physics for the SM-limit check.
Each (quark, flavor) entry is set once after forming the full product
g_ν g_q / (√2 G_F (Q² + m²)).

## Conventions

No convention conversion was needed: the description already uses the gauge
coupling g′ directly (not α′), MeV masses, and a photon kinetic-mixing parameter
in the sense of conventions section 6.
