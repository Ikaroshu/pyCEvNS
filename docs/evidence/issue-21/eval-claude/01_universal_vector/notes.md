# Notes: flavor-universal vector Z′ (m = 45 MeV, g′ = 7e-5)

## Description

A universal vector Z′ of mass 45 MeV couples with g′ = 7e-5 to all three
left-handed neutrino flavors and to up and down quarks. No other couplings.

## Derivation

1. **Lorentz structure.** Vector exchange, so structure `V`. No scalar entries.
   No axial coupling is mentioned, so nothing is dropped.

2. **Couplings as written.** Lagrangian in the form of conventions section 3,

       L ⊃ Z′_μ ( g′ Σ_α ν̄_α γ^μ P_L ν_α + g′ ū γ^μ u + g′ d̄ γ^μ d ),

   so g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = g′ = 7e-5, all off-diagonal g_ν = 0,
   and g_u = g_d = g′ = 7e-5. The description says "left-handed neutrino
   flavors", so g_ν is already the left-handed coupling; no factor of 2
   conversion is needed. The quark coupling is a plain vector coupling, so
   g_q = g′ directly (no chiral averaging). No kinetic mixing (ε = 0), since
   "there are no other couplings".

3. **Matching** (conventions section 3, tree-level Z′ exchange):

       C^{uV}_{αα}(Q²) = C^{dV}_{αα}(Q²) = g′² / ( √2 G_F (Q² + m²) ),
       m = 45 MeV,  m² = 2025 MeV².

   Flavor-diagonal and identical for e, μ, τ; off-diagonal entries zero.

4. **Reference values** (Cs-133 points of example 01, Q² = 1238, 3714, 6190 MeV²;
   √2 G_F = 1.6495e-11 MeV⁻², g′² = 4.9e-9):

   | Q² [MeV²] | C^{uV} = C^{dV} (each diagonal flavor) |
   |---|---|
   | 1238 | 9.104e-02 |
   | 3714 | 5.176e-02 |
   | 6190 | 3.616e-02 |

5. **Sign and interference.** g_ν g_q = +g′² > 0, so the new-physics shift of the
   nuclear vector charge, ΔQ = (2Z + N + Z + 2N) C = 3A·C, is positive, while
   the SM charge Q_SM ≈ 0.04 Z − 0.5 N is negative: destructive interference.
   For Cs-133 (A = 133) at Q² = 1238 MeV², ΔQ ≈ 399 × 0.0910 ≈ 36.3, close in
   magnitude to the tree-level |Q_SM| ≈ 36.9. This parameter point therefore
   sits near the charge-cancellation region at low recoil, with the
   cancellation weakening at higher Q² (ΔQ ≈ 20.7 and 14.4 at the other two
   points). This is a property of the described point and is reported as given;
   nothing was tuned.

6. **Perturbativity.** |g′| = 7e-5 ≪ √(4π).

7. **Card.** The schema expresses this exactly as one vector mediator with
   diagonal neutrino couplings g′ and quark couplings g_u = g_d = g′, so
   `card.json` is provided alongside `plugin.py`. The plugin computes the same
   coefficients from the exported constants `GF` and `SQRT2`, independently of
   the card.

8. **Anomaly remark.** The universal charge assignment (Q′ = +1 for all listed
   fields) is a toy model and is anomalous as a gauge symmetry; this is the
   model author's concern and does not affect the low-energy matching.
