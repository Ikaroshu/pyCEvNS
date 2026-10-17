# Universal vector Z′, m = 45 MeV, g′ = 7e-5

## Description

A universal vector Z′ of mass 45 MeV couples with g′ = 7e-5 to all three
left-handed neutrino flavors and to up and down quarks. No other couplings.

## Derivation

1. **Lorentz structure.** A vector boson exchanged between the left-handed
   neutrino current and the quark vector current: structure `V`. No scalar,
   axial or tensor piece is present, so the interaction lies inside the V/S
   basis.

2. **Couplings as written.** The Lagrangian is

       L ⊃ Z′_μ ( g′ Σ_α ν̄_α γ^μ P_L ν_α + g′ ū γ^μ u + g′ d̄ γ^μ d ),

   so g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = g′, all off-diagonal g_ν^{αβ} = 0,
   g_u = g_d = g′. The neutrino coupling is already to the left-handed current,
   so no factor of 2 conversion is needed (conventions section 3). The quark
   coupling is purely vector (g_L = g_R = g′), so g_q = (g_L + g_R)/2 = g′ and
   no axial part is dropped. No kinetic mixing. This is the "universal (toy)"
   charge assignment of conventions section 7 with Q′ = +1 for every field.

3. **Matching** (conventions section 3, tree-level Z′ exchange):

       C^{uV}_{αα}(Q²) = C^{dV}_{αα}(Q²) = g′² / ( √2 G_F (Q² + m²) ),
       m = 45 MeV, g′ = 7e-5.

   All off-diagonal and all scalar coefficients are zero. The plugin computes
   this directly from the exported `GF` and `SQRT2`; the card holds the same
   mediator entry so the framework can cross-check them.

4. **Numerical values** at the Cs-133 reference points (Q² = 2 M T for
   T = 5, 15, 25 keV), using g′² = 4.9e-9, m² = 2025 MeV²,
   √2 G_F = 1.6495e-11 MeV⁻²:

   | Q² (MeV²) | C^{uV} = C^{dV} |
   |---|---|
   | 1238 | 9.104e-02 |
   | 3714 | 5.176e-02 |
   | 6190 | 3.616e-02 |

5. **Sign and interference.** g_ν g_q = +g′² > 0, so C > 0 for both quarks and
   the new-physics nuclear charge shift ΔQ = (3Z + 3N) C = 3A·C is positive,
   while the SM charge Q_SM ≈ 0.04 Z − 0.5 N is negative. The interference is
   destructive. For Cs-133 at Q² = 1238 MeV², ΔQ ≈ 3·133·0.09104 ≈ 36.3,
   compared with the tree-level |Q_SM| = 36.9 (38.3 with the framework's
   radiative corrections). The described point therefore sits very close to
   the charge-cancellation coupling at low recoil (tree-level cancellation at
   this Q² occurs near g′ ≈ 7.06e-5 for m = 45 MeV), so the coherent rate is
   strongly suppressed at low recoil energies and recovers at higher Q² where
   the propagator suppresses C. This is a property of the described point, which
   is preserved exactly; no parameter has been adjusted.

6. **Perturbativity.** |g′| = 7e-5 ≪ √(4π). Mass positive. With `g_prime`
   zeroed the plugin returns all-zero coefficients, which is the SM limit.

7. **Card-expressible.** Yes: one vector mediator with flavor-diagonal
   universal neutrino couplings and equal u, d couplings. `card.json` is
   provided.

## Convention conversions

None. The description is already in the framework's normalization
(left-handed neutrino coupling, vector quark coupling, mass in MeV).

## Files

- `plugin.py`: coefficient plugin, `PARAMETERS = {"g_prime": 7e-5, "mass_mev": 45.0}`, `COUPLINGS = ("g_prime",)`.
- `card.json`: one `V` mediator, m = 45 MeV, g_ν = g_u = g_d = 7e-5.
- `notes.md`: this derivation.
