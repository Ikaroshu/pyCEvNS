# Notes: muon antineutrinos through a universal vector Z′

## Description

Muon antineutrinos scatter through a universal vector Z′ of mass 30 MeV with g′ = 8e-5. The same coupling applies to all left-handed neutrino flavors and to both u and d quarks, with no other couplings.

## Derivation

1. **Lorentz structure.** A vector boson exchanged between the neutrino current and the quark vector current: structure `V` only. No scalar, axial or tensor pieces are described, and no kinetic mixing is mentioned ("no other couplings"), so `kinetic_mixing` = 0.

2. **Lagrangian.** Taking the description literally in the form of conventions section 3,

       L ⊃ Z′_μ ( g′ Σ_α ν̄_α γ^μ P_L ν_α + g′ ū γ^μ u + g′ d̄ γ^μ d ),   m = 30 MeV, g′ = 8e-5.

   So g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = g′, all off-diagonal g_ν zero, and g_u = g_d = g′. This is the "universal (toy)" assignment of conventions section 7 with all U(1)′ charges equal to +1. The coupling is already in the left-handed-current normalization, so no factor-2 conversion is needed.

3. **Matching** (conventions section 3, tree-level Z′ exchange):

       C^{uV}_{αα}(Q²) = C^{dV}_{αα}(Q²) = g′² / ( √2 G_F (Q² + m²) ),   α = e, μ, τ,

   with the propagator written as 1/(Q² + m²) for spacelike Q² = 2 M T. All off-diagonal and all scalar coefficients are zero.

4. **Antineutrinos.** The incoming particles are muon antineutrinos. Per conventions section 3 ("Antineutrinos: the framework handles them; you supply the same C"), the coefficient matrix is unchanged: the framework folds the antineutrino flux with the same C^{fV}. The muon-flavor entry `mm` is the one that matters for the described beam, but the description states that the coupling is universal, so `ee` and `tt` are filled with the same value. The framework's flux selection picks out the relevant flavor.

5. **Sign and interference.** g_ν g_q = +g′² > 0, so the new-physics nuclear charge shift ΔQ = 3(Z + N) C is positive while Q_SM ≈ 0.04 Z − 0.5 N is negative: destructive interference. With g′ = 8e-5 the coupling sits close to the degenerate point noted in worked example 01 (ΔQ ≈ −2 Q_SM near g′ ≈ 8.1e-5 at Q² = 1238 MeV²), where the total charge has flipped sign and the rate is close to the Standard Model rate again. At g′ = 8e-5 exactly, ΔQ at Q² = 1238 MeV² is (8e-5/1e-4)² × 113.1 ≈ 72.4 versus |Q_SM| = 36.9 (tree level), so the total charge is about +35.5, slightly below |Q_SM| in magnitude. This is a statement about the described point, which is preserved exactly; no parameter was adjusted.

6. **Numerical check** at the reference Q² of example 01 (Cs-133, T = 5, 15, 25 keV): scaling the example values by (8e-5/1e-4)² = 0.64 gives

   | coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
   |---|---|---|---|
   | C^{uV}_{αα} = C^{dV}_{αα} | 1.8147e-01 | 8.4090e-02 | 5.4724e-02 |

7. **Perturbativity.** |g′| = 8e-5 ≪ √(4π).

8. **Card-expressible.** Yes: one vector mediator entry with constant couplings, so `card.json` is provided and the plugin computes the same coefficients independently from the exported constants `GF` and `SQRT2`.

## Conventions used

No conversion was needed; the description is already in the normalization of conventions section 3. The plugin exposes `g_prime` as its single coupling (set to zero in the Standard Model-limit check) and `mass_mev` as the mediator mass.
