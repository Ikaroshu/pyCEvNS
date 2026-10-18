# Gauged L_μ−L_τ Z′, m = 10 MeV, g′ = 1e-4, kinetic mixing ε = 1e-3

## Inputs taken from the description

- Mediator: a gauge boson of U(1)_{L_μ−L_τ}, so the Lorentz structure is vector (`V`).
- Mass m = 10 MeV.
- Gauge coupling g′ = 1e-4.
- Kinetic mixing with the photon ε = 1e-3, given explicitly.

All three inputs are fixed by the description, so no questions are needed.

## Derivation

1. **Charges** (conventions section 7, L_μ−L_τ row): Q′_{ν_e} = 0, Q′_{ν_μ} = +1, Q′_{ν_τ} = −1, Q′_u = Q′_d = 0.
   The neutrino coupling is the left-handed doublet coupling, g_ν^{αα} = g′ Q′_{ν_α}:
   g_ν^{ee} = 0, g_ν^{mm} = +1e-4, g_ν^{tt} = −1e-4, all off-diagonal entries zero.
2. **Quark couplings** come only from kinetic mixing (conventions section 6):
   g_q = 0 + ε e Q_q^em, with e = 0.303, Q_u^em = +2/3, Q_d^em = −1/3, so
   g_u = +2.02e-4, g_d = −1.01e-4 (using e = E_CHARGE at full precision).
3. **Matching** (conventions section 3):
   C^{qV}_{αβ}(Q²) = g_ν^{αβ} g_q / (√2 G_F (Q² + m²)).
   Hence
   C^{uV}_{mm} = +g′ ε e (2/3) / (√2 G_F (Q² + m²)),
   C^{dV}_{mm} = −g′ ε e (1/3) / (√2 G_F (Q² + m²)),
   C^{qV}_{tt} = −C^{qV}_{mm}, and all ee and off-diagonal entries vanish.
   Scalar entries are zero.
4. **Sign of ε.** The description gives ε = +1e-3 with no Lagrangian convention, so it is taken as the framework's `kinetic_mixing` convention: the Z′ acquires +ε e Q f̄ γ^μ f, i.e. the same sign as the loop-induced mixing of section 6 (ε = +κ g′). With ε g′ > 0 the ν_μ nuclear-charge shift ΔQ = Z g′ ε e / (√2 G_F (Q² + m²)) is positive while Q_SM is negative, so the interference is destructive for ν_μ (and ν̄_μ) and constructive for ν_τ. If the user's convention has the opposite sign of ε, flip `kinetic_mixing` to −1e-3; the magnitude of the coefficients is unchanged.
5. **Loop-induced mixing is not added.** The explicit ε = 1e-3 is treated as the total low-energy photon mixing, as in worked example 03 A. The μ/τ-loop contribution would be κ g′ ≈ 1.44e-6, about 0.14 percent of the given value, and adding it on top of an explicitly specified ε would double-count if the stated ε already represents the full mixing. The Q² dependence of the loop piece (example 06) is likewise negligible at this level.
6. **Physics check.** The mixing couples the Z′ to electric charge, so only protons are affected: 2 g_u + g_d = ε e and g_u + 2 g_d = 0. Electron neutrinos see no new physics; the COHERENT prompt ν_μ and delayed ν̄_μ fluxes do.
7. **Perturbativity.** |g′| = 1e-4 and |ε| = 1e-3 are far below √(4π).
8. **Anomalies.** L_μ−L_τ is anomaly-free with the SM fermion content.

## Numerical values at the Cs-133 reference points (Q² = 1238, 3714, 6190 MeV²)

Prefactor g′ ε e (2/3) / (√2 G_F) = 1e-4 × 1e-3 × 0.303 × (2/3) / (1.64951e-11 MeV⁻²) ≈ 1.2246e+03 MeV².
With m² = 100 MeV²:

| coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
|---|---|---|---|
| C^{uV}_mm = −C^{uV}_tt | ≈ 9.15e-01 | ≈ 3.21e-01 | ≈ 1.95e-01 |
| C^{dV}_mm = −C^{dV}_tt | ≈ −4.58e-01 | ≈ −1.60e-01 | ≈ −9.74e-02 |

(Rounded for reading; the plugin uses the exported constants at full precision.)

## Card

Expressible: one vector mediator with `nu_couplings` mm = +1e-4, tt = −1e-4, zero direct `quark_couplings`, and `kinetic_mixing` = 1e-3 written explicitly (a card has no loop default). `card.json` is provided.

## Plugin

`plugin.py` computes the same coefficients from `GF`, `SQRT2`, `E_CHARGE` and `Q_EM` exported by `nuagent.physics.couplings`. `COUPLINGS = ("g_prime", "kinetic_mixing")`; zeroing either removes all new physics, so the SM-limit check passes.
