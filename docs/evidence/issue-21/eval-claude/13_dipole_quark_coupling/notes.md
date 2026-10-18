# Universal vector mediator with dipole quark form factor

## Description as read

- Vector mediator Z′, mass m = 30 MeV.
- Neutrino coupling g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = g′ = 1e-4 to the left-handed current, off-diagonal entries zero.
- Quark couplings g_u(Q²) = g_d(Q²) = g′ F(Q²) with the dipole form factor
  F(Q²) = 1 / (1 + Q²/Λ²)², Λ = 200 MeV.
- All other couplings vanish: no scalar structure, no kinetic mixing, no contacts.

## Derivation

1. Lorentz structure: vector exchange, structure `V`. No scalar entries.
2. Neutrino coupling taken as written for the left-handed current (conventions section 3), so g_ν = g′ with no factor-2 conversion.
3. Quark coupling: the description gives the vector coupling to each quark directly, g_q(Q²) = g′ F(Q²). There is no chiral split, so no averaging and no dropped axial part.
4. Matching (conventions section 3) with the Q²-dependent quark coupling inserted:

       C^{uV}_{αα}(Q²) = C^{dV}_{αα}(Q²) = g′² F(Q²) / (√2 G_F (Q² + m²)),   α = e, μ, τ,

   with F(Q²) = 1/(1 + Q²/Λ²)². Off-diagonal flavor entries are zero.
5. Sign: g_ν g_q = +g′² F(Q²) > 0 for all Q² (F is positive), so the new-physics nuclear charge shift is positive while the SM charge is negative. The interference is destructive, as in example 01, but suppressed by F(Q²) at finite Q².
6. Size of the form-factor suppression at the Cs-133 reference points (Q² = 1238, 3714, 6190 MeV², Λ² = 40000 MeV²):

   | Q² (MeV²) | 1 + Q²/Λ² | F(Q²) |
   |---|---|---|
   | 1238 | 1.03095 | 0.94091 |
   | 3714 | 1.09285 | 0.83733 |
   | 6190 | 1.15475 | 0.74991 |

   The coefficients are those of example 01 multiplied by F: approximately 2.668e-01, 1.100e-01, 6.413e-02 for both quarks and all three diagonal flavors. A percent-to-tens-of-percent effect across the recoil range, which no constant card coupling can reproduce.
7. Reduction check: at Q² → 0, F → 1 and the plugin reduces exactly to the flavor-universal Z′ of example 01 (g′ = 1e-4, m = 30 MeV). At Λ → ∞ the plugin reduces to example 01 at every Q².

## Why there is no card.json

The card schema stores `quark_couplings` as constant numbers. The quark coupling here is g′ F(Q²), which varies with Q² by about 25 percent over the COHERENT recoil range, so a card with constant g_u = g_d = g′ would fail the 1e-6 cross-check against the plugin at every nonzero Q². Absorbing F(Q²) into a different constant is not possible either. This is a plugin-only translation, as in example 06.

## Plugin parameters

- `g_prime` = 1e-4 (coupling; zeroed by the SM-limit check, |g′| ≪ √(4π)).
- `mass_mev` = 30.0.
- `lambda_mev` = 200.0 (form-factor scale, not a coupling, so not listed in `COUPLINGS`).

No convention conversion was needed. No scalar, axial, tensor or kinetic-mixing contributions.
