# Universal vector mediator with dipole-form-factor quark couplings

Description: a vector mediator of mass m = 30 MeV couples with g′ = 1e-4 to all three left-handed neutrino flavors, and to u and d quarks with g′ times a dipole form factor F(Q²) = 1/(1 + Q²/Λ²)², Λ = 200 MeV. All other couplings vanish.

## Derivation

1. Lorentz structure: vector exchange, structure `V`. No scalar, axial or tensor piece.
2. Neutrino couplings as written: g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = g′ = 1e-4, left-handed current, off-diagonal entries zero. No convention conversion needed.
3. Quark couplings as written: g_u(Q²) = g_d(Q²) = g′ F(Q²) with F(Q²) = 1/(1 + Q²/Λ²)². F is a pure vector coupling (no chiral split), so no axial part is dropped. F(0) = 1 and F decreases monotonically with Q²; Q² is spacelike and non-negative, so F is never singular.
4. Matching (conventions section 3), with the Q²-dependent quark coupling inserted:

       C^{uV}_{αα}(Q²) = C^{dV}_{αα}(Q²) = g′² F(Q²) / ( √2 G_F (Q² + m²) ),   α = e, μ, τ.

   All off-diagonal and all scalar entries are zero.
5. Sign: g_ν g_q = g′² F > 0, so the new-physics nuclear charge shift ΔQ = (3Z + 3N) C is positive while Q_SM is negative. The interference is destructive, as in example 01. The form factor only reduces the magnitude of ΔQ with Q² and does not change the sign.
6. Reduction check: at Q² → 0 the plugin equals the universal Z′ of example 01 (F = 1). At the reference points the coefficients are those of example 01 multiplied by F(Q²):

   | Q² (MeV²) | F(Q²) | C^{uV} = C^{dV} (all diagonal flavors), approximate |
   |---|---|---|
   | 1238 | 0.9409 | 2.668e-01 |
   | 3714 | 0.8373 | 1.100e-01 |
   | 6190 | 0.7499 | 6.412e-02 |

   These are hand-rounded; the plugin computes them from the exported constants at full precision.
7. Perturbativity: |g′| = 1e-4 ≪ √(4π). The mass is positive. Λ is a scale parameter, not a coupling, so it is in `PARAMETERS` but not in `COUPLINGS`; with g′ = 0 every coefficient vanishes and the Standard Model limit holds.

## Why no card.json

The card schema describes a mediator by a mass and constant quark couplings (optionally plus a constant kinetic mixing). The description's quark couplings carry the momentum dependence F(Q²) = 1/(1 + Q²/Λ²)², which varies by about 25 percent over the COHERENT recoil range. No constant g_u, g_d can reproduce this, and the contact entries are also constants. This is therefore a plugin-only translation, analogous to example 06; the expected report label is `plugin-only`.

## Parameters

- `g_prime` = 1e-4 (coupling; in `COUPLINGS`)
- `mass_mev` = 30.0
- `lambda_mev` = 200.0 (dipole scale Λ)
