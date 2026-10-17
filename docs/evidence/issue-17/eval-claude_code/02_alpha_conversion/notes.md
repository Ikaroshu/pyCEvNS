# Universal vector Z′, m = 45 MeV, α′ = g′²/(4π) = 3.899296105751435e-10

## Derivation

1. **Lorentz structure.** A vector Z′ exchanged between the left-handed neutrino current and the quark vector currents: structure `V` only. No scalar, axial or tensor pieces are described, so no scalar entries and nothing outside the basis.

2. **Convention conversion (conventions section 10).** The description gives the coupling as a fine-structure-type constant, α′ = g′²/(4π). Inverting,

       g′² = 4π α′ = 4π × 3.899296105751435e-10 = 4.9000e-9,
       g′  = √(4π α′) = 7.0e-5   (positive root, as stated: "positive coupling g′").

   Check: (7.0e-5)² / (4π) = 4.9e-9 / 12.566370614359… = 3.8992961057514e-10, which reproduces the given α′ to double precision. The plugin keeps α′ itself as the parameter, so the described point is preserved exactly, and forms g′² = 4π α′ internally; the card, which needs the Lagrangian coupling, carries g′ = 7.0e-5.

3. **Couplings as written.** Universal: g_ν^{ee} = g_ν^{μμ} = g_ν^{ττ} = g′, all off-diagonal g_ν^{αβ} = 0; g_u = g_d = g′. No kinetic mixing, no other couplings. This is the "universal (toy)" assignment of conventions section 7 with Q′ = +1 for all neutrinos and both quarks (anomalous as a gauge theory, but that is the model author's concern).

4. **Matching (conventions section 3).**

       C^{uV}_{αα}(Q²) = C^{dV}_{αα}(Q²) = g_ν g_q / (√2 G_F (Q² + m²)) = g′² / (√2 G_F (Q² + m²)) = 4π α′ / (√2 G_F (Q² + 45²)),

   for α = e, μ, τ; all off-diagonal and all scalar coefficients are zero. Propagator is 1/(Q² + m²) with Q² = 2 M T ≥ 0 in MeV². The mediator form is used (not a contact term) because a mass is given.

5. **Sign and interference.** g_ν g_q = +g′² > 0, so the new-physics nuclear charge shift ΔQ = (3Z + 3N) C is positive, opposite to the negative SM charge Q_SM ≈ 0.04 Z − 0.5 N. The interference is destructive. For reference, at Q² = 1238 MeV² (Cs-133, T = 5 keV): C ≈ 4.9e-9 / (1.41421 × 1.16638e-11 × (1238 + 2025)) ≈ 9.10e-2, so ΔQ ≈ 3 × 133 × 0.091 ≈ 36.3 against Q_SM ≈ −36.9: this point sits very close to the charge cancellation for Cs at low recoil, and the coherent rate there is strongly suppressed. This is a consequence of the described point, which is preserved exactly; nothing was tuned.

6. **Sanity.** g′ = 7.0e-5 ≪ √(4π) ≈ 3.54; mass positive. With `alpha_prime` zeroed the plugin returns all-zero coefficients, i.e. the SM limit.

## Card

Expressible: one vector mediator with mass 45 MeV, diagonal universal neutrino couplings g′ = 7.0e-5 and quark couplings g_u = g_d = 7.0e-5, zero kinetic mixing, no contacts. The card's g′ is the exact square root of 4π α′ at double precision, so the plugin/card cross-check agrees to far better than 1e-6.

## Files

- `plugin.py`: coefficient plugin, parameters `alpha_prime` (coupling, default 3.899296105751435e-10) and `mass_mev` (45.0).
- `card.json`: equivalent mediator card with g′ = 7.0e-5.
- `notes.md`: this derivation.
