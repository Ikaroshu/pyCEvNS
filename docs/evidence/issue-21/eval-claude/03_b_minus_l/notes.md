# Notes: U(1)_{B−L} gauge boson, m = 25 MeV, g′ = 3e-5

## Description

"The U(1) B−L gauge boson has mass 25 MeV and gauge coupling g′ = 3e-5, with no kinetic mixing."

The description fixes the Lorentz structure (a gauge boson, hence vector exchange), the mass, the single gauge coupling, and the absence of kinetic mixing. The fermion couplings follow from the B−L charge assignment, so nothing is missing and no questions are needed.

## Derivation

1. **Lorentz structure.** A U(1) gauge boson couples to the vector current of each charged fermion: L ⊃ g′ Z′_μ Σ_f Q′_f f̄ γ^μ f. Structure `V`. The neutrino coupling acts on the left-handed current, g′ Q′_ν ν̄ γ^μ P_L ν, since the lepton doublet carries the charge; a right-handed neutrino, if present, does not scatter at CEvNS experiments. There is no axial part to drop (the charge is the same for both chiralities of each quark).

2. **Charges** (conventions section 7, B−L row): Q′ = −1 for every lepton doublet, Q′ = +1/3 for every quark. Hence
   - g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = −g′ = −3e-5, off-diagonal entries zero (B−L is flavor-universal and diagonal in the mass basis);
   - g_u = g_d = +g′/3 = 1e-5.

3. **Kinetic mixing.** The description says none, so ε = 0 and no electromagnetic-charge-proportional term is added to the quark couplings. The card field `kinetic_mixing` is written explicitly as 0.0.

4. **Matching** (conventions section 3):

       C^{qV}_{αα}(Q²) = g_ν g_q / (√2 G_F (Q² + m²)) = −g′² / (3 √2 G_F (Q² + m²)),   q = u, d,  α = e, μ, τ.

   With g′ = 3e-5, m = 25 MeV: the numerator g_ν g_q = −3e-10. At the first reference point Q² = 1238 MeV² (Cs-133, T = 5 keV), Q² + m² = 1863 MeV², so C ≈ −3e-10 / (1.4142 × 1.16638e-11 × 1863) ≈ −9.76e-3. At Q² = 3714 MeV², C ≈ −4.19e-3; at Q² = 6190 MeV², C ≈ −2.68e-3. These are hand estimates; the plugin computes the values from the exported constants.

5. **Sign and interference.** g_ν g_q < 0, so the new-physics nuclear charge shift ΔQ = (3Z + 3N) C = −(Z + N) g′²/(√2 G_F (Q² + m²)) is negative, the same sign as the SM charge Q_SM ≈ 0.04 Z − 0.5 N < 0. The interference is constructive; there is no cancellation region for any g′. The relative sign between the neutrino and quark couplings is physical and is kept in both the plugin (via Q_NU = −1, Q_QUARK = +1/3) and the card (signed couplings).

6. **Perturbativity.** |g′| = 3e-5 ≪ √(4π). All couplings and the positive mass are within the framework's bounds.

7. **Card expressibility.** One vector mediator with constant couplings: fully expressible. `card.json` lists the mediator with the signed neutrino couplings −3e-5 on the diagonal, quark couplings +1e-5, mass 25 MeV, kinetic mixing 0. The plugin computes the same coefficients independently from `g_prime`, `mass_mev` and the exported `GF`, `SQRT2`.

8. **Scalar sector.** None; all `S` entries are zero. No contact terms.

## Parameter point

| parameter | value |
|---|---|
| g_prime | 3e-5 |
| mass_mev | 25.0 |

`COUPLINGS = ("g_prime",)`: with g′ → 0 all coefficients vanish and the SM is recovered.

## Anomalies

U(1)_{B−L} is anomaly-free once three right-handed neutrinos are added; this is the standard assumption and does not affect the CEvNS coefficients.
