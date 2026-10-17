# U(1)_{B−L} gauge boson: m = 25 MeV, g′ = 3e-5, no kinetic mixing

## Description

"The U(1) B−L gauge boson has mass 25 MeV and gauge coupling g′ = 3e-5, with no kinetic mixing."

The description fixes the Lorentz structure (gauge boson, vector), the mass, the gauge
coupling and the charge assignment (B−L is a standard assignment, conventions section 7).
Nothing is missing, so a plugin and a card are produced.

## Derivation

1. **Lorentz structure.** A gauge boson couples to vector currents, so structure `V`.
   The B−L current is purely vector on quarks (g_L = g_R = g′/3), so no axial piece is dropped.
   On leptons the lepton doublet carries charge −1; only the left-handed neutrino current
   exists in the SM and it is the one that scatters, so g_ν = g′ Q′_L with no chirality factor.

2. **Charges** (conventions section 7, B−L row):
   Q′_{ν_e} = Q′_{ν_μ} = Q′_{ν_τ} = −1,  Q′_u = Q′_d = +1/3.
   Hence, with g′ = 3e-5,

       g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = −g′ = −3e-5,   off-diagonal entries 0,
       g_u = g_d = +g′/3 = 1e-5.

   B−L with three right-handed neutrinos is anomaly-free; no anomaly caveat needed.

3. **Kinetic mixing.** Stated as zero, so `kinetic_mixing = 0.0` and no ε e Q_q^em shift of
   the quark couplings.

4. **Matching** (conventions section 3):

       C^{uV}_{αα}(Q²) = C^{dV}_{αα}(Q²) = g_ν g_q / (√2 G_F (Q² + m²))
                        = −g′² / (3 √2 G_F (Q² + m²)),   m = 25 MeV.

   No convention conversion is needed: the description gives the gauge coupling directly,
   not α′ or a chiral coupling.

5. **Sign and interference.** g_ν g_q = −g′²/3 < 0, so the nuclear charge shift
   ΔQ = (3Z + 3N) C = −A g′² / (√2 G_F (Q² + m²)) is negative, the same sign as
   Q_SM ≈ 0.04 Z − 0.5 N < 0. Interference with the SM is constructive; there is no
   cancellation region. The relative sign between the neutrino and quark couplings is
   physical and is kept in both plugin and card (dropping it would flip to destructive).

6. **Card.** One vector mediator entry with the signed couplings above; the schema expresses
   the interaction completely, so `card.json` is written.

## Reference values (for reading, rounded)

Cs-133 reference points Q² = 1238, 3714, 6190 MeV² (T = 5, 15, 25 keV):

| coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
|---|---|---|---|
| C^{uV}_αα = C^{dV}_αα (α = e, μ, τ) | −9.762e-03 | −4.191e-03 | −2.669e-03 |

At the first point ΔQ = 3·133·C ≈ −3.9 against Q_SM ≈ −36.9 (tree level), i.e. a modest
constructive enhancement of the coherent rate. All off-diagonal and all scalar entries are zero.

## Files

- `plugin.py`: `PARAMETERS = {"g_prime": 3e-5, "mass_mev": 25.0}`, `COUPLINGS = ("g_prime",)`;
  builds g_ν = −g′, g_q = g′/3 from the exported charge constants and the exported `GF`, `SQRT2`.
- `card.json`: the same mediator written with explicit signed couplings.
- Validation: `nuagent validate --plugin plugin.py --card card.json`.
