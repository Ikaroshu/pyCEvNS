# Protophobic vector mediator, m = 20 MeV

## Description

A protophobic vector mediator of mass 20 MeV with a universal left-handed neutrino
coupling g_ν = 1e-4 and vector nucleon couplings g_p = 0, g_n = 3e-4. No other
interactions.

## Derivation

1. **Lorentz structure.** Vector exchange, structure `V`. The neutrino coupling is
   stated as left-handed, which is exactly the current in the framework's basis, so
   g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = 1e-4 and all off-diagonal entries are zero. No
   chirality or normalization conversion is needed. No scalar entries.

2. **Nucleon to quark couplings** (conventions section 3). The framework's nuclear
   vector charge is Q = Z g_p + N g_n with g_p = 2 g_u + g_d and g_n = g_u + 2 g_d.
   Inverting:

       g_u = (2 g_p − g_n) / 3 = (0 − 3e-4) / 3 = −1e-4
       g_d = (2 g_n − g_p) / 3 = (6e-4 − 0) / 3 = +2e-4

   Check: 2 g_u + g_d = −2e-4 + 2e-4 = 0 = g_p and g_u + 2 g_d = −1e-4 + 4e-4 = 3e-4 = g_n.
   The quark couplings are vector couplings; the description gives vector nucleon
   couplings, so there is no axial part to drop.

3. **Matching** (conventions section 3):

       C^{uV}_{αα}(Q²) = g_ν g_u / (√2 G_F (Q² + m²)) = −1e-8 / (√2 G_F (Q² + 400 MeV²))
       C^{dV}_{αα}(Q²) = g_ν g_d / (√2 G_F (Q² + m²)) = +2e-8 / (√2 G_F (Q² + 400 MeV²))

   for α = e, μ, τ. The plugin computes g_u and g_d from the parameters g_p and g_n
   with the same formulas and uses the exported GF and SQRT2.

4. **Sign and interference.** The nuclear charge shift is
   ΔQ = Z (2 C^u + C^d) + N (C^u + 2 C^d) = N g_ν g_n / (√2 G_F (Q² + m²)), with no
   proton contribution, as required for a protophobic mediator. Since g_ν g_n > 0, ΔQ is
   positive while Q_SM ≈ 0.04 Z − 0.5 N is negative: destructive interference with the
   Standard Model. At the first reference point (Q² = 1238 MeV², Cs-133 with N = 78),
   C^{uV} ≈ −0.370, C^{dV} ≈ +0.740, and ΔQ ≈ +86.6, which exceeds |Q_SM| ≈ 36.9, so at
   this point the total charge has flipped sign and the coherent rate is enhanced
   rather than cancelled.

5. **Perturbativity.** All couplings are of order 1e-4, far below √(4π).

6. **Card.** The interaction is a single vector mediator with constant couplings, so
   the schema expresses it exactly. `card.json` carries the converted quark couplings
   g_u = −1e-4, g_d = +2e-4, the universal neutrino coupling, mass 20 MeV, and zero
   kinetic mixing.

## Parameters

| parameter | value | role |
|---|---|---|
| g_nu | 1e-4 | universal left-handed neutrino coupling (coupling) |
| g_p | 0 | vector proton coupling (coupling) |
| g_n | 3e-4 | vector neutron coupling (coupling) |
| mass_mev | 20 | mediator mass (not a coupling) |

The described point is preserved exactly; nothing was tuned.
