# Protophobic vector mediator: derivation

## Description as given

A vector mediator of mass m = 20 MeV, universal left-handed neutrino coupling
g_ν = 1e-4 (same for ν_e, ν_μ, ν_τ, no off-diagonal entries), and vector
nucleon couplings g_p = 0 and g_n = 3e-4. No other interactions.

## 1. Lorentz structure

Vector exchange, so structure `V`. The neutrino coupling is stated to be to the
left-handed current, which is exactly the g_ν ν̄ γ^μ P_L ν normalization of
conventions section 3, so g_ν = 1e-4 with no conversion. No scalar, axial,
pseudoscalar or tensor pieces are present.

## 2. Nucleon couplings to quark couplings

Nucleon couplings are not card fields, but they convert exactly to quark
couplings because the framework's nuclear vector charge is Q = Z g_p + N g_n
with g_p = 2 g_u + g_d and g_n = g_u + 2 g_d (conventions section 3). Inverting:

    g_u = (2 g_p − g_n) / 3 = (0 − 3e-4) / 3 = −1e-4
    g_d = (2 g_n − g_p) / 3 = (6e-4 − 0) / 3 = +2e-4

Check: 2 g_u + g_d = −2e-4 + 2e-4 = 0 = g_p; g_u + 2 g_d = −1e-4 + 4e-4 = 3e-4 = g_n.
This is the protophobic case g_u = −g_n/3, g_d = 2 g_n/3 quoted in the conventions.

## 3. Matching

Tree-level exchange (conventions section 3):

    C^{fV}_{αα}(Q²) = g_ν g_f / ( √2 G_F (Q² + m²) ),   α = e, μ, τ

so

    C^{uV}_{αα}(Q²) = (1e-4)(−1e-4) / ( √2 G_F (Q² + 400 MeV²) ) = −1e-8 / ( √2 G_F (Q² + 400) )
    C^{dV}_{αα}(Q²) = (1e-4)(+2e-4) / ( √2 G_F (Q² + 400 MeV²) ) = +2e-8 / ( √2 G_F (Q² + 400) )

Off-diagonal flavor entries and all scalar entries are zero. The plugin keeps
g_nu, g_p, g_n and mass_mev as parameters and computes g_u, g_d from g_p, g_n
with the inversion above, so that the described nucleon-level point is preserved
exactly; the card carries the equivalent quark couplings.

## 4. Numerical check at the reference points (Cs-133, Q² = 1238, 3714, 6190 MeV²)

With √2 G_F = 1.6495e-11 MeV⁻²:

| coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
|---|---|---|---|
| C^{uV}_αα | −3.7011e-01 | −1.4735e-01 | −9.1990e-02 |
| C^{dV}_αα | +7.4022e-01 | +2.9470e-01 | +1.8398e-01 |

Values rounded for reading; the plugin uses the exported constants.

## 5. Sign and interference

The nuclear charge shift is ΔQ = Z (2 C^u + C^d) + N (C^u + 2 C^d). The proton
combination 2 C^u + C^d vanishes by construction (protophobic), and the neutron
combination C^u + 2 C^d equals g_ν g_n / (√2 G_F (Q² + m²)), so
ΔQ = N g_ν g_n / (√2 G_F (Q² + m²)) with g_ν g_n = +3e-8 > 0. For Cs-133 (N = 78) at Q² = 1238 MeV² this gives
ΔQ ≈ +86.6, positive, while Q_SM ≈ −36.9 is negative: destructive interference.
At this point |ΔQ| > 2 |Q_SM|, so the total charge has flipped sign and the
coherent rate exceeds the SM rate at low recoil (ratio ≈ (49.7/36.9)² ≈ 1.8 at
tree level at the first reference point, falling with Q² as the propagator
drops).

## 6. Card

The interaction is a single vector mediator with constant couplings, so the
schema expresses it exactly: one `V` mediator entry with mass 20 MeV, diagonal
neutrino couplings 1e-4, quark couplings u = −1e-4, d = +2e-4, and no kinetic
mixing. No contact terms (a mass is given, so the mediator form is used).

## 7. Conventions and assumptions

- No convention conversion was needed: the description already uses the
  left-handed neutrino current and vector (not chiral) nucleon couplings.
- The nucleon couplings are taken as pure vector couplings at Q² → 0 of the
  nucleon, which is what the framework's Q = Z g_p + N g_n charge assumes; no
  axial coupling was given or dropped.
- Perturbativity: all couplings are O(1e-4), far below √(4π).
- Anomaly cancellation for a protophobic U(1) is the model author's concern.
