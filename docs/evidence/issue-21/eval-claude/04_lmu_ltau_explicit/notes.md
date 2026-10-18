# L_μ−L_τ Z′ (m = 10 MeV, g′ = 1e-4) with explicit kinetic mixing ε = +1e-3

## Description as read

A gauged U(1)_{L_μ−L_τ} vector boson Z′ with mass 10 MeV, gauge coupling g′ = 1e-4, and photon kinetic mixing ε = +1e-3 given directly in the conventions of this pack (conventions section 6). No direct quark couplings. This is the pattern of worked example 03, description A (explicit mixing), at a different parameter point.

## Derivation

1. **Lorentz structure.** A gauge boson couples through vector currents: structure `V`. The neutrino current is the left-handed doublet current, so g_ν is used as written (conventions section 3).

2. **Neutrino couplings (conventions section 7).** L_μ−L_τ charges are Q′_{ν_e} = 0, Q′_{ν_μ} = +1, Q′_{ν_τ} = −1, flavor-diagonal in the mass basis. Hence

       g_ν^{ee} = 0,   g_ν^{mm} = +g′ = +1e-4,   g_ν^{tt} = −g′ = −1e-4,

   and all off-diagonal entries vanish.

3. **Quark couplings.** The description states there are no direct quark couplings (Q′_u = Q′_d = 0), so the `quark_couplings` entries are zero. The only quark coupling is the electromagnetic one induced by kinetic mixing (conventions section 6):

       g_q = ε e Q_q^em,   g_u = ε e (2/3),   g_d = ε e (−1/3),   e = E_CHARGE = 0.303.

   Numerically g_u = +2.02e-4 and g_d = −1.01e-4 (using e = 0.303).

4. **Mixing value and sign.** The description gives ε = +1e-3 explicitly "in the conventions of this pack", i.e. it is already the low-energy coupling ε in the sense g_q → g_q + ε e Q_q^em, with the sign relative to g′ fixed as positive. No Lagrangian-convention conversion is needed. Because the description fixes the total mixing, the loop-induced piece κ g′ ≈ 1.44e-6 (section 6) is **not** added on top: the stated ε is taken as the complete mixing at the parameter point, exactly as in example 03 A. (If instead the user intended ε to be an additional UV contribution, the loop term would shift ε by about 0.14 percent; this reading was not adopted because the description gives a single explicit value.) The mixing is treated as a Q²-independent constant, which is what the card field represents.

5. **Matching (conventions section 3).**

       C^{qV}_{αβ}(Q²) = g_ν^{αβ} g_q / (√2 G_F (Q² + m²)),

   so

       C^{uV}_{mm} = +g′ ε e (2/3) / (√2 G_F (Q² + m²)),
       C^{dV}_{mm} = −g′ ε e (1/3) / (√2 G_F (Q² + m²)),
       C^{uV}_{tt} = −C^{uV}_{mm},   C^{dV}_{tt} = −C^{dV}_{mm},
       C^{qV}_{ee} = 0,   all off-diagonal and all scalar entries = 0.

   At the Cs-133 reference points (Q² = 1238, 3714, 6190 MeV²) with m² = 100 MeV²: C^{uV}_{mm} ≈ 9.15e-1, 3.21e-1, 1.95e-1 and C^{dV}_{mm} is minus half of that. (Cross-check against example 03 A, which has the same product g′ ε = 1e-7: its first-point value 7.476e-1 at Q² + m² = 1638 MeV² rescales by 1638/1338 to 9.15e-1 here.)

6. **Physics check.** The mixing-induced coupling is proportional to electric charge, so only protons feel it: 2 C^u + C^d ∝ 2(2/3) − 1/3 = 1 and C^u + 2 C^d ∝ 2/3 − 2/3 = 0. The ν_μ nuclear charge shift ΔQ^{mm} = Z g′ ε e / (√2 G_F (Q² + m²)) is positive for ε g′ > 0, while Q_SM is negative: destructive interference for ν_μ and ν̄_μ. Electron neutrinos see no new physics. For COHERENT the prompt ν_μ and delayed ν̄_μ are affected; the delayed ν_e are not. No axial coupling is dropped, since the induced coupling is purely vector.

7. **Perturbativity.** |g′| = 1e-4 and |ε| = 1e-3 are far below √(4π).

8. **SM limit.** `COUPLINGS = ("g_prime", "kinetic_mixing")`; zeroing either removes the entire new-physics contribution, so the SM-limit check is satisfied.

## Card

Expressible: one vector mediator with the neutrino flavor matrix above, zero direct quark couplings and `kinetic_mixing: 0.001`. The card carries the mixing explicitly (a card has no κ g′ default). The plugin computes the same coefficients independently from the exported constants `GF`, `SQRT2`, `E_CHARGE` and `Q_EM`.

## Files

- `plugin.py`: coefficient plugin, parameters `g_prime = 1e-4`, `mass_mev = 10.0`, `kinetic_mixing = 1e-3`.
- `card.json`: equivalent model card.
- `notes.md`: this derivation.
