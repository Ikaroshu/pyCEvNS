# Notes: gauged L_μ−L_τ Z′ with negative total kinetic mixing

Description: L_μ−L_τ Z′, mass 15 MeV, gauge coupling g′ = 6e-5, total photon kinetic mixing ε = −5e-4 in the pack's convention, no direct quark couplings.

## Derivation

1. **Lorentz structure.** Gauge boson exchange: structure `V`, left-handed neutrino current only (conventions section 3).

2. **Neutrino couplings** (conventions section 7, L_μ−L_τ row): Q′_{ν_e} = 0, Q′_{ν_μ} = +1, Q′_{ν_τ} = −1, so

       g_ν^{ee} = 0,   g_ν^{mm} = +g′ = +6e-5,   g_ν^{tt} = −g′ = −6e-5,

   all off-diagonal entries zero (the charges are flavor-diagonal).

3. **Quark couplings.** The description states no direct quark couplings (Q′_q = 0), so the quarks couple only through the kinetic mixing (conventions section 6):

       g_u = ε e (+2/3),   g_d = ε e (−1/3),   e = 0.303.

   The description gives the *total* mixing ε = −5e-4 in the pack's convention. "Total" means the loop-induced piece κ g′ (≈ +8.67e-7) is already included in the quoted number, so it is **not** added again; ε = −5e-4 is used as is. This is the same treatment as example 03 A (explicit mixing), with a different sign.

4. **Matching** (conventions section 3), with the propagator 1/(Q² + m²):

       C^{uV}_{mm} = +g′ · ε e (2/3) / (√2 G_F (Q² + m²)),
       C^{dV}_{mm} = −g′ · ε e (1/3) / (√2 G_F (Q² + m²)),
       C^{qV}_{tt} = −C^{qV}_{mm},   C^{qV}_{ee} = 0.

   Numerically at Q² = 1238 MeV² (m² = 225 MeV²): √2 G_F (Q² + m²) = 2.4134e-8, g′ ε e = −9.09e-9, so C^{uV}_{mm} ≈ −0.2511 and C^{dV}_{mm} ≈ +0.1256; both fall as 1/(Q² + m²).

5. **Sign and interference.** The induced nuclear charge is proton-only: 2 C^u + C^d ∝ 2(2/3) − 1/3 = 1 and C^u + 2 C^d ∝ 2/3 − 2/3 = 0. For ν_μ the shift is ΔQ^{μμ} = Z g′ ε e / (√2 G_F (Q² + m²)). Because ε g′ < 0 here, ΔQ^{μμ} is **negative**, the same sign as Q_SM ≈ 0.04 Z − 0.5 N < 0: the interference with the Standard Model is **constructive** for ν_μ and ν̄_μ (the COHERENT prompt and delayed muon-flavor fluxes), the opposite of the loop-only case in section 6 and example 03 B. For ν_τ the shift has the opposite sign, but there is no ν_τ flux at COHERENT. Electron neutrinos see no new physics.

   The sign of ε relative to g′ is part of the model and is taken from the description: ε = −5e-4 with g′ = +6e-5 in the pack's convention (g_q → g_q + ε e Q_q^em).

6. **Perturbativity.** |g′| = 6e-5 and |ε| = 5e-4 are both far below √(4π). Both are listed in `COUPLINGS`; zeroing either removes all new physics, so the SM-limit check passes.

7. **Card.** Card-expressible: one vector mediator with the signed neutrino couplings, zero direct quark couplings, and `kinetic_mixing: -0.0005` written explicitly (a card has no κ g′ default). The plugin computes the same coefficients independently from the exported constants `GF`, `SQRT2`, `E_CHARGE` and `Q_EM`.

No convention conversion was needed: the mixing is already quoted in the pack's convention and the coupling is the gauge coupling g′ directly. No axial quark couplings were dropped (there are no direct quark couplings). The Q² dependence of the loop mixing (example 06) is not applied, because the description fixes a constant total ε.
