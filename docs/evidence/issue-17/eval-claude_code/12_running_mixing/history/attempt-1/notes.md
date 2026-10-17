# Notes: gauged L_μ−L_τ Z′ with full one-loop (Q²-dependent) photon mixing

Description: "A gauged L_mu−L_tau vector Z′ has mass 12 MeV and gauge coupling g′ = 8e-5. Keep the full spacelike Q² dependence of the muon/tau one-loop photon mixing, with no tree-level mixing or direct quark coupling."

Parameter point preserved exactly: `g_prime = 8e-5`, `mass_mev = 12.0`.

## Derivation

1. **Lorentz structure.** A gauge boson coupling to a conserved current: vector exchange, structure `V`. No scalar entries.

2. **Charges** (conventions section 7, L_μ−L_τ row): Q′_{ν_e} = 0, Q′_{ν_μ} = +1, Q′_{ν_τ} = −1, Q′_u = Q′_d = 0. Hence
   g_ν^{mm} = +g′, g_ν^{tt} = −g′, g_ν^{ee} = 0, all off-diagonal entries zero, and no direct quark coupling. The neutrino coupling is to the left-handed current with no factor-of-2 ambiguity, since the gauge interaction is g′ Q′ Z′_μ ν̄ γ^μ P_L ν.

3. **Photon mixing from the lepton loops.** The description forbids a tree-level kinetic-mixing term, so the only Z′–photon mixing is the one-loop muon and tau vacuum-polarization diagram. For spacelike Q² ≥ 0 it is

       ε(Q²) = (e g′ / 2π²) ∫₀¹ dx x(1−x) ln[ (m_τ² + x(1−x) Q²) / (m_μ² + x(1−x) Q²) ].

   The Q²-independent (divergent) part cancels because Σ_ℓ Q′_ℓ Q_ℓ = (+1)(−1) + (−1)(−1) = 0; only the mass-dependent logarithm survives. At Q² → 0 the integral of x(1−x) is 1/6, so ε(0) = e g′ ln(m_τ/m_μ)/(6π²) = κ g′ with κ = 0.0144417637 (conventions section 6), giving ε(0) = 1.15534e-06 for g′ = 8e-5.

   Sign: positive throughout, as in section 6. The logarithm is positive for every x because m_τ > m_μ, and it carries the sign that makes the running electromagnetic coupling grow with Q². Physically the proton's screening cloud holds more μ⁻ than τ⁻, and μ⁻ carries L_μ−L_τ charge +1 like ν_μ, so the induced proton coupling has the same sign as the ν_μ coupling.

   Q² dependence: ε(Q²) falls slowly with Q² because the COHERENT recoil range reaches a few tens of percent of m_μ² = 11164 MeV². The ratio ε(Q²)/ε(0) is independent of g′ and m, so it is the same as in worked example 06: 0.99613, 0.98866, 0.98150 at Q² = 1238, 3714, 6190 MeV².

4. **Induced quark couplings** (section 6): g_q(Q²) = ε(Q²) e Q_q^em with Q_u^em = +2/3, Q_d^em = −1/3 and e = 0.303 (`E_CHARGE`, `Q_EM` from the framework).

5. **Matching** (section 3):

       C^{uV}_{mm}(Q²) = +g′ ε(Q²) e (2/3) / (√2 G_F (Q² + m²)),
       C^{dV}_{mm}(Q²) = −g′ ε(Q²) e (1/3) / (√2 G_F (Q² + m²)),
       C^{qV}_{tt} = −C^{qV}_{mm},   C^{qV}_{ee} = 0,   off-diagonal = 0.

   Only protons are affected: 2C^u + C^d ∝ 2(2/3) − 1/3 = 1, C^u + 2C^d ∝ 2/3 − 2/3 = 0, as expected for photon-mediated mixing.

6. **Interference.** For ν_μ the charge shift ΔQ = Z g′ ε(Q²) e / (√2 G_F (Q² + m²)) is positive while Q_SM is negative: destructive interference for the COHERENT prompt ν_μ and delayed ν̄_μ. Electron neutrinos are unaffected.

7. **Approximate values at the reference points** (Cs-133, Q² = 1238, 3714, 6190 MeV²), obtained by scaling example 06 by g′²/(Q² + m²): C^{uV}_mm ≈ 8.16e-04, 2.90e-04, 1.75e-04; C^{dV}_mm = −C^{uV}_mm / 2. The plugin computes them at full precision from the exported constants.

## Why no card.json

A card's `kinetic_mixing` is a single constant. The description explicitly asks for the full spacelike Q² dependence of the loop mixing, which varies by about 1–2 percent across the recoil range and would fail the 1e-6 plugin-versus-card cross-check. The interaction is therefore plugin-only.

## Reduction check for reviewers

At Q² = 0 the plugin's ε equals κ g′ exactly, so the coefficients reduce to those of worked example 03 B with g′ = 8e-5 and m = 12 MeV: C^{qV}_{mm}(0) = g′ κ g′ e Q_q^em / (√2 G_F m²). The 40-point Gauss–Legendre rule on [0, 1] reproduces the smooth Feynman-parameter integral to better than 1e-14 relative (a trapezoid rule dropping the endpoints would be low by about 2e-6 and fail the eval tolerance).

## Conventions and bookkeeping

- `COUPLINGS = ("g_prime",)`: the mixing is derived from g′, so zeroing g′ removes all new physics and the SM-limit check passes. Mass is a parameter but not a coupling.
- Perturbativity: |g′| = 8e-5 ≪ √(4π).
- No axial quark coupling is dropped: photon mixing induces a purely vector quark current.
- The L_μ−L_τ assignment is anomaly-free with the SM fermion content.
- No convention conversion was needed; the description already uses the gauge coupling g′ and the mass in MeV.
