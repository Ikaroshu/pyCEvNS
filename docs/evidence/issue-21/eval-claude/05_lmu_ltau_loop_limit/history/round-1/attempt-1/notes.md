# Notes: gauged L_μ−L_τ Z′ with loop-induced photon mixing (Q² → 0 limit)

## Description

Z′ of mass m = 15 MeV, gauge coupling g′ = 6e-5, gauging U(1)_{L_μ−L_τ}. Quark couplings
arise only from the positive loop-induced kinetic mixing with the photon, evaluated in the
Q² → 0 limit (Q² ≪ m_μ²). No tree-level kinetic mixing, no direct quark charge.

## Derivation

1. **Lorentz structure.** Z′ is a vector boson coupling to vector currents: structure `V`.
   The neutrino coupling is to the left-handed lepton doublet, so g_ν is the full
   coefficient of ν̄ γ^μ P_L ν with no factor-2 conversion needed.

2. **Charges (conventions §7).** Q′_{ν_e} = 0, Q′_{ν_μ} = +1, Q′_{ν_τ} = −1, Q′_u = Q′_d = 0.
   Hence g_ν^{ee} = 0, g_ν^{mm} = +g′ = 6e-5, g_ν^{tt} = −g′ = −6e-5, off-diagonal zero.
   Direct quark couplings vanish. (L_μ−L_τ is anomaly-free; nothing to flag.)

3. **Loop-induced mixing (conventions §6).** Muon and tau loops mix Z′ with the photon.
   At Q² ≪ m_μ² the mixing is a constant,

       ε = +κ g′,   κ = e ln(m_τ/m_μ)/(6π²) = 0.0144417637…,

   so ε = κ · 6e-5 = 8.66505823932692e-07. The sign is a physics result: the
   m-independent part of the loop cancels because Σ_ℓ Q′_ℓ Q_ℓ = 0, and the surviving
   ln(m_τ²/m_μ²) term is positive. The description's "positive loop-induced photon mixing"
   matches this. In the plugin ε is computed from the exported `KAPPA_LMU_LTAU` times
   `g_prime`; the card has no default and carries the numerical value at full precision so
   the plugin/card cross-check passes at 1e-6.

4. **Induced quark couplings.** g_q = ε e Q_q^em with e = 0.303:
   g_u = +(2/3) ε e, g_d = −(1/3) ε e. Nucleon level: g_p = 2g_u + g_d = ε e, g_n = g_u + 2g_d = 0.
   Only protons couple, as expected for photon-mediated mixing.

5. **Matching (conventions §3).**

       C^{qV}_{αβ}(Q²) = g_ν^{αβ} g_q / (√2 G_F (Q² + m²)),

   giving C^{uV}_{mm} = +g′ ε e (2/3)/(√2 G_F (Q² + m²)),
   C^{dV}_{mm} = −g′ ε e (1/3)/(√2 G_F (Q² + m²)),
   C^{qV}_{tt} = −C^{qV}_{mm}, C^{qV}_{ee} = 0, all off-diagonal and scalar entries zero.
   The propagator is 1/(Q² + m²) with Q² = 2MT spacelike.

6. **Momentum dependence of the mixing.** The description fixes the Q² → 0 limit, so ε is
   constant and no running of the loop function with Q² is included. This is why the card
   can express the interaction (a `kinetic_mixing` number); the Q²-dependent case would need
   a plugin only.

7. **Interference.** For ν_μ the nuclear charge shift is ΔQ = Z g′ ε e/(√2 G_F (Q² + m²)) > 0
   since g′ ε = κ g′² > 0, opposite in sign to Q_SM < 0: destructive interference for the
   COHERENT prompt ν_μ and delayed ν̄_μ. ν_e sees no new physics. The ττ entries do not
   contribute at COHERENT (no ν_τ flux) but are included for completeness.

8. **SM limit and perturbativity.** `g_prime` is the only coupling; ε is derived from it, so
   zeroing `COUPLINGS` removes all new physics. |g′| = 6e-5 ≪ √(4π).

## Files

- `plugin.py`: coefficients from exported constants (`KAPPA_LMU_LTAU`, `E_CHARGE`, `Q_EM`,
  `GF`, `SQRT2`); parameters `g_prime = 6e-5`, `mass_mev = 15.0`.
- `card.json`: one vector mediator, ν couplings (0, +6e-5, −6e-5), zero direct quark
  couplings, `kinetic_mixing` = κ g′ = 8.66505823932692e-07.
