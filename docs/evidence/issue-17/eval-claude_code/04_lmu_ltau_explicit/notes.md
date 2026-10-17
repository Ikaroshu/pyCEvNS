# Notes: gauged L_μ−L_τ Z′ with explicit kinetic mixing

Description: Z′ of U(1)_{L_μ−L_τ}, mass m = 10 MeV, gauge coupling g′ = 1e-4,
photon kinetic mixing ε = +1e-3 in the conventions of this pack, no direct
quark couplings.

## Derivation

1. **Lorentz structure.** A gauge boson couples through vector currents, so
   structure `V` only. No scalar, axial or tensor pieces arise.

2. **Neutrino couplings** (conventions section 7, L_μ−L_τ row). The lepton
   doublet charges are Q′_{ν_e} = 0, Q′_{ν_μ} = +1, Q′_{ν_τ} = −1, so

       g_ν^{ee} = 0,   g_ν^{mm} = +g′ = +1e-4,   g_ν^{tt} = −g′ = −1e-4,

   flavor-diagonal; all off-diagonal entries vanish. The neutrino current is
   left-handed as required.

3. **Quark couplings.** The description states there are no direct quark
   couplings (Q′_u = Q′_d = 0, consistent with L_μ−L_τ). The only quark
   coupling comes from kinetic mixing (conventions section 6):

       g_q = ε e Q_q^em,   g_u = +(2/3) ε e,   g_d = −(1/3) ε e,

   with e = E_CHARGE = 0.303 and ε = +1e-3 as given. Because the description
   gives ε explicitly "in the conventions of this pack", no sign conversion is
   needed; ε is used as written with its positive sign. The loop-induced value
   κ g′ ≈ 1.44e-6 is *not* added: the explicit ε supersedes it, exactly as in
   worked example 03, description A.

4. **Matching** (conventions section 3):

       C^{qV}_{αβ}(Q²) = g_ν^{αβ} g_q / (√2 G_F (Q² + m²)).

   Hence

       C^{uV}_{mm} = +g′ ε e (2/3) / (√2 G_F (Q² + m²)),
       C^{dV}_{mm} = −g′ ε e (1/3) / (√2 G_F (Q² + m²)),
       C^{qV}_{tt} = −C^{qV}_{mm},
       C^{qV}_{ee} = 0,   all off-diagonal and scalar entries = 0.

5. **Physics check.** The induced nucleon couplings are
   g_p = 2 g_u + g_d = ε e and g_n = g_u + 2 g_d = 0: only protons couple, as
   expected for photon-like mixing. For ν_μ the nuclear charge shift
   ΔQ^{μμ} = Z g′ ε e / (√2 G_F (Q² + m²)) is positive (g′ ε > 0), opposite in
   sign to Q_SM ≈ 0.04 Z − 0.5 N < 0, so the interference is destructive for
   ν_μ and ν̄_μ. For ν_τ the shift has the opposite sign (constructive), but
   COHERENT has no ν_τ flux. Electron neutrinos see no new physics.

   Magnitude at the first reference point (Q² = 1238 MeV², Cs-133):
   C^{uV}_{mm} = (1e-4)(1e-3)(0.303)(2/3) / (√2 · 1.16638e-11 · (1238 + 100))
   ≈ 0.915, C^{dV}_{mm} ≈ −0.458, ΔQ^{μμ} ≈ 55 · 1.37 ≈ 75, which exceeds
   |Q_SM| ≈ 37, so the ν_μ rate is enhanced at this point.

6. **Perturbativity.** |g′| = 1e-4 and |ε| = 1e-3 are far below √(4π).

7. **Card.** Expressible as a single vector mediator with the signed neutrino
   couplings, zero direct quark couplings and `kinetic_mixing` = 0.001 written
   explicitly (a card has no κ g′ default). The plugin computes the same
   coefficients from the exported constants `E_CHARGE` and `Q_EM` and does not
   read the card.

8. **SM limit.** `COUPLINGS = ("g_prime", "kinetic_mixing")`; zeroing either one
   removes all new physics, since every coefficient is proportional to g′ ε.

## Files

- `plugin.py`: coefficient plugin with `PARAMETERS`, `COUPLINGS`, `coefficients`.
- `card.json`: equivalent model card.
- `notes.md`: this derivation.
