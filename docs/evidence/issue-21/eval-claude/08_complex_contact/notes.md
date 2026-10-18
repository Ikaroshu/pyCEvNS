# Notes: contact vector NSI on down quarks with a complex e–τ entry

Description: contact vector NSI on down quarks with ε^{dV}_{μμ} = −0.05 and
ε^{dV}_{eτ} = 0.02 + 0.03i, with its Hermitian partner. All other entries vanish.

## Derivation

1. **Lorentz structure.** The description says "vector NSI", so structure `V`.
   No mass is given and the interaction is stated as epsilons, so the contact
   form of conventions section 5 applies: the epsilons are added directly to
   C^{dV} and are Q²-independent.

2. **Coefficients.** In the framework normalization C^{fV} = ε^{fL} + ε^{fR},
   and a contact ε^{fV} is exactly that combination, so no conversion is needed:

   - C^{dV}_{μμ} = −0.05 (real, diagonal) at every Q².
   - C^{dV}_{eτ} = 0.02 + 0.03i at every Q².
   - C^{dV}_{τe} = (C^{dV}_{eτ})* = 0.02 − 0.03i by Hermiticity; the plugin and
     the framework fill this partner automatically when `et` is set.
   - All up-quark entries, all other down-quark entries and all scalar entries
     are zero.

3. **Parameters.** Plugin parameters are real, so the complex e–τ entry is
   built inside `coefficients` from `eps_d_et_re = 0.02` and
   `eps_d_et_im = 0.03`. The μμ entry is `eps_d_mm = −0.05`. All three are
   listed in `COUPLINGS`; zeroing them reproduces the Standard Model, and each
   is far below the perturbativity bound √(4π) ≈ 3.54.

4. **Interference with the SM** (Cs-133, Z = 55, N = 78, tree-level
   Q_SM ≈ −36.93). For a down-quark-only contact term the nuclear charge shift
   is ΔQ^{αβ} = (Z + 2N) C^{dV}_{αβ} = 211 C^{dV}_{αβ}.

   - μμ: ΔQ^{μμ} = 211 × (−0.05) = −10.55, the same sign as Q_SM, so the
     interference is constructive and the ν_μ and ν̄_μ coherent rate rises to
     about (47.48/36.93)² ≈ 1.65 times the SM at tree level.
   - eτ: |ΔQ^{eτ}| = 211 × |0.02 + 0.03i| = 211 × 0.03606 ≈ 7.61. This
     flavor-changing entry does not interfere with the SM (no SM
     flavor-changing neutral current); it adds incoherently as |Q^{eτ}|² to the
     ν_e rate and, with equal magnitude |Q^{τe}|², to the ν_τ rate. Only the
     modulus of the complex entry matters for the rate; the phase is kept as
     given so the coefficient matches the description exactly.

5. **Card.** The schema expresses this interaction directly through one
   `contacts` entry on fermion `d`, with the complex off-diagonal entry
   written as `"et": [0.02, 0.03]` (`[re, im]`). The plugin computes the same
   coefficients independently of the card.

No convention conversion was needed: the description already uses the
framework's ε^{fV} normalization and the e–τ ordering (`et` = C_{eτ}).
