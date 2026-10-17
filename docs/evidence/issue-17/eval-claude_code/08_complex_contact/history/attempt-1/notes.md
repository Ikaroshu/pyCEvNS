# Notes: contact vector NSI on down quarks (complex off-diagonal)

## Description

Contact vector NSI on down quarks with ε^{dV}_{μμ} = −0.05 and ε^{dV}_{eτ} = 0.02 + 0.03i,
plus its Hermitian partner ε^{dV}_{τe} = 0.02 − 0.03i. All other entries vanish.

## Derivation

1. **Lorentz structure.** The description gives vector NSI epsilons, so the structure is `V`.
   No mediator mass is given, so the contact form of conventions section 5 applies: the
   constant ε^{fV}_{αβ} is added directly to C^{fV}_{αβ}, with no Q² dependence.
2. **Quark.** Only the down quark is coupled. C^{uV} = 0 for every flavor entry.
3. **Flavor entries.** In the framework's normalization C^{fV} = ε^{fL} + ε^{fR} is the NSI
   vector combination, which is exactly what ε^{dV} denotes. So
   - C^{dV}_{μμ}(Q²) = −0.05 (real diagonal, as required),
   - C^{dV}_{eτ}(Q²) = 0.02 + 0.03i,
   - C^{dV}_{τe}(Q²) = 0.02 − 0.03i, filled by the framework as the conjugate partner when the
     `et` entry is set (conventions section 8; `set("V","d","et",…)` in the plugin API).
   - ee, tt, em, mt entries and all scalar entries are zero.
4. **Convention check.** The card and plugin key `et` means the (e, τ) entry C_{eτ}, matching the
   description's ε_{eτ}. No convention conversion was needed; the description is already in the
   framework's ε^{fV} normalization.
5. **Physics remarks.** The μμ entry interferes with the SM: ΔQ^{μμ} = (Z + 2N) ε^{dV}_{μμ} is
   negative, the same sign as Q_SM ≈ 0.04 Z − 0.5 N, so the muon-flavor rate is enhanced
   (constructive interference). The eτ entry is flavor-changing and enters incoherently through
   Σ_β |Q^{αβ}|², adding |Q^{eτ}|² = (Z + 2N)² |ε_{eτ}|² with |ε_{eτ}|² = 0.0013 to the ν_e rate
   (and the same to the ν_τ rate, which the COHERENT flux does not contain). Only the modulus of the
   complex entry affects the rate; the phase is carried faithfully but is unobservable in CEvNS.
6. **Perturbativity.** All couplings are far below √(4π).

## Plugin

`plugin.py` exposes the three real parameters `eps_dV_mm`, `eps_dV_et_re`, `eps_dV_et_im` at the
described point (−0.05, 0.02, 0.03). The complex eτ entry is built inside `coefficients` from the
separate real and imaginary parameters, as required by the plugin API. All three are listed in
`COUPLINGS`, so zeroing them reproduces the Standard Model. Each entry is set once, so no summing
before `set` was needed.

## Card

`card.json` expresses the same interaction with a single `contacts` entry on fermion `d`, with
`mm: -0.05` and `et: [0.02, 0.03]`. The schema supports contact vector entries with complex
off-diagonals, so the card is complete and should cross-validate against the plugin.
