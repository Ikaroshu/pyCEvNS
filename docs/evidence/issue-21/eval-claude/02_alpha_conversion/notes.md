# Notes: universal vector Z′, m = 45 MeV, α′ = g′²/4π

## Description as given

A universal vector Z′ of mass 45 MeV with α′ = g′²/(4π) = 3.899296105751435e-10. The positive coupling g′ is the same for all left-handed neutrino flavors and for the u and d quarks; no other couplings.

## Derivation

1. **Lorentz structure.** Vector exchange, structure `V`. No scalar entries.

2. **Coupling conversion (conventions section 10).** The description gives α′, not g′. Inverting α′ = g′²/(4π):

       g′ = sqrt(4π α′) = sqrt(12.566370614359172 × 3.899296105751435e-10)
          = sqrt(4.9000000e-9) = 7.0e-5.

   The product 4π α′ equals 4.9e-9 to all given digits, so g′ = 7e-5 exactly at double precision. The description says g′ is positive, so the positive root is taken. Perturbativity: 7e-5 ≪ √(4π).

3. **Couplings in the framework's Lagrangian (conventions section 3).**
   L ⊃ Z′_μ ( g_ν^{αβ} ν̄_α γ^μ P_L ν_β + g_u ū γ^μ u + g_d d̄ γ^μ d ) with
   g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = g′, off-diagonal g_ν = 0, g_u = g_d = g′.
   The neutrino coupling is already to the left-handed current, so no chirality factor is needed. The quark couplings are pure vector (no chiral split given), so no axial part is dropped.

4. **Matching.** Tree-level Z′ exchange gives

       C^{uV}_{αα}(Q²) = C^{dV}_{αα}(Q²) = g′² / ( √2 G_F (Q² + m²) ),   m = 45 MeV,

   for α = e, μ, τ, with the propagator 1/(Q² + m²) and Q² = 2 M T in MeV². The plugin computes this from the exported `GF` and `SQRT2`, independently of the card. Note that g′² = 4π α′ = 4.9e-9, so the numerator is just 4π α′.

5. **Sign and interference.** g_ν g_q = +g′² > 0, so the new-physics nuclear charge shift ΔQ = (3Z + 3N) C is positive, while the SM charge is negative (Q_SM ≈ 0.04 Z − 0.5 N). The interference is destructive. This is the "universal (toy)" assignment of conventions section 7, for which a cancellation of the charge is possible at some coupling; the described point is kept exactly as given and not adjusted.

6. **Card.** The interaction is one vector mediator with constant couplings, so the schema expresses it fully: structure V, mass 45, nu_couplings diagonal 7e-5, quark_couplings u = d = 7e-5, kinetic_mixing 0, no contacts. The card carries the converted g′ = 7e-5 rather than α′ because the schema has no field for α′.

## Parameters

| parameter | value | meaning |
|---|---|---|
| `g_prime` | 7e-5 | g′ = sqrt(4π α′), common coupling to ν_L (all flavors), u and d |
| `mass_mev` | 45.0 | Z′ mass in MeV |

`COUPLINGS = ("g_prime",)`; setting it to zero recovers the Standard Model.

## Convention conversion recorded

The only conversion is α′ → g′ = sqrt(4π α′). No chiral, nucleon-level or kinetic-mixing conversions are needed. No parameter was tuned; the described point (α′ = 3.899296105751435e-10, m = 45 MeV) is preserved exactly through g′ = 7e-5.
