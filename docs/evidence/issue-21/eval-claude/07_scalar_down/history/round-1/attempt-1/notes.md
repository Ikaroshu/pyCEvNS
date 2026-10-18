# Scalar φ, mass 20 MeV, flavor-universal Dirac neutrino coupling, down quarks only

## Description as given

A scalar φ of mass 20 MeV with Dirac couplings 2e-5 φ ν̄_α ν_α for α = e, μ, τ and
2e-5 φ d̄ d. All other couplings (up quark, off-diagonal neutrino flavors, charged
leptons) vanish.

## Derivation

1. **Lorentz structure.** φ ν̄ ν and φ d̄ d are both scalar bilinears, so the
   structure is `S`. No vector, axial, pseudoscalar or tensor piece appears.

2. **Lagrangian in the framework's form** (conventions section 4):

       L ⊃ φ ( g_ν^{αβ} ν̄_α ν_β + Σ_q g_q q̄ q ),   m = 20 MeV,

   with g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = 2e-5, all off-diagonal g_ν = 0,
   g_d = 2e-5 and g_u = 0. The description is already in this normalization,
   so no convention conversion is needed.

3. **Dirac versus Majorana.** The description says the coupling is Dirac:
   ν̄_α ν_α = ν̄_L ν_R + ν̄_R ν_L requires a right-handed Dirac partner, which
   is what makes the helicity-flipping scalar amplitude possible. For a Dirac
   coupling written as g_ν φ ν̄ ν the vertex is g_ν as written; the factor-2
   subtlety of section 4 applies only to a Majorana mass-type term written
   without the 1/2, which is not the case here.

4. **Matching** (conventions section 4, tree-level t-channel φ exchange):

       C^{dS}_{αα}(Q²) = g_ν g_d / ( √2 G_F (Q² + m²) ),   α = e, μ, τ,
       C^{uS}_{αβ}(Q²) = 0,
       C^{dS}_{αβ}(Q²) = 0 for α ≠ β.

   Numerically g_ν g_d = 4e-10 and m² = 400 MeV². All vector coefficients are
   zero, so there is no new-physics interference with the Standard Model; the
   scalar rate adds incoherently with its own kinematic factor, supplied by
   the framework.

5. **Reference values** (Cs-133, Q² = 2 M T at T = 5, 15, 25 keV; rounded):

   | coefficient | Q² = 1238 | Q² = 3714 | Q² = 6190 |
   |---|---|---|---|
   | C^{dS}_ee = C^{dS}_mm = C^{dS}_tt | 1.480e-02 | 5.894e-03 | 3.680e-03 |
   | C^{uS} (all flavors) | 0 | 0 | 0 |

   The plugin computes these at full precision from the exported `GF` and
   `SQRT2`.

6. **Nuclear scalar charge** (for orientation only; the framework forms it).
   With only g_d non-zero, c_N = m_N g_d f_d^N / m_d, giving
   c_p ≈ 8.2 g_d and c_n ≈ 9.0 g_d, i.e. the nucleus couples as
   roughly (8.2 Z + 9.0 N) g_d, about half the "17 A g_q" of a
   u+d-universal scalar because the up-quark term is absent.

7. **Perturbativity.** |g_ν| = |g_d| = 2e-5 ≪ √(4π). Mass positive.

8. **Card.** Fully expressible: one `S` mediator with diagonal universal
   neutrino couplings, `quark_couplings` u = 0, d = 2e-5. No kinetic mixing
   (vector-only field, omitted). The plugin and card are independent
   representations of the same matching formula.

## Parameters

`PARAMETERS = {"g_nu": 2e-5, "g_d": 2e-5, "mass_mev": 20.0}`,
`COUPLINGS = ("g_nu", "g_d")`. Setting both couplings to zero returns the
Standard Model. The described point is preserved exactly.
