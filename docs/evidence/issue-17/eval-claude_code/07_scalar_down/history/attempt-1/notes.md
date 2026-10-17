# Notes: scalar φ with universal neutrino coupling and down-quark-only coupling

## Description

"A scalar phi of mass 20 MeV has Dirac couplings 2e-5 phi nu-bar_alpha nu_alpha to all three
neutrino flavors and 2e-5 phi d-bar d to down quarks only. All other couplings vanish."

## Derivation

1. **Lorentz structure.** Scalar exchange, structure `S`. The description writes the couplings
   in exactly the form of conventions section 4, L ⊃ φ (g_ν^{αβ} ν̄_α ν_β + Σ_q g_q q̄ q), so no
   convention conversion is needed.

2. **Couplings as written.**
   - g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = 2e-5; off-diagonal entries zero.
   - g_d = 2e-5; g_u = 0 ("down quarks only"; "all other couplings vanish").
   - Mass m = 20 MeV.

3. **Neutrino nature.** The description states Dirac couplings, so ν̄_α ν_α pairs the SM
   left-handed neutrino with a right-handed Dirac partner. There is no Majorana 1/2 issue; g_ν is
   taken as written.

4. **Matching (conventions section 4).**

       C^{dS}_{αα}(Q²) = g_ν g_d / (√2 G_F (Q² + m²)),     α = e, μ, τ
       C^{uS}_{αβ}(Q²) = 0
       all vector coefficients = 0, all off-diagonal scalar entries = 0

   with g_ν g_d = 4e-10 and m² = 400 MeV².

5. **Reference values** (Cs-133, Q² = 1238, 3714, 6190 MeV², rounded to four digits; the plugin
   uses the exported constants at full precision):

   | coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
   |---|---|---|---|
   | C^{dS}_ee = C^{dS}_mm = C^{dS}_tt | 1.480e-02 | 5.894e-03 | 3.680e-03 |
   | C^{uS}_αβ, all C^{V} | 0 | 0 | 0 |

6. **Nucleon charges (formed by the framework, recorded for orientation).** With g_u = 0 only
   the down-quark scalar form factors contribute:
   c_p = m_p g_d f_d^p / m_d ≈ 8.20 g_d,  c_n = m_n g_d f_d^n / m_d ≈ 9.02 g_d,
   so the nucleus couples as roughly g_ν g_d (8.2 Z + 9.0 N)/(Q² + m²). This is about half the
   per-nucleon strength of the isoscalar case (c_N ≈ 17.1 g_q for g_u = g_d), and it is slightly
   neutron-favoring.

7. **Interference.** The scalar amplitude flips the neutrino helicity and does not interfere with
   the SM vector amplitude; the new-physics rate adds incoherently with the kinematic factor the
   framework supplies. The overall sign of g_ν g_d is therefore unobservable here.

8. **Perturbativity.** |g_ν| = |g_d| = 2e-5 ≪ √(4π).

9. **Card-expressible.** Yes: one scalar mediator entry with the three diagonal neutrino couplings,
   d = 2e-5 and u = 0. `card.json` is written accordingly.

## Files

- `plugin.py`: parameters `g_nu`, `g_d`, `mass_mev` (defaults 2e-5, 2e-5, 20.0); couplings
  `g_nu`, `g_d`. Sets only `S`/`d` diagonal entries.
- `card.json`: the equivalent single-mediator card.
