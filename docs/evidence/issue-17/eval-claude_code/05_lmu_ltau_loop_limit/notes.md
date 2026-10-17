# Notes: gauged L_μ − L_τ Z′, loop-induced photon mixing, Q² → 0 limit

Description: a gauged U(1)_{L_μ−L_τ} vector Z′ with mass m = 15 MeV and gauge
coupling g′ = 6e-5. Quarks couple only through the positive loop-induced
mixing with the photon evaluated at Q² ≪ m_μ²; no tree-level kinetic mixing
and no direct quark charge.

## Derivation

1. **Lorentz structure.** Gauge boson exchange: structure `V`. Only the
   left-handed neutrino current enters (conventions section 3). No scalar
   contribution.

2. **Neutrino couplings** (conventions section 7, L_μ−L_τ row). The lepton
   doublets carry Q′_{L_e} = 0, Q′_{L_μ} = +1, Q′_{L_τ} = −1, so

       g_ν^{ee} = 0,   g_ν^{mm} = +g′ = +6e-5,   g_ν^{tt} = −g′ = −6e-5,

   flavor-diagonal; all off-diagonal entries vanish. The assignment is
   anomaly-free.

3. **Quark couplings.** Quarks have Q′_q = 0, so there is no tree-level
   coupling. The muon and tau loops mix Z′ with the photon. In the
   Q² ≪ m_μ² limit (conventions section 6) the mass-independent part of the
   loop cancels because Σ_ℓ Q′_ℓ Q_ℓ = 0, leaving

       ε = +κ g′,   κ = e ln(m_τ/m_μ) / (6π²) = 0.0144417637,

   so ε = 0.0144417637 × 6e-5 = 8.66505822e-7. The induced quark coupling is
   electric-charge proportional:

       g_u = ε e Q_u = ε e (+2/3),   g_d = ε e Q_d = ε e (−1/3),   e = 0.303.

   The sign is positive and is a physics result, not a convention: the lighter
   muon dominates the running of the electromagnetic coupling, and μ⁻ carries
   L_μ−L_τ charge +1 like ν_μ. The description asks explicitly for the
   positive loop-induced mixing, which agrees with this. The plugin uses the
   exported `KAPPA_LMU_LTAU`, `E_CHARGE` and `Q_EM` rather than retyped
   numbers; the card has no default for the mixing, so the value κ g′ is written
   explicitly.

4. **Matching** (conventions section 3):

       C^{qV}_{αα}(Q²) = g_ν^{αα} g_q / ( √2 G_F (Q² + m²) ),   m = 15 MeV.

   Explicitly, with g′ = 6e-5,

       C^{uV}_mm = + g′ · κ g′ e (2/3) / (√2 G_F (Q² + m²)),
       C^{dV}_mm = − g′ · κ g′ e (1/3) / (√2 G_F (Q² + m²)),
       C^{uV}_tt = − C^{uV}_mm,   C^{dV}_tt = − C^{dV}_mm,
       C^{qV}_ee = 0.

   Since the quark coupling is proportional to electric charge, the induced
   neutron charge g_n = g_u + 2 g_d = 0 vanishes identically and only the
   proton charge g_p = 2 g_u + g_d = ε e survives. The nuclear charge shift is
   therefore ΔQ^{αα} = Z g_ν^{αα} ε e / (√2 G_F (Q² + m²)).

5. **Approximate magnitude** at the Cs-133 reference point Q² = 1238 MeV²
   (for orientation only; the plugin computes at full precision):

   | quantity | value |
   |---|---|
   | g_u | ≈ +1.75e-7 |
   | g_d | ≈ −8.75e-8 |
   | C^{uV}_mm | ≈ +4.35e-4 |
   | C^{dV}_mm | ≈ −2.18e-4 |
   | ΔQ^{mm} (Cs-133, Z = 55) | ≈ +0.036 |

   Q_SM ≈ −36.9 is negative, so for ν_μ (and ν̄_μ) the shift is destructive
   but at the 0.1 percent level of the charge at this point. For ν_τ the sign
   flips (constructive), but no ν_τ flux is present at COHERENT. ν_e is
   unaffected.

6. **Validity of the Q² → 0 limit.** The description fixes the Q² → 0 form of
   the mixing, so the constant κ is used at every Q². For reference, the
   COHERENT CsI recoil range has Q² ≲ 10⁴ MeV², comparable to m_μ² ≈ 1.1e4
   MeV², where the full loop function deviates from its Q² → 0 value; that
   case would need the running-mixing plugin of example 06 and is deliberately
   not used here, as instructed.

7. **Card-expressible:** yes. One vector mediator with nu_couplings
   mm = +6e-5, tt = −6e-5, zero direct quark couplings, and
   kinetic_mixing = κ g′ = 8.66505822e-7. The framework adds ε e Q_q^em to each
   quark coupling, reproducing step 3.

## Conventions and sign checks

- No convention conversion was needed: the description gives the gauge coupling
  g′ directly and the L_μ−L_τ charges are the standard ones.
- Perturbativity: |g′| = 6e-5 ≪ √(4π).
- The axial quark coupling is zero (photon-like vector coupling), so nothing is
  dropped.
- Pitfall avoided: the pyCEvNS factor-of-2 normalization is not used; the
  matching formula of conventions section 3 is.

## Validation

`nuagent validate --plugin plugin.py --card card.json` returned valid with no
findings: perturbativity, SM limit with COUPLINGS zeroed, finite non-negative
counts, schema-valid card, and plugin/card cross-check on 66 Q² points.
