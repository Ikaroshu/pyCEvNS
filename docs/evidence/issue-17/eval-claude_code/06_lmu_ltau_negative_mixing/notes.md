# L_μ−L_τ Z′ with negative total kinetic mixing

Description: gauged L_μ−L_τ vector Z′, m = 15 MeV, g′ = 6e-5, total photon
kinetic mixing ε = −5e-4 in the pack's convention (conventions section 6), no
direct quark couplings.

## Derivation

1. **Lorentz structure.** Gauge boson exchange: vector, structure `V`. No scalar
   entries. The neutrino coupling is to the left-handed lepton doublet, so g_ν
   as written enters the matching formula directly.

2. **Neutrino couplings** (conventions section 7, L_μ−L_τ row): Q′_{ν_e} = 0,
   Q′_{ν_μ} = +1, Q′_{ν_τ} = −1. Hence
   g_ν^{ee} = 0, g_ν^{mm} = +g′ = +6e-5, g_ν^{tt} = −g′ = −6e-5, off-diagonal
   entries zero (charges are flavor-diagonal in the mass basis).

3. **Quark couplings.** The U(1)′ charges of the quarks vanish and the
   description states there are no direct quark couplings, so the only quark
   coupling is through the photon kinetic mixing (section 6):
   g_q = ε e Q_q^em with e = 0.303 (E_CHARGE), Q_u = +2/3, Q_d = −1/3.
   With ε = −5e-4: g_u = −1.01e-4, g_d = +5.05e-5.

   The description gives ε as the **total** mixing already in the pack's
   convention, so it is used as given. No loop-induced κ g′ term is added on
   top: κ g′ = +8.665e-7 is part of the stated total, which means the tree-level
   piece of the mixing in this model is about −5.0087e-4. No convention
   conversion is needed because the value is already in the section 6 sign
   convention (g_q → g_q + ε e Q_q^em). The negative sign is a model input and
   is preserved exactly.

4. **Matching** (section 3):

       C^{qV}_{αβ}(Q²) = g_ν^{αβ} ε e Q_q^em / (√2 G_F (Q² + m²)),   m = 15 MeV.

   Explicitly, with g′ ε e = 6e-5 × (−5e-4) × 0.303 = −9.09e-9:

       C^{uV}_mm = −C^{uV}_tt = (2/3) g′ ε e / (√2 G_F (Q² + m²))
       C^{dV}_mm = −C^{dV}_tt = (−1/3) g′ ε e / (√2 G_F (Q² + m²))
       C^{uV}_ee = C^{dV}_ee = 0, all off-diagonal entries zero.

   Approximate values at the Cs-133 reference points (Q² = 1238, 3714, 6190
   MeV²; m² = 225 MeV²), rounded for reading:

   | coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
   |---|---|---|---|
   | C^{uV}_mm = −C^{uV}_tt | −2.511e-01 | −9.327e-02 | −5.727e-02 |
   | C^{dV}_mm = −C^{dV}_tt | +1.256e-01 | +4.663e-02 | +2.863e-02 |

5. **Physics check.** Photon-like mixing couples only to protons:
   2 C^u + C^d ∝ 2(2/3) − 1/3 = 1 and C^u + 2 C^d ∝ 2/3 − 2/3 = 0, so the
   neutron coupling vanishes. For ν_μ the nuclear charge shift is
   ΔQ^{μμ} = Z g′ ε e / (√2 G_F (Q² + m²)), which is **negative** because
   ε g′ < 0. Q_SM ≈ −36.9 for Cs-133 is also negative, so the interference is
   **constructive** for ν_μ (and ν̄_μ): at the first reference point
   ΔQ^{μμ} ≈ −20.7, giving a total charge ≈ −57.6 and a ν_μ rate about 2.4
   times the SM there. This is the opposite of the positive-ε cases of
   examples 03 and 06, where the ν_μ interference is destructive. For ν_τ the
   shift has the opposite sign (destructive), but there is no ν_τ flux at
   COHERENT. Electron neutrinos see no new physics.

6. **Card.** The card schema expresses this exactly: one vector mediator with
   the L_μ−L_τ neutrino charges times g′, zero direct quark couplings and
   `kinetic_mixing` = −5e-4. Because ε is a constant here (not the
   Q²-dependent loop function of example 06), plugin and card describe
   identical coefficients.

7. **Plugin.** PARAMETERS are the described point (g′ = 6e-5, m = 15 MeV,
   ε = −5e-4). Both `g_prime` and `kinetic_mixing` are listed in COUPLINGS;
   zeroing either removes all new physics, which satisfies the SM-limit check.
   All couplings are far below √(4π). Each coefficient entry is set exactly once.

8. **Dropped pieces.** None: the Z′ coupling to quarks via photon mixing is
   purely vector, so there is no axial part to discard. The Z′ also couples to
   charged muons and taus, which does not enter CEvNS.

## Validation

`nuagent validate --plugin plugin.py --card card.json` run in this directory;
result recorded in the final summary.
