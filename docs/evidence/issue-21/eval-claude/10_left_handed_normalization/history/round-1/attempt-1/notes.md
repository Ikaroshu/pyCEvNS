# Universal vector Z′ with a (1 − γ5) neutrino current

## Description as given

Vector Z′, mass m = 30 MeV. Neutrino interaction g Z′_μ ν̄ γ^μ (1 − γ5) ν for every flavor, g = 5e-5, explicitly without a factor 1/2. Vector coupling to u and to d quarks 1e-4 each. All other couplings vanish.

## Derivation

1. **Lorentz structure.** Vector exchange between a neutrino current and a quark vector current: structure `V`. No scalar, axial or tensor pieces, so the V/S basis is sufficient and a card can express the model.

2. **Neutrino coupling normalization.** The framework's convention (conventions.md section 3) writes the Z′ vertex as g_ν ν̄ γ^μ P_L ν with P_L = (1 − γ5)/2. The description writes g ν̄ γ^μ (1 − γ5) ν = 2g ν̄ γ^μ P_L ν. Therefore

       g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = 2 g = 2 × 5e-5 = 1e-4.

   Off-diagonal neutrino entries are zero ("every flavor", flavor-diagonal and universal). This is the pitfall called out in conventions section 3: a (g′/2)(1 − γ5) form gives g_ν = g′, the (1 − γ5) form without the 1/2 gives g_ν = 2g. Only the left-handed current appears either way, so nothing is dropped.

3. **Quark couplings.** Given directly as vector couplings: g_u = g_d = 1e-4. No chiral split is given, so there is no axial part to drop. No kinetic mixing (ε = 0), consistent with "all other couplings vanish".

4. **Matching.** Tree-level Z′ exchange (conventions section 3):

       C^{fV}_{αα}(Q²) = g_ν g_f / (√2 G_F (Q² + m²)),   f ∈ {u, d},  α ∈ {e, μ, τ},

   with g_ν = 1e-4, g_u = g_d = 1e-4, m = 30 MeV. Numerically this coincides with the flavor-universal example 01 (g′ = 1e-4 for neutrinos and quarks), so the expected coefficients at the Cs-133 reference points Q² = 1238, 3714, 6190 MeV² are 2.8355e-01, 1.3139e-01, 8.5506e-02 for both C^{uV} and C^{dV}, flavor-diagonal.

5. **Sign and interference.** g_ν g_q = +1e-8 > 0, so the new-physics nuclear charge shift is positive while the SM charge is negative: destructive interference, and at the described point the shift exceeds |Q_SM|, so the total charge has flipped sign and the rate is enhanced relative to the SM.

6. **Perturbativity.** All couplings are of order 1e-4, far below √(4π).

## Plugin parameters

`PARAMETERS = {"g": 5e-5, "g_u": 1e-4, "g_d": 1e-4, "mass_mev": 30.0}`; the plugin forms g_ν = 2 g internally. `COUPLINGS = ("g", "g_u", "g_d")`; zeroing them returns the SM. The card stores the already-converted left-handed coupling g_ν = 1e-4, since the schema's `nu_couplings` is defined for the P_L current.

## Card

One vector mediator entry with mass 30 MeV, `nu_couplings` ee = mm = tt = 1e-4, `quark_couplings` u = d = 1e-4, no kinetic mixing, no contacts.
