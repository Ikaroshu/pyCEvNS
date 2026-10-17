# Notes: universal vector Z′ with g ν̄ γ^μ (1 − γ5) ν neutrino coupling

## Description

Vector Z′, mass 30 MeV. Neutrino interaction g Z′_μ ν̄ γ^μ (1 − γ5) ν for every
flavor with g = 5e-5 and no factor 1/2. Vector coupling 1e-4 to each of u and d.
All other couplings vanish.

## Derivation

1. **Lorentz structure.** Vector boson exchange between a neutrino current and a
   quark vector current: structure `V`. No scalar part.

2. **Neutrino coupling normalization.** The framework's matching formula
   (conventions section 3) is written for Z′_μ g_ν ν̄ γ^μ P_L ν with
   P_L = (1 − γ5)/2. The description writes g ν̄ γ^μ (1 − γ5) ν = 2g ν̄ γ^μ P_L ν
   without the 1/2, so

       g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = 2g = 2 × 5e-5 = 1e-4.

   This is exactly the case called out in conventions section 3: "If it writes
   g ν̄ γ^μ (1 − γ5) ν without the 1/2, then g_ν = 2g." Off-diagonal entries
   are zero (universal, flavor-diagonal).

3. **Quark couplings.** Given directly as vector couplings: g_u = g_d = 1e-4.
   No chiral decomposition needed, no axial part to drop, no kinetic mixing.

4. **Matching.** From conventions section 3,

       C^{uV}_{αα}(Q²) = C^{dV}_{αα}(Q²) = g_ν g_q / (√2 G_F (Q² + m²))
                        = (1e-4)(1e-4) / (√2 G_F (Q² + (30 MeV)²)).

   Numerically this coincides with worked example 01 (g′ = 1e-4 universal), so
   at the Cs-133 reference points Q² = 1238, 3714, 6190 MeV² the coefficients
   are approximately 0.28355, 0.13139, 0.085506 for both quarks and all three
   diagonal flavors.

5. **Sign and interference.** g_ν g_q = +1e-8 > 0, so the new-physics nuclear
   charge shift ΔQ = (3Z + 3N) C is positive while Q_SM is negative:
   destructive interference with the SM, with the charge sign already flipped at
   low Q² for this coupling (as in example 01).

6. **Perturbativity.** All couplings are of order 1e-4, far below √(4π).

## Card

The interaction is a single flavor-diagonal vector mediator with constant
couplings, so the schema expresses it exactly. The card stores the already
converted left-handed coupling g_ν = 2g = 1e-4, not the raw g = 5e-5, because
the card's `nu_couplings` field is defined as the coefficient of the P_L current.

## Plugin

`plugin.py` keeps the description's own parameter g = 5e-5 as a `PARAMETERS`
entry and performs the factor-2 conversion inside `coefficients`, so the
described point is preserved exactly while matching the card. `COUPLINGS` lists
g and g_q; zeroing either removes all new physics, reproducing the SM limit.
