# Notes: light vector boson coupled to neutrinos and quarks

## Description after round 2

Original description: "A new light vector boson that couples to neutrinos."
Round-1 questions asked for the mass, the neutrino couplings and the quark
couplings. The round-2 free-text answer fixes all of them:

- mass m = 30 MeV
- left-handed neutrino coupling g_ν = 1e-4, the same for ν_e, ν_μ, ν_τ
  (flavor-diagonal, no off-diagonal entries)
- quark couplings g_u = g_d = 1e-4
- "nothing else": no kinetic mixing, no second mediator, no scalar part

The interaction is now fully determined within the V/S basis, so no further
questions are needed.

## Lagrangian

    L ⊃ Z′_μ ( g_ν Σ_α ν̄_α γ^μ P_L ν_α + g_u ū γ^μ u + g_d d̄ γ^μ d ),   m = 30 MeV

with g_ν = g_u = g_d = 1e-4. The answer says "left-handed neutrino flavors",
so the coupling is already in the P_L normalization of conventions section 3
and no factor-of-2 conversion is needed. The quark coupling is given as a
single vector coupling per quark (no chiral split), so g_q is used directly
and there is no dropped axial part.

## Matching (conventions section 3)

Tree-level Z′ exchange gives, for each quark f and each diagonal flavor α,

    C^{fV}_{αα}(Q²) = g_ν g_f / ( √2 G_F (Q² + m²) ).

With g_ν g_u = g_ν g_d = 1e-8 and m² = 900 MeV²,

    C^{uV}_{αα} = C^{dV}_{αα} = 1e-8 / ( √2 G_F (Q² + 900 MeV²) ).

Numerically, at the Cs-133 reference points Q² = 1238, 3714, 6190 MeV² this
is about 0.284, 0.131 and 0.086 (same as worked example 01, which describes the
identical parameter point). All off-diagonal entries and all scalar entries
are zero.

## Sign and interference

g_ν g_q = +1e-8 > 0, so the new-physics shift of the nuclear vector charge,
ΔQ = (3Z + 3N) C, is positive while the SM charge Q_SM ≈ 0.04 Z − 0.5 N is
negative. The interference with the Standard Model is therefore destructive.
At this coupling and the lowest reference Q² the shift exceeds |Q_SM|, so the
total charge has flipped sign and the rate is enhanced relative to the SM.
This is just the physics of the described point; nothing was tuned.

## Plugin

`plugin.py` exposes `g_nu`, `g_u`, `g_d` (couplings) and `mass_mev`, with
defaults equal to the described point. Setting all couplings to zero returns
the SM (all new-physics coefficients vanish). All couplings are far below the
perturbativity bound √(4π). The u and d entries are set once each, so no
summation issue with `Coefficients.set` arises.

## Card

`card.json` expresses the same interaction with one vector mediator entry:
m = 30 MeV, diagonal neutrino couplings 1e-4, g_u = g_d = 1e-4,
kinetic_mixing = 0. The schema covers it fully, so the plugin and card should
cross-validate.

## Assumptions

- Dirac or Majorana nature does not matter for a left-handed vector current.
- Anomaly cancellation for a universal U(1)′ is the model author's concern
  and not addressed here.
