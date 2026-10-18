# Notes: light vector boson coupled to neutrinos and u, d quarks

## Description as completed by the round-2 answers

Round 1 asked for the mediator mass, the neutrino couplings and the per-quark
couplings, since "a new light vector boson that couples to neutrinos" fixed only
the Lorentz structure. The round-2 free-text answer supplies everything:

- mass m = 30 MeV
- coupling 1e-4 to all three left-handed neutrino flavors (flavor-diagonal,
  universal, no off-diagonal entries)
- the same coupling 1e-4 to both u and d quarks
- "nothing else": no kinetic mixing, no scalar component, no other fermions

The interaction is therefore fully determined and no further questions are
needed. It coincides with worked example 01 (flavor-universal Z′).

## Lagrangian and matching

    L ⊃ Z′_μ ( g_ν Σ_α ν̄_α γ^μ P_L ν_α + g_u ū γ^μ u + g_d d̄ γ^μ d ),
    g_ν = g_u = g_d = 1e-4,   m = 30 MeV.

Structure `V`. Tree-level Z′ exchange (conventions section 3):

    C^{qV}_{αα}(Q²) = g_ν g_q / ( √2 G_F (Q² + m²) ),   q ∈ {u, d},  α ∈ {e, μ, τ}.

No convention conversion is required: the answer gives the coupling to the
left-handed neutrino current directly (g_ν = 1e-4, not g_ν = 2 × 1e-4), and it
gives the quark vector coupling per quark, so no chiral averaging is needed and
no axial part is dropped. Off-diagonal flavor entries and all scalar entries are
zero.

## Plugin and card

`plugin.py` exposes `g_nu`, `g_u`, `g_d` as couplings (all 1e-4) and `mass_mev`
= 30. With the couplings zeroed every coefficient vanishes, reproducing the SM.
Keeping g_u and g_d as separate parameters preserves the described point exactly
while leaving the per-quark freedom that the conventions ask for. All couplings
are far below the perturbativity bound √(4π).

`card.json` expresses the same interaction with a single V mediator entry,
`kinetic_mixing` = 0 and no contacts, so the cross-validated label applies.

## Sign and size

g_ν g_q = +1e-8 > 0, so the new-physics shift to the nuclear vector charge,
ΔQ = (3Z + 3N) C^{V}, is positive, while the SM charge Q_SM ≈ 0.04 Z − 0.5 N is
negative: the interference is destructive. For Cs-133 at the first reference
point (Q² = 1238 MeV²), C ≈ 0.2836 and ΔQ ≈ 113, exceeding |Q_SM| ≈ 37, so the
total charge changes sign and the coherent rate is well above the SM at this
point (example 01 quotes a factor of about 4.3). This is a consequence of the
described point and has not been adjusted.
