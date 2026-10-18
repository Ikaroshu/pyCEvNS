# Light vector boson coupled to neutrinos and quarks

## Description as completed by the round-2 answers

The original description ("a new light vector boson that couples to neutrinos") fixed only the Lorentz structure. The round-2 answers supply the rest and are treated as part of the description:

| input | value |
|---|---|
| mediator mass m | 30.0 MeV |
| neutrino coupling g_ν | 1e-4, flavor-diagonal, equal for ee, mm, tt; no off-diagonal entries |
| up-quark coupling g_u | 1e-4 |
| down-quark coupling g_d | 1e-4 |
| kinetic mixing ε | none; quark couplings are direct |

Nothing is left open, so no `questions.json` is written this round.

## Lagrangian

    L ⊃ Z′_μ ( g_ν Σ_α ν̄_α γ^μ P_L ν_α + g_u ū γ^μ u + g_d d̄ γ^μ d ),   m = 30 MeV.

The neutrino coupling is taken as the left-handed current coupling of conventions section 3, with g_ν = 1e-4 exactly as answered (no factor-2 rewriting is needed because the answer is already stated as the coupling g_ν). The quark couplings are vector couplings g_q = (g_L + g_R)/2 as the card schema defines them; no axial part is specified, so none is dropped.

## Matching (conventions section 3)

Tree-level Z′ exchange at spacelike momentum transfer gives

    C^{qV}_{αα}(Q²) = g_ν g_q / ( √2 G_F (Q² + m²) ),   q ∈ {u, d},   α ∈ {e, μ, τ},

with all off-diagonal flavor entries zero and all scalar entries zero. With g_u = g_d = g_ν = 1e-4 the u and d coefficients are identical, and the model coincides numerically with the universal-vector worked example 01. No convention conversion was needed: the answers are given directly in the framework's normalization.

The plugin keeps g_ν, g_u and g_d as three separate parameters so that the described point is preserved exactly while each coupling remains independently adjustable. `COUPLINGS` lists all three; zeroing them returns the Standard Model.

## Sign and interference

g_ν g_q = +1e-8 > 0, so the new-physics nuclear charge shift ΔQ = (3Z + 3N) C is positive, while the SM charge Q_SM ≈ 0.04 Z − 0.5 N is negative. The interference is destructive, and at the described point the shift exceeds |Q_SM| over the COHERENT recoil range, so the total charge changes sign (as in example 01, where the same numbers give ΔQ ≈ 113 against |Q_SM| ≈ 37 at Q² = 1238 MeV² for Cs-133). This is a property of the described point and is reported, not adjusted.

## Reference values

For Cs-133 at Q² = 1238, 3714, 6190 MeV² the plugin gives C^{uV}_{αα} = C^{dV}_{αα} ≈ 2.8355e-01, 1.3139e-01, 8.5506e-02, matching the tabulated values of example 01.

## Card

The interaction is a single vector mediator with constant couplings, so the schema expresses it exactly: one `V` mediator entry with mass 30 MeV, diagonal neutrino couplings 1e-4, quark couplings u = d = 1e-4, and `kinetic_mixing` 0.0 written explicitly because the answers state there is none.

## Sanity

All couplings are 1e-4 ≪ √(4π). The mass is positive. The propagator is 1/(Q² + m²) as required.
