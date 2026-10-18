# Notes: universal vector Z′, m = 30 MeV, g′ = 8e-5, muon antineutrinos

## Reading of the description

- Mediator: a single vector boson Z′, so Lorentz structure `V`. No scalar, axial or tensor part is described.
- Mass: m = 30 MeV, given directly.
- Neutrino coupling: "the same coupling applies to all left-handed neutrino flavors", so the coupling is to the left-handed current ν̄_α γ^μ P_L ν_α with g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = g′ = 8e-5 and no off-diagonal entries. This is the universal (toy) charge assignment of conventions section 7 with Q′_ν = +1.
- Quark coupling: the same g′ to both u and d, so g_u = g_d = g′ = 8e-5 (vector couplings; no chiral split is described, so there is no dropped axial part). "No other couplings" means kinetic mixing is zero and no contact terms.
- Antineutrinos: the beam is ν̄_μ, but the coefficient C is the same for neutrinos and antineutrinos (conventions section 3, "Antineutrinos: the framework handles them; you supply the same C"). The framework selects the ν̄_μ flux and applies the antineutrino kinematics. The ee and tt entries are filled because the description makes the coupling universal; they are inert for a pure ν̄_μ flux and matter only for the other flavor components of the experiment's flux.

## Matching (conventions section 3)

For L ⊃ Z′_μ ( g_ν ν̄_α γ^μ P_L ν_α + g_q q̄ γ^μ q ), tree-level exchange gives

    C^{qV}_{αα}(Q²) = g_ν g_q / ( √2 G_F (Q² + m²) ) = g′² / ( √2 G_F (Q² + m²) ),   q ∈ {u, d},  α ∈ {e, μ, τ}.

No convention conversion was needed: g′ is already the coefficient of the left-handed current and of the quark vector current, with no 1/2 or (1 − γ5) factors to absorb. All off-diagonal and all scalar coefficients are zero.

Reference values at the example-01 grid (Q² = 1238, 3714, 6190 MeV²), obtained from the example-01 numbers scaled by (8e-5 / 1e-4)² = 0.64:

| coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
|---|---|---|---|
| C^{uV}_αα = C^{dV}_αα | 1.8147e-01 | 8.4090e-02 | 5.4724e-02 |

## Sign and interference

g_ν g_q = +g′² > 0, so the new-physics nuclear charge shift ΔQ = 3(Z + N) C is positive while the SM charge Q_SM ≈ 0.04 Z − 0.5 N is negative: destructive interference. At g′ = 8e-5 and the first reference point, ΔQ ≈ 72.4 against Q_SM ≈ −36.9 (tree level), so the total charge has flipped sign and its magnitude is close to the SM one. This coupling sits near the degenerate point of the universal-Z′ limit band noted in example 01 (ΔQ ≈ −2 Q_SM near g′ ≈ 8.1e-5); the described point g′ = 8e-5 is kept exactly as given and was not adjusted.

## Card

Card-expressible: yes, one `V` mediator with mass 30 MeV, diagonal neutrino couplings 8e-5, quark couplings u = d = 8e-5, kinetic mixing 0 and no contacts. `card.json` is written alongside `plugin.py`; the plugin computes the same coefficients from the exported constants without reading the card.

## Sanity

- |g′| = 8e-5 ≪ √(4π): perturbative.
- With `g_prime` set to zero the plugin returns all-zero coefficients, so the SM limit is recovered.
- Anomaly cancellation for the universal U(1)′ is not addressed by the description (the universal assignment is anomalous as a gauge symmetry); this is the model author's concern and does not affect the translation.
