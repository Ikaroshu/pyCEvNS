# Notes: gauged L_μ−L_τ Z′ with full one-loop (Q²-dependent) photon mixing

## Description point

- Mediator: vector Z′ of U(1)_{L_μ−L_τ}, mass m = 12 MeV, gauge coupling g′ = 8e-5.
- No tree-level kinetic mixing, no direct quark coupling. The only coupling to quarks is the one-loop γ–Z′ mixing from the muon and tau loops, kept with its full spacelike Q² dependence.
- Parameter point preserved exactly in `PARAMETERS = {"g_prime": 8e-5, "mass_mev": 12.0}`; `COUPLINGS = ("g_prime",)`.

## Lorentz structure and charges

Structure `V` (vector exchange). L_μ−L_τ charges (conventions section 7): Q′_{ν_e} = 0, Q′_{ν_μ} = +1, Q′_{ν_τ} = −1, Q′_u = Q′_d = 0. The neutrino coupling is left-handed, g_ν^{αα} = g′ Q′_{ν_α}, flavor-diagonal. The charged leptons carry the same charges as their doublet partners: Q′_μ = +1, Q′_τ = −1, with electric charge Q_ℓ = −1.

## Loop-induced mixing

A charged lepton ℓ with couplings e Q_ℓ (photon) and g′ Q′_ℓ (Z′) gives a vacuum-polarization-type γ–Z′ two-point function proportional to ∫₀¹ dx x(1−x) ln[(m_ℓ² + x(1−x)Q²)/μ²] for spacelike q² = −Q². Summing over μ and τ, the scheme-dependent, mass-independent piece cancels because Σ_ℓ Q′_ℓ Q_ℓ = 0, leaving a finite, scale-independent result:

    ε(Q²) = (e g′ / 2π²) ∫₀¹ dx x(1−x) ln[ (m_τ² + x(1−x) Q²) / (m_μ² + x(1−x) Q²) ].

Reduction check: at Q² → 0 the integral is (1/6) ln(m_τ²/m_μ²) = (1/3) ln(m_τ/m_μ), so ε(0) = e g′ ln(m_τ/m_μ)/(6π²) = κ g′ with κ = 0.0144417637 (conventions section 6), giving ε(0) = 1.1553411e-06 for g′ = 8e-5. The plugin reproduces the constant-mixing card of a `zprime_lmu_ltau` model with `kinetic_mixing = κ g′` in that limit.

Sign: positive throughout, as fixed in conventions section 6 (the ln(m²) term carries the sign that makes the running electromagnetic coupling grow with Q²; the lighter muon dominates and has Q′ = +1). For ν_μ this makes the new-physics proton coupling the same sign as the ν_μ coupling: destructive interference with the SM for ν_μ, constructive for ν_τ (whose charge is −1). The Q² dependence is mild over the COHERENT range (Q² ≲ 0.6 m_μ²), a slow decrease of ε at the percent level.

## Matching

With g_q(Q²) = ε(Q²) e Q_q^{em} (Q_u = +2/3, Q_d = −1/3, e = 0.303) and conventions section 3,

    C^{qV}_{αα}(Q²) = g′ Q′_{ν_α} · ε(Q²) e Q_q^{em} / ( √2 G_F (Q² + m²) ),

so C^{qV}_{mm} = −C^{qV}_{tt}, C^{qV}_{ee} = 0, all off-diagonal and scalar entries zero. The up-quark entries are positive for mm and the down-quark entries negative (ratio −1/2).

## Numerics

The Feynman-parameter integrand is smooth on [0, 1]; 40-point Gauss–Legendre on [0, 1] reproduces the exact integral to far better than 1e-6 relative (as in worked example 06). All constants (e, m_μ, m_τ, Q_EM, G_F, √2) are the exported ones from `nuagent.physics.couplings`.

## Why no card.json

The card's `kinetic_mixing` field is a constant ε. The description explicitly asks for the full Q² dependence of the loop mixing, which a card cannot carry, so this is a plugin-only translation. No tree-level mixing or direct quark coupling was requested, so none is added.

## Convention conversions

None needed: the description gives g′ and the mass directly in the framework's normalization. Perturbativity: |g′| = 8e-5 ≪ √(4π).
