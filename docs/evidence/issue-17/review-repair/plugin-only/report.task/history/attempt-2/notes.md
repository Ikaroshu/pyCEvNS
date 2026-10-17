# L_μ−L_τ Z′ (m = 12 MeV, g′ = 8e-5) with full one-loop Q²-dependent photon mixing

## Outcome: plugin-only (no card.json)

The description asks for the *full spacelike Q² dependence* of the muon/tau loop mixing. A model card's `kinetic_mixing` field is a single constant, so no card can represent this interaction. Per `framework.md` this is a legitimate plugin-only translation; the harness label is `plugin-only`.

### Correction relative to attempt 1

Attempt 1 shipped an empty placeholder `card.json` (`{}`), which the validator read as zero quark couplings and reported a plugin/card mismatch on every V u and V d entry. The correct resolution is not to repair the card but to omit it: the Q²-dependent mixing is exactly the case `conventions.md` section 6 and example 06 send to a coefficient plugin. The plugin itself was already correct and is kept unchanged in physics content; only the `Q_EM` import is taken from the exported constants.

## Derivation

1. **Lorentz structure**: vector exchange, structure `V`.
2. **Charges** (conventions section 7, L_μ−L_τ): Q′_{ν_e} = 0, Q′_{ν_μ} = +1, Q′_{ν_τ} = −1, Q′_u = Q′_d = 0. Hence g_ν^{mm} = +g′, g_ν^{tt} = −g′, g_ν^{ee} = 0, all off-diagonal zero. No tree-level kinetic mixing and no direct quark coupling, as the description states.
3. **Loop-induced mixing**: the μ and τ loops give, for spacelike Q² ≥ 0,

       ε(Q²) = (e g′ / 2π²) ∫₀¹ dx x(1−x) ln[ (m_τ² + x(1−x) Q²) / (m_μ² + x(1−x) Q²) ].

   The m-independent, divergent part of the vacuum polarisation cancels because Σ_ℓ Q′_ℓ Q_ℓ = 0. The logarithm's sign is fixed by the physics (running electromagnetic coupling grows with Q²), giving ε > 0 for all Q², as in section 6. At Q² → 0 the integral of x(1−x) is 1/6 and ε → κ g′ with κ = e ln(m_τ/m_μ)/(6π²) = 0.0144417637, i.e. ε(0) = 1.1553411e-06 for g′ = 8e-5.
4. **Quark couplings**: g_q(Q²) = ε(Q²) e Q_q^em, with Q_u = +2/3, Q_d = −1/3 taken from the exported `Q_EM`.
5. **Matching** (conventions section 3):

       C^{uV}_{mm}(Q²) = + g′ ε(Q²) e (2/3) / (√2 G_F (Q² + m²)),
       C^{dV}_{mm}(Q²) = − g′ ε(Q²) e (1/3) / (√2 G_F (Q² + m²)),
       C^{qV}_{tt} = − C^{qV}_{mm},   C^{qV}_{ee} = 0,   off-diagonal = 0,   scalar = 0.

   Numerical values: m = 12 MeV, g′ = 8e-5, no convention conversion needed (g′ is the gauge coupling multiplying the L_μ−L_τ current, so g_ν = g′ Q′).
6. **Physics check**: 2 C^u + C^d ∝ 2(2/3) − 1/3 = 1 and C^u + 2 C^d ∝ 2/3 − 2/3 = 0, so only protons couple, as expected for photon-mediated mixing. For ν_μ the shift ΔQ = Z g′ ε e/(√2 G_F (Q² + m²)) is positive while Q_SM < 0: destructive interference for the COHERENT prompt ν_μ and delayed ν̄_μ. ν_e sees no new physics.
7. **Q² dependence**: over the COHERENT recoil range Q² reaches a fraction of m_μ² = 11164 MeV², so ε(Q²)/ε(0) falls at the percent level (0.99613, 0.98866, 0.98150 at Q² = 1238, 3714, 6190 MeV², independent of g′ and m). This is the correction a constant card cannot carry.
8. **Quadrature**: the integrand is smooth on [0, 1]; 40-point Gauss-Legendre reproduces the integral to better than 1e-14 relative. A trapezoid rule dropping the endpoints would be low by about 2e-6 and would fail the 1e-6 tolerance.

## Reduction check a reviewer can verify

Replacing `loop_mixing(q2, g)` by the constant `KAPPA_LMU_LTAU * g` (the Q² → 0 limit) must reproduce the example 03 B plugin with m = 12 MeV and g′ = 8e-5. Equivalently, the plugin's coefficients equal the constant-mixing ones multiplied by the g′-independent ratio ε(Q²)/ε(0) listed above.

## Plugin parameters

`PARAMETERS = {"g_prime": 8e-5, "mass_mev": 12.0}`, `COUPLINGS = ("g_prime",)`. The mixing is derived from g′, so zeroing g′ removes every new-physics coefficient (SM limit). |g′| = 8e-5 ≪ √(4π): perturbative.
