# Example 06: L_μ−L_τ with the full one-loop, Q²-dependent kinetic mixing (plugin-only)

Description given to the agent: "The L_μ−L_τ Z′ of example 03 B (mass 10 MeV, g′ = 1e-4), but keep the full one-loop momentum dependence of the photon mixing instead of its Q² → 0 limit."

## Derivation

1. Structure `V`, charges as in example 03.
2. The muon and tau loops give, for spacelike Q²,

       ε(Q²) = (e g′ / 2π²) ∫₀¹ dx x(1−x) ln[ (m_τ² + x(1−x) Q²) / (m_μ² + x(1−x) Q²) ],

   which reduces to κ g′ = e g′ ln(m_τ/m_μ)/(6π²) = 1.444176e-06 at Q² → 0 (the integral of x(1−x) is 1/6) and falls slowly with Q² because Q² reaches a fraction of m_μ² = 11164 MeV² over the COHERENT recoil range. The effect is at the percent level, which is exactly the kind of correction a card cannot carry: ε(Q²)/ε(0) = 0.99613, 0.98866, 0.98150 at the reference points. The sign is positive throughout, as in section 6.
3. Quark couplings g_q(Q²) = ε(Q²) e Q_q^em; matching as in example 03 with ε replaced by ε(Q²).
4. Card-expressible: no. A card's `kinetic_mixing` is a constant. This is a plugin-only case and the report will be labeled `plugin-only`.
5. Built-in cross-check the plugin must satisfy: at Q² → 0 its coefficients equal those of example 03 B. A plugin for a plugin-only case should state such a reduction in `notes.md` so a reviewer can verify it.
6. Quadrature: the integrand is smooth on [0, 1], so 40-point Gauss-Legendre reproduces the exact integral to better than 1e-14 relative; a trapezoid rule that drops the endpoints is low by about 2e-6 and fails the 1e-6 eval tolerance.

Reference points: Q² = 2 M T for Cs-133 (M = 123.8 GeV) at T = 5, 15, 25 keV, i.e. Q² = 1238, 3714, 6190 MeV². SM tree-level nuclear charge for Cs-133: Q_SM = -36.93 (negative; with the framework's radiative corrections it is -38.28). Tabulated values are rounded to five digits for reading; eval expectations are generated at full precision from the exported constants.

## Expected coefficients

| coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
|---|---|---|---|
| C^{uV}_mm = −C^{uV}_tt | 1.3167e-03 | 4.5844e-04 | 2.7597e-04 |
| C^{dV}_mm = −C^{dV}_tt | -6.5833e-04 | -2.2922e-04 | -1.3798e-04 |

## plugin.py

```python
import numpy as np
from nuagent.physics.couplings import Coefficients, GF, SQRT2, E_CHARGE, M_MU, M_TAU

PARAMETERS = {"g_prime": 1e-4, "mass_mev": 10.0}
COUPLINGS = ("g_prime",)
Q_EM = {"u": 2.0 / 3.0, "d": -1.0 / 3.0}
_T, _W = np.polynomial.legendre.leggauss(40)
_X = (_T + 1.0) / 2.0          # nodes on [0, 1]
_WX = _W / 2.0                 # weights on [0, 1]

def loop_mixing(q2: np.ndarray, g: float) -> np.ndarray:
    q = np.asarray(q2, dtype=float)[:, None]
    x = _X[None, :]
    integrand = x * (1 - x) * np.log((M_TAU ** 2 + x * (1 - x) * q) / (M_MU ** 2 + x * (1 - x) * q))
    return E_CHARGE * g / (2 * np.pi ** 2) * (integrand * _WX[None, :]).sum(axis=1)

def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    eps = loop_mixing(q2, g)
    prop = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        g_q = eps * E_CHARGE * Q_EM[quark]
        c.set("V", quark, "mm", +g * g_q * prop)
        c.set("V", quark, "tt", -g * g_q * prop)
    return c
```

## notes.md

Why no card: Q²-dependent mixing. Reduction check: at Q² = 0 the plugin reproduces example 03 B. 40-point Gauss-Legendre over the Feynman parameter. Sign positive from the loop, as in section 6.
