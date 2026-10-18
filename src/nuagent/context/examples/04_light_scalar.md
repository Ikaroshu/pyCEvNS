# Example 04: light scalar mediator

Description given to the agent: "A light scalar φ of mass 10 MeV with a Yukawa coupling y = 1e-5 to all neutrino flavors and the same coupling y to up and down quarks."

## Derivation

1. Structure `S`. "Yukawa coupling to neutrinos" is read as y ν̄_α ν_α (Dirac, flavor-diagonal); recorded as an assumption.
2. g_ν^{αα} = y, g_u = g_d = y.
3. Scalar coefficient (section 4): C^{qS}_{αα} = y² / (√2 G_F (Q² + m²)).
4. Nucleon charges, built by the framework: c_p = m_p y (f_u^p/m_u + f_d^p/m_d) = 17.08 y, c_n = 17.09 y; nuclear Q_S = y (Z c_p + N c_n)/(Q² + m²). The scalar term does not interfere with the SM.
5. Card-expressible: yes.

Reference points: Q² = 2 M T for Cs-133 (M = 123.8 GeV) at T = 5, 15, 25 keV, i.e. Q² = 1238, 3714, 6190 MeV². SM tree-level nuclear charge for Cs-133: Q_SM = -36.93 (negative; with the framework's radiative corrections it is -38.28). Tabulated values are rounded to five digits for reading; eval expectations are generated at full precision from the exported constants.

## Expected coefficients

| coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
|---|---|---|---|
| C^{uS}_αα = C^{dS}_αα | 4.5309e-03 | 1.5895e-03 | 9.6382e-04 |
| Q_S^αα (MeV⁻², framework-derived) | 1.6980e-10 | 5.9570e-11 | 3.6121e-11 |

Vector entries are zero.

## card.json

```json
{
  "name": "light_scalar_example",
  "mediators": [
    {
      "structure": "S",
      "mass_mev": 10.0,
      "nu_couplings": {
        "ee": 1e-05,
        "mm": 1e-05,
        "tt": 1e-05,
        "em": 0,
        "et": 0,
        "mt": 0
      },
      "quark_couplings": {
        "u": 1e-05,
        "d": 1e-05
      }
    }
  ],
  "contacts": []
}
```

## plugin.py

```python
import numpy as np
from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"y": 1e-5, "mass_mev": 10.0}
COUPLINGS = ("y",)

def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    value = params["y"] ** 2 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        for pair in ("ee", "mm", "tt"):
            c.set("S", quark, pair, value)
    return c
```

## notes.md

Scalar structure; a Dirac $\bar\nu \nu$ coupling is assumed and stated. Matching (conventions section 4): $C^{qS}_{\alpha\alpha} = y^2 / (\sqrt{2} G_F (Q^2 + m^2))$. The quark-level couplings are given, so the nucleon charges are left to the framework. There is no interference with the SM.
