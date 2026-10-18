# Example 02: B−L gauge boson

Description given to the agent: "The gauge boson of U(1)_{B−L} with gauge coupling g′ = 1e-4 and mass 10 MeV."

## Derivation

1. Structure `V` (gauge boson).
2. Charges (conventions section 7): Q′ = −1 for every lepton doublet, +1/3 for every quark. So g_ν^{αα} = −g′ for α = e, μ, τ; g_u = g_d = +g′/3.
3. Matching: C^{qV}_{αα} = (−g′)(g′/3) / (√2 G_F (Q² + m²)) = −g′² / (3√2 G_F (Q² + m²)).
4. Sign: g_ν g_q < 0, so ΔQ = (3Z + 3N) C = −(Z + N) g′²/(√2 G_F (Q² + m²)) is negative, the same sign as Q_SM: constructive interference, no cancellation region. At the first reference point ΔQ = -60.262.
5. Card-expressible: yes. Write the mediator entry with the signed couplings (the `zprime_u1` template does the same from charges when run through the CLI).

Reference points: Q² = 2 M T for Cs-133 (M = 123.8 GeV) at T = 5, 15, 25 keV, i.e. Q² = 1238, 3714, 6190 MeV². SM tree-level nuclear charge for Cs-133: Q_SM = -36.93 (negative; with the framework's radiative corrections it is -38.28). Tabulated values are rounded to five digits for reading; eval expectations are generated at full precision from the exported constants.

## Expected coefficients

| coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
|---|---|---|---|
| C^{uV}_αα = C^{dV}_αα (α = e, μ, τ) | -1.5103e-01 | -5.2984e-02 | -3.2127e-02 |

## card.json

```json
{
  "name": "zprime_b_minus_l_example",
  "mediators": [
    {
      "structure": "V",
      "mass_mev": 10.0,
      "nu_couplings": {
        "ee": -0.0001,
        "mm": -0.0001,
        "tt": -0.0001,
        "em": 0,
        "et": 0,
        "mt": 0
      },
      "quark_couplings": {
        "u": 3.3333333333333335e-05,
        "d": 3.3333333333333335e-05
      },
      "kinetic_mixing": 0.0
    }
  ],
  "contacts": []
}
```

## plugin.py

```python
import numpy as np
from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_prime": 1e-4, "mass_mev": 10.0}
COUPLINGS = ("g_prime",)
Q_NU, Q_QUARK = -1.0, 1.0 / 3.0

def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    value = (g * Q_NU) * (g * Q_QUARK) / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)
    return c
```

## notes.md

Charges from the lepton doublet and the quark B−L numbers: $g_\nu = -g'$, $g_u = g_d = g'/3$. The relative sign between the neutrino and quark couplings is physical and kept, so the interference is constructive. Common mistake to avoid: dropping the sign of $Q'_\nu$, which turns constructive into destructive interference.
