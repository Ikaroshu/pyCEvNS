# Example 01: flavor-universal vector mediator

Description given to the agent: "A new vector boson Z′ of mass 30 MeV that couples with the same strength g′ = 1e-4 to all three left-handed neutrino flavors and to up and down quarks."

## Derivation

1. Lorentz structure: vector exchange, so structure `V`.
2. Couplings as written: g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = g′, off-diagonal zero; g_u = g_d = g′.
3. Matching (conventions section 3): C^{uV}_{αα} = C^{dV}_{αα} = g′² / (√2 G_F (Q² + m²)).
4. Sign: g_ν g_q = +g′² > 0, so the nuclear charge shift ΔQ = (3Z + 3N) C is positive while Q_SM is negative: destructive interference. At the first reference point ΔQ = 113.1, larger than |Q_SM| = 36.9, so the total charge has changed sign and the rate is about (76.2/36.9)² ≈ 4.3 times the SM there. Two couplings are worth knowing at this Q² (tree level; with radiative corrections in parentheses): exact cancellation of the charge, hence almost no coherent rate, near g′ ≈ 5.71e-05 (5.82e-05), which is excluded; and ΔQ = −2 Q_SM near g′ ≈ 8.08e-05 (8.23e-05), where the rate equals the SM again. The second is the origin of the degenerate band in universal-Z′ limits.
5. Card-expressible: yes, one mediator entry.

Reference points: Q² = 2 M T for Cs-133 (M = 123.8 GeV) at T = 5, 15, 25 keV, i.e. Q² = 1238, 3714, 6190 MeV². SM tree-level nuclear charge for Cs-133: Q_SM = -36.93 (negative; with the framework's radiative corrections it is -38.28). Tabulated values are rounded to five digits for reading; eval expectations are generated at full precision from the exported constants.

## Expected coefficients

| coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
|---|---|---|---|
| C^{uV}_ee = C^{uV}_mm = C^{uV}_tt | 2.8355e-01 | 1.3139e-01 | 8.5506e-02 |
| C^{dV}_ee = C^{dV}_mm = C^{dV}_tt | 2.8355e-01 | 1.3139e-01 | 8.5506e-02 |

All off-diagonal entries and all scalar entries are zero.

## card.json

```json
{
  "name": "zprime_universal_example",
  "mediators": [
    {
      "structure": "V",
      "mass_mev": 30.0,
      "nu_couplings": {
        "ee": 0.0001,
        "mm": 0.0001,
        "tt": 0.0001,
        "em": 0,
        "et": 0,
        "mt": 0
      },
      "quark_couplings": {
        "u": 0.0001,
        "d": 0.0001
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

PARAMETERS = {"g_prime": 1e-4, "mass_mev": 30.0}
COUPLINGS = ("g_prime",)

def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    value = params["g_prime"] ** 2 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)
    return c
```

## notes.md (what the agent should record)

Structure V, with the left-handed neutrino coupling taken as written. Matching (conventions section 3): $C^{qV}_{\alpha\alpha} = g'^2 / (\sqrt{2} G_F (Q^2 + m^2))$ for $q = u, d$. Since $g_\nu g_q > 0$ the interference with the SM is destructive. No convention conversion is needed.
