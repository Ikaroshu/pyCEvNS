# Example 05: contact NSI with a flavor-changing entry

Description given to the agent: "Vector NSI on up quarks with ε^{uV}_{ee} = 0.1 and ε^{uV}_{eμ} = 0.05, all other epsilons zero."

## Derivation

1. Structure `V`, contact form: the description gives epsilons directly, so C^{uV}_{ee} = 0.1 and C^{uV}_{eμ} = 0.05 at every Q²; C^{uV}_{μe} = 0.05 by Hermiticity; all d-quark and other entries zero.
2. The ee entry interferes with the SM: for ν_e, Q^{ee} = Q_SM + Z·2·0.1 + N·0.1 = Q_SM + 18.8 = -18.13; the shift is positive against a negative Q_SM, destructive.
3. The eμ entry adds incoherently: |Q^{eμ}|² with Q^{eμ} = (2Z + N)·0.05 = 9.4; it also contributes to the ν_μ rate as |Q^{μe}|² with the same magnitude.
4. Card-expressible: yes, through `contacts`.

Reference points: Q² = 2 M T for Cs-133 (M = 123.8 GeV) at T = 5, 15, 25 keV, i.e. Q² = 1238, 3714, 6190 MeV². SM tree-level nuclear charge for Cs-133: Q_SM = -36.93 (negative; with the framework's radiative corrections it is -38.28). Tabulated values are rounded to five digits for reading; eval expectations are generated at full precision from the exported constants.

## Expected coefficients (Q²-independent)

| coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
|---|---|---|---|
| C^{uV}_ee | 1.0000e-01 | 1.0000e-01 | 1.0000e-01 |
| C^{uV}_em = conj(C^{uV}_me) | 5.0000e-02 | 5.0000e-02 | 5.0000e-02 |

## card.json

```json
{
  "name": "contact_nsi_example",
  "mediators": [],
  "contacts": [
    {
      "fermion": "u",
      "epsilon": {
        "ee": 0.1,
        "mm": 0,
        "tt": 0,
        "em": 0.05,
        "et": 0,
        "mt": 0
      }
    }
  ]
}
```

A complex off-diagonal entry is written as `"em": [0.03, 0.04]`.

## plugin.py

```python
import numpy as np
from nuagent.physics.couplings import Coefficients

PARAMETERS = {"eps_u_ee": 0.1, "eps_u_em": 0.05}
COUPLINGS = ("eps_u_ee", "eps_u_em")

def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    c.set("V", "u", "ee", params["eps_u_ee"])
    c.set("V", "u", "em", params["eps_u_em"])   # me is filled as the conjugate
    return c
```

## notes.md

Contact form because epsilons were given; Hermitian partner noted; the eμ entry contributes incoherently to both the ν_e and ν_μ rates.
