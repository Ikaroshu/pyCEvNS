# Example 03: L_μ−L_τ gauge boson with kinetic mixing

Description A (explicit mixing): "A gauged L_μ−L_τ Z′ with mass 20 MeV, gauge coupling 5e-5 and kinetic mixing with the photon of 2e-3."
Description B (loop mixing): "A gauged L_μ−L_τ Z′ with mass 10 MeV and gauge coupling 1e-4; quark couplings arise only through the loop-induced kinetic mixing with the photon."

## Derivation

1. Structure `V`.
2. Charges (conventions section 7): Q′_{ν_μ} = +1, Q′_{ν_τ} = −1, Q′_{ν_e} = 0, Q′_q = 0. So g_ν^{mm} = +g′, g_ν^{tt} = −g′, g_ν^{ee} = 0; no direct quark coupling.
3. Kinetic mixing (section 6): g_u = ε e (2/3), g_d = ε e (−1/3). For A, ε = 2e-3 as given. For B, ε = κ g′ with κ = 0.0144417637, so ε = 1.4441763732e-06; the sign is positive, as derived in section 6.
4. Matching: C^{uV}_{mm} = g′ ε e (2/3)/(√2 G_F (Q² + m²)), C^{dV}_{mm} = −g′ ε e (1/3)/(√2 G_F (Q² + m²)); the ττ entries are the negatives of the μμ entries; ee entries are zero.
5. Physics check: only protons are affected (2 C^u + C^d ∝ 2·(2/3) − 1/3 = 1, C^u + 2 C^d ∝ 2/3 − 2/3 = 0), as expected for photon-mediated mixing. For ν_μ the shift ΔQ = Z g′ ε e/(√2 G_F (Q² + m²)) is positive when ε g′ > 0, opposite to Q_SM: destructive. Electron neutrinos see no new physics; the COHERENT prompt ν_μ and delayed ν̄_μ do.
6. Card-expressible: yes. In a card the mixing is always written explicitly; there is no default. Only the CLI template `zprime_lmu_ltau` fills κ g′ when the field is omitted.

Reference points: Q² = 2 M T for Cs-133 (M = 123.8 GeV) at T = 5, 15, 25 keV, i.e. Q² = 1238, 3714, 6190 MeV². SM tree-level nuclear charge for Cs-133: Q_SM = -36.93 (negative; with the framework's radiative corrections it is -38.28). Tabulated values are rounded to five digits for reading; eval expectations are generated at full precision from the exported constants.

## Expected coefficients, description A (ε = 2e-3, m = 20 MeV, g′ = 5e-5)

| coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
|---|---|---|---|
| C^{uV}_mm = −C^{uV}_tt | 7.4762e-01 | 2.9767e-01 | 1.8583e-01 |
| C^{dV}_mm = −C^{dV}_tt | -3.7381e-01 | -1.4883e-01 | -9.2914e-02 |

## Expected coefficients, description B (ε = κ g′ = 1.444176e-06, m = 10 MeV, g′ = 1e-4)

| coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
|---|---|---|---|
| C^{uV}_mm = −C^{uV}_tt | 1.3218e-03 | 4.6370e-04 | 2.8117e-04 |
| C^{dV}_mm = −C^{dV}_tt | -6.6089e-04 | -2.3185e-04 | -1.4058e-04 |

## card.json (description A)

```json
{
  "name": "zprime_lmu_ltau_example_a",
  "mediators": [
    {
      "structure": "V",
      "mass_mev": 20.0,
      "nu_couplings": {
        "ee": 0,
        "mm": 5e-05,
        "tt": -5e-05,
        "em": 0,
        "et": 0,
        "mt": 0
      },
      "quark_couplings": {
        "u": 0.0,
        "d": 0.0
      },
      "kinetic_mixing": 0.002
    }
  ],
  "contacts": []
}
```

## card.json (description B)

```json
{
  "name": "zprime_lmu_ltau_example_b",
  "mediators": [
    {
      "structure": "V",
      "mass_mev": 10.0,
      "nu_couplings": {
        "ee": 0,
        "mm": 0.0001,
        "tt": -0.0001,
        "em": 0,
        "et": 0,
        "mt": 0
      },
      "quark_couplings": {
        "u": 0.0,
        "d": 0.0
      },
      "kinetic_mixing": 1.4441763732211533e-06
    }
  ],
  "contacts": []
}
```

The B card carries ε at full precision so the plugin-versus-card cross-check passes at 1e-6; a rounded value such as 1.444e-6 fails it.

## plugin.py (description A)

```python
import numpy as np
from nuagent.physics.couplings import Coefficients, GF, SQRT2, E_CHARGE

PARAMETERS = {"g_prime": 5e-5, "mass_mev": 20.0, "kinetic_mixing": 2e-3}
COUPLINGS = ("g_prime", "kinetic_mixing")
Q_EM = {"u": 2.0 / 3.0, "d": -1.0 / 3.0}

def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    eps = params["kinetic_mixing"]
    prop = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        g_q = eps * E_CHARGE * Q_EM[quark]
        c.set("V", quark, "mm", +g * g_q * prop)
        c.set("V", quark, "tt", -g * g_q * prop)
    return c
```

## plugin.py (description B)

```python
import numpy as np
from nuagent.physics.couplings import Coefficients, GF, SQRT2, E_CHARGE, M_MU, M_TAU

PARAMETERS = {"g_prime": 1e-4, "mass_mev": 10.0}
COUPLINGS = ("g_prime",)
Q_EM = {"u": 2.0 / 3.0, "d": -1.0 / 3.0}
KAPPA = E_CHARGE * np.log(M_TAU / M_MU) / (6 * np.pi ** 2)

def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    eps = KAPPA * g
    prop = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        g_q = eps * E_CHARGE * Q_EM[quark]
        c.set("V", quark, "mm", +g * g_q * prop)
        c.set("V", quark, "tt", -g * g_q * prop)
    return c
```

In B the mixing is derived from g′, so `g_prime` is the only coupling and setting it to zero removes everything, which is what the SM-limit check needs.

## notes.md

Charges of L_μ−L_τ; quark coupling only via ε e Q^em; proton-only coupling noted; destructive interference for ν_μ; for B, ε = κ g′ with the sign from the loop (section 6) and the value written at full precision in the card.
