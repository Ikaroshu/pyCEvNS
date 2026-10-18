# Example 07: a description whose form is open, then assumed values

Description given to the agent: "A new light scalar that couples to electron neutrinos."

## What the description fixes and what it leaves open

1. Lorentz structure: scalar exchange, so structure `S` (conventions section 4). Fixed.
2. Neutrino side: only the ee entry. Fixed.
3. Quark side: not stated. Whether the scalar couples to u, d or both, and with which relative couplings, changes the nuclear scalar charge; it is part of the form of the interaction, not a number to assume. This is a question.
4. Mass and coupling values: not given. These are numbers, so they are never asked for; once the form is fixed they take the assumed defaults of conventions section 12.

Round 1 is therefore a questions outcome: write `questions.json` and `notes.md`, and no `plugin.py` or `card.json`.

## questions.json

```json
{
  "questions": [
    {
      "id": "quark_couplings",
      "question": "Which quarks does the scalar couple to, and with which relative couplings?",
      "unit": null,
      "suggested": "equal scalar couplings to u and d quarks",
      "reason": "Equal u and d couplings are the usual benchmark for a light scalar; the description names no quarks."
    }
  ]
}
```

The question asks about form only. Its `suggested` names a form, its `reason` says where the suggestion comes from, and the suggestion is never treated as part of the description.

## notes.md for round 1

Structure S with only the $ee$ neutrino entry, as stated. Open: the quark side of the interaction, which fixes the nuclear scalar charge $Q_S = g_\nu (Z c_p + N c_n)/(Q^2 + m^2)$ through $c_N = m_N \sum_q g_q f_q^N / m_q$. The mass and the coupling values will take the assumed defaults once the form is known.

## After the answer

The answer "equal scalar couplings to u and d quarks" arrives in `TASK.md` under `## Answers (round 2)` and is then part of the description. The form is now fixed: $g_\nu^{ee} = g_\nu$, $g_u = g_d = g_q$, and every other coupling zero. The mass and couplings take the defaults of conventions section 12 (10 MeV, and 1e-5 for each scalar coupling), and the coefficient is

$$C^{qS}_{ee}(Q^2) = \frac{g_\nu g_q}{\sqrt{2} G_F (Q^2 + m^2)}, \qquad q = u, d.$$

```python
import numpy as np
from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"mass_mev": 10.0, "g_nu": 1e-5, "g_q": 1e-5}
COUPLINGS = ("g_nu", "g_q")

def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    value = params["g_nu"] * params["g_q"] / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        c.set("S", quark, "ee", value)
    return c
```

The card has one `S` mediator at 10 MeV with ν coupling `ee` = 1e-5 and quark couplings u = d = 1e-5, as in example 04.

## notes.md for round 2

Structure S with the $ee$ neutrino entry and equal $u$ and $d$ couplings, from the description and the answer. Matching (conventions section 4): $C^{qS}_{ee} = g_\nu g_q / (\sqrt{2} G_F (Q^2 + m^2))$. A Dirac $\bar\nu\nu$ coupling is assumed and stated. The scalar term does not interfere with the SM.

The assumed mass and couplings appear only in `PARAMETERS`, not in the notes.
