"""Universal vector mediator with dipole-form-factor quark couplings (plugin-only).

Derivation and the reason no card exists are in notes.md.
"""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_prime": 1e-4, "mass_mev": 30.0, "lambda_mev": 200.0}
COUPLINGS = ("g_prime",)


def dipole_form_factor(q2: np.ndarray, lambda_mev: float) -> np.ndarray:
    """F(Q^2) = 1 / (1 + Q^2 / Lambda^2)^2, equal to 1 at Q^2 = 0."""
    return 1.0 / (1.0 + np.asarray(q2, dtype=float) / lambda_mev**2) ** 2


def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    g_nu = g
    g_q = g * dipole_form_factor(q2, params["lambda_mev"])
    value = g_nu * g_q / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)
    return c
