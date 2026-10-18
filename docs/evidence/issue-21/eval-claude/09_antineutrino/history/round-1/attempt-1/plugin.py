"""Flavor-universal vector Z' (m = 30 MeV, g' = 8e-5) coupled to left-handed neutrinos and to u, d quarks."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_prime": 8e-5, "mass_mev": 30.0}
COUPLINGS = ("g_prime",)


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    g_nu = params["g_prime"]
    g_q = params["g_prime"]
    value = g_nu * g_q / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)
    return c
