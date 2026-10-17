"""Flavor-universal vector Z' (45 MeV, g' = 7e-5) coupled to left-handed neutrinos and to u, d quarks."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_prime": 7e-5, "mass_mev": 45.0}
COUPLINGS = ("g_prime",)


def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    g_nu = params["g_prime"]
    g_u = params["g_prime"]
    g_d = params["g_prime"]
    propagator = 1.0 / (q2 + params["mass_mev"] ** 2)
    value_u = g_nu * g_u * propagator / (SQRT2 * GF)
    value_d = g_nu * g_d * propagator / (SQRT2 * GF)
    for pair in ("ee", "mm", "tt"):
        c.set("V", "u", pair, value_u)
        c.set("V", "d", pair, value_d)
    return c
