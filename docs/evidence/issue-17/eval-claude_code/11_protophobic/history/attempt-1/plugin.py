"""Protophobic vector mediator; derivation in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_nu": 1e-4, "g_p": 0.0, "g_n": 3e-4, "mass_mev": 20.0}
COUPLINGS = ("g_nu", "g_p", "g_n")


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    g_nu = params["g_nu"]
    g_p = params["g_p"]
    g_n = params["g_n"]
    g_u = (2.0 * g_p - g_n) / 3.0
    g_d = (2.0 * g_n - g_p) / 3.0
    propagator = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark, g_q in (("u", g_u), ("d", g_d)):
        value = g_nu * g_q * propagator
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)
    return c
