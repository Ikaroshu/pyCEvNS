"""Flavor-universal light vector mediator; derivation in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_nu": 1e-4, "g_u": 1e-4, "g_d": 1e-4, "mass_mev": 30.0}
COUPLINGS = ("g_nu", "g_u", "g_d")


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    propagator = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    quark_couplings = {"u": params["g_u"], "d": params["g_d"]}
    for quark, g_q in quark_couplings.items():
        value = params["g_nu"] * g_q * propagator
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)
    return c
