"""Scalar mediator with universal Dirac neutrino coupling and a down-quark-only coupling."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_nu": 2e-5, "g_d": 2e-5, "mass_mev": 20.0}
COUPLINGS = ("g_nu", "g_d")


def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    value = params["g_nu"] * params["g_d"] / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for pair in ("ee", "mm", "tt"):
        c.set("S", "d", pair, value)
    return c
