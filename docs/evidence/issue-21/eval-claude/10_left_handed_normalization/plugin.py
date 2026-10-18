"""Universal vector Z' with neutrino coupling written as g nubar gamma^mu (1 - gamma5) nu (no 1/2).

Derivation in notes.md.
"""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

# g is the coefficient of nubar gamma^mu (1 - gamma5) nu as written in the description.
# g_u and g_d are the vector quark couplings. mass_mev is the Z' mass.
PARAMETERS = {"g": 5e-5, "g_u": 1e-4, "g_d": 1e-4, "mass_mev": 30.0}
COUPLINGS = ("g", "g_u", "g_d")


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    # (1 - gamma5) = 2 P_L, so the left-handed neutrino coupling is g_nu = 2 g.
    g_nu = 2.0 * params["g"]
    propagator = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    quark_couplings = {"u": params["g_u"], "d": params["g_d"]}
    for quark, g_q in quark_couplings.items():
        value = g_nu * g_q * propagator
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)
    return c
