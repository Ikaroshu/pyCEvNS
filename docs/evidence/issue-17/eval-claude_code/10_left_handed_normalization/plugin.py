"""Universal vector Z' with neutrino coupling g nu-bar gamma^mu (1 - gamma5) nu (no 1/2).

Derivation in notes.md.
"""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

# g: coefficient of nu-bar gamma^mu (1 - gamma5) nu as written in the description.
# g_q: vector coupling to each of u and d.
PARAMETERS = {"g": 5e-5, "g_q": 1e-4, "mass_mev": 30.0}
COUPLINGS = ("g", "g_q")


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    # (1 - gamma5) = 2 P_L, so the left-handed neutrino coupling is g_nu = 2 g.
    g_nu = 2.0 * params["g"]
    value = g_nu * params["g_q"] / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)
    return c
