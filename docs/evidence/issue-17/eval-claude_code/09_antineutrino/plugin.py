"""Universal vector Z' (m = 30 MeV, g' = 8e-5) for muon antineutrino scattering; derivation in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_prime": 8e-5, "mass_mev": 30.0}
COUPLINGS = ("g_prime",)


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    g_nu = params["g_prime"]
    g_u = params["g_prime"]
    g_d = params["g_prime"]
    propagator = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    values = {"u": g_nu * g_u * propagator, "d": g_nu * g_d * propagator}
    for quark in ("u", "d"):
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, values[quark])
    return c
