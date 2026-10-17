"""U(1)_{B-L} gauge boson, m = 25 MeV, g' = 3e-5, no kinetic mixing; derivation in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_prime": 3e-5, "mass_mev": 25.0}
COUPLINGS = ("g_prime",)

# B-L charges: lepton doublet (hence every neutrino) -1, every quark +1/3.
Q_NU = -1.0
Q_QUARK = 1.0 / 3.0


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    g_nu = g * Q_NU
    g_q = g * Q_QUARK
    value = g_nu * g_q / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)
    return c
