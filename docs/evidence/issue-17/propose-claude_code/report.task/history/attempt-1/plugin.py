import numpy as np

from nuagent.physics.couplings import Coefficients, E_CHARGE, GF, Q_EM, SQRT2

PARAMETERS = {"g_prime": 1e-4, "kinetic_mixing": 1e-3, "mass_mev": 10.0}
COUPLINGS = ("g_prime", "kinetic_mixing")


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    g_prime = params["g_prime"]
    propagator = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    nu_charges = {"mm": 1.0, "tt": -1.0}
    for quark in ("u", "d"):
        g_q = params["kinetic_mixing"] * E_CHARGE * Q_EM[quark]
        for pair, charge in nu_charges.items():
            c.set("V", quark, pair, g_prime * charge * g_q * propagator)
    return c
