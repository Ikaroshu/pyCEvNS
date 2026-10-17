import numpy as np

from nuagent.physics.couplings import Coefficients, E_CHARGE, GF, Q_EM, SQRT2

PARAMETERS = {"g_prime": 1e-4, "mass_mev": 10.0, "kinetic_mixing": 1e-3}
COUPLINGS = ("g_prime", "kinetic_mixing")


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    denominator = SQRT2 * GF * (q2 + params["mass_mev"] ** 2)
    for quark in ("u", "d"):
        g_quark = params["kinetic_mixing"] * E_CHARGE * Q_EM[quark]
        value = params["g_prime"] * g_quark / denominator
        c.set("V", quark, "mm", value)
        c.set("V", quark, "tt", -value)
    return c
