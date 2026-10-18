import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"mass_mev": 30.0, "g_nu": 1e-4, "g_u": 1e-4, "g_d": 1e-4}
COUPLINGS = ("g_nu", "g_u", "g_d")


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    denominator = SQRT2 * GF * (q2 + params["mass_mev"] ** 2)
    for quark in ("u", "d"):
        value = params["g_nu"] * params[f"g_{quark}"] / denominator
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)
    return c
