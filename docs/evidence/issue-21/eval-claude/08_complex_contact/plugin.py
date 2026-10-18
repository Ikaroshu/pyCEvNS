"""Contact vector NSI on down quarks with a real mu-mu entry and a complex e-tau entry."""

import numpy as np

from nuagent.physics.couplings import Coefficients

PARAMETERS = {
    "eps_d_mm": -0.05,
    "eps_d_et_re": 0.02,
    "eps_d_et_im": 0.03,
}
COUPLINGS = ("eps_d_mm", "eps_d_et_re", "eps_d_et_im")


def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    c.set("V", "d", "mm", params["eps_d_mm"])
    eps_et = params["eps_d_et_re"] + 1j * params["eps_d_et_im"]
    c.set("V", "d", "et", eps_et)  # te is filled as the complex conjugate
    return c
