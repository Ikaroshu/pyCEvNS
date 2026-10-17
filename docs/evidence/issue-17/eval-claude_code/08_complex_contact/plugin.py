"""Contact vector NSI on down quarks with one real diagonal and one complex off-diagonal entry."""

import numpy as np

from nuagent.physics.couplings import Coefficients

PARAMETERS = {
    "eps_dV_mm": -0.05,
    "eps_dV_et_re": 0.02,
    "eps_dV_et_im": 0.03,
}
COUPLINGS = ("eps_dV_mm", "eps_dV_et_re", "eps_dV_et_im")


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    ones = np.ones_like(q2, dtype=float)
    eps_mm = params["eps_dV_mm"] * ones
    eps_et = (params["eps_dV_et_re"] + 1j * params["eps_dV_et_im"]) * ones
    c.set("V", "d", "mm", eps_mm)
    c.set("V", "d", "et", eps_et)
    return c
