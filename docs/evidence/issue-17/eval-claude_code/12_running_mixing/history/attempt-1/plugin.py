"""L_mu - L_tau Z' with the full one-loop, Q^2-dependent photon mixing.

Derivation and checks are in notes.md.
"""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2, E_CHARGE, M_MU, M_TAU, Q_EM

PARAMETERS = {"g_prime": 8e-5, "mass_mev": 12.0}
COUPLINGS = ("g_prime",)

# 40-point Gauss-Legendre quadrature on the Feynman parameter interval [0, 1].
_T, _W = np.polynomial.legendre.leggauss(40)
_X = (_T + 1.0) / 2.0
_WX = _W / 2.0


def loop_mixing(q2: np.ndarray, g: float) -> np.ndarray:
    """Spacelike one-loop Z'-photon mixing eps(Q^2) from the muon and tau loops."""
    q = np.asarray(q2, dtype=float)[:, None]
    x = _X[None, :]
    integrand = x * (1.0 - x) * np.log(
        (M_TAU ** 2 + x * (1.0 - x) * q) / (M_MU ** 2 + x * (1.0 - x) * q)
    )
    return E_CHARGE * g / (2.0 * np.pi ** 2) * (integrand * _WX[None, :]).sum(axis=1)


def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    eps = loop_mixing(q2, g)
    prop = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        g_q = eps * E_CHARGE * Q_EM[quark]
        c.set("V", quark, "mm", +g * g_q * prop)
        c.set("V", quark, "tt", -g * g_q * prop)
    return c
