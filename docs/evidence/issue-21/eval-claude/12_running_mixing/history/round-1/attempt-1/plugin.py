"""L_mu - L_tau Z' with the full one-loop, Q^2-dependent photon mixing (plugin-only).

Derivation and conventions are in notes.md.
"""

import numpy as np

from nuagent.physics.couplings import (
    Coefficients,
    GF,
    SQRT2,
    E_CHARGE,
    M_MU,
    M_TAU,
    Q_EM,
)

PARAMETERS = {"g_prime": 8e-5, "mass_mev": 12.0}
COUPLINGS = ("g_prime",)

# L_mu - L_tau charges of the lepton doublets (neutrinos and charged leptons alike).
_Q_PRIME = {"e": 0.0, "m": +1.0, "t": -1.0}
_PAIRS = {"ee": "e", "mm": "m", "tt": "t"}

# Gauss-Legendre nodes/weights mapped from [-1, 1] to the Feynman parameter x in [0, 1].
_T, _W = np.polynomial.legendre.leggauss(40)
_X = (_T + 1.0) / 2.0
_WX = _W / 2.0


def loop_mixing(q2: np.ndarray, g_prime: float) -> np.ndarray:
    """Spacelike one-loop gamma-Z' mixing eps(Q^2) from the muon and tau loops.

    eps(Q^2) = (e g' / 2 pi^2) * Int_0^1 dx x(1-x) ln[(m_tau^2 + x(1-x) Q^2) / (m_mu^2 + x(1-x) Q^2)]
    which reduces to kappa g' = e g' ln(m_tau/m_mu) / (6 pi^2) at Q^2 -> 0.
    """
    q = np.asarray(q2, dtype=float)[:, None]
    x = _X[None, :]
    xx = x * (1.0 - x)
    integrand = xx * np.log((M_TAU**2 + xx * q) / (M_MU**2 + xx * q))
    integral = (integrand * _WX[None, :]).sum(axis=1)
    return E_CHARGE * g_prime / (2.0 * np.pi**2) * integral


def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    g_prime = params["g_prime"]
    mass = params["mass_mev"]

    eps = loop_mixing(q2, g_prime)
    prop = 1.0 / (SQRT2 * GF * (q2 + mass**2))

    for quark in ("u", "d"):
        # No tree-level mixing and no direct quark charge: the quark coupling is purely loop-induced.
        g_q = eps * E_CHARGE * Q_EM[quark]
        for pair, flavor in _PAIRS.items():
            g_nu = g_prime * _Q_PRIME[flavor]
            value = g_nu * g_q * prop
            c.set("V", quark, pair, value)
    return c
