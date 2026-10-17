"""U(1)_{L_mu - L_tau} Z' with loop-induced photon mixing in the Q^2 -> 0 limit.

Derivation in notes.md. Quarks couple only through the muon/tau loop mixing
epsilon = +kappa g', kappa = e ln(m_tau/m_mu) / (6 pi^2) (conventions section 6).
"""

import numpy as np

from nuagent.physics.couplings import (
    Coefficients,
    E_CHARGE,
    GF,
    KAPPA_LMU_LTAU,
    Q_EM,
    SQRT2,
)

PARAMETERS = {"g_prime": 6e-5, "mass_mev": 15.0}
COUPLINGS = ("g_prime",)

# L_mu - L_tau charges of the lepton doublets (conventions section 7).
Q_NU = {"ee": 0.0, "mm": 1.0, "tt": -1.0}


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    m = params["mass_mev"]

    # Loop-induced kinetic mixing with the photon, Q^2 << m_mu^2 limit.
    epsilon = KAPPA_LMU_LTAU * g
    g_quark = {q: epsilon * E_CHARGE * Q_EM[q] for q in ("u", "d")}

    propagator = 1.0 / (SQRT2 * GF * (q2 + m**2))
    for quark in ("u", "d"):
        for pair in ("mm", "tt"):
            g_nu = g * Q_NU[pair]
            c.set("V", quark, pair, g_nu * g_quark[quark] * propagator)
    return c
