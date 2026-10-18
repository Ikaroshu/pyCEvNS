"""L_mu - L_tau Z' with loop-induced photon mixing in the Q^2 -> 0 limit.

Derivation in notes.md.
"""

import numpy as np

from nuagent.physics.couplings import (
    Coefficients,
    GF,
    SQRT2,
    E_CHARGE,
    Q_EM,
    KAPPA_LMU_LTAU,
)

PARAMETERS = {"g_prime": 6e-5, "mass_mev": 15.0}
COUPLINGS = ("g_prime",)

# L_mu - L_tau charges of the lepton doublets (neutrinos); quarks are uncharged.
Q_PRIME_NU = {"ee": 0.0, "mm": +1.0, "tt": -1.0}


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    # Positive loop-induced kinetic mixing at Q^2 << m_mu^2 (conventions section 6).
    eps = KAPPA_LMU_LTAU * g
    prop = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        g_q = eps * E_CHARGE * Q_EM[quark]
        for pair, charge in Q_PRIME_NU.items():
            g_nu = g * charge
            c.set("V", quark, pair, g_nu * g_q * prop)
    return c
