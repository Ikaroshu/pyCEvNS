"""L_mu - L_tau Z' with an explicitly given total photon kinetic mixing; derivation in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2, E_CHARGE, Q_EM

PARAMETERS = {"g_prime": 6e-5, "mass_mev": 15.0, "kinetic_mixing": -5e-4}
COUPLINGS = ("g_prime", "kinetic_mixing")

# L_mu - L_tau charges of the lepton doublets (conventions section 7).
NU_CHARGES = {"ee": 0.0, "mm": +1.0, "tt": -1.0}


def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    eps = params["kinetic_mixing"]
    prop = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        g_q = eps * E_CHARGE * Q_EM[quark]
        for pair, charge in NU_CHARGES.items():
            g_nu = g * charge
            c.set("V", quark, pair, g_nu * g_q * prop)
    return c
