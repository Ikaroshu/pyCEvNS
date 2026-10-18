"""Gauged U(1)_{L_mu - L_tau} Z' with explicit photon kinetic mixing; derivation in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2, E_CHARGE, Q_EM

PARAMETERS = {"g_prime": 1e-4, "mass_mev": 10.0, "kinetic_mixing": 1e-3}
COUPLINGS = ("g_prime", "kinetic_mixing")

# L_mu - L_tau charges of the lepton doublets (section 7 of conventions.md).
Q_NU = {"ee": 0.0, "mm": +1.0, "tt": -1.0}


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    eps = params["kinetic_mixing"]
    prop = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        # No direct quark charge under L_mu - L_tau; the quark coupling is
        # entirely the mixing-induced electromagnetic piece.
        g_q = eps * E_CHARGE * Q_EM[quark]
        for pair, charge in Q_NU.items():
            g_nu = g * charge
            c.set("V", quark, pair, g_nu * g_q * prop)
    return c
