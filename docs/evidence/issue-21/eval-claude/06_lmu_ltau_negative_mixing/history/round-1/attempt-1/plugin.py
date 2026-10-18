"""Gauged L_mu - L_tau Z' with explicit (total) photon kinetic mixing; derivation in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2, E_CHARGE, Q_EM

PARAMETERS = {"g_prime": 6e-5, "mass_mev": 15.0, "kinetic_mixing": -5e-4}
COUPLINGS = ("g_prime", "kinetic_mixing")

# L_mu - L_tau charges of the lepton doublets; quarks carry no charge.
Q_NU = {"ee": 0.0, "mm": +1.0, "tt": -1.0}


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    eps = params["kinetic_mixing"]
    prop = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        g_q = eps * E_CHARGE * Q_EM[quark]  # induced quark coupling, no direct piece
        for pair, charge in Q_NU.items():
            if charge == 0.0:
                continue
            c.set("V", quark, pair, (g * charge) * g_q * prop)
    return c
