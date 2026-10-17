"""Universal vector Z' specified by alpha' = g'^2 / (4 pi); derivation in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"alpha_prime": 3.899296105751435e-10, "mass_mev": 45.0}
COUPLINGS = ("alpha_prime",)


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    g_prime_sq = 4.0 * np.pi * params["alpha_prime"]  # g'^2, with g' > 0
    value = g_prime_sq / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)
    return c
