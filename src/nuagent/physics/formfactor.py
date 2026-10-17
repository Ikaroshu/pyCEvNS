"""Helm nuclear form factor, ported from legacy `HelmFormFactor`."""

import numpy as np
from scipy.special import spherical_jn

from nuagent.physics.couplings import HBAR_C_MEV_M

HELM_SKIN_FM = 0.9


def helm_form_factor_squared(q_mev: np.ndarray, radius_fm: float) -> np.ndarray:
    """Squared Helm form factor at momentum transfer |q| in MeV for nuclear radius `radius_fm`; q must be positive."""
    r = radius_fm * 1e-15 / HBAR_C_MEV_M
    s = HELM_SKIN_FM * 1e-15 / HBAR_C_MEV_M
    r0 = np.sqrt(5 / 3 * r**2 - 5 * s**2)
    x = np.asarray(q_mev) * r0
    return (3 * spherical_jn(1, x) / x * np.exp(-((np.asarray(q_mev) * s) ** 2) / 2)) ** 2
