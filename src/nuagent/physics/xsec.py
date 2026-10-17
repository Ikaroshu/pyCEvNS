"""Master differential cross section for coherent neutrino scattering on spin-0 nuclei."""

from dataclasses import dataclass

import numpy as np

from nuagent.physics.couplings import GF, Coefficients, scalar_charge, vector_charge
from nuagent.physics.formfactor import helm_form_factor_squared

# Incoming flavor -> row alpha of the coefficient matrices. An antineutrino uses the conjugate matrix,
# which leaves sum_beta |Q^{alpha beta}|^2 unchanged, so it shares the row of its neutrino.
FLAVOR_ROW = {"e": 0, "mu": 1, "tau": 2, "ebar": 0, "mubar": 1, "taubar": 2}


@dataclass(frozen=True)
class Isotope:
    z: int
    n: int
    mass_mev: float
    helm_radius_fm: float

    def q2(self, t: np.ndarray) -> np.ndarray:
        """Q^2 = 2 M T in MeV^2 for recoil energies `t` in MeV."""
        return 2 * self.mass_mev * np.asarray(t, dtype=float)


def dsigma_dt(e_nu: np.ndarray, t: np.ndarray, flavor: str, isotope: Isotope, coefficients: Coefficients) -> np.ndarray:
    """dsigma/dT in MeV^-3 for neutrino energy `e_nu` and recoil energy `t` (MeV), summed over final flavors.

    `t` is a scalar or a 1D array, `coefficients` is the total (SM plus new physics) evaluated on
    q2 = isotope.q2(t), and `e_nu` broadcasts against `t`. The result has the broadcast shape. The
    formula is valid only above threshold, e_nu >= (T + sqrt(T^2 + 2 M T)) / 2; the caller integrates there.
    """
    e_nu = np.asarray(e_nu, dtype=float)
    t = np.asarray(t, dtype=float)
    m = isotope.mass_mev
    row = FLAVOR_ROW[flavor]
    vector = np.sum(np.abs(vector_charge(coefficients, isotope.z, isotope.n)[row]) ** 2, axis=0).reshape(t.shape)
    scalar = np.sum(np.abs(scalar_charge(coefficients, isotope.z, isotope.n)[row]) ** 2, axis=0).reshape(t.shape)
    form_factor = helm_form_factor_squared(np.sqrt(2 * m * t), isotope.helm_radius_fm)
    kinematics = 1 - t / e_nu + t**2 / (2 * e_nu**2) - m * t / (2 * e_nu**2)
    return (GF**2 * m / np.pi * vector * kinematics + m**2 * t * scalar / (4 * np.pi * e_nu**2)) * form_factor
