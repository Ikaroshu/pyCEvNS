"""Operator basis, physical constants, Standard Model coefficients and the Coefficients container.

Conventions are those of docs/agent/conventions.md, which is normative.
"""

from dataclasses import dataclass

import numpy as np

GF = 1.16638e-11  # Fermi constant, MeV^-2
SQRT2 = np.sqrt(2.0)
E_CHARGE = 0.303  # electromagnetic coupling e, alpha = e^2 / 4 pi
SIN2_THETA_W = 0.2312

M_U = 2.2  # MeV, current quark masses used in the scalar nucleon charges
M_D = 4.7
M_P = 938.27  # MeV
M_N = 939.56
M_MU = 105.658369
M_TAU = 1776.99

HBAR_C_MEV_M = 6.58212e-22 * 2.998e8  # hbar c in MeV m, the legacy value

QUARK_MASS_MEV = {"u": M_U, "d": M_D}
Q_EM = {"u": 2.0 / 3.0, "d": -1.0 / 3.0}
# Scalar quark contents f_q^N of the proton and neutron.
SCALAR_CONTENT_P = {"u": 0.0208, "d": 0.0411}
SCALAR_CONTENT_N = {"u": 0.0189, "d": 0.0451}

# Radiative corrections to the SM neutral-current couplings (Barranco et al. 2005, as in legacy).
RHO_NC = 1.0086
KAPPA_NU = 0.9978
LAMBDA_UL = -0.0031
LAMBDA_DL = -0.0025
LAMBDA_DR = 7.5e-5
LAMBDA_UR = LAMBDA_DR / 2

SM_VECTOR = {
    "u": RHO_NC * (0.5 - 4.0 / 3.0 * KAPPA_NU * SIN2_THETA_W) + LAMBDA_UL + LAMBDA_UR,
    "d": RHO_NC * (-0.5 + 2.0 / 3.0 * KAPPA_NU * SIN2_THETA_W) + LAMBDA_DL + LAMBDA_DR,
}

# Loop-induced kinetic mixing of the U(1)_{L_mu - L_tau} boson: epsilon = KAPPA_LMU_LTAU * g'.
KAPPA_LMU_LTAU = E_CHARGE * np.log(M_TAU / M_MU) / (6 * np.pi**2)

STRUCTURES = ("V", "S")
QUARKS = ("u", "d")
FLAVORS = ("e", "mu", "tau")
PAIR_INDEX = {"ee": (0, 0), "mm": (1, 1), "tt": (2, 2), "em": (0, 1), "et": (0, 2), "mt": (1, 2)}


def flavor_matrix(entries: dict[str, complex]) -> np.ndarray:
    """Hermitian 3x3 matrix from the six entries ee, mm, tt, em, et, mt; absent entries are zero."""
    matrix = np.zeros((3, 3), dtype=complex)
    for pair, value in entries.items():
        i, j = PAIR_INDEX[pair]
        matrix[i, j] = value
        matrix[j, i] = np.conj(value)
    return matrix


@dataclass
class Coefficients:
    """Dimensionless coefficients C^{qX}_{alpha beta}(q2) on a grid of q2 = 2 M T in MeV^2.

    `values[structure, quark]` is a complex array shaped (3, 3, len(q2)), indexed by
    (alpha, beta) in the order e, mu, tau.
    """

    q2: np.ndarray
    values: dict[tuple[str, str], np.ndarray]

    @classmethod
    def zeros(cls, q2: np.ndarray) -> "Coefficients":
        q2 = np.atleast_1d(np.asarray(q2, dtype=float))
        values = {(s, q): np.zeros((3, 3, q2.size), dtype=complex) for s in STRUCTURES for q in QUARKS}
        return cls(q2, values)

    def set(self, structure: str, quark: str, pair: str, value: complex | np.ndarray) -> None:
        """Overwrite one entry with a scalar or an array shaped like q2; an off-diagonal entry also sets its conjugate."""
        i, j = PAIR_INDEX[pair]
        if i == j and np.any(np.imag(value) != 0):
            raise ValueError(f"diagonal entry {structure} {quark} {pair} must be real")
        matrix = self.values[structure, quark]
        matrix[i, j] = value
        matrix[j, i] = np.conj(value)

    def __add__(self, other: "Coefficients") -> "Coefficients":
        return Coefficients(self.q2, {key: value + other.values[key] for key, value in self.values.items()})


def sm(q2: np.ndarray) -> Coefficients:
    """Standard Model vector coefficients with the legacy radiative corrections, flavor-diagonal."""
    c = Coefficients.zeros(q2)
    for quark in QUARKS:
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, SM_VECTOR[quark])
    return c


def mediator_coefficient(g_nu: complex | np.ndarray, g_q: float, mass_mev: float, q2: np.ndarray) -> np.ndarray:
    """C = g_nu g_q / (sqrt2 GF (q2 + m^2)) for a vector or scalar mediator; broadcasts g_nu against q2."""
    return np.multiply.outer(np.asarray(g_nu) * g_q, 1.0 / (SQRT2 * GF * (np.asarray(q2) + mass_mev**2)))


def kinetic_mixing_coupling(epsilon: float, quark: str) -> float:
    """Quark vector coupling epsilon e Q_q^em induced by kinetic mixing with the photon."""
    return epsilon * E_CHARGE * Q_EM[quark]


def vector_charge(c: Coefficients, z: int, n: int) -> np.ndarray:
    """Nuclear vector charge Q^{alpha beta}(q2) = Z (2 C^u + C^d) + N (C^u + 2 C^d), shaped (3, 3, len(q2))."""
    cu, cd = c.values["V", "u"], c.values["V", "d"]
    return z * (2 * cu + cd) + n * (cu + 2 * cd)


def scalar_charge(c: Coefficients, z: int, n: int) -> np.ndarray:
    """Nuclear scalar charge Q_S^{alpha beta}(q2) in MeV^-2, shaped (3, 3, len(q2))."""
    return (
        SQRT2
        * GF
        * sum(
            c.values["S", q] * (z * M_P * SCALAR_CONTENT_P[q] + n * M_N * SCALAR_CONTENT_N[q]) / QUARK_MASS_MEV[q]
            for q in QUARKS
        )
    )
