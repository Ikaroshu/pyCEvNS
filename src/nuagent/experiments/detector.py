"""Isotope mixtures and the COHERENT CsI detector response."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nuagent.physics.xsec import Isotope

MEV_PER_KG = 5.6095887e29
SECONDS_PER_DAY = 86400
CSI_QUENCHING_FACTOR = 0.0878
CSI_LIGHT_YIELD_PE_PER_KEV_EE = 13.348
CSI_PE_PER_MEV = CSI_QUENCHING_FACTOR * CSI_LIGHT_YIELD_PE_PER_KEV_EE * 1000


def unit_efficiency(t: float | np.ndarray) -> np.ndarray:
    return np.ones_like(t, dtype=float)


def csi_efficiency_pe(pe: float | np.ndarray) -> np.ndarray:
    """Released acceptance curve, including the half-height 5 <= PE < 6 step."""
    pe = np.asarray(pe)
    step = np.where(pe < 5, 0.0, np.where(pe < 6, 0.5, 1.0))
    return 0.6655 / (1 + np.exp(-0.4942 * (pe - 10.8507))) * step


def csi_efficiency(t: float | np.ndarray) -> np.ndarray:
    return csi_efficiency_pe(np.asarray(t) * CSI_PE_PER_MEV)


@dataclass(frozen=True)
class Detector:
    isotopes: tuple[Isotope, ...]
    number_fractions: tuple[float, ...]
    efficiency: Callable[[float | np.ndarray], np.ndarray] = unit_efficiency
    efficiency_steps_mev: tuple[float, ...] = ()
    mass_kg: float | None = None
    quenching_factor: float | None = None
    light_yield_pe_per_kev_ee: float | None = None

    def recoil_mev(self, pe: np.ndarray) -> np.ndarray:
        return np.asarray(pe) / (self.quenching_factor * self.light_yield_pe_per_kev_ee * 1000)

    def targets_per_kg(self) -> float:
        """Number fractions are per nucleus, so normalize by the mixture's mean mass."""
        mean_mass_mev = sum(f * i.mass_mev for f, i in zip(self.number_fractions, self.isotopes))
        return MEV_PER_KG / mean_mass_mev


def csi_detector(helm_radius_fm: float = 5.5) -> Detector:
    return Detector(
        isotopes=(Isotope(55, 78, 123800.0, helm_radius_fm), Isotope(53, 74, 118210.0, helm_radius_fm)),
        number_fractions=(0.5, 0.5),
        efficiency=csi_efficiency,
        efficiency_steps_mev=(5 / CSI_PE_PER_MEV, 6 / CSI_PE_PER_MEV),
        mass_kg=14.6,
        quenching_factor=CSI_QUENCHING_FACTOR,
        light_yield_pe_per_kev_ee=CSI_LIGHT_YIELD_PE_PER_KEV_EE,
    )
