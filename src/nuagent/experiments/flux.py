"""Tabulated neutrino spectra and exact monochromatic lines."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nuagent.physics.couplings import HBAR_C_MEV_M


@dataclass(frozen=True)
class NeutrinoFlux:
    energy_mev: np.ndarray
    continuous: dict[str, np.ndarray]
    lines: dict[str, tuple[tuple[float, float], ...]]
    normalization_cm2_s: float

    @property
    def flavors(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.continuous, *self.lines)))

    @property
    def energy_max_mev(self) -> float:
        return max([self.energy_mev[-1], *(energy for lines in self.lines.values() for energy, _ in lines)])

    def integrate(
        self,
        lower_mev: float,
        upper_mev: float,
        flavor: str,
        weight: Callable[[np.ndarray], np.ndarray],
    ) -> float:
        """Integrate a weight against the flux, in natural units per second.

        Interpolate the *weighted* tabulation at both edges, as legacy does;
        evaluating the weight after interpolating the flux changes the quadrature.
        """
        if upper_mev <= lower_mev:
            return 0.0
        total = sum(
            strength * float(np.asarray(weight(np.asarray(energy))))
            for energy, strength in self.lines.get(flavor, ())
            if lower_mev < energy <= upper_mev
        )
        lower = max(lower_mev, self.energy_mev[0])
        upper = min(upper_mev, self.energy_mev[-1])
        if flavor in self.continuous and lower < upper:
            weighted = self.continuous[flavor] * weight(self.energy_mev)
            interior = (self.energy_mev > lower) & (self.energy_mev < upper)
            energies = np.concatenate(([lower], self.energy_mev[interior], [upper]))
            values = np.concatenate((
                [np.interp(lower, self.energy_mev, weighted)],
                weighted[interior],
                [np.interp(upper, self.energy_mev, weighted)],
            ))
            total += np.trapezoid(values, energies)
        return float(total * self.normalization_cm2_s * (100 * HBAR_C_MEV_M) ** 2)


def coherent_flux() -> NeutrinoFlux:
    """Legacy COHERENT Michel tabulation and 29 MeV prompt muon-neutrino line."""
    energy = np.linspace(0.001, 52.0, 100)
    electron = (3 * (energy / (2 / 3 * 52)) ** 2 - 2 * (energy / (2 / 3 * 52)) ** 3) / 29.25
    antimuon = (3 * (energy / 52) ** 2 - 2 * (energy / 52) ** 3) / 26
    return NeutrinoFlux(energy, {"e": electron, "mubar": antimuon}, {"mu": ((29.0, 1.0),)}, 1.13e7)
