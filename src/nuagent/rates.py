"""Flux folding, detector response and signal counts per analysis bin."""

from collections.abc import Callable

import numpy as np
from scipy.integrate import quad

from nuagent.experiments.detector import Detector, SECONDS_PER_DAY
from nuagent.experiments.flux import NeutrinoFlux
from nuagent.experiments.registry import Experiment
from nuagent.physics.couplings import Coefficients, sm
from nuagent.physics.xsec import Isotope, dsigma_dt

CoefficientFunction = Callable[[np.ndarray], Coefficients]
CrossSection = Callable[[np.ndarray, np.ndarray, str, Isotope], np.ndarray]


def differential_rate(
    t: float,
    flux: NeutrinoFlux,
    detector: Detector,
    exposure_kg_day: float,
    coefficients: CoefficientFunction = Coefficients.zeros,
    *,
    flavor: str | None = None,
    raw_cross_section: CrossSection | None = None,
) -> float:
    """Events per MeV of recoil, including efficiency and exposure.

    Coefficients supply new physics only. A raw cross section supplies the full
    interaction, including the SM, and bypasses coefficient construction.
    """
    total = 0.0
    for fraction, isotope in zip(detector.number_fractions, detector.isotopes):
        if raw_cross_section is None:
            q2 = np.atleast_1d(isotope.q2(t))
            full = sm(q2) + coefficients(q2)

        def cross_section(energy: np.ndarray, incoming: str) -> np.ndarray:
            if raw_cross_section is not None:
                return raw_cross_section(energy, np.asarray(t), incoming, isotope)
            return dsigma_dt(energy, np.asarray(t), incoming, isotope, full)

        minimum = (t + np.sqrt(t**2 + 2 * isotope.mass_mev * t)) / 2
        for incoming in (flux.flavors if flavor is None else (flavor,)):
            total += fraction * flux.integrate(
                minimum, flux.energy_max_mev, incoming, lambda energy: cross_section(energy, incoming)
            )
    return float(total * detector.efficiency(t) * exposure_kg_day * detector.targets_per_kg() * SECONDS_PER_DAY)


def integrated_counts(
    experiment: Experiment,
    coefficients: CoefficientFunction = Coefficients.zeros,
    *,
    raw_cross_section: CrossSection | None = None,
) -> np.ndarray:
    """Integrate signal bins, resolving steps and monochromatic recoil endpoints.

    Validation uses these values to report invalid external plugin outputs.
    Normal physics callers use predicted_counts, which rejects invalid counts.
    """
    edges = experiment.recoil_edges_mev()
    breaks = list(experiment.detector.efficiency_steps_mev)
    for isotope in experiment.detector.isotopes:
        for lines in experiment.flux.lines.values():
            breaks.extend(2 * energy**2 / (isotope.mass_mev + 2 * energy) for energy, _ in lines)

    def integrand(t: float) -> float:
        return differential_rate(
            t, experiment.flux, experiment.detector, experiment.exposure_kg_day,
            coefficients, raw_cross_section=raw_cross_section,
        )

    return np.array([
        quad(integrand, lower, upper, points=[p for p in breaks if lower < p < upper], epsabs=1e-7, epsrel=1e-6)[0]
        for lower, upper in zip(edges[:-1], edges[1:])
    ])


def predicted_counts(
    experiment: Experiment,
    coefficients: CoefficientFunction = Coefficients.zeros,
    *,
    raw_cross_section: CrossSection | None = None,
) -> np.ndarray:
    """Finite, non-negative signal counts per analysis bin."""
    counts = integrated_counts(experiment, coefficients, raw_cross_section=raw_cross_section)
    invalid = np.flatnonzero(~np.isfinite(counts) | (counts < 0))
    if invalid.size:
        index = int(invalid[0])
        raise ValueError(f"non-finite or negative predicted count in bin {index}: {counts[index]}")
    return counts
