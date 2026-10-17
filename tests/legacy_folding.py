"""Fold new-package cross sections with legacy fluxes and detectors, the way legacy `rates` does.

Legacy `rates(er, ...)` integrates the flux times 1, 1/E and 1/E^2 with `NeutrinoFlux.integrate` from
E_min(T) = (T + sqrt(T^2 + 2 M T)) / 2 to the flux maximum, per isotope, then sums with the isotope
fractions and multiplies by the detector efficiency. The vector and scalar cross sections are linear
in those three weights, and `integrate` is linear in its weight (trapezoid sums on the flux grid with
linear interpolation at the edges, plus delta lines strictly inside the bounds), so folding the new
dsigma/dT as the weight over the same bounds reproduces the legacy combination exactly.

`integrate` caches its weighted grid by weight-function identity. Every fold here passes a fresh
closure that the cache keeps alive, so a cached entry is never reused for a different weight.
"""

from collections.abc import Callable

import numpy as np
from pyCEvNS.constants import mev_per_kg

from nuagent.physics.couplings import Coefficients
from nuagent.physics.xsec import Isotope, dsigma_dt

EXPOSURE_KG_DAY = 4466


def detector_isotopes(detector, helm_radius_fm: float) -> list[Isotope]:
    return [Isotope(int(z), int(n), float(m), helm_radius_fm) for z, n, m in zip(detector.z, detector.n, detector.m)]


def fold(
    t: float,
    flavor: str,
    flux,
    detector,
    isotopes: list[Isotope],
    cross_section: Callable[[np.ndarray, Isotope], np.ndarray],
) -> float:
    """Legacy-style differential rate at recoil energy t for a cross section `cross_section(e_nu, isotope)`."""
    total = 0.0
    for fraction, isotope in zip(detector.frac, isotopes):
        e_min = 0.5 * (np.sqrt(t**2 + 2 * t * isotope.mass_mev) + t)
        total += fraction * flux.integrate(
            e_min, flux.ev_max, flavor, weight_function=lambda e, iso=isotope: cross_section(e, iso)
        )
    if detector.detectoin_efficiency is not None:
        total *= detector.detectoin_efficiency(t)
    return total


def fold_coefficients(
    t: float,
    flavor: str,
    flux,
    detector,
    isotopes: list[Isotope],
    total_coefficients: Callable[[np.ndarray], Coefficients],
) -> float:
    """Fold `xsec.dsigma_dt` with the total coefficients `total_coefficients(q2)` evaluated once per isotope."""
    at_t = {isotope: total_coefficients(isotope.q2(t)) for isotope in isotopes}
    return fold(t, flavor, flux, detector, isotopes, lambda e, iso: dsigma_dt(e, t, flavor, iso, at_t[iso]))


def events_per_mev(rate: float, detector) -> float:
    """Convert a legacy rate to events per MeV of recoil for the README exposure, as legacy `events` scales it."""
    return rate * EXPOSURE_KG_DAY * mev_per_kg * 24 * 60 * 60 / np.dot(detector.m, detector.frac)
