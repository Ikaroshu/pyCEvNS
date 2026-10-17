"""Build new detector records from legacy definitions used by numerical oracles."""

import numpy as np
from pyCEvNS.constants import mev_per_kg

from nuagent.experiments.detector import Detector, csi_efficiency, unit_efficiency
from nuagent.physics.xsec import Isotope

EXPOSURE_KG_DAY = 4466


def detector_isotopes(detector, helm_radius_fm: float) -> list[Isotope]:
    return [Isotope(int(z), int(n), float(m), helm_radius_fm) for z, n, m in zip(detector.z, detector.n, detector.m)]


def production_detector(detector, helm_radius_fm: float) -> Detector:
    efficiency = csi_efficiency if detector.det_type == "csi" else unit_efficiency
    return Detector(tuple(detector_isotopes(detector, helm_radius_fm)), tuple(detector.frac), efficiency)


def events_per_mev(rate: float, detector) -> float:
    """Scale the legacy oracle into events per MeV at the README exposure."""
    return rate * EXPOSURE_KG_DAY * mev_per_kg * 24 * 60 * 60 / np.dot(detector.m, detector.frac)
