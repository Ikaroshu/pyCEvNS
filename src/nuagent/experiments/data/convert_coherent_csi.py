"""Convert COHERENT release 10.5281/zenodo.1228631 without network access.

Sources: data_coincidence_beamOn.txt, data_anticoincidence_beamOn.txt,
promptPDF.txt and coherent_parameters.yaml. Conversion date: 2026-10-17.
Run: uv run python src/nuagent/experiments/data/convert_coherent_csi.py
"""

import argparse
import re
from pathlib import Path

import numpy as np

from nuagent.experiments.detector import csi_efficiency_pe

DATA_DIRECTORY = Path(__file__).parent
PE_EDGES = np.arange(6.0, 32.0, 2.0)
HEADER = """Source: https://zenodo.org/records/1228631 (DOI 10.5281/zenodo.1228631)
Release files: data_coincidence_beamOn.txt, data_anticoincidence_beamOn.txt,
promptPDF.txt, coherent_parameters.yaml; conversion date: 2026-10-17.
Analysis: 6 <= PE < 30, edges 6,8,...,30; full release 0 <= time < 12 us.
Raw beamOn C and AC counts summed over all 24 time centers 0.25,...,11.75 us.
AC scale = 1 (same duration/exposure); no acceptance correction to observed/AC.
Prompt neutrons: efficiency at each 1-PE center, times 7.47594 GWhr, then rebin.
Timing integrated in full; no timing rescaling. Paper likelihood uses time < 6 us.
pe_lower,pe_upper,observed,steady_state,prompt_neutrons"""


def observed_spectrum(path: Path) -> np.ndarray:
    """Read the released 2 PE by 0.5 us histogram and sum its full time range."""
    rows = np.loadtxt(path, delimiter=",")
    expected_coordinates = np.array([(pe, time) for pe in np.arange(1, 50, 2) for time in np.arange(0.25, 12, 0.5)])
    if rows.shape != (600, 3) or not np.array_equal(rows[:, :2], expected_coordinates):
        raise ValueError(f"unexpected PE/time grid in {path}")
    counts = rows[:, 2]
    if not np.all(np.isfinite(counts) & (counts >= 0) & (counts == np.floor(counts))):
        raise ValueError(f"invalid raw counts in {path}")
    return np.histogram(rows[:, 0], bins=PE_EDGES, weights=counts)[0].astype(int)


def convert(source: Path) -> np.ndarray:
    observed = observed_spectrum(source / "data_coincidence_beamOn.txt")
    steady_state = observed_spectrum(source / "data_anticoincidence_beamOn.txt")
    prompt = np.loadtxt(source / "promptPDF.txt", delimiter=",")
    if prompt.shape != (50, 2) or not np.array_equal(prompt[:, 0], np.arange(0.5, 50, 1)):
        raise ValueError("unexpected prompt neutron PE grid")
    if not np.all(np.isfinite(prompt[:, 1]) & (prompt[:, 1] >= 0)):
        raise ValueError("invalid prompt neutron expectation")
    parameters = (source / "coherent_parameters.yaml").read_text()
    exposure_gw_hr = float(re.search(r"beamExposure:\n  name: [^\n]+\n  value: (\S+)", parameters)[1])
    neutrons = np.histogram(
        prompt[:, 0], bins=PE_EDGES,
        weights=prompt[:, 1] * csi_efficiency_pe(prompt[:, 0]) * exposure_gw_hr,
    )[0]
    return np.column_stack((PE_EDGES[:-1], PE_EDGES[1:], observed, steady_state, neutrons))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DATA_DIRECTORY / "coherent_csi_release")
    parser.add_argument("--output", type=Path, default=DATA_DIRECTORY / "coherent_csi_2017.csv")
    args = parser.parse_args()
    np.savetxt(args.output, convert(args.source), delimiter=",", fmt=("%g", "%g", "%d", "%d", "%.16g"), header=HEADER)


if __name__ == "__main__":
    main()
