"""Named experiments with their response, exposure and released observations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from nuagent.experiments.detector import Detector, csi_detector
from nuagent.experiments.flux import NeutrinoFlux, coherent_flux


@dataclass(frozen=True)
class Background:
    name: str
    counts: np.ndarray
    normalization_prior: float


@dataclass(frozen=True)
class Experiment:
    name: str
    citations: tuple[str, ...]
    data_release: str
    flux: NeutrinoFlux
    detector: Detector
    exposure_kg_day: float
    bin_edges: np.ndarray
    bin_unit: Literal["PE", "MeV"]
    observed: np.ndarray
    backgrounds: tuple[Background, ...]
    signal_normalization_prior: float
    time_window_us: tuple[float, float]

    def recoil_edges_mev(self) -> np.ndarray:
        return self.detector.recoil_mev(self.bin_edges) if self.bin_unit == "PE" else self.bin_edges


def names() -> tuple[str, ...]:
    return ("coherent_csi_2017",)


def get(name: str) -> Experiment:
    if name not in names():
        raise ValueError(f"unknown experiment {name!r}; valid experiments: {', '.join(names())}")
    data = np.loadtxt(Path(__file__).parent / "data" / "coherent_csi_2017.csv", delimiter=",")
    return Experiment(
        name=name,
        citations=("https://doi.org/10.5281/zenodo.1228631", "https://arxiv.org/abs/1804.09459", "https://arxiv.org/abs/1708.01294"),
        data_release="10.5281/zenodo.1228631",
        flux=coherent_flux(),
        detector=csi_detector(),
        exposure_kg_day=4466.0,
        bin_edges=np.concatenate((data[:, 0], data[-1:, 1])),
        bin_unit="PE",
        observed=data[:, 2],
        backgrounds=(
            Background("steady_state", data[:, 3], 1 / np.sqrt(np.sum(data[:, 3]))),
            Background("prompt_neutrons", data[:, 4], 0.25),
        ),
        signal_normalization_prior=0.28,
        time_window_us=(0.0, 12.0),
    )
