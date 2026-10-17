"""Loaders for user-authored coefficient plugins and raw cross-section plugins, by explicit file path.

Plugins run in-process with no sandbox; see docs/agent/plugin_api.md for the contract.
"""

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

from nuagent.physics.couplings import Coefficients
from nuagent.physics.xsec import Isotope


@dataclass(frozen=True)
class CoefficientPlugin:
    path: Path
    coefficients: Callable[[np.ndarray, dict[str, float]], Coefficients]
    parameters: dict[str, float]
    couplings: tuple[str, ...]


@dataclass(frozen=True)
class RawCrossSectionPlugin:
    path: Path
    dsigma_dt: Callable[[np.ndarray, np.ndarray, str, Isotope, dict[str, float]], np.ndarray]
    parameters: dict[str, float]
    couplings: tuple[str, ...]


def load_module(path: Path) -> ModuleType:
    """Execute the Python file at `path` as a fresh module."""
    spec = importlib.util.spec_from_file_location(f"nuagent_plugin_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_coefficient_plugin(path: Path) -> CoefficientPlugin:
    module = load_module(path)
    return CoefficientPlugin(path, module.coefficients, dict(module.PARAMETERS), tuple(module.COUPLINGS))


def load_raw_plugin(path: Path) -> RawCrossSectionPlugin:
    module = load_module(path)
    return RawCrossSectionPlugin(path, module.dsigma_dt, dict(module.PARAMETERS), tuple(module.COUPLINGS))
