"""Spec Verification test 2: the new vector cross section against legacy `NeutrinoNucleusElasticVector.rates`.

The production dR/dT folds the new flux and detector response, compared pointwise at 50 recoil energies
spanning each detector's threshold range. Rates are compared in events per MeV for the README exposure
(4466 kg day), the units of the
integrand of legacy `events`; the raw legacy rate is of order 1e-30, so an absolute floor of 1e-6 in raw
units would make every comparison pass.

Two legacy issues are accounted for, not fixed:
- The legacy light-mediator mapping epsilon = g / (2 sqrt2 G_F m^2) is half the physical matching
  C = g_nu g_f / (sqrt2 G_F (q^2 + m^2)) (conventions.md section 3), so legacy is given g = 2 g_nu g_f.
- Legacy `NSIparameters.ed()` divides `gu` instead of `gd`, so with a mediator the legacy d-quark
  coefficients are the u-quark ones. The comparison uses `CorrectedNSIparameters`, whose `ed()` divides
  `gd`; `test_legacy_ed_uses_gu` pins the legacy behavior.
"""

import numpy as np
import pytest
from experiment_fixtures import detector_isotopes, events_per_mev, production_detector
from pyCEvNS.constants import gf
from pyCEvNS.detectors import Detector
from pyCEvNS.events import HelmFormFactor, NeutrinoNucleusElasticVector, eff_coherent
from pyCEvNS.flux import NeutrinoFlux, NeutrinoFluxFactory
from pyCEvNS.parameters import NSIparameters

from nuagent.experiments.flux import NeutrinoFlux as ProductionFlux
from nuagent.experiments.flux import coherent_flux
from nuagent.physics.couplings import GF, PAIR_INDEX, SQRT2, sm, vector_charge
from nuagent.physics.modelcard import Contact, FlavorMatrix, Mediator, ModelCard, QuarkCouplings
from nuagent.rates import differential_rate

HELM_RADIUS_FM = {"csi": 5.5, "ar": 4.7 * (40 / 133) ** (1 / 3)}
OFF_DIAGONAL = ("em", "et", "mt")


class CorrectedNSIparameters(NSIparameters):
    """Legacy NSIparameters with `ed()` dividing `gd`, as its docstring intends."""

    def ed(self):
        if self.mz != 0:
            for pair in self.epd:
                self.epd[pair] = self.gd[pair] / (2 * np.sqrt(2) * gf * self.mz**2)
        return np.array(
            [
                [self.epd["ee"], self.epd["em"], self.epd["et"]],
                [np.conj(self.epd["em"]), self.epd["mm"], self.epd["mt"]],
                [np.conj(self.epd["et"]), np.conj(self.epd["mt"]), self.epd["tt"]],
            ]
        )


def coherent_all_flavors():
    """The legacy COHERENT tabulation with the nu_e spectrum given to ebar, tau and taubar and the 29 MeV line to tau."""
    ev = np.linspace(0.001, 52, 100)
    de = (3 * (ev / (2 / 3 * 52)) ** 2 - 2 * (ev / (2 / 3 * 52)) ** 3) / 29.25
    return NeutrinoFlux(
        continuous_fluxes={"ev": ev, "ebar": de, "tau": de, "taubar": de},
        delta_fluxes={"tau": [(29, 1)]},
        norm=1.13e7,
    )


def random_flavor_matrix(rng: np.random.Generator, scale: float) -> dict:
    entries = {pair: scale * rng.uniform(-1, 1) for pair in ("ee", "mm", "tt")}
    entries |= {pair: (scale * rng.uniform(-0.5, 0.5), scale * rng.uniform(-0.5, 0.5)) for pair in OFF_DIAGONAL}
    return entries


def as_complex(value) -> complex:
    return complex(*value) if isinstance(value, tuple) else value


def draw_point(rng: np.random.Generator, kind: str, mass_mev: float) -> tuple[ModelCard, NSIparameters]:
    if kind == "contact":
        eps = {q: random_flavor_matrix(rng, 0.3) for q in ("u", "d")}
        card = ModelCard(contacts=[Contact(fermion=q, epsilon=FlavorMatrix(**eps[q])) for q in eps])
        legacy = CorrectedNSIparameters()
        for pair in PAIR_INDEX:
            legacy.epu[pair] = as_complex(eps["u"][pair])
            legacy.epd[pair] = as_complex(eps["d"][pair])
        return card, legacy
    # Scale so that |C| is of order 0.3 at q^2 ~ 2000 MeV^2, the middle of the CsI and Ar ranges.
    scale = np.sqrt(0.3 * SQRT2 * GF * (mass_mev**2 + 2000))
    g_nu = random_flavor_matrix(rng, scale)
    g_q = {q: scale * rng.uniform(-1, 1) for q in ("u", "d")}
    card = ModelCard(
        mediators=[
            Mediator(
                structure="V",
                mass_mev=mass_mev,
                nu_couplings=FlavorMatrix(**g_nu),
                quark_couplings=QuarkCouplings(**g_q),
            )
        ]
    )
    legacy = CorrectedNSIparameters(mz=mass_mev)
    for pair in PAIR_INDEX:
        legacy.gu[pair] = 2 * as_complex(g_nu[pair]) * g_q["u"]
        legacy.gd[pair] = 2 * as_complex(g_nu[pair]) * g_q["d"]
    return card, legacy


def far_from_cancellation(card: ModelCard) -> bool:
    """|Q^{alpha alpha}| above a tenth of its SM value for every flavor at every recoil energy of both detectors."""
    for name in ("csi", "ar"):
        detector = Detector(name)
        t = np.linspace(detector.er_min, detector.er_max, 50)
        for isotope in detector_isotopes(detector, HELM_RADIUS_FM[name]):
            q2 = isotope.q2(t)
            total = np.abs(np.diagonal(vector_charge(sm(q2) + card.coefficients(q2), isotope.z, isotope.n)))
            standard = np.abs(np.diagonal(vector_charge(sm(q2), isotope.z, isotope.n)))
            if np.any(total < 0.1 * standard):
                return False
    return True


def nsi_points() -> list[tuple[str, ModelCard, NSIparameters]]:
    rng = np.random.default_rng(20261017)
    kinds = [("mediator", m) for m in (5.0, 30.0, 300.0, 5.0, 30.0, 300.0, 30.0)] + [("contact", 0.0)] * 3
    points = []
    for kind, mass in kinds:
        card, legacy = draw_point(rng, kind, mass)
        while not far_from_cancellation(card):
            card, legacy = draw_point(rng, kind, mass)
        points.append((f"{kind}-{mass:g}-{len(points)}", card, legacy))
    return points


POINTS = nsi_points()
FLAVOR_FLUXES = [("coherent", f) for f in ("e", "mu", "mubar")] + [
    ("all_flavors", f) for f in ("ebar", "tau", "taubar")
]


@pytest.fixture(scope="module")
def fluxes():
    return {"coherent": NeutrinoFluxFactory().get("coherent"), "all_flavors": coherent_all_flavors()}


def test_points_cover_off_diagonal_u_and_d():
    for _, card, _ in POINTS:
        c = card.coefficients(np.array([1000.0]))
        for quark in ("u", "d"):
            assert np.all(np.abs(c.values["V", quark][[0, 0, 1], [1, 2, 2]]) > 0)


def test_legacy_ed_uses_gu():
    legacy = NSIparameters(mz=10.0)
    legacy.gu["ee"] = 1.0
    assert legacy.ed()[0, 0] == legacy.eu()[0, 0] != 0


@pytest.mark.parametrize("detector_name", ["csi", "ar"])
@pytest.mark.parametrize(("label", "card", "legacy_nsi"), POINTS, ids=[p[0] for p in POINTS])
def test_vector_rates_match_legacy(fluxes, detector_name, label, card, legacy_nsi):
    detector = Detector(detector_name, efficiency=eff_coherent if detector_name == "csi" else None)
    radius = HELM_RADIUS_FM[detector_name]
    new_detector = production_detector(detector, radius)
    new_flux = coherent_flux()
    extra_flux = ProductionFlux(
        new_flux.energy_mev,
        {f: new_flux.continuous["e"] for f in ("ebar", "tau", "taubar")},
        {"tau": ((29.0, 1.0),)},
        new_flux.normalization_cm2_s,
    )
    production_fluxes = {"coherent": new_flux, "all_flavors": extra_flux}
    legacy = NeutrinoNucleusElasticVector(legacy_nsi, HelmFormFactor(radius))
    recoil = np.linspace(detector.er_min, detector.er_max, 50)

    for flux_name, flavor in FLAVOR_FLUXES:
        flux = fluxes[flux_name]
        expected = np.array(
            [events_per_mev(np.real(legacy.rates(t, flavor, flux, detector)), detector) for t in recoil]
        )
        new = np.array(
            [
                differential_rate(t, production_fluxes[flux_name], new_detector, 4466, card.coefficients, flavor=flavor)
                for t in recoil
            ]
        )
        assert np.any(expected > 0)
        np.testing.assert_allclose(new, expected, rtol=1e-4, atol=1e-6, err_msg=f"{detector_name} {flavor}")
