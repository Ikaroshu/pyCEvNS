"""Declarative model card: schema, conversion to Coefficients, and the shipped templates."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nuagent.physics.couplings import (
    KAPPA_LMU_LTAU,
    PAIR_INDEX,
    QUARKS,
    Coefficients,
    flavor_matrix,
    kinetic_mixing_coupling,
    mediator_coefficient,
)

OFF_DIAGONAL_DESCRIPTION = "Off-diagonal entry: a real number, or [re, im] for a complex value. The conjugate partner is filled by the framework."


class CardModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FlavorMatrix(CardModel):
    """Hermitian 3x3 flavor matrix given by its six independent entries; an absent entry is zero."""

    ee: float = Field(0.0, description="Diagonal (e, e) entry, real.")
    mm: float = Field(0.0, description="Diagonal (mu, mu) entry, real.")
    tt: float = Field(0.0, description="Diagonal (tau, tau) entry, real.")
    em: float | tuple[float, float] = Field(0.0, description=f"(e, mu) entry. {OFF_DIAGONAL_DESCRIPTION}")
    et: float | tuple[float, float] = Field(0.0, description=f"(e, tau) entry. {OFF_DIAGONAL_DESCRIPTION}")
    mt: float | tuple[float, float] = Field(0.0, description=f"(mu, tau) entry. {OFF_DIAGONAL_DESCRIPTION}")

    def matrix(self) -> np.ndarray:
        entries = {pair: getattr(self, pair) for pair in PAIR_INDEX}
        return flavor_matrix({pair: complex(*v) if isinstance(v, tuple) else v for pair, v in entries.items()})


class QuarkCouplings(CardModel):
    """Vector (g_q, the average of the chiral couplings) or scalar quark couplings of a mediator."""

    u: float = Field(0.0, description="Coupling to the up quark.")
    d: float = Field(0.0, description="Coupling to the down quark.")


class Mediator(CardModel):
    """One new boson exchanged between the neutrino and the quarks."""

    structure: Literal["V", "S"] = Field(description="Lorentz structure: V (vector) or S (scalar).")
    mass_mev: float = Field(gt=0, description="Mediator mass in MeV, positive.")
    nu_couplings: FlavorMatrix = Field(
        default_factory=FlavorMatrix,
        description="Neutrino couplings g_nu^{alpha beta} (left-handed current for V, nu-bar nu for S).",
    )
    quark_couplings: QuarkCouplings = Field(default_factory=QuarkCouplings, description="Quark couplings g_u, g_d.")
    kinetic_mixing: float = Field(
        0.0,
        description="Vector only: kinetic mixing epsilon with the photon, adding epsilon e Q_q^em to each quark coupling.",
    )

    @model_validator(mode="after")
    def kinetic_mixing_needs_vector(self) -> "Mediator":
        if self.structure != "V" and self.kinetic_mixing != 0:
            raise ValueError("kinetic_mixing is allowed only for structure V")
        return self


class Contact(CardModel):
    """Constant vector coefficients epsilon^{fV}_{alpha beta} added to C^{fV}."""

    fermion: Literal["u", "d"] = Field(description="Quark the contact term couples to.")
    epsilon: FlavorMatrix = Field(description="Hermitian matrix of epsilon^{fV}_{alpha beta}.")


class ModelCard(CardModel):
    """New-physics contribution to the CEvNS operator basis; the Standard Model is added by the framework."""

    name: str = Field("", description="Free-form label for the model.")
    mediators: list[Mediator] = Field(default_factory=list, description="Vector and scalar mediators.")
    contacts: list[Contact] = Field(default_factory=list, description="Contact vector interactions.")

    def coefficients(self, q2: np.ndarray) -> Coefficients:
        """New-physics coefficients on the q2 grid (MeV^2)."""
        c = Coefficients.zeros(q2)
        for mediator in self.mediators:
            g_nu = mediator.nu_couplings.matrix()
            for quark in QUARKS:
                g_q = getattr(mediator.quark_couplings, quark) + kinetic_mixing_coupling(mediator.kinetic_mixing, quark)
                c.values[mediator.structure, quark] += mediator_coefficient(g_nu, g_q, mediator.mass_mev, c.q2)
        for contact in self.contacts:
            c.values["V", contact.fermion] += contact.epsilon.matrix()[:, :, None]
        return c


def u1_card(
    g_prime: float,
    mass_mev: float,
    *,
    q_nu_e: float = 0.0,
    q_nu_mu: float = 0.0,
    q_nu_tau: float = 0.0,
    q_u: float = 0.0,
    q_d: float = 0.0,
    kinetic_mixing: float = 0.0,
    name: str = "",
) -> ModelCard:
    """Vector-mediator card for a gauged U(1)' with coupling g' and charges of the lepton doublets and quarks."""
    mediator = Mediator(
        structure="V",
        mass_mev=mass_mev,
        nu_couplings=FlavorMatrix(ee=g_prime * q_nu_e, mm=g_prime * q_nu_mu, tt=g_prime * q_nu_tau),
        quark_couplings=QuarkCouplings(u=g_prime * q_u, d=g_prime * q_d),
        kinetic_mixing=kinetic_mixing,
    )
    return ModelCard(name=name, mediators=[mediator])


def universal_card(structure: Literal["V", "S"], params: dict[str, float], name: str) -> ModelCard:
    g_q = params.get("g_q", 0.0)
    g_nu = params.get("g_nu", 0.0)
    mediator = Mediator(
        structure=structure,
        mass_mev=params["mass_mev"],
        nu_couplings=FlavorMatrix(ee=g_nu, mm=g_nu, tt=g_nu),
        quark_couplings=QuarkCouplings(u=params.get("g_u", g_q), d=params.get("g_d", g_q)),
    )
    return ModelCard(name=name, mediators=[mediator])


def lmu_ltau_card(params: dict[str, float]) -> ModelCard:
    g_prime = params.get("g_prime", 0.0)
    return u1_card(
        g_prime,
        params["mass_mev"],
        q_nu_mu=1.0,
        q_nu_tau=-1.0,
        kinetic_mixing=params.get("kinetic_mixing", KAPPA_LMU_LTAU * g_prime),
        name="zprime_lmu_ltau",
    )


CONTACT_PARAMETERS = tuple(f"eps_{q}_{pair}" for q in QUARKS for pair in PAIR_INDEX) + tuple(
    f"eps_{q}_{pair}_im" for q in QUARKS for pair in ("em", "et", "mt")
)


def contact_card(params: dict[str, float]) -> ModelCard:
    contacts = []
    for quark in QUARKS:
        entries = {pair: params.get(f"eps_{quark}_{pair}", 0.0) for pair in PAIR_INDEX}
        for pair in ("em", "et", "mt"):
            entries[pair] = (entries[pair], params.get(f"eps_{quark}_{pair}_im", 0.0))
        contacts.append(Contact(fermion=quark, epsilon=FlavorMatrix(**entries)))
    return ModelCard(name="contact_nsi", contacts=contacts)


@dataclass(frozen=True)
class Template:
    build: Callable[[dict[str, float]], ModelCard]
    parameters: tuple[str, ...]
    couplings: tuple[str, ...]
    description: str


U1_PARAMETERS = ("g_prime", "mass_mev", "q_nu_e", "q_nu_mu", "q_nu_tau", "q_u", "q_d", "kinetic_mixing")

TEMPLATES = {
    "zprime_universal": Template(
        lambda p: universal_card("V", p, "zprime_universal"),
        ("g_nu", "g_u", "g_d", "g_q", "mass_mev"),
        ("g_nu", "g_u", "g_d", "g_q"),
        "Vector mediator with flavor-universal g_nu and quark couplings g_u, g_d; g_q sets both.",
    ),
    "zprime_u1": Template(
        lambda p: u1_card(**({"g_prime": 0.0} | p), name="zprime_u1"),
        U1_PARAMETERS,
        ("g_prime", "kinetic_mixing"),
        "Gauged U(1)' with coupling g_prime and charges q_nu_e, q_nu_mu, q_nu_tau, q_u, q_d.",
    ),
    "zprime_lmu_ltau": Template(
        lmu_ltau_card,
        ("g_prime", "mass_mev", "kinetic_mixing"),
        ("g_prime", "kinetic_mixing"),
        "Gauged L_mu - L_tau; kinetic_mixing defaults to the loop value KAPPA_LMU_LTAU * g_prime.",
    ),
    "scalar_universal": Template(
        lambda p: universal_card("S", p, "scalar_universal"),
        ("g_nu", "g_u", "g_d", "g_q", "mass_mev"),
        ("g_nu", "g_u", "g_d", "g_q"),
        "Scalar mediator with flavor-universal g_nu and quark couplings g_u, g_d; g_q sets both.",
    ),
    "contact_nsi": Template(
        contact_card,
        CONTACT_PARAMETERS,
        CONTACT_PARAMETERS,
        "Contact vector epsilons eps_<quark>_<pair>, with eps_<quark>_<pair>_im for off-diagonal imaginary parts.",
    ),
}


def template_card(name: str, params: dict[str, float]) -> ModelCard:
    """Model card of a shipped template at a parameter point; absent couplings are zero."""
    if name not in TEMPLATES:
        raise ValueError(f"unknown template {name!r}; valid templates: {', '.join(TEMPLATES)}")
    template = TEMPLATES[name]
    unknown = sorted(set(params) - set(template.parameters))
    if unknown:
        raise ValueError(
            f"unknown parameters {unknown} for template {name!r}; valid parameters: {', '.join(template.parameters)}"
        )
    return template.build(params)
