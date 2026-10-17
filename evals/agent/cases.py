"""Executable eval seeds and full-precision answers. Never copy this module into agent context."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from nuagent.physics.couplings import Coefficients, E_CHARGE, GF, KAPPA_LMU_LTAU, M_MU, M_TAU, Q_EM, SQRT2

Q2_MEV2 = np.array([1238.0, 3714.0, 6190.0])
ALPHA_G_PRIME = np.sqrt(4 * np.pi * 3.899296105751435e-10)


@dataclass(frozen=True)
class NotesRequirement:
    label: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    description: str
    card_expected: bool
    expected_coefficients: Coefficients | None
    notes_requirements: tuple[NotesRequirement, ...] = ()


def mediator_coefficients(
    structure: str,
    mass_mev: float,
    neutrinos: dict[str, float],
    quarks: dict[str, float | np.ndarray],
) -> Coefficients:
    """Match the stated couplings directly, independently of the model-card implementation."""
    c = Coefficients.zeros(Q2_MEV2)
    for quark, g_q in quarks.items():
        for pair, g_nu in neutrinos.items():
            c.set(structure, quark, pair, g_nu * g_q / (SQRT2 * GF * (Q2_MEV2 + mass_mev**2)))
    return c


def universal_coefficients(mass_mev: float, g_nu: float, g_u: float, g_d: float) -> Coefficients:
    return mediator_coefficients("V", mass_mev, dict.fromkeys(("ee", "mm", "tt"), g_nu), {"u": g_u, "d": g_d})


def lmu_ltau_coefficients(mass_mev: float, g_prime: float, mixing: float | np.ndarray) -> Coefficients:
    return mediator_coefficients(
        "V", mass_mev, {"mm": g_prime, "tt": -g_prime},
        {quark: mixing * E_CHARGE * charge for quark, charge in Q_EM.items()},
    )


def running_mixing(g_prime: float) -> np.ndarray:
    """Full spacelike one-loop integral, evaluated by adaptive quadrature at the reference points."""
    integrals = [
        quad(
            lambda x: x * (1 - x) * np.log((M_TAU**2 + x * (1 - x) * q2) / (M_MU**2 + x * (1 - x) * q2)),
            0.0, 1.0, epsabs=1e-12, epsrel=1e-12,
        )[0]
        for q2 in Q2_MEV2
    ]
    return E_CHARGE * g_prime / (2 * np.pi**2) * np.array(integrals)


def contact_coefficients() -> Coefficients:
    c = Coefficients.zeros(Q2_MEV2)
    c.set("V", "d", "mm", -0.05)
    c.set("V", "d", "et", 0.02 + 0.03j)
    return c


def dipole_coefficients() -> Coefficients:
    form_factor = 1 / (1 + Q2_MEV2 / 200.0**2) ** 2
    return mediator_coefficients(
        "V", 30.0, dict.fromkeys(("ee", "mm", "tt"), 1e-4),
        {"u": 1e-4 * form_factor, "d": 1e-4 * form_factor},
    )


MISSING_INFORMATION = r"\b(?:missing|unspecified|not\s+(?:given|specified)|unknown|need(?:ed)?|require[ds]?|provide|specify|what|which)\b"

CASES = (
    EvalCase(
        "01_universal_vector",
        "A universal vector Z′ of mass 45 MeV couples with g′ = 7e-5 to all three left-handed neutrino flavors "
        "and to up and down quarks. There are no other couplings.",
        True, universal_coefficients(45.0, 7e-5, 7e-5, 7e-5),
    ),
    EvalCase(
        "02_alpha_conversion",
        "A universal vector Z′ of mass 45 MeV has alpha′ = g′²/(4 pi) = 3.899296105751435e-10. "
        "Its positive coupling g′ is the same for all left-handed neutrino flavors and both u and d quarks, "
        "with no other couplings.",
        True, universal_coefficients(45.0, ALPHA_G_PRIME, ALPHA_G_PRIME, ALPHA_G_PRIME),
    ),
    EvalCase(
        "03_b_minus_l",
        "The U(1) B−L gauge boson has mass 25 MeV and gauge coupling g′ = 3e-5, with no kinetic mixing.",
        True, universal_coefficients(25.0, -3e-5, 3e-5 / 3, 3e-5 / 3),
    ),
    EvalCase(
        "04_lmu_ltau_explicit",
        "A gauged L_mu−L_tau vector Z′ has mass 10 MeV, gauge coupling g′ = 1e-4 and photon kinetic "
        "mixing epsilon = +1e-3 in the conventions of this pack. There are no direct quark couplings.",
        True, lmu_ltau_coefficients(10.0, 1e-4, 1e-3),
    ),
    EvalCase(
        "05_lmu_ltau_loop_limit",
        "A gauged L_mu−L_tau vector Z′ has mass 15 MeV and gauge coupling g′ = 6e-5. Use only the "
        "positive loop-induced photon mixing in the Q² → 0 limit; there is no tree-level mixing or direct quark coupling.",
        True, lmu_ltau_coefficients(15.0, 6e-5, KAPPA_LMU_LTAU * 6e-5),
    ),
    EvalCase(
        "06_lmu_ltau_negative_mixing",
        "A gauged L_mu−L_tau vector Z′ has mass 15 MeV, gauge coupling g′ = 6e-5, total photon kinetic "
        "mixing epsilon = −5e-4 in the pack's convention, and no direct quark couplings.",
        True, lmu_ltau_coefficients(15.0, 6e-5, -5e-4),
    ),
    EvalCase(
        "07_scalar_down",
        "A scalar phi of mass 20 MeV has Dirac couplings 2e-5 phi nu-bar_alpha nu_alpha to all three "
        "neutrino flavors and 2e-5 phi d-bar d to down quarks only. All other couplings vanish.",
        True, mediator_coefficients("S", 20.0, dict.fromkeys(("ee", "mm", "tt"), 2e-5), {"d": 2e-5}),
    ),
    EvalCase(
        "08_complex_contact",
        "Contact vector NSI on down quarks has epsilon^{dV}_{mu mu} = −0.05 and "
        "epsilon^{dV}_{e tau} = 0.02 + 0.03i, with its Hermitian partner. All other entries vanish.",
        True, contact_coefficients(),
    ),
    EvalCase(
        "09_antineutrino",
        "Muon antineutrinos scatter through a universal vector Z′ of mass 30 MeV with g′ = 8e-5. "
        "The same coupling applies to all left-handed neutrino flavors and to both u and d quarks, with no other couplings.",
        True, universal_coefficients(30.0, 8e-5, 8e-5, 8e-5),
    ),
    EvalCase(
        "10_left_handed_normalization",
        "A universal vector Z′ has mass 30 MeV. Its neutrino interaction is "
        "g Z′_mu nu-bar gamma^mu (1 − gamma5) nu for every flavor, with g = 5e-5 and no factor 1/2. "
        "Its vector coupling to each of u and d quarks is 1e-4. All other couplings vanish.",
        True, universal_coefficients(30.0, 2 * 5e-5, 1e-4, 1e-4),
    ),
    EvalCase(
        "11_protophobic",
        "A protophobic vector mediator has mass 20 MeV, universal left-handed neutrino coupling g_nu = 1e-4, "
        "and vector nucleon couplings g_p = 0 and g_n = 3e-4. There are no other interactions.",
        True, universal_coefficients(20.0, 1e-4, (2 * 0.0 - 3e-4) / 3, (2 * 3e-4 - 0.0) / 3),
    ),
    EvalCase(
        "12_running_mixing",
        "A gauged L_mu−L_tau vector Z′ has mass 12 MeV and gauge coupling g′ = 8e-5. "
        "Keep the full spacelike Q² dependence of the muon/tau one-loop photon mixing, "
        "with no tree-level mixing or direct quark coupling.",
        False, lmu_ltau_coefficients(12.0, 8e-5, running_mixing(8e-5)),
    ),
    EvalCase(
        "13_dipole_quark_coupling",
        "A universal vector mediator has mass 30 MeV and coupling g′ = 1e-4 to all left-handed neutrino flavors. "
        "Each u and d quark coupling is g′ times the dipole form factor 1/(1 + Q²/Lambda²)², "
        "with Lambda = 200 MeV. All other couplings vanish.",
        False, dipole_coefficients(),
    ),
    EvalCase(
        "14_under_specified",
        "A new light vector boson that couples to neutrinos.",
        False, None,
        (
            NotesRequirement("missing mediator mass", (r"\bmass\b", MISSING_INFORMATION)),
            NotesRequirement("missing neutrino coupling", (r"\b(?:g_nu|neutrino couplings?)\b", MISSING_INFORMATION)),
            NotesRequirement("missing quark coupling", (r"\b(?:g_q|quark couplings?)\b", MISSING_INFORMATION)),
        ),
    ),
    EvalCase(
        "15_magnetic_moment",
        "A neutrino magnetic moment of 1e-10 Bohr magnetons.",
        False, None,
        (
            NotesRequirement(
                "outside the vector/scalar basis",
                (r"\b(?:outside|beyond|unsupported|not supported)\b", r"\bbasis\b", r"\b(?:v\s*[,/]?\s*(?:and\s+)?s|vector[ /,]+(?:and\s+)?scalar)\b"),
            ),
            NotesRequirement(
                "user-authored raw cross-section route",
                (r"\braw[- ]cross[- ]section\b", r"\b(?:user[- ](?:authored|written)|written by (?:the )?user)\b"),
            ),
        ),
    ),
)
