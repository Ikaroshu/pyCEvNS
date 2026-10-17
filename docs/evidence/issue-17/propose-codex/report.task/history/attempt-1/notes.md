# Gauged L_mu minus L_tau with explicit photon mixing

The described point is m = 10 MeV, g_prime = +1e-4, and epsilon = +1e-3. Following context/conventions.md sections 6 and 7 and worked example 03 A, kinetic mixing means constant low-energy photon mixing, with induced coupling +epsilon e Q_EM. The supplied epsilon is the mixing used in full; no additional loop-induced default or momentum-dependent loop contribution is added. Hypercharge mixing and Z–Z-prime mass mixing are not specified.

The L_mu minus L_tau charges of (nu_e, nu_mu, nu_tau) are (0, +1, -1), so the left-handed neutrino coupling matrix is diag(0, g_prime, -g_prime). Quarks have zero direct charge. Photon mixing gives g_u = +(2/3) epsilon e and g_d = -(1/3) epsilon e. The plugin imports E_CHARGE and Q_EM rather than redefining them.

For each quark f, spacelike vector exchange generates

    L_eff = -g_nu g_f / (Q² + m²) (nu_bar gamma^mu P_L nu)(f_bar gamma_mu f).

Matching to the framework convention -sqrt(2) GF C times these currents yields

    C_mm^(fV)(Q²) = g_prime epsilon e Q_EM[f] / (sqrt(2) GF (Q² + m²)),
    C_tt^(fV)(Q²) = -C_mm^(fV)(Q²).

Here Q² = q2 is nonnegative in MeV² and m² = 100 MeV². GF has units MeV^-2, so the coefficients are dimensionless. There is no extra factor of two. All electron-flavor, off-diagonal, and scalar new-physics coefficients vanish. The finite-mass propagator is retained, without a contact approximation. Antineutrinos use the same coefficients; their treatment belongs to the framework.

As a charge check, 2 g_u + g_d = epsilon e and g_u + 2 g_d = 0. Thus the nuclear shift is proton-only:

    Delta Q_mm = Z g_prime epsilon e / (sqrt(2) GF (Q² + m²)),
    Delta Q_tt = -Delta Q_mm.

Positive g_prime and epsilon give a positive muon-flavor shift, opposite to the negative Standard Model nuclear charge for CsI, hence destructive interference; the tau-flavor shift has the opposite sign. Electron neutrinos receive no new contribution.

The card represents one vector mediator with zero direct quark couplings and explicit kinetic_mixing = 1e-3. The plugin independently computes the coefficients from parameters and exported constants, without reading the card or any files. Each populated entry is set once. COUPLINGS contains g_prime and kinetic_mixing; zeroing them removes all new physics. Standard Model terms and cross sections are supplied only by the framework.

Validation uses nuagent validate --plugin plugin.py --card card.json. Agreement between the two representations is a finite-grid consistency check within the framework's spin-0 vector/scalar treatment, not an independent proof of the physical interpretation.

Validation result: exit status 0, valid = true, and no findings. The card schema, perturbativity, zero-coupling Standard Model limit, and finite nonnegative counts passed. The plugin/card coefficient cross-check passed on 66 Q² points spanning the selected recoil ranges. No observed data or likelihood values were inspected, and the described parameter point was preserved.
