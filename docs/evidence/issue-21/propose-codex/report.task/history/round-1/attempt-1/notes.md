# Gauged L_mu minus L_tau

The described point is m = 10 MeV, g_prime = +1e-4 and kinetic_mixing = +1e-3. Following conventions sections 6 and 7 and worked example 03 A, the supplied kinetic mixing is the constant low-energy photon mixing epsilon. Its sign is positive relative to g_prime, with induced coupling +epsilon e Q_EM. It is used directly, without adding or substituting the loop-induced default; no momentum-dependent loop mixing was requested.

The gauge boson contributes to the vector structure. The neutrino charges (e, mu, tau) are (0, +1, -1), so the left-handed neutrino couplings are (0, +g_prime, -g_prime). Direct quark charges vanish. Photon mixing produces

    g_u = epsilon e (2/3) = +2.02e-4,
    g_d = epsilon e (-1/3) = -1.01e-4.

For spacelike Q squared = q2 >= 0, integrating out the vector gives the effective interaction -g_nu g_q/(q2 + m^2) times the neutrino and quark currents. Matching to -sqrt(2) G_F C times those currents yields

    C_mm^(qV)(q2) = g_prime epsilon e Q_EM[q] / (sqrt(2) G_F (q2 + m^2)),
    C_tt^(qV)(q2) = -C_mm^(qV)(q2).

All electron-flavor, off-diagonal and scalar entries vanish. Coefficients are dimensionless: G_F has units MeV^-2 and q2 + m^2 has units MeV^2. The full mediator propagator is retained. The plugin imports the framework constants, computes these coefficients independently of the card, and supplies only new physics; the framework adds the Standard Model and handles antineutrinos.

The induced nucleon couplings satisfy 2 g_u + g_d = epsilon e and g_u + 2 g_d = 0. Thus the nuclear shift for muon neutrinos is +Z g_prime epsilon e/(sqrt(2) G_F (q2 + m^2)), with the opposite shift for tau neutrinos. The muon-neutrino shift interferes destructively with the negative Standard Model nuclear charge; electron neutrinos have no new contribution. This sign statement concerns the interference term, not a guarantee of rate suppression at every momentum.

The card expresses this interaction exactly using one vector mediator with zero direct quark couplings and explicit kinetic mixing. Both coupling parameters are listed in COUPLINGS; zeroing them removes all new coefficients. No missing inputs require questions.json.
