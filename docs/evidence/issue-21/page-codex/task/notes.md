The round-2 answers complete the interaction: a vector mediator of mass
30.0 MeV has left-handed neutrino couplings g_nu = 1e-4 for ee, mm and tt,
no off-diagonal couplings, and vector quark couplings g_u = g_d = 1e-4.
These are the supplied values, not benchmark suggestions. The current
normalizations are those explicitly requested in the round-1 questions.

In the conventions of context/conventions.md, the interaction is

    L_int = Zprime_mu [g_nu sum_alpha nu-bar_alpha gamma^mu P_L nu_alpha
                      + g_u u-bar gamma^mu u + g_d d-bar gamma^mu d].

For spacelike momentum transfer Q² >= 0, exchange gives the effective
operator -g_nu g_f/(Q² + m²) times the neutrino and quark currents.
Matching to -sqrt(2) G_F C^{fV}_{alpha alpha} times those currents gives

    C^{fV}_{alpha alpha}(Q²) = g_nu g_f / [sqrt(2) G_F (Q² + m²)]
                            = 1e-8 / [sqrt(2) G_F (Q² + 900 MeV²)],
    f = u, d; alpha = e, mu, tau.

All off-diagonal vector entries and all scalar entries vanish. The flavor
matrices are real and diagonal, hence Hermitian. Couplings are dimensionless;
G_F has units MeV^-2 and Q² and m² have units MeV², so C is dimensionless.
The finite-mass propagator is retained without a contact approximation.
No factor-of-two conversion is needed for the stated left-handed neutrino
current and vector quark currents.

The positive coupling products give positive diagonal vector coefficients.
At the supplied point C^{uV} = C^{dV} = C, so the nuclear charge shift is
Delta Q = 3(Z + N) C > 0. This has the opposite sign to the negative Standard
Model nuclear charge for Cs and I and therefore gives a destructive
interference term; this does not imply that the total rate is below the
Standard Model rate when the new amplitude is large.

The supplied quark couplings are used directly as the effective vector
couplings. No additional kinetic mixing or contact interaction is specified
or added. The schema expresses this interaction with one V mediator.
plugin.py computes the coefficients independently of card.json using exported
GF and SQRT2. Each entry is set once, and only the new-physics contribution
is returned. The framework adds the Standard Model and handles nuclear and
detector physics. No ultraviolet charge assignment is specified.

No inputs remain missing, so no questions.json is needed. Validation checks
the plugin and card representations and the framework sanity conditions;
agreement cannot independently prove the physical interpretation or matching.
