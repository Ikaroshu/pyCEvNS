# Universal light vector interaction

The round-2 answer fixes the mass to 30 MeV and the common positive coupling to g_prime = 1e-4. The neutrino coupling is flavor diagonal and equal for all three left-handed flavors. The up and down quarks have the same vector coupling. "Nothing else" sets flavor-changing couplings, kinetic mixing, scalar interactions, and additional contact interactions to zero. All previous questions are resolved.

In the supplied conventions the interaction is

    L_int = g_prime Zprime_mu [sum_alpha nu-bar_alpha gamma^mu P_L nu_alpha
                              + u-bar gamma^mu u + d-bar gamma^mu d].

Here alpha runs over e, mu, tau. The neutrino coupling multiplies P_L directly; no factor-of-two conversion is needed. The quark couplings multiply full vector currents, with no axial contribution.

For spacelike momentum transfer Q², exchange produces the current-current term

    -g_prime² / (Q² + mass_mev²) [nu-bar_alpha gamma^mu P_L nu_alpha]
                                [f-bar gamma_mu f],  f = u, d.

Matching to -sqrt(2) G_F C times these currents gives

    C^{uV}_{alpha beta}(Q²) = C^{dV}_{alpha beta}(Q²)
                          = delta_{alpha beta} g_prime²
                            / [sqrt(2) G_F (Q² + mass_mev²)].

Every scalar and flavor-changing coefficient is zero. Energies are in MeV, so Q² and mass_mev² have units MeV²; the exported G_F has units MeV^-2, making C dimensionless. The full propagator is retained without a contact approximation.

The coefficient is positive. Its nuclear charge shift is 3(Z + N) C, opposite to the negative Standard Model nuclear charge for the relevant Cs and I nuclei. Thus the interference term is destructive; this alone does not imply the total rate is below the Standard Model rate when the new amplitude is large. The framework adds the Standard Model once and handles antineutrinos and detector response.

This is a phenomenological vector interaction within the supported basis, not a specified ultraviolet gauge completion. A gauged interpretation of this restricted charge assignment would need anomaly cancellation and electroweak completion; no additional interactions are inferred here.

The card represents one constant-coupling vector mediator. The plugin independently computes the coefficients using exported GF and SQRT2, without reading the card or any other file. Each coefficient is set once. PARAMETERS preserves the described mass and common coupling exactly, and COUPLINGS identifies the common coupling for the Standard Model-limit check.
