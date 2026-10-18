# Notes: new light vector boson coupling to neutrinos

Description: "A new light vector boson that couples to neutrinos."

## What the description fixes

1. Lorentz structure: a vector boson, so structure `V` (conventions section 3).
   Only the left-handed neutrino current `nu-bar gamma^mu P_L nu` scatters, so the
   relevant neutrino coupling is g_nu in `g_nu Z'_mu nu-bar_alpha gamma^mu P_L nu_beta`.

## What is missing

The vector matching of conventions section 3,

    C^{qV}_{alpha beta}(Q^2) = g_nu^{alpha beta} g_q / ( sqrt(2) G_F (Q^2 + m^2) ),

needs every one of the following, and the description gives none of them:

1. Mediator mass m. "Light" is not a number.
2. Neutrino flavor structure: which diagonal entries (ee, mm, tt) and whether any
   off-diagonal entries (em, et, mt) are present, with their relative strengths.
   "Couples to neutrinos" does not say universal, single-flavor or otherwise.
3. Neutrino coupling value g_nu, in the left-handed normalization above.
4. Quark couplings g_u and g_d. The description mentions no coupling to quarks at
   all. A boson that couples only to neutrinos gives no coherent elastic
   neutrino-nucleus scattering at tree level, so quark couplings are essential.
   They may come from direct U(1)' charges, from nucleon-level couplings
   (converted with g_u = (2 g_p - g_n)/3, g_d = (2 g_n - g_p)/3), or from kinetic
   mixing with the photon (g_q = epsilon e Q_q^em, with the sign of epsilon part of
   the model). Any of these forms fixes g_u and g_d; the questions ask for g_u
   and g_d separately, as conventions section 11 requires, and accept a kinetic
   mixing value instead.

## Outcome

This is an under-specified description within the V/S basis. `questions.json`
asks for the mass, the flavor structure, the neutrino coupling and the two quark
couplings. No `plugin.py` or `card.json` is written, because every coefficient
would otherwise rest on invented numbers.

The suggested values (10 MeV mass, coupling 1e-4) are conventional CEvNS
light-mediator benchmarks only, and are not part of the description. No
suggestion is offered for the flavor structure or the quark couplings, since no
single conventional choice exists.

## After the answers

Once the mass, flavor entries, g_nu, g_u and g_d are given, the translation is a
plugin with `PARAMETERS` holding those values, `COUPLINGS` listing g_nu, g_u and
g_d, and C^{uV} and C^{dV} set for each stated flavor entry from the formula
above, together with a card with one `V` mediator (as in example 01). The sign of
g_nu g_q then decides whether the interference with the Standard Model nuclear
charge is constructive (g_nu g_q < 0) or destructive (g_nu g_q > 0).
