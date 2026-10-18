# Incomplete vector interaction

The description specifies a light vector boson coupled to neutrinos, so the relevant structure is V. It does not specify a numerical mediator mass, the strengths or flavor pattern of the neutrino couplings, or the vector couplings to up and down quarks. The word "light" does not determine a mass. Coupling to neutrinos does not establish flavor universality or imply that all quark couplings vanish.

Using the left-handed neutrino current and quark vector currents of context/conventions.md, tree-level exchange would give

    C^{fV}_{alpha beta}(Q²) = g_nu^{alpha beta} g_f / [sqrt(2) G_F (Q² + m²)],  f = u, d.

This follows by matching the exchange term -g_nu^{alpha beta} g_f/(Q² + m²) times the two currents to the effective interaction -sqrt(2) G_F C^{fV}_{alpha beta} times those currents. The neutrino coupling matrix must be Hermitian, with real diagonal entries. The relative signs of the neutrino and quark couplings determine the sign of the new contribution; no sign is fixed by the description.

questions.json requests the missing mass, neutrino coupling matrix, and each quark coupling separately. Quark couplings may be explicitly zero or supplied through a specified mixing mechanism; none is inferred here. The suggested mass of 10 MeV is only a conventional benchmark from the supplied context, not part of the described interaction or an adopted parameter value.

A fully specified constant-coupling vector interaction can be expressed by the card schema. The present obstacle is missing physical inputs, not a basis or schema limitation. No plugin.py or card.json is produced. This questions outcome has no predictions or physics verification label, and plugin validation is not applicable.
