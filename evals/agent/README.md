# Agent eval set (seed list, draft)

Not part of the context pack: nothing here is copied into a task directory. Every case is a variant of a worked example with different numbers, flavors or quarks, so that passing requires translation rather than lookup. Expected coefficients are generated at full precision from the constants exported by `nuagent.physics.couplings` at Q² ∈ {1238, 3714, 6190} MeV² (Cs-133 at T = 5, 15, 25 keV), never copied from the rounded tables in the examples. Each case records: description, whether a card is expected, expected coefficients (or the expected no-output behavior).

1. Universal Z′, m = 45 MeV, g′ = 7e-5 to all neutrino flavors and both quarks. Card expected.
2. Universal Z′ as in 1 but the coupling given as α′ = g′²/4π = 3.899296105751435e-10 (conversion test; expected coefficients computed from this α′, so g′ = √(4π α′)). Card expected.
3. B−L gauge boson, m = 25 MeV, g′ = 3e-5. Card expected.
4. The spec's Outcome description: L_μ−L_τ Z′, m = 10 MeV, g′ = 1e-4, kinetic mixing 1e-3. Card expected.
5. L_μ−L_τ Z′, m = 15 MeV, g′ = 6e-5, loop-induced mixing (κ g′, sign positive). Card expected with the explicit value.
6. L_μ−L_τ Z′, m = 15 MeV, g′ = 6e-5, kinetic mixing −5e-4 (sign test). Card expected.
7. Scalar φ, m = 20 MeV, coupling 2e-5 to all neutrinos and to d quarks only. Card expected.
8. Contact vector NSI on d quarks: ε^{dV}_{mm} = −0.05, ε^{dV}_{eτ} = 0.02 + 0.03i. Card expected (complex `et` entry).
9. Antineutrino phrasing: "muon antineutrinos scattering through a universal Z′ of mass 30 MeV with g′ = 8e-5". Expected coefficients equal the neutrino case. Card expected.
10. Normalization test: a universal Z′, m = 30 MeV, with the neutrino current written as g ν̄ γ^μ (1 − γ5) ν, g = 5e-5 (so g_ν = 1e-4), quark coupling 1e-4. Card expected.
11. Nucleon-level input: a protophobic vector mediator, m = 20 MeV, g_ν = 1e-4 universal, g_p = 0, g_n = 3e-4 (conversion to g_u = −1e-4, g_d = 2e-4). Card expected.
12. Plugin-only: L_μ−L_τ Z′, m = 12 MeV, g′ = 8e-5, with the full Q²-dependent one-loop mixing. No card expected.
13. Plugin-only: a universal vector mediator, m = 30 MeV, g′ = 1e-4, whose quark coupling carries a dipole form factor 1/(1 + Q²/Λ²)² with Λ = 200 MeV. No card expected.
14. Under-specified: "a new light vector boson that couples to neutrinos". Expected: no plugin, `notes.md` names the missing mass, g_ν and g_q.
15. Out of basis: "a neutrino magnetic moment of 1e-10 Bohr magnetons". Expected: no plugin, `notes.md` says the structure is outside the V, S basis and names the user-authored raw cross-section route (consistent with the pack's output contract).

Scoring (spec, Agent eval set): physics accuracy at 1e-6 relative on every coefficient; expressibility recognition from the presence of `card.json` against the expectation above.
