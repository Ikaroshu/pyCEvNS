# Gauged U(1)_{L_μ−L_τ} Z′ with kinetic mixing

Description: "a gauged L_mu minus L_tau Z prime with mass 10 MeV, gauge coupling 1e-4 and kinetic mixing 1e-3".

## Parameter point (preserved exactly)

- m_Z′ = 10 MeV
- g′ = 1e-4
- ε = 1e-3 (kinetic mixing with the photon)

## Derivation

1. Lorentz structure: gauge boson exchange, so structure `V`.
2. Charges (conventions §7): Q′_{ν_e} = 0, Q′_{ν_μ} = +1, Q′_{ν_τ} = −1, Q′_u = Q′_d = 0. The neutrino charge is the charge of the lepton doublet, so the left-handed neutrino current couples with
   g_ν^{μμ} = +g′ = 1e-4, g_ν^{ττ} = −g′ = −1e-4, all other entries zero (flavor-diagonal).
3. Quarks have no direct L_μ−L_τ charge; they couple only through kinetic mixing (conventions §6):
   g_q = ε e Q_q^em, so g_u = ε e (2/3) = 2.02e-4, g_d = ε e (−1/3) = −1.01e-4 (e = 0.303).
4. Matching (conventions §3):
   C^{qV}_{αα}(Q²) = g_ν^{αα} g_q / (√2 G_F (Q² + m²)),
   so C^{uV}_{μμ} = −C^{uV}_{ττ} = g′ ε e (2/3) / (√2 G_F (Q² + m²)), and C^{dV}_{μμ} = −C^{dV}_{ττ} = −g′ ε e (1/3) / (√2 G_F (Q² + m²)).
   C^{qV}_{ee} = 0, all off-diagonal and scalar entries zero.
5. Nuclear charge: since 2Q_u + Q_d = 1 and Q_u + 2Q_d = 0, ΔQ^{μμ} = Z g′ ε e / (√2 G_F (Q² + m²)). Only protons respond (a dark-photon-like coupling); ΔQ^{ττ} = −ΔQ^{μμ}.

## Sign conventions

- The kinetic mixing is taken as the stated total low-energy value ε = +1e-3, entering as g_q → ε e Q_q^em with the framework's sign (kinetic term −(ε/2) F_μν Z′^μν with D_μ = ∂_μ + i e Q A_μ). The description gives the mixing as an explicit number, so the loop-induced value κ g′ = 1.444e-6 (conventions §6) is **not** added on top: the stated ε is interpreted as the full effective mixing. If the user intended 1e-3 as a tree-level mixing in addition to the μ/τ loop, the total would be ε = 1e-3 + 1.44417637e-6; please say so if that is the intent.
- With ε > 0 and g′ > 0, g_ν^{μμ} g_p > 0 has sign opposite to the (negative) SM charge: destructive interference for ν_μ and ν̄_μ; ν_τ would interfere constructively but there is no τ flux at COHERENT; ν_e is unaffected.

## Representation

- Card-expressible: one V mediator with nu_couplings mm = 1e-4, tt = −1e-4, quark_couplings zero, kinetic_mixing = 1e-3.
- The plugin computes the same coefficients independently from the exported constants `E_CHARGE`, `Q_EM`, `GF`, `SQRT2`. `COUPLINGS` = (g_prime, kinetic_mixing); zeroing them gives the SM.
- Mass is given, so the mediator (propagator) form is used, not a contact term. Q²-dependence of the loop-induced mixing (example 06) is irrelevant here because ε is given as a constant.
- Anomaly: L_μ−L_τ is anomaly-free with the SM fermion content.
