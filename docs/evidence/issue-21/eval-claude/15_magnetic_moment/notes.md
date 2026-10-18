# Notes: neutrino magnetic moment μ_ν = 1e-10 μ_B

Outcome: **notes-only**. The interaction is outside the vector/scalar operator basis of this framework, so no `plugin.py`, `card.json` or `questions.json` is produced. The description itself is complete (one number, one structure); nothing is missing, so this is not a questions outcome either.

## 1. What the description specifies

A neutrino magnetic (dipole) moment of magnitude μ_ν = 1e-10 μ_B, where μ_B = e/(2 m_e) is the Bohr magneton. The operator is

    L ⊃ −(μ_ν / 2) ν̄_α σ^{μν} ν_β F_μν + h.c.,       σ^{μν} = (i/2)[γ^μ, γ^ν],

a dimension-5 dipole coupling of the neutrino to the photon. Scattering on the nucleus proceeds by photon exchange with the nuclear electric charge Z e.

Because the description quotes a single number with no flavor structure, the natural reading is an *effective* magnetic moment μ_eff² = Σ_β |μ_αβ|² applied equally to every incoming flavor α (ν_e, ν_μ, ν̄_μ at COHERENT). Whether the neutrino is Dirac (diagonal and transition moments allowed) or Majorana (transition moments only) does not change the coherent rate at this level, since the final-state neutrino is unobserved and summed over.

## 2. Why the V/S basis cannot express it

The framework accepts only

- `V`: −√2 G_F C^{fV}(Q²) [ν̄_α γ^μ P_L ν_β][f̄ γ_μ f], coherent nuclear charge Q = Z(2C^u + C^d) + N(C^u + 2C^d), interfering with the Standard Model, and
- `S`: a scalar [ν̄_α ν_β][q̄ q] contact, helicity-flipping, added incoherently with its own fixed kinematic factor.

The dipole operator differs from both in three ways that cannot be absorbed into a Q²-dependent coefficient:

1. **Lorentz structure.** σ^{μν} F_μν is a tensor-type (dipole) structure. The schema and `Coefficients.set` accept only `V` and `S`; axial, pseudoscalar and tensor are rejected (conventions section 2).
2. **Kinematics.** The dipole cross section carries the photon pole, 1/Q² = 1/(2 M T), and a recoil dependence (1/T − 1/E_ν + T/(4E_ν²)) that is not the spin-0 vector factor (1 − M T/(2E_ν²) + …) nor the scalar factor the framework uses. No choice of C^{fV}(Q²) or C^{qS}(Q²) reproduces this T and E_ν dependence, because the coefficients may depend on Q² but not separately on E_ν.
3. **Interference.** The dipole amplitude flips the neutrino helicity, so for massless external neutrinos it does not interfere with the SM vector amplitude; it adds incoherently. A vector coefficient would wrongly interfere with Q_SM. The scalar slot is also incoherent but has the wrong recoil spectrum (point 2), and the scalar nuclear charge is built from quark scalar charges f_q^N, not from the electric charge Z.

The dipole rate also couples to the nuclear *electric* charge Z only (neutrons do not contribute at leading order), whereas the V and S charges are both built from u and d quark counts of protons and neutrons. One could tune g_u, g_d to make the vector charge ∝ Z (g_p ≠ 0, g_n = 0), but points 1–3 still exclude a faithful mapping.

## 3. The user-authored raw-cross-section route

`plugin_api.md` provides an escape hatch for exactly this case: a raw plugin exposing

    dsigma_dt(e_nu, t, flavor, isotope, params) -> np.ndarray      # MeV^-3

together with `PARAMETERS` and `COUPLINGS`. It must return the **complete** differential cross section, Standard Model included, with the experiment's form factor. Per the output contract and `framework.md`, this plugin is written by the user, not by the agent; the report is labeled `raw-cross-section` and skips the SM-limit and card cross-checks. The agent does not author it. For the user's reference, the physics it needs to contain is:

**Standard Model part**, obtained from the framework (do not re-derive):

    nuagent.physics.xsec.dsigma_dt(e_nu, t, flavor, isotope,
                                   nuagent.physics.couplings.sm(isotope.q2(t)))

**Magnetic-moment part**, added incoherently (no interference term):

    dσ_μ/dT = α Z² μ_ν² [ 1/T − 1/E_ν + T/(4 E_ν²) ] F²(Q²),       Q² = 2 M T = isotope.q2(t),

with μ_ν in natural units (MeV⁻¹). Equivalently, with μ_ν written in Bohr magnetons,

    dσ_μ/dT = (π α² / m_e²) (μ_ν/μ_B)² Z² [ 1/T − 1/E_ν + T/(4 E_ν²) ] F²(Q²).

The two forms agree because μ_B = e/(2 m_e) gives μ_ν² = (μ_ν/μ_B)² π α / m_e². The term T/(4E_ν²) is often dropped; it is below the percent level at COHERENT energies and may be kept or omitted consistently. F(Q²) should be the same Helm form factor the framework uses, built from `isotope.helm_radius_fm`, so that the SM and dipole terms share one nuclear form factor. The dipole term is the same for neutrinos and antineutrinos and for every flavor under the effective-moment reading of section 1; it vanishes for `flavor` values whose moment the user wants to set to zero.

Inputs the plugin author needs that the framework does not export: m_e = 0.51099895 MeV and the fine-structure constant α. For consistency with the framework's `E_CHARGE = 0.303`, use α = E_CHARGE²/(4π) ≈ 7.306e-3 (versus 1/137.036 = 7.297e-3, a 0.1 percent difference in α, 0.2 percent in the rate).

**Parameter point to preserve.** `PARAMETERS = {"mu_nu_bohr": 1e-10}` with `COUPLINGS = ("mu_nu_bohr",)`. Numerically:

| quantity | value |
|---|---|
| μ_B = e/(2 m_e) | 0.303 / (2 × 0.51099895 MeV) = 0.29648 MeV⁻¹ |
| μ_ν = 1e-10 μ_B | 2.9648e-11 MeV⁻¹ |

The described value 1e-10 μ_B is kept exactly; nothing was tuned or rescaled.

## 4. Summary

- Structure: dipole (tensor-type) ν̄ σ^{μν} ν F_μν, photon exchange, helicity-flipping, 1/T spectrum, Z² scaling.
- Not representable as `V` or `S` coefficients: wrong Lorentz structure, wrong interference pattern, and an E_ν-dependent kinematic factor that a C(Q²) cannot carry.
- Route: user-authored raw cross-section plugin as described in `plugin_api.md`, SM part from the framework plus the incoherent dipole term above, at μ_ν = 1e-10 μ_B.
- Files produced: this `notes.md` only.
