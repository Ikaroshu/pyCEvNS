# Neutrino magnetic moment, μ_ν = 1e-10 μ_B

## Outcome: notes only (no plugin.py, no card.json)

The described interaction is outside the vector/scalar operator basis of this
framework, so no coefficient plugin and no model card are produced. This is
the notes-only outcome described in `framework.md` and `conventions.md` §2.

## Why the V/S basis cannot express it

The framework's effective Lagrangian (conventions §2) contains only

    −√2 G_F Σ C^{fV}_{αβ} [ν̄_α γ^μ P_L ν_β] [f̄ γ_μ f]      (vector, contact or mediator)

and the scalar structure of §4. Both are four-fermion neutrino–quark operators
whose nuclear charge is a Q²-dependent combination of Z and N with the spin-0
coherent kinematics supplied by the framework.

A neutrino magnetic moment is the dimension-5 dipole operator

    L ⊃ −(μ_ν / 2) ν̄ σ^{μν} ν F_μν   (Dirac; the Majorana case needs a transition moment ν̄_α σ^{μν} ν_β with α ≠ β)

which couples the neutrino to the photon, not to a new vector or scalar current.
Scattering on the nucleus proceeds through t-channel photon exchange with the
nuclear electric charge Z. The resulting cross section

    dσ/dT = (π α² μ_ν² / m_e²) Z² (1/T − 1/E_ν + T/(4 E_ν²)) F²(Q²)
          ≈ (π α² μ_ν² / m_e²) Z² (1/T − 1/E_ν) F²(Q²)

has three properties that no V or S coefficient can reproduce:

1. **Tensor (σ^{μν}) Lorentz structure.** The schema accepts only `V` and `S`;
   axial, pseudoscalar and tensor are rejected (conventions §2). The dipole
   operator is a tensor coupling to the photon field strength.
2. **Helicity flip, no interference with the SM, and a different kinematic
   factor.** The rate goes like 1/T rather than the (1 − M T / 2E_ν²) coherent
   factor used for V, and it is not the scalar kinematic factor either.
   It adds incoherently to the SM rate.
3. **Wrong Q² dependence for any mediator trick.** One might try to mimic the
   1/T ∝ 1/Q² enhancement with a massless vector mediator (propagator
   1/(Q² + m²) with m → 0), but the schema requires m > 0, the rate would scale
   as 1/Q⁴ with the coherent Z-and-N charge rather than Z² alone, and the
   interference pattern with the SM would be wrong. The physics is not a
   neutrino–quark contact or mediator current at all.

The nuclear coupling is also purely electromagnetic (Z only), while the
framework's vector charge Q = Z(2C^u + C^d) + N(C^u + 2C^d) always mixes
protons and neutrons through the quark coefficients. Choosing C^d = −2C^u
(protophilic) would isolate Z but still leaves the wrong Lorentz structure and
kinematics.

## Parameter point as described

    μ_ν = 1e-10 μ_B,    μ_B = e / (2 m_e)

with e = E_CHARGE = 0.303 and m_e = 0.511 MeV, so μ_B ≈ 0.2965 MeV⁻¹ and
μ_ν ≈ 2.965e-11 MeV⁻¹. Flavor is not specified; a flavor-universal
(or ν_μ- and ν_e-equal) effective moment is the natural reading for a
COHERENT-type stopped-pion flux with ν_μ, ν̄_μ and ν_e components. If the user
intends a flavor-specific or transition moment, that should be stated. The
point itself (1e-10 μ_B) is preserved and not tuned.

## Route the user can take

Per `plugin_api.md`, physics outside the V/S basis (magnetic moments are the
named example) goes through the **user-authored raw cross-section plugin**:
a module exposing

    dsigma_dt(e_nu, t, flavor, isotope, params) -> np.ndarray   # MeV⁻³

plus `PARAMETERS` (e.g. `{"mu_nu_bohr": 1e-10}`) and `COUPLINGS`. That
function must return the *complete* differential cross section, i.e. the
framework's SM CEvNS term

    nuagent.physics.xsec.dsigma_dt(e_nu, t, flavor, isotope,
                                   nuagent.physics.couplings.sm(isotope.q2(t)))

added incoherently to the magnetic-moment term

    (π α² μ_ν² / m_e²) · isotope.z² · (1/t − 1/e_nu + t/(4 e_nu²)) · F²(isotope.q2(t))

with α = E_CHARGE²/4π and F the Helm form factor at `isotope.helm_radius_fm`.
Its report would carry the weaker `raw-cross-section` label, skipping the
SM-limit and card cross-checks. The framework documentation states that this
raw plugin is written by the user, not by the agent, so it is not produced here.
