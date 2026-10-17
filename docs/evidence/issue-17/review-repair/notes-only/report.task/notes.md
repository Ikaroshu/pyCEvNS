# Neutrino magnetic moment μ_ν = 1e-10 μ_B — notes-only outcome

## Outcome

No `plugin.py` and no `card.json` are produced. A neutrino magnetic moment is
outside the vector/scalar operator basis of this framework, so it cannot be
translated into coefficients `C^{fV}` or `C^{qS}`. The only supported route is
the user-authored raw cross-section plugin described in `context/plugin_api.md`.

## Why the previous attempt is discarded

The files in `history/attempt-1/` were a seeded fixture: a flavor-universal
45 MeV Z′ coefficient plugin together with an empty card `{}`. They do not
describe a magnetic moment at all, and the validation findings (plugin/card
mismatch on V u and V d) simply reflect that the plugin set non-zero vector
coefficients while the empty card set none. The correct repair is not to make
the card match the Z′ plugin, because neither file represents the described
interaction. The correct repair is to withdraw both files.

## Why the V/S basis cannot express a magnetic moment

The framework's effective Lagrangian (conventions.md, section 2) contains only

    −√2 G_F C^{fV}_{αβ}(Q²) [ν̄_α γ^μ P_L ν_β][f̄ γ_μ f]        (vector)

and the scalar operator of section 4. Both are four-fermion contact-type
operators between the neutrino current and a quark current, with a propagator
`1/(Q² + m²)` for a massive mediator.

A neutrino magnetic moment is the dimension-5 dipole operator

    L ⊃ −(μ_ν / 2) ν̄ σ^{μν} ν F_μν     (Dirac; a transition moment ν̄_α σ^{μν} ν_β for Majorana)

which couples the neutrino to the photon, not to a quark vector or scalar
current. Its contribution to coherent elastic neutrino–nucleus scattering
differs from the V/S operators in every respect that matters here:

1. **Lorentz structure.** The operator is tensor (σ^{μν}) on the neutrino line
   and couples to the electromagnetic field strength. Axial, pseudoscalar and
   tensor structures are rejected by the card schema and by `Coefficients.set`,
   which accepts only `"V"` and `"S"`.
2. **Momentum dependence.** Photon exchange gives a massless propagator
   `1/Q²`, and the dipole vertex supplies a further factor of q, so the
   differential cross section behaves as `(1/T − 1/E_ν + T/(4E_ν²))` rather than
   the `(1 − T/T_max)`-type factor of the vector channel. The framework's
   kinematic factor for V and for S is fixed internally and cannot be replaced
   from a coefficient plugin.
3. **Nuclear charge.** The dipole couples to the nuclear electric charge Z
   times the charge form factor, not to the weak vector charge
   `Z(2C_u + C_d) + N(C_u + 2C_d)` or the scalar charge built from `f_q^N`.
   There is no choice of `C^{uV}`, `C^{dV}` that reproduces a pure-Z coupling
   with `1/Q²` momentum dependence together with the different kinematic factor.
4. **Interference.** The magnetic-moment amplitude flips the neutrino helicity
   and therefore does not interfere with the Standard Model vector amplitude.
   The scalar channel also flips helicity, but with a different kinematic
   factor and a different nuclear charge, so faking the dipole through `S`
   coefficients would give the wrong recoil spectrum.

Attempting to encode μ_ν as an effective `C^{fV}(Q²)` with a `1/Q²` propagator
would also break the framework's SM-limit and finiteness checks at Q² → 0 and
would misrepresent the physics. Per `context/framework.md`, a magnetic moment is
a legitimate notes-only outcome.

## The description, for the record

- Parameter: μ_ν = 1e-10 μ_B (Bohr magnetons), where
  μ_B = e / (2 m_e) ≈ 5.788e-11 MeV/T in SI-ish units, or in natural units
  μ_B = e/(2 m_e) with e = 0.303 and m_e = 0.511 MeV.
- Flavor: not specified. For COHERENT CsI the flux contains ν_μ, ν̄_μ and ν_e,
  so the user must decide whether 1e-10 μ_B applies to one flavor or to all
  (an effective μ_ν per flavor, or a transition-moment matrix for Majorana
  neutrinos).
- Dirac vs Majorana: not specified. At the level of the CEvNS rate the
  distinction does not change the formula once an effective μ_ν per incoming
  flavor is fixed.

## The user-authored raw cross-section route

If predictions for this point are wanted, a user (not the agent) writes a raw
plugin exposing

    dsigma_dt(e_nu, t, flavor, isotope, params) -> np.ndarray   # MeV⁻³

with `PARAMETERS = {"mu_nu_bohr": 1e-10}` and `COUPLINGS = ("mu_nu_bohr",)`.
The function must return the **complete** differential cross section: the
framework's Standard Model term obtained from

    nuagent.physics.xsec.dsigma_dt(e_nu, t, flavor, isotope,
                                   nuagent.physics.couplings.sm(isotope.q2(t)))

plus the incoherent magnetic-moment term, which for a spin-0 nucleus is

    dσ_μ/dT = (π α² μ_ν² / m_e²) Z² (1/T − 1/E_ν + T/(4 E_ν²)) F_ch(Q²)²

with μ_ν in units of μ_B, α = e²/4π, m_e the electron mass, Z = `isotope.z`,
and the Helm charge form factor evaluated at Q² = `isotope.q2(t)` with
`isotope.helm_radius_fm`. The two pieces add without interference. The user
owns the normalization of this term, the choice of flavor(s) to which μ_ν is
applied, and the form factor. The harness labels such a report
`raw-cross-section`, which skips the SM-limit check and the plugin/card
cross-check.

## Information that would be needed if a basis extension existed

Even with a raw plugin, the user should state:

1. Which neutrino flavor(s) carry μ_ν = 1e-10 μ_B (ν_e only, ν_μ only, or
   flavor-universal), and whether antineutrinos are included.
2. Whether a Dirac diagonal moment or a Majorana transition-moment matrix is
   intended.

No tuning was performed, no evaluation or observed data was inspected, and no
validation command was run because there is no plugin to validate.
