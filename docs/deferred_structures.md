# Deferred Lorentz structures: tensor, axial, pseudoscalar

The prototype accepts the vector (V) and scalar (S) structures and rejects axial (A), pseudoscalar (P) and tensor (T) at the card schema. This note records why adding them is a physics decision rather than a schema change, so that whoever picks this up starts from the right question. Statements marked *inference* have not been checked against the literature.

## What the prototype assumes

The master cross section in `src/nuagent/physics/xsec.py` treats every target isotope as a spin-0 nucleus and writes the rate as a vector term plus a scalar term,

    dσ/dT = (G_F² M/π) Σ_β |Q_V^{αβ}|² (kinematics) F² + (M² T / 4π E²) Σ_β |Q_S^{αβ}|² F²,

with no cross term. Two facts make this form sufficient for V and S:

- Coherence. The vector and scalar quark currents reduce, at the nucleon level, to the nucleon number operator. Every nucleon contributes with the same sign, the nucleus couples through a charge like Z g_p + N g_n, and the rate scales as A². Only Z, N, the mass and a form factor enter; nothing else about the nuclear wavefunction is needed.
- No interference. The scalar amplitude flips the neutrino helicity and the vector amplitude does not, so the two add in quadrature.

The same code maps antineutrino flavors onto the neutrino rows of the coefficient matrices on the explicit assumption that the rate is otherwise unchanged (`FLAVOR_ROW` in `xsec.py`). That holds for V and S on a spin-0 target.

## Axial and pseudoscalar: spin, not number

The axial and pseudoscalar quark currents reduce at the nucleon level to the nucleon spin operator. In a spin-0 nucleus the spins are paired and the coherent sum is exactly zero. In an odd-A nucleus such as ¹³³Cs or ¹²⁷I one unpaired nucleon remains, so the amplitude is of order one instead of order A: roughly 1/A relative to the coherent vector amplitude, roughly 1/A² in the rate. Computing that residual needs a spin structure factor from a nuclear-structure calculation, an input of a different kind from Z and N.

In this framework the honest answer for A and P on the shipped targets is therefore zero, with that ceiling stated. The physics becomes relevant for electron targets, where there is no coherence to lose, or for a deliberately spin-dependent analysis.

## Tensor: the spin-1/2 nucleus question

The tensor current q̄σ^{μν}q also reduces at the nucleon level to spin-dependent operators, so by the argument above it should behave like A and P on a spin-0 target (*inference*). The published COHERENT analyses of neutrino generalized interactions instead derive the cross section by treating the whole nucleus as one spin-1/2 Dirac particle. That treatment produces a coherent tensor charge Z δ^p + N δ^n built from nucleon tensor charges (δ_u^p = 0.54, δ_d^p = −0.23, δ_u^n = −0.23, δ_d^n = 0.54 in those analyses) and an A²-enhanced tensor rate.

Whether that enhancement is physical or an artifact of the spin-1/2 idealization is the first thing to settle, and it decides the shape of the whole task:

- If the coherent tensor piece vanishes for spin 0, there is no formula to add. Tensor joins A and P as a documented zero with a spin-structure-factor ceiling.
- If it survives, a spin-1/2 formula has to be reconciled with a code in which every other term assumes spin 0, and nobody has yet written down how the two pictures agree.

A natural place to check is the effective-field-theory treatment of CEvNS with explicit nuclear responses, Hoferichter, Menéndez and Schwenk, [arXiv:2007.08529](https://arxiv.org/abs/2007.08529), which separates coherent from spin-dependent responses operator by operator (*to be verified; the conclusion drawn here is not confirmed from that source*).

## Interference and antineutrinos

The tensor amplitude flips the neutrino helicity, like the scalar one, so scalar and tensor interfere. Those analyses include a scalar-tensor cross term whose sign flips between neutrinos and antineutrinos. Adding tensor therefore breaks both structural assumptions above at once: the rate is no longer a sum of squares, and antineutrinos no longer share the neutrino rate. The `FLAVOR_ROW` shortcut and the summed-over-β structure in `dsigma_dt` would both need rework, not a new additive term.

## Normalization conversion

Those analyses write the Lagrangian as (G_F/√2) ν̄Γν q̄Γ(C + iγ5 D)q for a Dirac neutrino with both chiralities available. The prototype's normative convention (`docs/agent/conventions.md`) is −√2 G_F C [ν̄γ^μ P_L ν][f̄γ_μ f] with a left-handed neutrino. Converting between them involves factors of 2 and √2 and a chirality projection, and this project has already met one factor-2 mismatch in the legacy mediator mapping. Any tensor formula must be written in the prototype's normalization with the conversion stated explicitly, in the spec, before code exists.

## Testing without an oracle

For V and S the legacy package served as a numerical oracle, and wrong transcriptions failed the equivalence tests. There is no tensor implementation anywhere in this repository; the only legacy axial code is the charged-current quasi-elastic class on free nucleons, which is unrelated. An implementer who codes the formula and writes the test from the code checks their transcription against itself. The test must anchor to the formula as written in the spec, evaluated by hand at a few points, and that is a weaker check than the project has had so far.

## Current handling and revisit condition

Today a description that needs A, P or T is a notes-only outcome for the agent (`conventions.md`, section 2), the axial part of chiral Z′ quark couplings is dropped and recorded (`conventions.md`, section 3), and users have the raw cross-section plugin route at verification level `raw-cross-section`. No user-facing path is blocked.

Revisit when a concrete use case asks for one of these structures; the use case also fixes the target nucleus and therefore the spin picture. If the work goes ahead, it touches: `STRUCTURES` and a tensor nuclear charge in `couplings.py`; the cross section, interference and antineutrino handling in `xsec.py`; the structure enum in `modelcard.py` and the generated `card_schema.json`; `conventions.md`, `framework.md` and `plugin_api.md` in `src/nuagent/context/` with their mirror under `docs/agent/`; the out-of-basis rubric in `evals/agent/`; the rejection test in `tests/test_modelcard.py`; and the design spec.
