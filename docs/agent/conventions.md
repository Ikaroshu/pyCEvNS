# Conventions for new-physics coefficients

This file is normative. Every model card, every coefficient plugin, and every derivation in `notes.md` follows it. When a paper you are translating uses a different convention, convert to this one and say so in `notes.md`.

## 1. Units and constants

Natural units with energies in MeV. The framework provides these constants; do not redefine them.

| symbol | value | meaning |
|---|---|---|
| `GF` | 1.16638e-11 MeV⁻² | Fermi constant |
| `SQRT2` | √2 at double precision | √2 |
| `E_CHARGE` | 0.303 | electromagnetic coupling e (α = e²/4π) |
| `SIN2_THETA_W` | 0.2312 | sin²θ_W |
| `M_U`, `M_D` | 2.2, 4.7 MeV | current quark masses used in scalar nucleon charges |
| `M_P`, `M_N` | 938.27, 939.56 MeV | nucleon masses |
| `M_MU`, `M_TAU` | 105.658369, 1776.99 MeV | charged lepton masses |

The values in this table are the ones exported by `nuagent.physics.couplings`, at full precision; use the exported names in plugins rather than retyping numbers. The module also exports `Q_EM` (the quark electric charges `{"u": 2/3, "d": -1/3}`) and `KAPPA_LMU_LTAU` (κ of section 6).

Momentum transfer: the framework passes you `q2`, a numpy array of the squared spacelike momentum transfer Q² = 2 M T ≥ 0 in MeV², where M is the nuclear mass and T the nuclear recoil energy. A propagator is always `1 / (q2 + mass**2)`.

## 2. The operator basis

The framework describes neutrino scattering on quark f ∈ {u, d} between flavors α, β ∈ {e, μ, τ} with the effective Lagrangian

    L_eff ⊃ −√2 G_F Σ_{α,β,f} C^{fV}_{αβ}(Q²) [ν̄_α γ^μ P_L ν_β] [f̄ γ_μ f]          (vector)

and, for scalar exchange, with the coefficient defined in section 4. The coefficients C are dimensionless, complex, and Hermitian in the flavor indices: C_{βα} = (C_{αβ})*.

The vector coefficient is the usual NSI combination C^{fV} = ε^{fL} + ε^{fR}. In this normalization the Standard Model values are the tree-level Z couplings

    C^{uV}_SM = +1/2 − (4/3) sin²θ_W ≈ +0.1917,     C^{dV}_SM = −1/2 + (2/3) sin²θ_W ≈ −0.3459,

flavor-diagonal. The framework adds radiative corrections to these itself. You never write SM values; you write only the new-physics contribution, and the framework forms the total C = C_SM + C_NP once.

The nuclear vector charge, for the entry C_{αβ} (the operator ν̄_α … ν_β turns an incoming ν_β into an outgoing ν_α; rates are unchanged by this labeling because |C_{αβ}| = |C_{βα}|), is

    Q^{αβ} = Z (2 C^{uV}_{αβ} + C^{dV}_{αβ}) + N (C^{uV}_{αβ} + 2 C^{dV}_{αβ}),

and the rate is proportional to Σ_β |Q^{αβ}|². In the SM, Q^{αα} ≈ 0.04 Z − 0.5 N, which is negative. A new-physics contribution with the same sign as the SM charge interferes constructively; one with the opposite sign interferes destructively and can produce a cancellation at some coupling.

Structures accepted in this prototype: `V` (vector) and `S` (scalar). Axial, pseudoscalar and tensor are rejected by the schema; if a description needs them, say so in `notes.md` and stop.

## 3. Vector mediator matching

For a new vector boson Z′ with the interaction

    L ⊃ Z′_μ ( g_ν^{αβ} ν̄_α γ^μ P_L ν_β + g_f f̄ γ^μ f ),       mass m > 0,

tree-level exchange gives

    C^{fV}_{αβ}(Q²) = g_ν^{αβ} g_f / ( √2 G_F (Q² + m²) ).

Derivation: integrating out Z′ produces −(g_ν g_f)/(Q² + m²) [ν̄γ^μP_Lν][f̄γ_μf] for spacelike exchange; compare with −√2 G_F C [ν̄γ^μP_Lν][f̄γ_μf].

Anchor you can check in one line: insert the SM Z itself, g_ν = g/(2 cos θ_W), g_f = g g_V^f/(2 cos θ_W), m = M_Z, and use G_F/√2 = g²/(8 M_W²). Here g_V^f = T₃ − 2 Q sin²θ_W, the normalization in which the Z couples as −(g/2cos θ_W) Z_μ f̄ γ^μ (g_V − g_A γ5) f. The result is C^{fV} = g_V^f, the SM value above. A formula that returns g_V^f / 2 is missing a factor of 2.

Known pitfall: the legacy `pyCEvNS` package mapped ε = g/(2√2 G_F m²), half of the expression above if its `g` meant g_ν g_f. Do not copy it.

Notes on the neutrino coupling:
- Only the left-handed neutrino current appears. If a model writes g_ν ν̄ γ^μ ν for a Dirac neutrino with both chiralities, the right-handed part does not scatter on the SM-produced left-handed neutrinos; use the left-handed coupling.
- If a model writes the coupling as (g′/2) ν̄ γ^μ (1 − γ5) ν, that is g′ ν̄ γ^μ P_L ν, so g_ν = g′. If it writes g ν̄ γ^μ (1 − γ5) ν without the 1/2, then g_ν = 2g.
- Off-diagonal g_ν^{αβ} with α ≠ β is allowed and must be Hermitian. It enters incoherently (no interference with the SM) because the SM has no flavor-changing neutral current.
- Antineutrinos: the framework handles them; you supply the same C.

Quark couplings: give g_u and g_d separately. If a model gives chiral quark couplings Z′_μ (g_L q̄_L γ^μ q_L + g_R q̄_R γ^μ q_R), the vector coupling is g_q = (g_L + g_R)/2. Note the 1/2 here against C = ε^L + ε^R with no 1/2: these are different objects, and confusing them is a second factor-2 trap. The axial part (g_R − g_L)/2 contributes nothing to coherent scattering on the spin-0 formula used in this prototype; keep the vector part and record the dropped axial coupling in `notes.md`.

Nucleon-level inputs: nucleon couplings are not card fields, but they convert exactly to quark couplings, because the framework's nuclear charge is Q = Z g_p + N g_n with g_p = 2 g_u + g_d and g_n = g_u + 2 g_d. Given g_p and g_n:

    g_u = (2 g_p − g_n) / 3,     g_d = (2 g_n − g_p) / 3.

A protophobic Z′ (g_p = 0) therefore has g_u = −g_n/3 and g_d = 2 g_n/3; a coupling to baryon number (g_p = g_n = g_B) has g_u = g_d = g_B/3.

## 4. Scalar mediator matching

For a scalar φ with

    L ⊃ φ ( g_ν^{αβ} ν̄_α ν_β + Σ_q g_q q̄ q ),       mass m > 0,

the scalar coefficient is defined, dimensionless like the vector one, as

    C^{qS}_{αβ}(Q²) = g_ν^{αβ} g_q / ( √2 G_F (Q² + m²) ),

and the framework forms the nuclear scalar charge

    Q_S^{αβ}(Q²) = √2 G_F Σ_q C^{qS}_{αβ}(Q²) ( Z m_p f_q^p + N m_n f_q^n ) / m_q,

with f_u^p = 0.0208, f_d^p = 0.0411, f_u^n = 0.0189, f_d^n = 0.0451. For a single mediator this is g_ν (Z c_p + N c_n)/(Q² + m²) with c_N = m_N Σ_q g_q f_q^N / m_q. Numerically, for g_u = g_d = g_q, c_p ≈ c_n ≈ 17.1 g_q, so the nucleus couples as roughly 17 A g_q.

The scalar amplitude flips the neutrino helicity and does not interfere with the SM or with vector new physics. Its rate is proportional to Σ_β |Q_S^{αβ}|² with a different kinematic factor, which the framework supplies. You do not write cross sections.

Notes:
- ν̄_α ν_β requires a right-handed neutrino (Dirac) or a Majorana mass-type coupling. State which in `notes.md`; the coefficient is the same at this level.
- If a model gives nucleon-level scalar couplings y_p and y_n, convert to quark couplings by solving the two linear equations c_p = y_p and c_n = y_n, that is m_p (g_u f_u^p/m_u + g_d f_d^p/m_d) = y_p and m_n (g_u f_u^n/m_u + g_d f_d^n/m_d) = y_n. For y_p = y_n = y the solution is g_u = 0.058970 y and g_d = 0.058122 y. This reproduces the framework's nuclear scalar charge exactly.
- Only u and d scalar couplings can be represented. A model whose scalar couples to heavier quarks with non-negligible weight (for example a Higgs-portal scalar with g_q ∝ m_q, where heavy quarks dominate the nucleon coupling through the gluon term) cannot be expressed here: say so in `notes.md` and stop.
- For a Majorana neutrino the statement that the coefficient is unchanged holds when the mass-type term is written (g_ν/2) φ ν̄ ν; written without the 1/2, the vertex doubles and g_ν must be doubled accordingly.

## 5. Contact interactions

A card may list `contacts`: constant ε^{fV}_{αβ} that are added directly to C^{fV}. Use them only when the description gives epsilons rather than a mediator. If a mass is given, use the mediator form in both the plugin and the card, however heavy the mediator: the contact form differs by Q²/m², which reaches about 1 percent at m = 1 GeV at the top of the COHERENT recoil range and fails the 1e-6 cross-check for any mass below about 100 GeV. Contact scalar entries are not supported.

## 6. Kinetic mixing

For a vector mediator, the card field `kinetic_mixing` ε adds an electromagnetic-charge-proportional coupling to every charged fermion:

    g_q → g_q + ε e Q_q^em,       Q_u^em = +2/3,   Q_d^em = −1/3,   e = 0.303.

This is the low-energy coupling of a Z′ that mixes with the photon. The sign of ε relative to the other couplings is part of the model and must be stated in `notes.md`. Mapping from Lagrangian conventions: with a kinetic term −(ε_L/2) F_μν Z′^μν and covariant derivative D_μ = ∂_μ + i e Q A_μ, the Z′ acquires +ε_L e Q f̄ γ^μ f, so ε = ε_L; with D_μ = ∂_μ − i e Q A_μ the sign flips. Mixing with hypercharge ε_Y gives ε = ε_Y cos θ_W for m ≪ M_Z. Models with Z–Z′ mass mixing, such as the dark-hypercharge and dark-Z cases of arXiv:1803.01224 (its equations 7 and 8), give Z′ couplings proportional to hypercharge or to the Z current, neutrinos included; take g_ν and g_q from the paper's coupling formula rather than from `kinetic_mixing` alone.

Loop-induced mixing for U(1)_{L_μ−L_τ}: muon and tau loops generate, at Q² ≪ m_μ²,

    ε = + κ g′,     κ = e ln(m_τ/m_μ) / (6 π²) = 0.0144417637.

The sign is a physics result, not a convention. The induced quark coupling is g_q = e Q_q Σ_ℓ (g′ Q′_ℓ)(e Q_ℓ) ln(m_ℓ²)/(12π²): the m-independent part of the loop cancels because Σ_ℓ Q′_ℓ Q_ℓ = 0, and the ln(m²) term carries the sign that makes the running electromagnetic coupling grow with Q². With Q′_μ = +1, Q′_τ = −1 and Q_ℓ = −1 the sum is +ln(m_τ²/m_μ²) > 0, so ε = +κ g′. Intuition: the proton's screening cloud holds more μ⁻ than τ⁻ because the muon is lighter, and μ⁻ carries L_μ−L_τ charge +1 like ν_μ, so the induced proton coupling has the same sign as the ν_μ coupling and the interference with the SM is destructive for ν_μ. The template `zprime_lmu_ltau` fills ε = κ g′ when no `kinetic_mixing` is given; a card has no such default, so write the value explicitly (κ times g′, at full precision). The comparison with the limit curve of arXiv:1803.01224 in milestone M4 is a magnitude check; a disagreement there points to a normalization difference, not to the sign. For Q² comparable to m_μ² the mixing depends on Q²; that case needs a coefficient plugin (example 06).

## 7. U(1)′ charge assignments

For a gauged U(1)′ with gauge coupling g′ and charges Q′: g_ν^{αα} = g′ Q′_{ν_α}, g_q = g′ Q′_q, flavor-diagonal unless the charges are flavor off-diagonal in the mass basis (rare; then give the matrix). Common assignments:

| model | Q′_{ν_e} | Q′_{ν_μ} | Q′_{ν_τ} | Q′_u | Q′_d | note |
|---|---|---|---|---|---|---|
| B − L | −1 | −1 | −1 | +1/3 | +1/3 | g_ν g_q < 0: constructive with the SM |
| universal (toy) | +1 | +1 | +1 | +1 | +1 | g_ν g_q > 0: destructive, cancellation possible |
| L_μ − L_τ | 0 | +1 | −1 | 0 | 0 | quarks only via kinetic mixing |
| B − 3L_τ | 0 | 0 | −3 | +1/3 | +1/3 | |
| B − 3L_e | −3 | 0 | 0 | +1/3 | +1/3 | |

The neutrino charge is the charge of the lepton doublet. Anomaly cancellation is the model author's concern, not the framework's; mention in `notes.md` if a requested assignment is anomalous.

## 8. Flavor matrices

A flavor matrix is given by six entries `ee, mm, tt, em, et, mt`. Diagonal entries are real. Off-diagonal entries may be complex, written as `[re, im]`; the framework fills the conjugate partner. `em` means the (e, μ) entry, that is C_{eμ}.

## 9. What the framework does for you

Given your coefficients, the framework: adds the SM; forms the nuclear charges for each isotope of the detector; evaluates the differential cross section with the Helm form factor; folds it with the experiment's neutrino flux for each flavor; applies efficiency, quenching, exposure and binning; and compares with observed counts. You never need the flux, the detector, the form factor, or the SM couplings.

## 10. Perturbativity and sanity

Every dimensionless coupling you write must satisfy |g| ≤ √(4π) ≈ 3.54. Masses are in MeV and positive. If a description gives a coupling in a different normalization (for example α′ = g′²/4π), convert and show the conversion.

## 11. When to stop and ask

Ask only about the **form** of the interaction: which particles couple (which quarks, which neutrino flavors), the Lorentz structure (vector or scalar), the flavor structure, or a sign or convention that changes the physics. Never ask for a numeric value. If the form is open, write `questions.json` (shape in `plugin_api.md`) and `notes.md` saying what is open, and produce no plugin or card. A question's `suggested` is then usually a short string naming the suggested form, for example "universal vector coupling to u and d quarks, all three flavors", with `reason` saying where it comes from; never present a suggestion as part of the description. The user's answers arrive in `TASK.md` under `## Answers (round N)` and are then part of the description; ask again only if they still leave the form undetermined. Example 07 shows such a translation. Physics outside the V and S basis is not a question: it is notes-only (`framework.md`).

## 12. Assumed values

When the form is fixed but a numeric value is missing, do not ask: take the benchmark default below, declare it in `PARAMETERS` like any other value, and proceed. The user changes it afterwards by re-evaluating the plugin at other parameter values.

| missing value | assumed default |
|---|---|
| light-mediator mass | 10 MeV |
| vector gauge coupling: each of g_ν and g_q, or g′ of a U(1)′ | 1e-4 |
| scalar couplings: each of g_ν and g_q | 1e-5 |
| contact NSI ε | 0.1 |
| kinetic mixing ε | 0, unless the model fixes it (for L_μ−L_τ the loop value κ g′ of section 6) |

The assumed values live in `PARAMETERS`, which the page shows as the parameter form; do not list them again in `notes.md`.

## 13. Writing notes.md

Write `notes.md` in Markdown. Write math in LaTeX between `$...$` for inline formulas and `$$...$$` for displayed ones, for example $C^{uV}_{\mu\mu} = g_\nu g_u / (\sqrt{2} G_F (Q^2 + m^2))$, so that the page and the report render it. Do not put the notes or their formulas in code fences.
