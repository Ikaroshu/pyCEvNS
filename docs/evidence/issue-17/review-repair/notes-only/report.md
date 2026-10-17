Outcome: **notes-only**

No physics verification label or predictions: the agent reports missing information or physics outside the supported basis.

## Description

A neutrino magnetic moment of 1e-10 Bohr magnetons.

## Provenance

```json
{
  "package": "nuagent",
  "package_version": "0.1.0",
  "source_path": "/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/issue-17-agent-harness/src/nuagent/agent/report.py",
  "git_revision": "543b84d30985ca4f3e75a0055f5664f9fc111921",
  "git_tree": "42dce7f80339b1a6a18d18be3fb96de9d1b6dfca",
  "source_dirty": false,
  "data_releases": {
    "coherent_csi_2017": "10.5281/zenodo.1228631"
  },
  "max_turns": 16,
  "backend": "seeded_then_claude_code"
}
```

## notes.md

```text
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

```

## Validation and derivation history

### Attempt 1

```json
{
  "valid": false,
  "findings": [
    "plugin/card mismatch: V u (all entries, rtol=1e-6, atol=0)",
    "plugin/card mismatch: V d (all entries, rtol=1e-6, atol=0)"
  ],
  "checks": [
    "couplings are perturbative",
    "SM-limit counts checked with COUPLINGS zeroed (rtol=1e-9, atol=0)",
    "proposed counts checked for finiteness and non-negativity",
    "ModelCard schema valid",
    "couplings are perturbative",
    "card SM limit holds by construction",
    "card counts checked for finiteness and non-negativity",
    "plugin/card cross-check on 66 Q2 points spanning the selected recoil ranges"
  ],
  "notes": "Seeded inconsistent card for correction-path verification; this first attempt is a test fixture, not model output."
}
```

### Attempt 2

```json
{
  "valid": true,
  "findings": [],
  "checks": [
    "notes-only outcome; no predictions or physics verification label"
  ],
  "notes": "# Neutrino magnetic moment \u03bc_\u03bd = 1e-10 \u03bc_B \u2014 notes-only outcome\n\n## Outcome\n\nNo `plugin.py` and no `card.json` are produced. A neutrino magnetic moment is\noutside the vector/scalar operator basis of this framework, so it cannot be\ntranslated into coefficients `C^{fV}` or `C^{qS}`. The only supported route is\nthe user-authored raw cross-section plugin described in `context/plugin_api.md`.\n\n## Why the previous attempt is discarded\n\nThe files in `history/attempt-1/` were a seeded fixture: a flavor-universal\n45 MeV Z\u2032 coefficient plugin together with an empty card `{}`. They do not\ndescribe a magnetic moment at all, and the validation findings (plugin/card\nmismatch on V u and V d) simply reflect that the plugin set non-zero vector\ncoefficients while the empty card set none. The correct repair is not to make\nthe card match the Z\u2032 plugin, because neither file represents the described\ninteraction. The correct repair is to withdraw both files.\n\n## Why the V/S basis cannot express a magnetic moment\n\nThe framework's effective Lagrangian (conventions.md, section 2) contains only\n\n    \u2212\u221a2 G_F C^{fV}_{\u03b1\u03b2}(Q\u00b2) [\u03bd\u0304_\u03b1 \u03b3^\u03bc P_L \u03bd_\u03b2][f\u0304 \u03b3_\u03bc f]        (vector)\n\nand the scalar operator of section 4. Both are four-fermion contact-type\noperators between the neutrino current and a quark current, with a propagator\n`1/(Q\u00b2 + m\u00b2)` for a massive mediator.\n\nA neutrino magnetic moment is the dimension-5 dipole operator\n\n    L \u2283 \u2212(\u03bc_\u03bd / 2) \u03bd\u0304 \u03c3^{\u03bc\u03bd} \u03bd F_\u03bc\u03bd     (Dirac; a transition moment \u03bd\u0304_\u03b1 \u03c3^{\u03bc\u03bd} \u03bd_\u03b2 for Majorana)\n\nwhich couples the neutrino to the photon, not to a quark vector or scalar\ncurrent. Its contribution to coherent elastic neutrino\u2013nucleus scattering\ndiffers from the V/S operators in every respect that matters here:\n\n1. **Lorentz structure.** The operator is tensor (\u03c3^{\u03bc\u03bd}) on the neutrino line\n   and couples to the electromagnetic field strength. Axial, pseudoscalar and\n   tensor structures are rejected by the card schema and by `Coefficients.set`,\n   which accepts only `\"V\"` and `\"S\"`.\n2. **Momentum dependence.** Photon exchange gives a massless propagator\n   `1/Q\u00b2`, and the dipole vertex supplies a further factor of q, so the\n   differential cross section behaves as `(1/T \u2212 1/E_\u03bd + T/(4E_\u03bd\u00b2))` rather than\n   the `(1 \u2212 T/T_max)`-type factor of the vector channel. The framework's\n   kinematic factor for V and for S is fixed internally and cannot be replaced\n   from a coefficient plugin.\n3. **Nuclear charge.** The dipole couples to the nuclear electric charge Z\n   times the charge form factor, not to the weak vector charge\n   `Z(2C_u + C_d) + N(C_u + 2C_d)` or the scalar charge built from `f_q^N`.\n   There is no choice of `C^{uV}`, `C^{dV}` that reproduces a pure-Z coupling\n   with `1/Q\u00b2` momentum dependence together with the different kinematic factor.\n4. **Interference.** The magnetic-moment amplitude flips the neutrino helicity\n   and therefore does not interfere with the Standard Model vector amplitude.\n   The scalar channel also flips helicity, but with a different kinematic\n   factor and a different nuclear charge, so faking the dipole through `S`\n   coefficients would give the wrong recoil spectrum.\n\nAttempting to encode \u03bc_\u03bd as an effective `C^{fV}(Q\u00b2)` with a `1/Q\u00b2` propagator\nwould also break the framework's SM-limit and finiteness checks at Q\u00b2 \u2192 0 and\nwould misrepresent the physics. Per `context/framework.md`, a magnetic moment is\na legitimate notes-only outcome.\n\n## The description, for the record\n\n- Parameter: \u03bc_\u03bd = 1e-10 \u03bc_B (Bohr magnetons), where\n  \u03bc_B = e / (2 m_e) \u2248 5.788e-11 MeV/T in SI-ish units, or in natural units\n  \u03bc_B = e/(2 m_e) with e = 0.303 and m_e = 0.511 MeV.\n- Flavor: not specified. For COHERENT CsI the flux contains \u03bd_\u03bc, \u03bd\u0304_\u03bc and \u03bd_e,\n  so the user must decide whether 1e-10 \u03bc_B applies to one flavor or to all\n  (an effective \u03bc_\u03bd per flavor, or a transition-moment matrix for Majorana\n  neutrinos).\n- Dirac vs Majorana: not specified. At the level of the CEvNS rate the\n  distinction does not change the formula once an effective \u03bc_\u03bd per incoming\n  flavor is fixed.\n\n## The user-authored raw cross-section route\n\nIf predictions for this point are wanted, a user (not the agent) writes a raw\nplugin exposing\n\n    dsigma_dt(e_nu, t, flavor, isotope, params) -> np.ndarray   # MeV\u207b\u00b3\n\nwith `PARAMETERS = {\"mu_nu_bohr\": 1e-10}` and `COUPLINGS = (\"mu_nu_bohr\",)`.\nThe function must return the **complete** differential cross section: the\nframework's Standard Model term obtained from\n\n    nuagent.physics.xsec.dsigma_dt(e_nu, t, flavor, isotope,\n                                   nuagent.physics.couplings.sm(isotope.q2(t)))\n\nplus the incoherent magnetic-moment term, which for a spin-0 nucleus is\n\n    d\u03c3_\u03bc/dT = (\u03c0 \u03b1\u00b2 \u03bc_\u03bd\u00b2 / m_e\u00b2) Z\u00b2 (1/T \u2212 1/E_\u03bd + T/(4 E_\u03bd\u00b2)) F_ch(Q\u00b2)\u00b2\n\nwith \u03bc_\u03bd in units of \u03bc_B, \u03b1 = e\u00b2/4\u03c0, m_e the electron mass, Z = `isotope.z`,\nand the Helm charge form factor evaluated at Q\u00b2 = `isotope.q2(t)` with\n`isotope.helm_radius_fm`. The two pieces add without interference. The user\nowns the normalization of this term, the choice of flavor(s) to which \u03bc_\u03bd is\napplied, and the form factor. The harness labels such a report\n`raw-cross-section`, which skips the SM-limit check and the plugin/card\ncross-check.\n\n## Information that would be needed if a basis extension existed\n\nEven with a raw plugin, the user should state:\n\n1. Which neutrino flavor(s) carry \u03bc_\u03bd = 1e-10 \u03bc_B (\u03bd_e only, \u03bd_\u03bc only, or\n   flavor-universal), and whether antineutrinos are included.\n2. Whether a Dirac diagonal moment or a Majorana transition-moment matrix is\n   intended.\n\nNo tuning was performed, no evaluation or observed data was inspected, and no\nvalidation command was run because there is no plugin to validate.\n"
}
```

## Backend CLI records

Model, cost and usage fields, when supplied by the CLI, are retained below verbatim. Codex exec has no native max-turns flag; max_turns bounds Claude Code only.

### Attempt 1

```json
{
  "final_text": "seeded first attempt",
  "exit_status": 0,
  "raw_output": "{\"fixture\": true, \"source\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/issue-17-agent-harness/docs/evidence/issue-17/eval-claude_code/01_universal_vector/plugin.py\"}",
  "stderr": "",
  "command": [],
  "error": false
}
```

### Attempt 2

```json
{
  "final_text": "The corrected outcome is notes-only. The described interaction, a neutrino magnetic moment of 1e-10 Bohr magnetons, lies outside the vector/scalar basis, so no plugin or card can represent it.\n\n**Files produced**\n\n- `notes.md` only. It explains why the dipole operator cannot be expressed as vector or scalar coefficients, and points to the user-authored raw cross-section route.\n- No `plugin.py` and no `card.json`, as the output contract requires for out-of-basis physics.\n\n**Why the previous attempt was withdrawn rather than patched**\n\nThe history attempt was a seeded fixture, a flavor-universal 45 MeV Z\u2032 plugin paired with an empty card. The validation mismatch on the up and down vector coefficients only reflected that the plugin set values while the empty card set none. Neither file described a magnetic moment, so making them agree would have preserved the wrong physics.\n\n**What the notes cover**\n\n- The dipole operator has tensor structure and couples to the photon, giving a massless propagator and a different recoil kinematic factor and a pure-charge nuclear coupling. None of these can be mimicked by the framework's fixed vector or scalar kinematics.\n- The raw plugin interface and the magnetic-moment cross-section formula the user would need, with the Standard Model term obtained from the framework's own function.\n- Two open questions for the user: which flavors carry the moment, and whether a Dirac diagonal moment or a Majorana transition moment is intended.\n\nNo validate command was run because there is no plugin to validate, and no parameters were tuned.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":51677,\"stop_reason\":\"end_turn\",\"session_id\":\"af83a20a-ece8-4964-8775-8916296cfe70\",\"total_cost_usd\":0.5612745,\"usage\":{\"input_tokens\":98,\"cache_creation_input_tokens\":17515,\"cache_read_input_tokens\":42378,\"output_tokens\":3988,\"output_tokens_details\":{\"thinking_tokens\":266},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":17515,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":550,\"cache_read_input_tokens\":20737,\"cache_creation_input_tokens\":2784,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":2784},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":98,\"outputTokens\":3988,\"cacheReadInputTokens\":42378,\"cacheCreationInputTokens\":17515,\"webSearchRequests\":0,\"costUSD\":0.5612745,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":266,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01FQ2VosP1heZJHRqmx6J7MR\",\"tool_input\":{\"command\":\"ls -R /private/tmp/nuagent-issue17-repair-notes-only/report.task\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":10,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"The corrected outcome is notes-only. The described interaction, a neutrino magnetic moment of 1e-10 Bohr magnetons, lies outside the vector/scalar basis, so no plugin or card can represent it.\\n\\n**Files produced**\\n\\n- `notes.md` only. It explains why the dipole operator cannot be expressed as vector or scalar coefficients, and points to the user-authored raw cross-section route.\\n- No `plugin.py` and no `card.json`, as the output contract requires for out-of-basis physics.\\n\\n**Why the previous attempt was withdrawn rather than patched**\\n\\nThe history attempt was a seeded fixture, a flavor-universal 45 MeV Z\u2032 plugin paired with an empty card. The validation mismatch on the up and down vector coefficients only reflected that the plugin set values while the empty card set none. Neither file described a magnetic moment, so making them agree would have preserved the wrong physics.\\n\\n**What the notes cover**\\n\\n- The dipole operator has tensor structure and couples to the photon, giving a massless propagator and a different recoil kinematic factor and a pure-charge nuclear coupling. None of these can be mimicked by the framework's fixed vector or scalar kinematics.\\n- The raw plugin interface and the magnetic-moment cross-section formula the user would need, with the Standard Model term obtained from the framework's own function.\\n- Two open questions for the user: which flavors carry the moment, and whether a Dirac diagonal moment or a Majorana transition moment is intended.\\n\\nNo validate command was run because there is no plugin to validate, and no parameters were tuned.\",\"ttft_ms\":1183,\"type\":\"result\",\"duration_ms\":51856,\"uuid\":\"86356c3a-3e6a-47ea-ab97-acaa318feace\",\"ttft_stream_ms\":1039,\"time_to_request_ms\":31,\"first_content_frame_ms\":1040,\"queued_turn_count\":0,\"result_index\":0}\n",
  "stderr": "",
  "command": [
    "claude",
    "-p",
    "--output-format",
    "json",
    "--max-turns",
    "16",
    "--restricted",
    "--tools",
    "Read,Write,Edit,Bash",
    "--permission-mode",
    "dontAsk",
    "--setting-sources",
    "",
    "--settings",
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/issue-17-agent-harness/.venv/bin/python3 -m nuagent.agent.tool_policy /private/tmp/nuagent-issue17-repair-notes-only/report.task\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
