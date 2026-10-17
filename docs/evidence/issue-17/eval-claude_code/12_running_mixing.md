Verification: **plugin-only**

SM-limit, perturbativity and count checks pass. There is no independent card cross-check; normalization and momentum dependence rest on the derivation.

## Description

A gauged L_mu−L_tau vector Z′ has mass 12 MeV and gauge coupling g′ = 8e-5. Keep the full spacelike Q² dependence of the muon/tau one-loop photon mixing, with no tree-level mixing or direct quark coupling.

## Provenance

```json
{
  "package": "nuagent",
  "package_version": "0.1.0",
  "source_path": "/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/issue-17-agent-harness/src/nuagent/agent/report.py",
  "git_revision": "934e7113687b7d2927d34462b77925c3fef56c73",
  "git_tree": "8d29d0e53eca3f2cb3350458bc84362237f6121c",
  "source_dirty": false,
  "data_releases": {
    "coherent_csi_2017": "10.5281/zenodo.1228631"
  },
  "max_turns": 16,
  "backend": "claude_code"
}
```

## plugin.py

```python
"""L_mu - L_tau Z' with the full one-loop, Q^2-dependent photon mixing.

Derivation and checks are in notes.md.
"""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2, E_CHARGE, M_MU, M_TAU, Q_EM

PARAMETERS = {"g_prime": 8e-5, "mass_mev": 12.0}
COUPLINGS = ("g_prime",)

# 40-point Gauss-Legendre quadrature on the Feynman parameter interval [0, 1].
_T, _W = np.polynomial.legendre.leggauss(40)
_X = (_T + 1.0) / 2.0
_WX = _W / 2.0


def loop_mixing(q2: np.ndarray, g: float) -> np.ndarray:
    """Spacelike one-loop Z'-photon mixing eps(Q^2) from the muon and tau loops."""
    q = np.asarray(q2, dtype=float)[:, None]
    x = _X[None, :]
    integrand = x * (1.0 - x) * np.log(
        (M_TAU ** 2 + x * (1.0 - x) * q) / (M_MU ** 2 + x * (1.0 - x) * q)
    )
    return E_CHARGE * g / (2.0 * np.pi ** 2) * (integrand * _WX[None, :]).sum(axis=1)


def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    eps = loop_mixing(q2, g)
    prop = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        g_q = eps * E_CHARGE * Q_EM[quark]
        c.set("V", quark, "mm", +g * g_q * prop)
        c.set("V", quark, "tt", -g * g_q * prop)
    return c

```

## notes.md

```text
# Notes: gauged L_μ−L_τ Z′ with full one-loop (Q²-dependent) photon mixing

Description: "A gauged L_mu−L_tau vector Z′ has mass 12 MeV and gauge coupling g′ = 8e-5. Keep the full spacelike Q² dependence of the muon/tau one-loop photon mixing, with no tree-level mixing or direct quark coupling."

Parameter point preserved exactly: `g_prime = 8e-5`, `mass_mev = 12.0`.

## Derivation

1. **Lorentz structure.** A gauge boson coupling to a conserved current: vector exchange, structure `V`. No scalar entries.

2. **Charges** (conventions section 7, L_μ−L_τ row): Q′_{ν_e} = 0, Q′_{ν_μ} = +1, Q′_{ν_τ} = −1, Q′_u = Q′_d = 0. Hence
   g_ν^{mm} = +g′, g_ν^{tt} = −g′, g_ν^{ee} = 0, all off-diagonal entries zero, and no direct quark coupling. The neutrino coupling is to the left-handed current with no factor-of-2 ambiguity, since the gauge interaction is g′ Q′ Z′_μ ν̄ γ^μ P_L ν.

3. **Photon mixing from the lepton loops.** The description forbids a tree-level kinetic-mixing term, so the only Z′–photon mixing is the one-loop muon and tau vacuum-polarization diagram. For spacelike Q² ≥ 0 it is

       ε(Q²) = (e g′ / 2π²) ∫₀¹ dx x(1−x) ln[ (m_τ² + x(1−x) Q²) / (m_μ² + x(1−x) Q²) ].

   The Q²-independent (divergent) part cancels because Σ_ℓ Q′_ℓ Q_ℓ = (+1)(−1) + (−1)(−1) = 0; only the mass-dependent logarithm survives. At Q² → 0 the integral of x(1−x) is 1/6, so ε(0) = e g′ ln(m_τ/m_μ)/(6π²) = κ g′ with κ = 0.0144417637 (conventions section 6), giving ε(0) = 1.15534e-06 for g′ = 8e-5.

   Sign: positive throughout, as in section 6. The logarithm is positive for every x because m_τ > m_μ, and it carries the sign that makes the running electromagnetic coupling grow with Q². Physically the proton's screening cloud holds more μ⁻ than τ⁻, and μ⁻ carries L_μ−L_τ charge +1 like ν_μ, so the induced proton coupling has the same sign as the ν_μ coupling.

   Q² dependence: ε(Q²) falls slowly with Q² because the COHERENT recoil range reaches a few tens of percent of m_μ² = 11164 MeV². The ratio ε(Q²)/ε(0) is independent of g′ and m, so it is the same as in worked example 06: 0.99613, 0.98866, 0.98150 at Q² = 1238, 3714, 6190 MeV².

4. **Induced quark couplings** (section 6): g_q(Q²) = ε(Q²) e Q_q^em with Q_u^em = +2/3, Q_d^em = −1/3 and e = 0.303 (`E_CHARGE`, `Q_EM` from the framework).

5. **Matching** (section 3):

       C^{uV}_{mm}(Q²) = +g′ ε(Q²) e (2/3) / (√2 G_F (Q² + m²)),
       C^{dV}_{mm}(Q²) = −g′ ε(Q²) e (1/3) / (√2 G_F (Q² + m²)),
       C^{qV}_{tt} = −C^{qV}_{mm},   C^{qV}_{ee} = 0,   off-diagonal = 0.

   Only protons are affected: 2C^u + C^d ∝ 2(2/3) − 1/3 = 1, C^u + 2C^d ∝ 2/3 − 2/3 = 0, as expected for photon-mediated mixing.

6. **Interference.** For ν_μ the charge shift ΔQ = Z g′ ε(Q²) e / (√2 G_F (Q² + m²)) is positive while Q_SM is negative: destructive interference for the COHERENT prompt ν_μ and delayed ν̄_μ. Electron neutrinos are unaffected.

7. **Approximate values at the reference points** (Cs-133, Q² = 1238, 3714, 6190 MeV²), obtained by scaling example 06 by g′²/(Q² + m²): C^{uV}_mm ≈ 8.16e-04, 2.90e-04, 1.75e-04; C^{dV}_mm = −C^{uV}_mm / 2. The plugin computes them at full precision from the exported constants.

## Why no card.json

A card's `kinetic_mixing` is a single constant. The description explicitly asks for the full spacelike Q² dependence of the loop mixing, which varies by about 1–2 percent across the recoil range and would fail the 1e-6 plugin-versus-card cross-check. The interaction is therefore plugin-only.

## Reduction check for reviewers

At Q² = 0 the plugin's ε equals κ g′ exactly, so the coefficients reduce to those of worked example 03 B with g′ = 8e-5 and m = 12 MeV: C^{qV}_{mm}(0) = g′ κ g′ e Q_q^em / (√2 G_F m²). The 40-point Gauss–Legendre rule on [0, 1] reproduces the smooth Feynman-parameter integral to better than 1e-14 relative (a trapezoid rule dropping the endpoints would be low by about 2e-6 and fail the eval tolerance).

## Conventions and bookkeeping

- `COUPLINGS = ("g_prime",)`: the mixing is derived from g′, so zeroing g′ removes all new physics and the SM-limit check passes. Mass is a parameter but not a coupling.
- Perturbativity: |g′| = 8e-5 ≪ √(4π).
- No axial quark coupling is dropped: photon mixing induces a purely vector quark current.
- The L_μ−L_τ assignment is anomaly-free with the SM fermion content.
- No convention conversion was needed; the description already uses the gauge coupling g′ and the mass in MeV.

```

## Validation and derivation history

### Attempt 1

```json
{
  "valid": true,
  "findings": [],
  "checks": [
    "couplings are perturbative",
    "SM-limit counts checked with COUPLINGS zeroed (rtol=1e-9, atol=0)",
    "proposed counts checked for finiteness and non-negativity"
  ],
  "notes": "# Notes: gauged L_\u03bc\u2212L_\u03c4 Z\u2032 with full one-loop (Q\u00b2-dependent) photon mixing\n\nDescription: \"A gauged L_mu\u2212L_tau vector Z\u2032 has mass 12 MeV and gauge coupling g\u2032 = 8e-5. Keep the full spacelike Q\u00b2 dependence of the muon/tau one-loop photon mixing, with no tree-level mixing or direct quark coupling.\"\n\nParameter point preserved exactly: `g_prime = 8e-5`, `mass_mev = 12.0`.\n\n## Derivation\n\n1. **Lorentz structure.** A gauge boson coupling to a conserved current: vector exchange, structure `V`. No scalar entries.\n\n2. **Charges** (conventions section 7, L_\u03bc\u2212L_\u03c4 row): Q\u2032_{\u03bd_e} = 0, Q\u2032_{\u03bd_\u03bc} = +1, Q\u2032_{\u03bd_\u03c4} = \u22121, Q\u2032_u = Q\u2032_d = 0. Hence\n   g_\u03bd^{mm} = +g\u2032, g_\u03bd^{tt} = \u2212g\u2032, g_\u03bd^{ee} = 0, all off-diagonal entries zero, and no direct quark coupling. The neutrino coupling is to the left-handed current with no factor-of-2 ambiguity, since the gauge interaction is g\u2032 Q\u2032 Z\u2032_\u03bc \u03bd\u0304 \u03b3^\u03bc P_L \u03bd.\n\n3. **Photon mixing from the lepton loops.** The description forbids a tree-level kinetic-mixing term, so the only Z\u2032\u2013photon mixing is the one-loop muon and tau vacuum-polarization diagram. For spacelike Q\u00b2 \u2265 0 it is\n\n       \u03b5(Q\u00b2) = (e g\u2032 / 2\u03c0\u00b2) \u222b\u2080\u00b9 dx x(1\u2212x) ln[ (m_\u03c4\u00b2 + x(1\u2212x) Q\u00b2) / (m_\u03bc\u00b2 + x(1\u2212x) Q\u00b2) ].\n\n   The Q\u00b2-independent (divergent) part cancels because \u03a3_\u2113 Q\u2032_\u2113 Q_\u2113 = (+1)(\u22121) + (\u22121)(\u22121) = 0; only the mass-dependent logarithm survives. At Q\u00b2 \u2192 0 the integral of x(1\u2212x) is 1/6, so \u03b5(0) = e g\u2032 ln(m_\u03c4/m_\u03bc)/(6\u03c0\u00b2) = \u03ba g\u2032 with \u03ba = 0.0144417637 (conventions section 6), giving \u03b5(0) = 1.15534e-06 for g\u2032 = 8e-5.\n\n   Sign: positive throughout, as in section 6. The logarithm is positive for every x because m_\u03c4 > m_\u03bc, and it carries the sign that makes the running electromagnetic coupling grow with Q\u00b2. Physically the proton's screening cloud holds more \u03bc\u207b than \u03c4\u207b, and \u03bc\u207b carries L_\u03bc\u2212L_\u03c4 charge +1 like \u03bd_\u03bc, so the induced proton coupling has the same sign as the \u03bd_\u03bc coupling.\n\n   Q\u00b2 dependence: \u03b5(Q\u00b2) falls slowly with Q\u00b2 because the COHERENT recoil range reaches a few tens of percent of m_\u03bc\u00b2 = 11164 MeV\u00b2. The ratio \u03b5(Q\u00b2)/\u03b5(0) is independent of g\u2032 and m, so it is the same as in worked example 06: 0.99613, 0.98866, 0.98150 at Q\u00b2 = 1238, 3714, 6190 MeV\u00b2.\n\n4. **Induced quark couplings** (section 6): g_q(Q\u00b2) = \u03b5(Q\u00b2) e Q_q^em with Q_u^em = +2/3, Q_d^em = \u22121/3 and e = 0.303 (`E_CHARGE`, `Q_EM` from the framework).\n\n5. **Matching** (section 3):\n\n       C^{uV}_{mm}(Q\u00b2) = +g\u2032 \u03b5(Q\u00b2) e (2/3) / (\u221a2 G_F (Q\u00b2 + m\u00b2)),\n       C^{dV}_{mm}(Q\u00b2) = \u2212g\u2032 \u03b5(Q\u00b2) e (1/3) / (\u221a2 G_F (Q\u00b2 + m\u00b2)),\n       C^{qV}_{tt} = \u2212C^{qV}_{mm},   C^{qV}_{ee} = 0,   off-diagonal = 0.\n\n   Only protons are affected: 2C^u + C^d \u221d 2(2/3) \u2212 1/3 = 1, C^u + 2C^d \u221d 2/3 \u2212 2/3 = 0, as expected for photon-mediated mixing.\n\n6. **Interference.** For \u03bd_\u03bc the charge shift \u0394Q = Z g\u2032 \u03b5(Q\u00b2) e / (\u221a2 G_F (Q\u00b2 + m\u00b2)) is positive while Q_SM is negative: destructive interference for the COHERENT prompt \u03bd_\u03bc and delayed \u03bd\u0304_\u03bc. Electron neutrinos are unaffected.\n\n7. **Approximate values at the reference points** (Cs-133, Q\u00b2 = 1238, 3714, 6190 MeV\u00b2), obtained by scaling example 06 by g\u2032\u00b2/(Q\u00b2 + m\u00b2): C^{uV}_mm \u2248 8.16e-04, 2.90e-04, 1.75e-04; C^{dV}_mm = \u2212C^{uV}_mm / 2. The plugin computes them at full precision from the exported constants.\n\n## Why no card.json\n\nA card's `kinetic_mixing` is a single constant. The description explicitly asks for the full spacelike Q\u00b2 dependence of the loop mixing, which varies by about 1\u20132 percent across the recoil range and would fail the 1e-6 plugin-versus-card cross-check. The interaction is therefore plugin-only.\n\n## Reduction check for reviewers\n\nAt Q\u00b2 = 0 the plugin's \u03b5 equals \u03ba g\u2032 exactly, so the coefficients reduce to those of worked example 03 B with g\u2032 = 8e-5 and m = 12 MeV: C^{qV}_{mm}(0) = g\u2032 \u03ba g\u2032 e Q_q^em / (\u221a2 G_F m\u00b2). The 40-point Gauss\u2013Legendre rule on [0, 1] reproduces the smooth Feynman-parameter integral to better than 1e-14 relative (a trapezoid rule dropping the endpoints would be low by about 2e-6 and fail the eval tolerance).\n\n## Conventions and bookkeeping\n\n- `COUPLINGS = (\"g_prime\",)`: the mixing is derived from g\u2032, so zeroing g\u2032 removes all new physics and the SM-limit check passes. Mass is a parameter but not a coupling.\n- Perturbativity: |g\u2032| = 8e-5 \u226a \u221a(4\u03c0).\n- No axial quark coupling is dropped: photon mixing induces a purely vector quark current.\n- The L_\u03bc\u2212L_\u03c4 assignment is anomaly-free with the SM fermion content.\n- No convention conversion was needed; the description already uses the gauge coupling g\u2032 and the mass in MeV.\n"
}
```

## Experiment predictions and likelihood comparison

Δ = −2 ln L(new physics) − (−2 ln L(SM)); positive Δ favors the SM. Timing is integrated out.

```json
[
  {
    "experiment": "coherent_csi_2017",
    "data_release": "10.5281/zenodo.1228631",
    "predicted_counts": [
      10.138260380578297,
      16.545799604359658,
      21.759782046395806,
      22.101819030245306,
      17.848134066105914,
      13.092454142584428,
      10.06667527315485,
      7.5716416499307115,
      5.596613693342689,
      4.078052043374353,
      2.9294878321083213,
      2.0717751889159826
    ],
    "predicted_total": 133.8004949510963,
    "observed_total": 855.0,
    "delta": 0.0015326498328072091,
    "new_physics": {
      "neg2_log_likelihood": 17.378970149266827,
      "alpha": 0.07131688687311229,
      "beta": {
        "steady_state": 0.001564404722838787,
        "prompt_neutrons": 0.0028937082601721456
      }
    },
    "sm": {
      "neg2_log_likelihood": 17.37743749943402,
      "alpha": 0.07067927935625874,
      "beta": {
        "steady_state": 0.0015139001341382585,
        "prompt_neutrons": 0.0028841964415516333
      }
    }
  }
]
```

## Backend CLI records

Model, cost and usage fields, when supplied by the CLI, are retained below verbatim. Codex exec has no native max-turns flag; max_turns bounds Claude Code only.

### Attempt 1

```json
{
  "final_text": "Validation passed with no findings. Nothing further is needed, so here is the recap.\n\n**Outcome:** plugin-only translation, validated cleanly.\n\nFiles produced in the task directory:\n\n- **plugin.py**: vector Z\u2032 with g_\u03bd^{mm} = +g\u2032, g_\u03bd^{tt} = \u2212g\u2032, no electron or quark charge. The quark couplings come solely from the one-loop muon/tau photon mixing \u03b5(Q\u00b2), evaluated as a Feynman-parameter integral with 40-point Gauss-Legendre quadrature, times e Q_q^em. Defaults are g\u2032 = 8e-5 and m = 12 MeV, preserved exactly. The only entry in COUPLINGS is g\u2032, since the mixing is derived from it.\n- **notes.md**: derivation (charges, loop mixing with its positive sign, matching formula, proton-only coupling, destructive interference for \u03bd_\u03bc), the Q\u00b2 \u2192 0 reduction check against the constant-mixing case, approximate reference-point values, and the reason no card exists.\n- **card.json**: deliberately not written. The schema's kinetic-mixing field is a constant, and the description asks for the full Q\u00b2 dependence, which varies at the percent level across the recoil range and would fail the card cross-check.\n\nThe validate command reported the plugin valid: couplings perturbative, Standard Model limit reproduced with g\u2032 zeroed, and counts finite and non-negative.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":85121,\"stop_reason\":\"end_turn\",\"session_id\":\"44449321-4d23-4a3a-8395-a83269479b5c\",\"total_cost_usd\":1.0191735,\"usage\":{\"input_tokens\":322,\"cache_creation_input_tokens\":29042,\"cache_read_input_tokens\":230654,\"output_tokens\":7549,\"output_tokens_details\":{\"thinking_tokens\":1124},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":29042,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":470,\"cache_read_input_tokens\":34839,\"cache_creation_input_tokens\":209,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":209},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":322,\"outputTokens\":7549,\"cacheReadInputTokens\":230654,\"cacheCreationInputTokens\":29042,\"webSearchRequests\":0,\"costUSD\":1.0191735,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":1124,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01HKNaVguq5x8xSArEKV6SKL\",\"tool_input\":{\"command\":\"ls -R /private/tmp/nuagent-issue17-eval-claude/12_running_mixing\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":33,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings. Nothing further is needed, so here is the recap.\\n\\n**Outcome:** plugin-only translation, validated cleanly.\\n\\nFiles produced in the task directory:\\n\\n- **plugin.py**: vector Z\u2032 with g_\u03bd^{mm} = +g\u2032, g_\u03bd^{tt} = \u2212g\u2032, no electron or quark charge. The quark couplings come solely from the one-loop muon/tau photon mixing \u03b5(Q\u00b2), evaluated as a Feynman-parameter integral with 40-point Gauss-Legendre quadrature, times e Q_q^em. Defaults are g\u2032 = 8e-5 and m = 12 MeV, preserved exactly. The only entry in COUPLINGS is g\u2032, since the mixing is derived from it.\\n- **notes.md**: derivation (charges, loop mixing with its positive sign, matching formula, proton-only coupling, destructive interference for \u03bd_\u03bc), the Q\u00b2 \u2192 0 reduction check against the constant-mixing case, approximate reference-point values, and the reason no card exists.\\n- **card.json**: deliberately not written. The schema's kinetic-mixing field is a constant, and the description asks for the full Q\u00b2 dependence, which varies at the percent level across the recoil range and would fail the card cross-check.\\n\\nThe validate command reported the plugin valid: couplings perturbative, Standard Model limit reproduced with g\u2032 zeroed, and counts finite and non-negative.\",\"ttft_ms\":788,\"type\":\"result\",\"duration_ms\":86840,\"uuid\":\"95badf18-051d-44d8-b33f-09bad95cc7fd\",\"ttft_stream_ms\":695,\"time_to_request_ms\":18,\"first_content_frame_ms\":695,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/issue-17-agent-harness/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/nuagent-issue17-eval-claude/12_running_mixing\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
