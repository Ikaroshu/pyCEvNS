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

## plugin.py

```python
"""L_mu - L_tau Z' (m = 12 MeV, g' = 8e-5) with the full one-loop, Q^2-dependent
Z'-photon mixing from the muon and tau loops.

Derivation, sign discussion and reduction checks are in notes.md.
"""

import numpy as np

from nuagent.physics.couplings import (
    Coefficients,
    GF,
    SQRT2,
    E_CHARGE,
    M_MU,
    M_TAU,
    Q_EM,
)

PARAMETERS = {"g_prime": 8e-5, "mass_mev": 12.0}
COUPLINGS = ("g_prime",)

# 40-point Gauss-Legendre quadrature mapped onto the Feynman-parameter interval [0, 1].
_T, _W = np.polynomial.legendre.leggauss(40)
_X = (_T + 1.0) / 2.0
_WX = _W / 2.0


def loop_mixing(q2: np.ndarray, g: float) -> np.ndarray:
    """Spacelike one-loop Z'-photon mixing eps(Q^2) induced by the mu and tau loops."""
    q = np.asarray(q2, dtype=float)[:, None]
    x = _X[None, :]
    integrand = x * (1.0 - x) * np.log(
        (M_TAU**2 + x * (1.0 - x) * q) / (M_MU**2 + x * (1.0 - x) * q)
    )
    return E_CHARGE * g / (2.0 * np.pi**2) * (integrand * _WX[None, :]).sum(axis=1)


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
# L_μ−L_τ Z′ (m = 12 MeV, g′ = 8e-5) with full one-loop Q²-dependent photon mixing

## Outcome: plugin-only (no card.json)

The description asks for the *full spacelike Q² dependence* of the muon/tau loop mixing. A model card's `kinetic_mixing` field is a single constant, so no card can represent this interaction. Per `framework.md` this is a legitimate plugin-only translation; the harness label is `plugin-only`.

### Correction relative to attempt 1

Attempt 1 shipped an empty placeholder `card.json` (`{}`), which the validator read as zero quark couplings and reported a plugin/card mismatch on every V u and V d entry. The correct resolution is not to repair the card but to omit it: the Q²-dependent mixing is exactly the case `conventions.md` section 6 and example 06 send to a coefficient plugin. The plugin itself was already correct and is kept unchanged in physics content; only the `Q_EM` import is taken from the exported constants.

## Derivation

1. **Lorentz structure**: vector exchange, structure `V`.
2. **Charges** (conventions section 7, L_μ−L_τ): Q′_{ν_e} = 0, Q′_{ν_μ} = +1, Q′_{ν_τ} = −1, Q′_u = Q′_d = 0. Hence g_ν^{mm} = +g′, g_ν^{tt} = −g′, g_ν^{ee} = 0, all off-diagonal zero. No tree-level kinetic mixing and no direct quark coupling, as the description states.
3. **Loop-induced mixing**: the μ and τ loops give, for spacelike Q² ≥ 0,

       ε(Q²) = (e g′ / 2π²) ∫₀¹ dx x(1−x) ln[ (m_τ² + x(1−x) Q²) / (m_μ² + x(1−x) Q²) ].

   The m-independent, divergent part of the vacuum polarisation cancels because Σ_ℓ Q′_ℓ Q_ℓ = 0. The logarithm's sign is fixed by the physics (running electromagnetic coupling grows with Q²), giving ε > 0 for all Q², as in section 6. At Q² → 0 the integral of x(1−x) is 1/6 and ε → κ g′ with κ = e ln(m_τ/m_μ)/(6π²) = 0.0144417637, i.e. ε(0) = 1.1553411e-06 for g′ = 8e-5.
4. **Quark couplings**: g_q(Q²) = ε(Q²) e Q_q^em, with Q_u = +2/3, Q_d = −1/3 taken from the exported `Q_EM`.
5. **Matching** (conventions section 3):

       C^{uV}_{mm}(Q²) = + g′ ε(Q²) e (2/3) / (√2 G_F (Q² + m²)),
       C^{dV}_{mm}(Q²) = − g′ ε(Q²) e (1/3) / (√2 G_F (Q² + m²)),
       C^{qV}_{tt} = − C^{qV}_{mm},   C^{qV}_{ee} = 0,   off-diagonal = 0,   scalar = 0.

   Numerical values: m = 12 MeV, g′ = 8e-5, no convention conversion needed (g′ is the gauge coupling multiplying the L_μ−L_τ current, so g_ν = g′ Q′).
6. **Physics check**: 2 C^u + C^d ∝ 2(2/3) − 1/3 = 1 and C^u + 2 C^d ∝ 2/3 − 2/3 = 0, so only protons couple, as expected for photon-mediated mixing. For ν_μ the shift ΔQ = Z g′ ε e/(√2 G_F (Q² + m²)) is positive while Q_SM < 0: destructive interference for the COHERENT prompt ν_μ and delayed ν̄_μ. ν_e sees no new physics.
7. **Q² dependence**: over the COHERENT recoil range Q² reaches a fraction of m_μ² = 11164 MeV², so ε(Q²)/ε(0) falls at the percent level (0.99613, 0.98866, 0.98150 at Q² = 1238, 3714, 6190 MeV², independent of g′ and m). This is the correction a constant card cannot carry.
8. **Quadrature**: the integrand is smooth on [0, 1]; 40-point Gauss-Legendre reproduces the integral to better than 1e-14 relative. A trapezoid rule dropping the endpoints would be low by about 2e-6 and would fail the 1e-6 tolerance.

## Reduction check a reviewer can verify

Replacing `loop_mixing(q2, g)` by the constant `KAPPA_LMU_LTAU * g` (the Q² → 0 limit) must reproduce the example 03 B plugin with m = 12 MeV and g′ = 8e-5. Equivalently, the plugin's coefficients equal the constant-mixing ones multiplied by the g′-independent ratio ε(Q²)/ε(0) listed above.

## Plugin parameters

`PARAMETERS = {"g_prime": 8e-5, "mass_mev": 12.0}`, `COUPLINGS = ("g_prime",)`. The mixing is derived from g′, so zeroing g′ removes every new-physics coefficient (SM limit). |g′| = 8e-5 ≪ √(4π): perturbative.

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
    "couplings are perturbative",
    "SM-limit counts checked with COUPLINGS zeroed (rtol=1e-9, atol=0)",
    "proposed counts checked for finiteness and non-negativity"
  ],
  "notes": "# L_\u03bc\u2212L_\u03c4 Z\u2032 (m = 12 MeV, g\u2032 = 8e-5) with full one-loop Q\u00b2-dependent photon mixing\n\n## Outcome: plugin-only (no card.json)\n\nThe description asks for the *full spacelike Q\u00b2 dependence* of the muon/tau loop mixing. A model card's `kinetic_mixing` field is a single constant, so no card can represent this interaction. Per `framework.md` this is a legitimate plugin-only translation; the harness label is `plugin-only`.\n\n### Correction relative to attempt 1\n\nAttempt 1 shipped an empty placeholder `card.json` (`{}`), which the validator read as zero quark couplings and reported a plugin/card mismatch on every V u and V d entry. The correct resolution is not to repair the card but to omit it: the Q\u00b2-dependent mixing is exactly the case `conventions.md` section 6 and example 06 send to a coefficient plugin. The plugin itself was already correct and is kept unchanged in physics content; only the `Q_EM` import is taken from the exported constants.\n\n## Derivation\n\n1. **Lorentz structure**: vector exchange, structure `V`.\n2. **Charges** (conventions section 7, L_\u03bc\u2212L_\u03c4): Q\u2032_{\u03bd_e} = 0, Q\u2032_{\u03bd_\u03bc} = +1, Q\u2032_{\u03bd_\u03c4} = \u22121, Q\u2032_u = Q\u2032_d = 0. Hence g_\u03bd^{mm} = +g\u2032, g_\u03bd^{tt} = \u2212g\u2032, g_\u03bd^{ee} = 0, all off-diagonal zero. No tree-level kinetic mixing and no direct quark coupling, as the description states.\n3. **Loop-induced mixing**: the \u03bc and \u03c4 loops give, for spacelike Q\u00b2 \u2265 0,\n\n       \u03b5(Q\u00b2) = (e g\u2032 / 2\u03c0\u00b2) \u222b\u2080\u00b9 dx x(1\u2212x) ln[ (m_\u03c4\u00b2 + x(1\u2212x) Q\u00b2) / (m_\u03bc\u00b2 + x(1\u2212x) Q\u00b2) ].\n\n   The m-independent, divergent part of the vacuum polarisation cancels because \u03a3_\u2113 Q\u2032_\u2113 Q_\u2113 = 0. The logarithm's sign is fixed by the physics (running electromagnetic coupling grows with Q\u00b2), giving \u03b5 > 0 for all Q\u00b2, as in section 6. At Q\u00b2 \u2192 0 the integral of x(1\u2212x) is 1/6 and \u03b5 \u2192 \u03ba g\u2032 with \u03ba = e ln(m_\u03c4/m_\u03bc)/(6\u03c0\u00b2) = 0.0144417637, i.e. \u03b5(0) = 1.1553411e-06 for g\u2032 = 8e-5.\n4. **Quark couplings**: g_q(Q\u00b2) = \u03b5(Q\u00b2) e Q_q^em, with Q_u = +2/3, Q_d = \u22121/3 taken from the exported `Q_EM`.\n5. **Matching** (conventions section 3):\n\n       C^{uV}_{mm}(Q\u00b2) = + g\u2032 \u03b5(Q\u00b2) e (2/3) / (\u221a2 G_F (Q\u00b2 + m\u00b2)),\n       C^{dV}_{mm}(Q\u00b2) = \u2212 g\u2032 \u03b5(Q\u00b2) e (1/3) / (\u221a2 G_F (Q\u00b2 + m\u00b2)),\n       C^{qV}_{tt} = \u2212 C^{qV}_{mm},   C^{qV}_{ee} = 0,   off-diagonal = 0,   scalar = 0.\n\n   Numerical values: m = 12 MeV, g\u2032 = 8e-5, no convention conversion needed (g\u2032 is the gauge coupling multiplying the L_\u03bc\u2212L_\u03c4 current, so g_\u03bd = g\u2032 Q\u2032).\n6. **Physics check**: 2 C^u + C^d \u221d 2(2/3) \u2212 1/3 = 1 and C^u + 2 C^d \u221d 2/3 \u2212 2/3 = 0, so only protons couple, as expected for photon-mediated mixing. For \u03bd_\u03bc the shift \u0394Q = Z g\u2032 \u03b5 e/(\u221a2 G_F (Q\u00b2 + m\u00b2)) is positive while Q_SM < 0: destructive interference for the COHERENT prompt \u03bd_\u03bc and delayed \u03bd\u0304_\u03bc. \u03bd_e sees no new physics.\n7. **Q\u00b2 dependence**: over the COHERENT recoil range Q\u00b2 reaches a fraction of m_\u03bc\u00b2 = 11164 MeV\u00b2, so \u03b5(Q\u00b2)/\u03b5(0) falls at the percent level (0.99613, 0.98866, 0.98150 at Q\u00b2 = 1238, 3714, 6190 MeV\u00b2, independent of g\u2032 and m). This is the correction a constant card cannot carry.\n8. **Quadrature**: the integrand is smooth on [0, 1]; 40-point Gauss-Legendre reproduces the integral to better than 1e-14 relative. A trapezoid rule dropping the endpoints would be low by about 2e-6 and would fail the 1e-6 tolerance.\n\n## Reduction check a reviewer can verify\n\nReplacing `loop_mixing(q2, g)` by the constant `KAPPA_LMU_LTAU * g` (the Q\u00b2 \u2192 0 limit) must reproduce the example 03 B plugin with m = 12 MeV and g\u2032 = 8e-5. Equivalently, the plugin's coefficients equal the constant-mixing ones multiplied by the g\u2032-independent ratio \u03b5(Q\u00b2)/\u03b5(0) listed above.\n\n## Plugin parameters\n\n`PARAMETERS = {\"g_prime\": 8e-5, \"mass_mev\": 12.0}`, `COUPLINGS = (\"g_prime\",)`. The mixing is derived from g\u2032, so zeroing g\u2032 removes every new-physics coefficient (SM limit). |g\u2032| = 8e-5 \u226a \u221a(4\u03c0): perturbative.\n"
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
  "final_text": "seeded first attempt",
  "exit_status": 0,
  "raw_output": "{\"fixture\": true, \"source\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/issue-17-agent-harness/docs/evidence/issue-17/eval-claude_code/12_running_mixing/plugin.py\"}",
  "stderr": "",
  "command": [],
  "error": false
}
```

### Attempt 2

```json
{
  "final_text": "Validation passes with no findings. The repair is complete.\n\n**What was wrong.** The first attempt shipped an empty placeholder card alongside a correct plugin. The validator read the empty card as zero quark couplings and flagged a mismatch on every up and down vector entry. The real issue is that no card can express this interaction: the schema's kinetic mixing field is a single constant, while the description asks for the full Q\u00b2 dependence of the muon/tau loop. This is the plugin-only case described in the conventions and in worked example 06.\n\n**Files produced** in the task directory:\n\n- `plugin.py`: vector Z\u2032 with g_\u03bd^{\u03bc\u03bc} = +g\u2032, g_\u03bd^{\u03c4\u03c4} = \u2212g\u2032, no electron or direct quark coupling. Quark couplings come from the spacelike one-loop mixing \u03b5(Q\u00b2), integrated over the Feynman parameter with 40-point Gauss-Legendre quadrature, times e Q_q^em using the exported constants. Defaults are g\u2032 = 8e-5 and m = 12 MeV, with g\u2032 as the only coupling so zeroing it recovers the Standard Model.\n- `notes.md`: derivation, positive sign of the loop mixing, proton-only coupling and destructive interference for \u03bd_\u03bc, the percent-level fall of \u03b5(Q\u00b2)/\u03b5(0) across the COHERENT recoil range, the Q\u00b2 \u2192 0 reduction to the constant-mixing case, and the explanation of why no card exists.\n- No `card.json`, by design.\n\nValidation output:\n\n```\n{\"valid\": true, \"findings\": [], \"checks\": [\"couplings are perturbative\", \"SM-limit counts checked with COUPLINGS zeroed (rtol=1e-9, atol=0)\", \"proposed counts checked for finiteness and non-negativity\"]}\n```\n\nThe expected harness label is `plugin-only`.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":80564,\"stop_reason\":\"end_turn\",\"session_id\":\"05078ea7-b4b1-4334-b597-b33561cef318\",\"total_cost_usd\":0.9844824999999999,\"usage\":{\"input_tokens\":258,\"cache_creation_input_tokens\":29976,\"cache_read_input_tokens\":184930,\"output_tokens\":6723,\"output_tokens_details\":{\"thinking_tokens\":566},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":29976,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":603,\"cache_read_input_tokens\":33658,\"cache_creation_input_tokens\":2324,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":2324},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":258,\"outputTokens\":6723,\"cacheReadInputTokens\":184930,\"cacheCreationInputTokens\":29976,\"webSearchRequests\":0,\"costUSD\":0.9844824999999999,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":566,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_014s7F4dAor2F3QNAYtZ4sBK\",\"tool_input\":{\"command\":\"ls -R /private/tmp/nuagent-issue17-repair-plugin-only/report.task\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":32,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passes with no findings. The repair is complete.\\n\\n**What was wrong.** The first attempt shipped an empty placeholder card alongside a correct plugin. The validator read the empty card as zero quark couplings and flagged a mismatch on every up and down vector entry. The real issue is that no card can express this interaction: the schema's kinetic mixing field is a single constant, while the description asks for the full Q\u00b2 dependence of the muon/tau loop. This is the plugin-only case described in the conventions and in worked example 06.\\n\\n**Files produced** in the task directory:\\n\\n- `plugin.py`: vector Z\u2032 with g_\u03bd^{\u03bc\u03bc} = +g\u2032, g_\u03bd^{\u03c4\u03c4} = \u2212g\u2032, no electron or direct quark coupling. Quark couplings come from the spacelike one-loop mixing \u03b5(Q\u00b2), integrated over the Feynman parameter with 40-point Gauss-Legendre quadrature, times e Q_q^em using the exported constants. Defaults are g\u2032 = 8e-5 and m = 12 MeV, with g\u2032 as the only coupling so zeroing it recovers the Standard Model.\\n- `notes.md`: derivation, positive sign of the loop mixing, proton-only coupling and destructive interference for \u03bd_\u03bc, the percent-level fall of \u03b5(Q\u00b2)/\u03b5(0) across the COHERENT recoil range, the Q\u00b2 \u2192 0 reduction to the constant-mixing case, and the explanation of why no card exists.\\n- No `card.json`, by design.\\n\\nValidation output:\\n\\n```\\n{\\\"valid\\\": true, \\\"findings\\\": [], \\\"checks\\\": [\\\"couplings are perturbative\\\", \\\"SM-limit counts checked with COUPLINGS zeroed (rtol=1e-9, atol=0)\\\", \\\"proposed counts checked for finiteness and non-negativity\\\"]}\\n```\\n\\nThe expected harness label is `plugin-only`.\",\"ttft_ms\":1675,\"type\":\"result\",\"duration_ms\":82305,\"uuid\":\"51084104-18b8-47a0-adf0-ee61edbae6c9\",\"ttft_stream_ms\":1546,\"time_to_request_ms\":36,\"first_content_frame_ms\":1546,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/issue-17-agent-harness/.venv/bin/python3 -m nuagent.agent.tool_policy /private/tmp/nuagent-issue17-repair-plugin-only/report.task\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
