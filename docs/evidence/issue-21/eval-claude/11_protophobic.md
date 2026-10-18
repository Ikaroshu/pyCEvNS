Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

A protophobic vector mediator has mass 20 MeV, universal left-handed neutrino coupling g_nu = 1e-4, and vector nucleon couplings g_p = 0 and g_n = 3e-4. There are no other interactions.

## Conversation

### Round 1: valid

## Provenance

```json
{
  "package": "nuagent",
  "package_version": "0.1.0",
  "source_path": "/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/m7-conversation/src/nuagent/agent/report.py",
  "git_revision": "9a0bb9100a177d1959606dfd94eeb4b92afeed81",
  "git_tree": "79c38e2695fcbf26a3535cbcb843611ccf137fd2",
  "source_dirty": false,
  "data_releases": {
    "coherent_csi_2017": "10.5281/zenodo.1228631"
  },
  "max_turns": 16,
  "rounds": 1,
  "backend": "claude"
}
```

## plugin.py

```python
"""Protophobic vector mediator; the derivation is in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_nu": 1e-4, "g_p": 0.0, "g_n": 3e-4, "mass_mev": 20.0}
COUPLINGS = ("g_nu", "g_p", "g_n")


def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    g_nu = params["g_nu"]
    g_p = params["g_p"]
    g_n = params["g_n"]
    g_u = (2.0 * g_p - g_n) / 3.0
    g_d = (2.0 * g_n - g_p) / 3.0
    propagator = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    quark_values = {"u": g_nu * g_u * propagator, "d": g_nu * g_d * propagator}
    for quark, value in quark_values.items():
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)
    return c

```

## card.json

```json
{
  "name": "protophobic_vector_mediator",
  "mediators": [
    {
      "structure": "V",
      "mass_mev": 20.0,
      "nu_couplings": {
        "ee": 0.0001,
        "mm": 0.0001,
        "tt": 0.0001,
        "em": 0,
        "et": 0,
        "mt": 0
      },
      "quark_couplings": {
        "u": -0.0001,
        "d": 0.0002
      },
      "kinetic_mixing": 0.0
    }
  ],
  "contacts": []
}

```

## notes.md

```text
# Protophobic vector mediator: derivation

## Description as given

A vector mediator of mass m = 20 MeV, universal left-handed neutrino coupling
g_ν = 1e-4 (same for ν_e, ν_μ, ν_τ, no off-diagonal entries), and vector
nucleon couplings g_p = 0 and g_n = 3e-4. No other interactions.

## 1. Lorentz structure

Vector exchange, so structure `V`. The neutrino coupling is stated to be to the
left-handed current, which is exactly the g_ν ν̄ γ^μ P_L ν normalization of
conventions section 3, so g_ν = 1e-4 with no conversion. No scalar, axial,
pseudoscalar or tensor pieces are present.

## 2. Nucleon couplings to quark couplings

Nucleon couplings are not card fields, but they convert exactly to quark
couplings because the framework's nuclear vector charge is Q = Z g_p + N g_n
with g_p = 2 g_u + g_d and g_n = g_u + 2 g_d (conventions section 3). Inverting:

    g_u = (2 g_p − g_n) / 3 = (0 − 3e-4) / 3 = −1e-4
    g_d = (2 g_n − g_p) / 3 = (6e-4 − 0) / 3 = +2e-4

Check: 2 g_u + g_d = −2e-4 + 2e-4 = 0 = g_p; g_u + 2 g_d = −1e-4 + 4e-4 = 3e-4 = g_n.
This is the protophobic case g_u = −g_n/3, g_d = 2 g_n/3 quoted in the conventions.

## 3. Matching

Tree-level exchange (conventions section 3):

    C^{fV}_{αα}(Q²) = g_ν g_f / ( √2 G_F (Q² + m²) ),   α = e, μ, τ

so

    C^{uV}_{αα}(Q²) = (1e-4)(−1e-4) / ( √2 G_F (Q² + 400 MeV²) ) = −1e-8 / ( √2 G_F (Q² + 400) )
    C^{dV}_{αα}(Q²) = (1e-4)(+2e-4) / ( √2 G_F (Q² + 400 MeV²) ) = +2e-8 / ( √2 G_F (Q² + 400) )

Off-diagonal flavor entries and all scalar entries are zero. The plugin keeps
g_nu, g_p, g_n and mass_mev as parameters and computes g_u, g_d from g_p, g_n
with the inversion above, so that the described nucleon-level point is preserved
exactly; the card carries the equivalent quark couplings.

## 4. Numerical check at the reference points (Cs-133, Q² = 1238, 3714, 6190 MeV²)

With √2 G_F = 1.6495e-11 MeV⁻²:

| coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
|---|---|---|---|
| C^{uV}_αα | −3.7011e-01 | −1.4735e-01 | −9.1990e-02 |
| C^{dV}_αα | +7.4022e-01 | +2.9470e-01 | +1.8398e-01 |

Values rounded for reading; the plugin uses the exported constants.

## 5. Sign and interference

The nuclear charge shift is ΔQ = Z (2 C^u + C^d) + N (C^u + 2 C^d). The proton
combination 2 C^u + C^d vanishes by construction (protophobic), and the neutron
combination C^u + 2 C^d equals g_ν g_n / (√2 G_F (Q² + m²)), so
ΔQ = N g_ν g_n / (√2 G_F (Q² + m²)) with g_ν g_n = +3e-8 > 0. For Cs-133 (N = 78) at Q² = 1238 MeV² this gives
ΔQ ≈ +86.6, positive, while Q_SM ≈ −36.9 is negative: destructive interference.
At this point |ΔQ| > 2 |Q_SM|, so the total charge has flipped sign and the
coherent rate exceeds the SM rate at low recoil (ratio ≈ (49.7/36.9)² ≈ 1.8 at
tree level at the first reference point, falling with Q² as the propagator
drops).

## 6. Card

The interaction is a single vector mediator with constant couplings, so the
schema expresses it exactly: one `V` mediator entry with mass 20 MeV, diagonal
neutrino couplings 1e-4, quark couplings u = −1e-4, d = +2e-4, and no kinetic
mixing. No contact terms (a mass is given, so the mediator form is used).

## 7. Conventions and assumptions

- No convention conversion was needed: the description already uses the
  left-handed neutrino current and vector (not chiral) nucleon couplings.
- The nucleon couplings are taken as pure vector couplings at Q² → 0 of the
  nucleon, which is what the framework's Q = Z g_p + N g_n charge assumes; no
  axial coupling was given or dropped.
- Perturbativity: all couplings are O(1e-4), far below √(4π).
- Anomaly cancellation for a protophobic U(1) is the model author's concern.

```

## Validation and derivation history

### Round 1, attempt 1

```json
{
  "valid": true,
  "findings": [],
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
  "notes": "# Protophobic vector mediator: derivation\n\n## Description as given\n\nA vector mediator of mass m = 20 MeV, universal left-handed neutrino coupling\ng_\u03bd = 1e-4 (same for \u03bd_e, \u03bd_\u03bc, \u03bd_\u03c4, no off-diagonal entries), and vector\nnucleon couplings g_p = 0 and g_n = 3e-4. No other interactions.\n\n## 1. Lorentz structure\n\nVector exchange, so structure `V`. The neutrino coupling is stated to be to the\nleft-handed current, which is exactly the g_\u03bd \u03bd\u0304 \u03b3^\u03bc P_L \u03bd normalization of\nconventions section 3, so g_\u03bd = 1e-4 with no conversion. No scalar, axial,\npseudoscalar or tensor pieces are present.\n\n## 2. Nucleon couplings to quark couplings\n\nNucleon couplings are not card fields, but they convert exactly to quark\ncouplings because the framework's nuclear vector charge is Q = Z g_p + N g_n\nwith g_p = 2 g_u + g_d and g_n = g_u + 2 g_d (conventions section 3). Inverting:\n\n    g_u = (2 g_p \u2212 g_n) / 3 = (0 \u2212 3e-4) / 3 = \u22121e-4\n    g_d = (2 g_n \u2212 g_p) / 3 = (6e-4 \u2212 0) / 3 = +2e-4\n\nCheck: 2 g_u + g_d = \u22122e-4 + 2e-4 = 0 = g_p; g_u + 2 g_d = \u22121e-4 + 4e-4 = 3e-4 = g_n.\nThis is the protophobic case g_u = \u2212g_n/3, g_d = 2 g_n/3 quoted in the conventions.\n\n## 3. Matching\n\nTree-level exchange (conventions section 3):\n\n    C^{fV}_{\u03b1\u03b1}(Q\u00b2) = g_\u03bd g_f / ( \u221a2 G_F (Q\u00b2 + m\u00b2) ),   \u03b1 = e, \u03bc, \u03c4\n\nso\n\n    C^{uV}_{\u03b1\u03b1}(Q\u00b2) = (1e-4)(\u22121e-4) / ( \u221a2 G_F (Q\u00b2 + 400 MeV\u00b2) ) = \u22121e-8 / ( \u221a2 G_F (Q\u00b2 + 400) )\n    C^{dV}_{\u03b1\u03b1}(Q\u00b2) = (1e-4)(+2e-4) / ( \u221a2 G_F (Q\u00b2 + 400 MeV\u00b2) ) = +2e-8 / ( \u221a2 G_F (Q\u00b2 + 400) )\n\nOff-diagonal flavor entries and all scalar entries are zero. The plugin keeps\ng_nu, g_p, g_n and mass_mev as parameters and computes g_u, g_d from g_p, g_n\nwith the inversion above, so that the described nucleon-level point is preserved\nexactly; the card carries the equivalent quark couplings.\n\n## 4. Numerical check at the reference points (Cs-133, Q\u00b2 = 1238, 3714, 6190 MeV\u00b2)\n\nWith \u221a2 G_F = 1.6495e-11 MeV\u207b\u00b2:\n\n| coefficient | Q\u00b2=1238 | Q\u00b2=3714 | Q\u00b2=6190 |\n|---|---|---|---|\n| C^{uV}_\u03b1\u03b1 | \u22123.7011e-01 | \u22121.4735e-01 | \u22129.1990e-02 |\n| C^{dV}_\u03b1\u03b1 | +7.4022e-01 | +2.9470e-01 | +1.8398e-01 |\n\nValues rounded for reading; the plugin uses the exported constants.\n\n## 5. Sign and interference\n\nThe nuclear charge shift is \u0394Q = Z (2 C^u + C^d) + N (C^u + 2 C^d). The proton\ncombination 2 C^u + C^d vanishes by construction (protophobic), and the neutron\ncombination C^u + 2 C^d equals g_\u03bd g_n / (\u221a2 G_F (Q\u00b2 + m\u00b2)), so\n\u0394Q = N g_\u03bd g_n / (\u221a2 G_F (Q\u00b2 + m\u00b2)) with g_\u03bd g_n = +3e-8 > 0. For Cs-133 (N = 78) at Q\u00b2 = 1238 MeV\u00b2 this gives\n\u0394Q \u2248 +86.6, positive, while Q_SM \u2248 \u221236.9 is negative: destructive interference.\nAt this point |\u0394Q| > 2 |Q_SM|, so the total charge has flipped sign and the\ncoherent rate exceeds the SM rate at low recoil (ratio \u2248 (49.7/36.9)\u00b2 \u2248 1.8 at\ntree level at the first reference point, falling with Q\u00b2 as the propagator\ndrops).\n\n## 6. Card\n\nThe interaction is a single vector mediator with constant couplings, so the\nschema expresses it exactly: one `V` mediator entry with mass 20 MeV, diagonal\nneutrino couplings 1e-4, quark couplings u = \u22121e-4, d = +2e-4, and no kinetic\nmixing. No contact terms (a mass is given, so the mediator form is used).\n\n## 7. Conventions and assumptions\n\n- No convention conversion was needed: the description already uses the\n  left-handed neutrino current and vector (not chiral) nucleon couplings.\n- The nucleon couplings are taken as pure vector couplings at Q\u00b2 \u2192 0 of the\n  nucleon, which is what the framework's Q = Z g_p + N g_n charge assumes; no\n  axial coupling was given or dropped.\n- Perturbativity: all couplings are O(1e-4), far below \u221a(4\u03c0).\n- Anomaly cancellation for a protophobic U(1) is the model author's concern.\n"
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
      10.226202154011277,
      6.824356436426806,
      3.3685345321206017,
      0.9915280891565783,
      0.1069449798632902,
      0.04761226311783086,
      0.2061838099520274,
      0.3549190781135437,
      0.4351245047479896,
      0.4497417904077724,
      0.41865975493860125,
      0.3620777184065254
    ],
    "predicted_total": 23.791885111262843,
    "observed_total": 855.0,
    "delta": 16.74642005871507,
    "new_physics": {
      "neg2_log_likelihood": 34.12385755814909,
      "alpha": -0.02429660708595615,
      "beta": {
        "steady_state": 0.08392100357692557,
        "prompt_neutrons": 0.03304430052962912
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

### Round 1, attempt 1

```json
{
  "final_text": "Validation passed with no findings. Plugin and card agree on all 66 Q\u00b2 points, the Standard Model limit holds with the couplings zeroed, and the couplings are perturbative.\n\nFiles produced in the task directory:\n\n- **plugin.py** keeps the described point as parameters g_nu = 1e-4, g_p = 0, g_n = 3e-4, mass_mev = 20. It converts nucleon to quark couplings with g_u = (2g_p \u2212 g_n)/3 and g_d = (2g_n \u2212 g_p)/3, then applies the tree-level vector matching from conventions section 3 to the three flavor-diagonal entries of C^{uV} and C^{dV}.\n- **card.json** expresses the same interaction as a single vector mediator with mass 20 MeV, diagonal neutrino couplings 1e-4, and quark couplings u = \u22121e-4 and d = +2e-4.\n- **notes.md** records the derivation: Lorentz structure, the nucleon-to-quark inversion with its check, the matching formula, coefficient values at the three Cs-133 reference points, and the sign analysis. The product of neutrino and neutron couplings is positive, so the new physics interferes destructively with the Standard Model and flips the sign of the nuclear charge at low recoil.\n\nNo questions.json was needed since the description fixes the mass, the couplings and the Lorentz structure completely.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":82600,\"stop_reason\":\"end_turn\",\"session_id\":\"bac55dc9-c65e-497a-a135-c6ddb0e76547\",\"total_cost_usd\":0.9210075,\"usage\":{\"input_tokens\":258,\"cache_creation_input_tokens\":26590,\"cache_read_input_tokens\":180710,\"output_tokens\":6829,\"output_tokens_details\":{\"thinking_tokens\":1036},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":26590,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":460,\"cache_read_input_tokens\":32297,\"cache_creation_input_tokens\":299,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":299},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":258,\"outputTokens\":6829,\"cacheReadInputTokens\":180710,\"cacheCreationInputTokens\":26590,\"webSearchRequests\":0,\"costUSD\":0.9210075,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":1036,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01Gmeqqo1ssaGpWPfuXpYeoF\",\"tool_input\":{\"command\":\"ls -R /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/11_protophobic\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":20,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings. Plugin and card agree on all 66 Q\u00b2 points, the Standard Model limit holds with the couplings zeroed, and the couplings are perturbative.\\n\\nFiles produced in the task directory:\\n\\n- **plugin.py** keeps the described point as parameters g_nu = 1e-4, g_p = 0, g_n = 3e-4, mass_mev = 20. It converts nucleon to quark couplings with g_u = (2g_p \u2212 g_n)/3 and g_d = (2g_n \u2212 g_p)/3, then applies the tree-level vector matching from conventions section 3 to the three flavor-diagonal entries of C^{uV} and C^{dV}.\\n- **card.json** expresses the same interaction as a single vector mediator with mass 20 MeV, diagonal neutrino couplings 1e-4, and quark couplings u = \u22121e-4 and d = +2e-4.\\n- **notes.md** records the derivation: Lorentz structure, the nucleon-to-quark inversion with its check, the matching formula, coefficient values at the three Cs-133 reference points, and the sign analysis. The product of neutrino and neutron couplings is positive, so the new physics interferes destructively with the Standard Model and flips the sign of the nuclear charge at low recoil.\\n\\nNo questions.json was needed since the description fixes the mass, the couplings and the Lorentz structure completely.\",\"ttft_ms\":870,\"type\":\"result\",\"duration_ms\":86421,\"uuid\":\"86d1b329-51c0-4580-b9b7-33afa8c0454a\",\"ttft_stream_ms\":754,\"time_to_request_ms\":24,\"first_content_frame_ms\":755,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/m7-conversation/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/11_protophobic\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
