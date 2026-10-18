Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

A universal vector Z′ of mass 45 MeV couples with g′ = 7e-5 to all three left-handed neutrino flavors and to up and down quarks. There are no other couplings.

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
"""Flavor-universal vector Z' coupling equally to left-handed neutrinos and u, d quarks.

Derivation in notes.md.
"""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_prime": 7e-5, "mass_mev": 45.0}
COUPLINGS = ("g_prime",)


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    g_nu = params["g_prime"]
    g_q = params["g_prime"]
    value = g_nu * g_q / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)
    return c

```

## card.json

```json
{
  "name": "zprime_universal_45mev",
  "mediators": [
    {
      "structure": "V",
      "mass_mev": 45.0,
      "nu_couplings": {
        "ee": 7e-5,
        "mm": 7e-5,
        "tt": 7e-5,
        "em": 0,
        "et": 0,
        "mt": 0
      },
      "quark_couplings": {
        "u": 7e-5,
        "d": 7e-5
      },
      "kinetic_mixing": 0.0
    }
  ],
  "contacts": []
}

```

## notes.md

```text
# Notes: flavor-universal vector Z′ (m = 45 MeV, g′ = 7e-5)

## Description

A universal vector Z′ of mass 45 MeV couples with g′ = 7e-5 to all three
left-handed neutrino flavors and to up and down quarks. No other couplings.

## Derivation

1. **Lorentz structure.** Vector exchange, so structure `V`. No scalar entries.
   No axial coupling is mentioned, so nothing is dropped.

2. **Couplings as written.** Lagrangian in the form of conventions section 3,

       L ⊃ Z′_μ ( g′ Σ_α ν̄_α γ^μ P_L ν_α + g′ ū γ^μ u + g′ d̄ γ^μ d ),

   so g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = g′ = 7e-5, all off-diagonal g_ν = 0,
   and g_u = g_d = g′ = 7e-5. The description says "left-handed neutrino
   flavors", so g_ν is already the left-handed coupling; no factor of 2
   conversion is needed. The quark coupling is a plain vector coupling, so
   g_q = g′ directly (no chiral averaging). No kinetic mixing (ε = 0), since
   "there are no other couplings".

3. **Matching** (conventions section 3, tree-level Z′ exchange):

       C^{uV}_{αα}(Q²) = C^{dV}_{αα}(Q²) = g′² / ( √2 G_F (Q² + m²) ),
       m = 45 MeV,  m² = 2025 MeV².

   Flavor-diagonal and identical for e, μ, τ; off-diagonal entries zero.

4. **Reference values** (Cs-133 points of example 01, Q² = 1238, 3714, 6190 MeV²;
   √2 G_F = 1.6495e-11 MeV⁻², g′² = 4.9e-9):

   | Q² [MeV²] | C^{uV} = C^{dV} (each diagonal flavor) |
   |---|---|
   | 1238 | 9.104e-02 |
   | 3714 | 5.176e-02 |
   | 6190 | 3.616e-02 |

5. **Sign and interference.** g_ν g_q = +g′² > 0, so the new-physics shift of the
   nuclear vector charge, ΔQ = (2Z + N + Z + 2N) C = 3A·C, is positive, while
   the SM charge Q_SM ≈ 0.04 Z − 0.5 N is negative: destructive interference.
   For Cs-133 (A = 133) at Q² = 1238 MeV², ΔQ ≈ 399 × 0.0910 ≈ 36.3, close in
   magnitude to the tree-level |Q_SM| ≈ 36.9. This parameter point therefore
   sits near the charge-cancellation region at low recoil, with the
   cancellation weakening at higher Q² (ΔQ ≈ 20.7 and 14.4 at the other two
   points). This is a property of the described point and is reported as given;
   nothing was tuned.

6. **Perturbativity.** |g′| = 7e-5 ≪ √(4π).

7. **Card.** The schema expresses this exactly as one vector mediator with
   diagonal neutrino couplings g′ and quark couplings g_u = g_d = g′, so
   `card.json` is provided alongside `plugin.py`. The plugin computes the same
   coefficients from the exported constants `GF` and `SQRT2`, independently of
   the card.

8. **Anomaly remark.** The universal charge assignment (Q′ = +1 for all listed
   fields) is a toy model and is anomalous as a gauge symmetry; this is the
   model author's concern and does not affect the low-energy matching.

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
  "notes": "# Notes: flavor-universal vector Z\u2032 (m = 45 MeV, g\u2032 = 7e-5)\n\n## Description\n\nA universal vector Z\u2032 of mass 45 MeV couples with g\u2032 = 7e-5 to all three\nleft-handed neutrino flavors and to up and down quarks. No other couplings.\n\n## Derivation\n\n1. **Lorentz structure.** Vector exchange, so structure `V`. No scalar entries.\n   No axial coupling is mentioned, so nothing is dropped.\n\n2. **Couplings as written.** Lagrangian in the form of conventions section 3,\n\n       L \u2283 Z\u2032_\u03bc ( g\u2032 \u03a3_\u03b1 \u03bd\u0304_\u03b1 \u03b3^\u03bc P_L \u03bd_\u03b1 + g\u2032 \u016b \u03b3^\u03bc u + g\u2032 d\u0304 \u03b3^\u03bc d ),\n\n   so g_\u03bd^{ee} = g_\u03bd^{mm} = g_\u03bd^{tt} = g\u2032 = 7e-5, all off-diagonal g_\u03bd = 0,\n   and g_u = g_d = g\u2032 = 7e-5. The description says \"left-handed neutrino\n   flavors\", so g_\u03bd is already the left-handed coupling; no factor of 2\n   conversion is needed. The quark coupling is a plain vector coupling, so\n   g_q = g\u2032 directly (no chiral averaging). No kinetic mixing (\u03b5 = 0), since\n   \"there are no other couplings\".\n\n3. **Matching** (conventions section 3, tree-level Z\u2032 exchange):\n\n       C^{uV}_{\u03b1\u03b1}(Q\u00b2) = C^{dV}_{\u03b1\u03b1}(Q\u00b2) = g\u2032\u00b2 / ( \u221a2 G_F (Q\u00b2 + m\u00b2) ),\n       m = 45 MeV,  m\u00b2 = 2025 MeV\u00b2.\n\n   Flavor-diagonal and identical for e, \u03bc, \u03c4; off-diagonal entries zero.\n\n4. **Reference values** (Cs-133 points of example 01, Q\u00b2 = 1238, 3714, 6190 MeV\u00b2;\n   \u221a2 G_F = 1.6495e-11 MeV\u207b\u00b2, g\u2032\u00b2 = 4.9e-9):\n\n   | Q\u00b2 [MeV\u00b2] | C^{uV} = C^{dV} (each diagonal flavor) |\n   |---|---|\n   | 1238 | 9.104e-02 |\n   | 3714 | 5.176e-02 |\n   | 6190 | 3.616e-02 |\n\n5. **Sign and interference.** g_\u03bd g_q = +g\u2032\u00b2 > 0, so the new-physics shift of the\n   nuclear vector charge, \u0394Q = (2Z + N + Z + 2N) C = 3A\u00b7C, is positive, while\n   the SM charge Q_SM \u2248 0.04 Z \u2212 0.5 N is negative: destructive interference.\n   For Cs-133 (A = 133) at Q\u00b2 = 1238 MeV\u00b2, \u0394Q \u2248 399 \u00d7 0.0910 \u2248 36.3, close in\n   magnitude to the tree-level |Q_SM| \u2248 36.9. This parameter point therefore\n   sits near the charge-cancellation region at low recoil, with the\n   cancellation weakening at higher Q\u00b2 (\u0394Q \u2248 20.7 and 14.4 at the other two\n   points). This is a property of the described point and is reported as given;\n   nothing was tuned.\n\n6. **Perturbativity.** |g\u2032| = 7e-5 \u226a \u221a(4\u03c0).\n\n7. **Card.** The schema expresses this exactly as one vector mediator with\n   diagonal neutrino couplings g\u2032 and quark couplings g_u = g_d = g\u2032, so\n   `card.json` is provided alongside `plugin.py`. The plugin computes the same\n   coefficients from the exported constants `GF` and `SQRT2`, independently of\n   the card.\n\n8. **Anomaly remark.** The universal charge assignment (Q\u2032 = +1 for all listed\n   fields) is a toy model and is anomalous as a gauge symmetry; this is the\n   model author's concern and does not affect the low-energy matching.\n"
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
      0.13184754856554987,
      0.6905458030784077,
      1.6873322305501948,
      2.5566407278909957,
      2.7460294999704398,
      2.510977126785431,
      2.289985986841688,
      1.9756864395317422,
      1.6353446151347812,
      1.3104642620690594,
      1.0208462072303037,
      0.7742586458535937
    ],
    "predicted_total": 19.329959093502186,
    "observed_total": 855.0,
    "delta": 14.545393032092662,
    "new_physics": {
      "neg2_log_likelihood": 31.92283053152668,
      "alpha": 0.1202741319517469,
      "beta": {
        "steady_state": 0.08369012558492817,
        "prompt_neutrons": 0.02812347153213654
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
  "final_text": "Validation passed with no findings. The plugin and card agree on all 66 Q\u00b2 cross-check points, the SM limit holds with the coupling zeroed, and the counts are finite and non-negative.\n\nFiles produced in the task directory:\n\n- **plugin.py** computes the flavor-diagonal vector coefficients C^{uV} = C^{dV} = g\u2032\u00b2 / (\u221a2 G_F (Q\u00b2 + m\u00b2)) for ee, mm and tt, with defaults g\u2032 = 7e-5 and m = 45 MeV and `g_prime` as the only coupling.\n- **card.json** expresses the same interaction as one vector mediator with neutrino couplings 7e-5 on each diagonal flavor, quark couplings g_u = g_d = 7e-5, and no kinetic mixing.\n- **notes.md** records the derivation: structure V, the left-handed neutrino coupling taken as written, the section 3 matching formula, reference coefficient values at the three Cs-133 Q\u00b2 points, and the sign analysis. The product g_\u03bd g_q is positive, so the interference with the Standard Model is destructive. At this point the new-physics charge shift for Cs-133 at low recoil is close in magnitude to the SM charge, so the described point sits near the charge-cancellation region. This was reported as a property of the given point and nothing was tuned.\n\nNo questions.json was written, since the description fixes the mass, the couplings and the Lorentz structure completely.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":119452,\"stop_reason\":\"end_turn\",\"session_id\":\"93ac169c-0bba-4d0e-8dde-e7b0e5cd8b54\",\"total_cost_usd\":0.74321925,\"usage\":{\"input_tokens\":194,\"cache_creation_input_tokens\":22804,\"cache_read_input_tokens\":119997,\"output_tokens\":5104,\"output_tokens_details\":{\"thinking_tokens\":792},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":22804,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":470,\"cache_read_input_tokens\":28511,\"cache_creation_input_tokens\":299,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":299},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":194,\"outputTokens\":5104,\"cacheReadInputTokens\":119997,\"cacheCreationInputTokens\":22804,\"webSearchRequests\":0,\"costUSD\":0.74321925,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":792,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01P4ZaMLPJrPYtSDbV3DnDzq\",\"tool_input\":{\"command\":\"find . -type f | head -50\",\"description\":\"List files in task directory\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":15,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings. The plugin and card agree on all 66 Q\u00b2 cross-check points, the SM limit holds with the coupling zeroed, and the counts are finite and non-negative.\\n\\nFiles produced in the task directory:\\n\\n- **plugin.py** computes the flavor-diagonal vector coefficients C^{uV} = C^{dV} = g\u2032\u00b2 / (\u221a2 G_F (Q\u00b2 + m\u00b2)) for ee, mm and tt, with defaults g\u2032 = 7e-5 and m = 45 MeV and `g_prime` as the only coupling.\\n- **card.json** expresses the same interaction as one vector mediator with neutrino couplings 7e-5 on each diagonal flavor, quark couplings g_u = g_d = 7e-5, and no kinetic mixing.\\n- **notes.md** records the derivation: structure V, the left-handed neutrino coupling taken as written, the section 3 matching formula, reference coefficient values at the three Cs-133 Q\u00b2 points, and the sign analysis. The product g_\u03bd g_q is positive, so the interference with the Standard Model is destructive. At this point the new-physics charge shift for Cs-133 at low recoil is close in magnitude to the SM charge, so the described point sits near the charge-cancellation region. This was reported as a property of the given point and nothing was tuned.\\n\\nNo questions.json was written, since the description fixes the mass, the couplings and the Lorentz structure completely.\",\"ttft_ms\":4512,\"type\":\"result\",\"duration_ms\":123236,\"uuid\":\"a68e3637-24d0-4a72-a426-9c40bb2e548d\",\"ttft_stream_ms\":3501,\"time_to_request_ms\":26,\"first_content_frame_ms\":3501,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/m7-conversation/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/01_universal_vector\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
