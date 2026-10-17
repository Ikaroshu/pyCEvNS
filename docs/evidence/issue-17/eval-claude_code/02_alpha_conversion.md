Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

A universal vector Z′ of mass 45 MeV has alpha′ = g′²/(4 pi) = 3.899296105751435e-10. Its positive coupling g′ is the same for all left-handed neutrino flavors and both u and d quarks, with no other couplings.

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
"""Universal vector Z' specified by alpha' = g'^2 / (4 pi); derivation in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"alpha_prime": 3.899296105751435e-10, "mass_mev": 45.0}
COUPLINGS = ("alpha_prime",)


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    g_prime_sq = 4.0 * np.pi * params["alpha_prime"]  # g'^2, with g' > 0
    value = g_prime_sq / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)
    return c

```

## card.json

```json
{
  "name": "zprime_universal_alpha_prime_45mev",
  "mediators": [
    {
      "structure": "V",
      "mass_mev": 45.0,
      "nu_couplings": {
        "ee": 7.0e-05,
        "mm": 7.0e-05,
        "tt": 7.0e-05,
        "em": 0,
        "et": 0,
        "mt": 0
      },
      "quark_couplings": {
        "u": 7.0e-05,
        "d": 7.0e-05
      },
      "kinetic_mixing": 0.0
    }
  ],
  "contacts": []
}

```

## notes.md

```text
# Universal vector Z′, m = 45 MeV, α′ = g′²/(4π) = 3.899296105751435e-10

## Derivation

1. **Lorentz structure.** A vector Z′ exchanged between the left-handed neutrino current and the quark vector currents: structure `V` only. No scalar, axial or tensor pieces are described, so no scalar entries and nothing outside the basis.

2. **Convention conversion (conventions section 10).** The description gives the coupling as a fine-structure-type constant, α′ = g′²/(4π). Inverting,

       g′² = 4π α′ = 4π × 3.899296105751435e-10 = 4.9000e-9,
       g′  = √(4π α′) = 7.0e-5   (positive root, as stated: "positive coupling g′").

   Check: (7.0e-5)² / (4π) = 4.9e-9 / 12.566370614359… = 3.8992961057514e-10, which reproduces the given α′ to double precision. The plugin keeps α′ itself as the parameter, so the described point is preserved exactly, and forms g′² = 4π α′ internally; the card, which needs the Lagrangian coupling, carries g′ = 7.0e-5.

3. **Couplings as written.** Universal: g_ν^{ee} = g_ν^{μμ} = g_ν^{ττ} = g′, all off-diagonal g_ν^{αβ} = 0; g_u = g_d = g′. No kinetic mixing, no other couplings. This is the "universal (toy)" assignment of conventions section 7 with Q′ = +1 for all neutrinos and both quarks (anomalous as a gauge theory, but that is the model author's concern).

4. **Matching (conventions section 3).**

       C^{uV}_{αα}(Q²) = C^{dV}_{αα}(Q²) = g_ν g_q / (√2 G_F (Q² + m²)) = g′² / (√2 G_F (Q² + m²)) = 4π α′ / (√2 G_F (Q² + 45²)),

   for α = e, μ, τ; all off-diagonal and all scalar coefficients are zero. Propagator is 1/(Q² + m²) with Q² = 2 M T ≥ 0 in MeV². The mediator form is used (not a contact term) because a mass is given.

5. **Sign and interference.** g_ν g_q = +g′² > 0, so the new-physics nuclear charge shift ΔQ = (3Z + 3N) C is positive, opposite to the negative SM charge Q_SM ≈ 0.04 Z − 0.5 N. The interference is destructive. For reference, at Q² = 1238 MeV² (Cs-133, T = 5 keV): C ≈ 4.9e-9 / (1.41421 × 1.16638e-11 × (1238 + 2025)) ≈ 9.10e-2, so ΔQ ≈ 3 × 133 × 0.091 ≈ 36.3 against Q_SM ≈ −36.9: this point sits very close to the charge cancellation for Cs at low recoil, and the coherent rate there is strongly suppressed. This is a consequence of the described point, which is preserved exactly; nothing was tuned.

6. **Sanity.** g′ = 7.0e-5 ≪ √(4π) ≈ 3.54; mass positive. With `alpha_prime` zeroed the plugin returns all-zero coefficients, i.e. the SM limit.

## Card

Expressible: one vector mediator with mass 45 MeV, diagonal universal neutrino couplings g′ = 7.0e-5 and quark couplings g_u = g_d = 7.0e-5, zero kinetic mixing, no contacts. The card's g′ is the exact square root of 4π α′ at double precision, so the plugin/card cross-check agrees to far better than 1e-6.

## Files

- `plugin.py`: coefficient plugin, parameters `alpha_prime` (coupling, default 3.899296105751435e-10) and `mass_mev` (45.0).
- `card.json`: equivalent mediator card with g′ = 7.0e-5.
- `notes.md`: this derivation.

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
    "proposed counts checked for finiteness and non-negativity",
    "ModelCard schema valid",
    "couplings are perturbative",
    "card SM limit holds by construction",
    "card counts checked for finiteness and non-negativity",
    "plugin/card cross-check on 66 Q2 points spanning the selected recoil ranges"
  ],
  "notes": "# Universal vector Z\u2032, m = 45 MeV, \u03b1\u2032 = g\u2032\u00b2/(4\u03c0) = 3.899296105751435e-10\n\n## Derivation\n\n1. **Lorentz structure.** A vector Z\u2032 exchanged between the left-handed neutrino current and the quark vector currents: structure `V` only. No scalar, axial or tensor pieces are described, so no scalar entries and nothing outside the basis.\n\n2. **Convention conversion (conventions section 10).** The description gives the coupling as a fine-structure-type constant, \u03b1\u2032 = g\u2032\u00b2/(4\u03c0). Inverting,\n\n       g\u2032\u00b2 = 4\u03c0 \u03b1\u2032 = 4\u03c0 \u00d7 3.899296105751435e-10 = 4.9000e-9,\n       g\u2032  = \u221a(4\u03c0 \u03b1\u2032) = 7.0e-5   (positive root, as stated: \"positive coupling g\u2032\").\n\n   Check: (7.0e-5)\u00b2 / (4\u03c0) = 4.9e-9 / 12.566370614359\u2026 = 3.8992961057514e-10, which reproduces the given \u03b1\u2032 to double precision. The plugin keeps \u03b1\u2032 itself as the parameter, so the described point is preserved exactly, and forms g\u2032\u00b2 = 4\u03c0 \u03b1\u2032 internally; the card, which needs the Lagrangian coupling, carries g\u2032 = 7.0e-5.\n\n3. **Couplings as written.** Universal: g_\u03bd^{ee} = g_\u03bd^{\u03bc\u03bc} = g_\u03bd^{\u03c4\u03c4} = g\u2032, all off-diagonal g_\u03bd^{\u03b1\u03b2} = 0; g_u = g_d = g\u2032. No kinetic mixing, no other couplings. This is the \"universal (toy)\" assignment of conventions section 7 with Q\u2032 = +1 for all neutrinos and both quarks (anomalous as a gauge theory, but that is the model author's concern).\n\n4. **Matching (conventions section 3).**\n\n       C^{uV}_{\u03b1\u03b1}(Q\u00b2) = C^{dV}_{\u03b1\u03b1}(Q\u00b2) = g_\u03bd g_q / (\u221a2 G_F (Q\u00b2 + m\u00b2)) = g\u2032\u00b2 / (\u221a2 G_F (Q\u00b2 + m\u00b2)) = 4\u03c0 \u03b1\u2032 / (\u221a2 G_F (Q\u00b2 + 45\u00b2)),\n\n   for \u03b1 = e, \u03bc, \u03c4; all off-diagonal and all scalar coefficients are zero. Propagator is 1/(Q\u00b2 + m\u00b2) with Q\u00b2 = 2 M T \u2265 0 in MeV\u00b2. The mediator form is used (not a contact term) because a mass is given.\n\n5. **Sign and interference.** g_\u03bd g_q = +g\u2032\u00b2 > 0, so the new-physics nuclear charge shift \u0394Q = (3Z + 3N) C is positive, opposite to the negative SM charge Q_SM \u2248 0.04 Z \u2212 0.5 N. The interference is destructive. For reference, at Q\u00b2 = 1238 MeV\u00b2 (Cs-133, T = 5 keV): C \u2248 4.9e-9 / (1.41421 \u00d7 1.16638e-11 \u00d7 (1238 + 2025)) \u2248 9.10e-2, so \u0394Q \u2248 3 \u00d7 133 \u00d7 0.091 \u2248 36.3 against Q_SM \u2248 \u221236.9: this point sits very close to the charge cancellation for Cs at low recoil, and the coherent rate there is strongly suppressed. This is a consequence of the described point, which is preserved exactly; nothing was tuned.\n\n6. **Sanity.** g\u2032 = 7.0e-5 \u226a \u221a(4\u03c0) \u2248 3.54; mass positive. With `alpha_prime` zeroed the plugin returns all-zero coefficients, i.e. the SM limit.\n\n## Card\n\nExpressible: one vector mediator with mass 45 MeV, diagonal universal neutrino couplings g\u2032 = 7.0e-5 and quark couplings g_u = g_d = 7.0e-5, zero kinetic mixing, no contacts. The card's g\u2032 is the exact square root of 4\u03c0 \u03b1\u2032 at double precision, so the plugin/card cross-check agrees to far better than 1e-6.\n\n## Files\n\n- `plugin.py`: coefficient plugin, parameters `alpha_prime` (coupling, default 3.899296105751435e-10) and `mass_mev` (45.0).\n- `card.json`: equivalent mediator card with g\u2032 = 7.0e-5.\n- `notes.md`: this derivation.\n"
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

### Attempt 1

```json
{
  "final_text": "Validation passed with no findings, including the plugin/card cross-check on 66 Q\u00b2 points.\n\n**What was produced**\n\n- `plugin.py`: vector-only coefficient plugin. It takes `alpha_prime` (default 3.899296105751435e-10, the described point, listed as the coupling) and `mass_mev` (45.0). It forms g\u2032\u00b2 = 4\u03c0 \u03b1\u2032 internally and sets C^{uV} = C^{dV} = g\u2032\u00b2 / (\u221a2 G_F (Q\u00b2 + m\u00b2)) on the three diagonal flavor entries. Off-diagonal and scalar entries are zero.\n- `card.json`: one vector mediator at 45 MeV with g_\u03bd = g_u = g_d = 7.0e-5 and no kinetic mixing. That value is \u221a(4\u03c0 \u03b1\u2032), which inverts the given \u03b1\u2032 to double precision.\n- `notes.md`: the derivation. It records the \u03b1\u2032 \u2192 g\u2032 conversion with its check, the universal charge assignment, the section-3 matching, and the sign of the interference. Since g_\u03bd g_q > 0 the new-physics charge is positive against a negative SM charge, so the interference is destructive. At the chosen point the shift nearly cancels the SM charge for cesium at low recoil, which is a property of the described point and was not tuned.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":68858,\"stop_reason\":\"end_turn\",\"session_id\":\"6999687e-e786-4c4b-8885-9498f840bbaf\",\"total_cost_usd\":0.6767315,\"usage\":{\"input_tokens\":194,\"cache_creation_input_tokens\":20408,\"cache_read_input_tokens\":109926,\"output_tokens\":4783,\"output_tokens_details\":{\"thinking_tokens\":873},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":20408,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":413,\"cache_read_input_tokens\":26119,\"cache_creation_input_tokens\":295,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":295},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":194,\"outputTokens\":4783,\"cacheReadInputTokens\":109926,\"cacheCreationInputTokens\":20408,\"webSearchRequests\":0,\"costUSD\":0.6767315,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":873,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_0162YzNPqPKvsjF5TH7UVhaf\",\"tool_input\":{\"command\":\"ls -R /private/tmp/nuagent-issue17-eval-claude/02_alpha_conversion\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":17,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings, including the plugin/card cross-check on 66 Q\u00b2 points.\\n\\n**What was produced**\\n\\n- `plugin.py`: vector-only coefficient plugin. It takes `alpha_prime` (default 3.899296105751435e-10, the described point, listed as the coupling) and `mass_mev` (45.0). It forms g\u2032\u00b2 = 4\u03c0 \u03b1\u2032 internally and sets C^{uV} = C^{dV} = g\u2032\u00b2 / (\u221a2 G_F (Q\u00b2 + m\u00b2)) on the three diagonal flavor entries. Off-diagonal and scalar entries are zero.\\n- `card.json`: one vector mediator at 45 MeV with g_\u03bd = g_u = g_d = 7.0e-5 and no kinetic mixing. That value is \u221a(4\u03c0 \u03b1\u2032), which inverts the given \u03b1\u2032 to double precision.\\n- `notes.md`: the derivation. It records the \u03b1\u2032 \u2192 g\u2032 conversion with its check, the universal charge assignment, the section-3 matching, and the sign of the interference. Since g_\u03bd g_q > 0 the new-physics charge is positive against a negative SM charge, so the interference is destructive. At the chosen point the shift nearly cancels the SM charge for cesium at low recoil, which is a property of the described point and was not tuned.\",\"ttft_ms\":3084,\"type\":\"result\",\"duration_ms\":70667,\"uuid\":\"80968a66-92a3-4a16-a28a-721ba9cec836\",\"ttft_stream_ms\":2248,\"time_to_request_ms\":22,\"first_content_frame_ms\":2248,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/issue-17-agent-harness/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/nuagent-issue17-eval-claude/02_alpha_conversion\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
