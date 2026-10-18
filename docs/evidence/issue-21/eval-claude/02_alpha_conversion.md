Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

A universal vector Z′ of mass 45 MeV has alpha′ = g′²/(4 pi) = 3.899296105751435e-10. Its positive coupling g′ is the same for all left-handed neutrino flavors and both u and d quarks, with no other couplings.

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
"""Universal vector Z' (45 MeV) with g' = sqrt(4 pi alpha'); derivation in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

# g' = sqrt(4 pi alpha') with alpha' = 3.899296105751435e-10 gives g' = 7e-5.
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
  "name": "zprime_universal_45mev_alpha_conversion",
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
# Notes: universal vector Z′, m = 45 MeV, α′ = g′²/4π

## Description as given

A universal vector Z′ of mass 45 MeV with α′ = g′²/(4π) = 3.899296105751435e-10. The positive coupling g′ is the same for all left-handed neutrino flavors and for the u and d quarks; no other couplings.

## Derivation

1. **Lorentz structure.** Vector exchange, structure `V`. No scalar entries.

2. **Coupling conversion (conventions section 10).** The description gives α′, not g′. Inverting α′ = g′²/(4π):

       g′ = sqrt(4π α′) = sqrt(12.566370614359172 × 3.899296105751435e-10)
          = sqrt(4.9000000e-9) = 7.0e-5.

   The product 4π α′ equals 4.9e-9 to all given digits, so g′ = 7e-5 exactly at double precision. The description says g′ is positive, so the positive root is taken. Perturbativity: 7e-5 ≪ √(4π).

3. **Couplings in the framework's Lagrangian (conventions section 3).**
   L ⊃ Z′_μ ( g_ν^{αβ} ν̄_α γ^μ P_L ν_β + g_u ū γ^μ u + g_d d̄ γ^μ d ) with
   g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = g′, off-diagonal g_ν = 0, g_u = g_d = g′.
   The neutrino coupling is already to the left-handed current, so no chirality factor is needed. The quark couplings are pure vector (no chiral split given), so no axial part is dropped.

4. **Matching.** Tree-level Z′ exchange gives

       C^{uV}_{αα}(Q²) = C^{dV}_{αα}(Q²) = g′² / ( √2 G_F (Q² + m²) ),   m = 45 MeV,

   for α = e, μ, τ, with the propagator 1/(Q² + m²) and Q² = 2 M T in MeV². The plugin computes this from the exported `GF` and `SQRT2`, independently of the card. Note that g′² = 4π α′ = 4.9e-9, so the numerator is just 4π α′.

5. **Sign and interference.** g_ν g_q = +g′² > 0, so the new-physics nuclear charge shift ΔQ = (3Z + 3N) C is positive, while the SM charge is negative (Q_SM ≈ 0.04 Z − 0.5 N). The interference is destructive. This is the "universal (toy)" assignment of conventions section 7, for which a cancellation of the charge is possible at some coupling; the described point is kept exactly as given and not adjusted.

6. **Card.** The interaction is one vector mediator with constant couplings, so the schema expresses it fully: structure V, mass 45, nu_couplings diagonal 7e-5, quark_couplings u = d = 7e-5, kinetic_mixing 0, no contacts. The card carries the converted g′ = 7e-5 rather than α′ because the schema has no field for α′.

## Parameters

| parameter | value | meaning |
|---|---|---|
| `g_prime` | 7e-5 | g′ = sqrt(4π α′), common coupling to ν_L (all flavors), u and d |
| `mass_mev` | 45.0 | Z′ mass in MeV |

`COUPLINGS = ("g_prime",)`; setting it to zero recovers the Standard Model.

## Convention conversion recorded

The only conversion is α′ → g′ = sqrt(4π α′). No chiral, nucleon-level or kinetic-mixing conversions are needed. No parameter was tuned; the described point (α′ = 3.899296105751435e-10, m = 45 MeV) is preserved exactly through g′ = 7e-5.

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
  "notes": "# Notes: universal vector Z\u2032, m = 45 MeV, \u03b1\u2032 = g\u2032\u00b2/4\u03c0\n\n## Description as given\n\nA universal vector Z\u2032 of mass 45 MeV with \u03b1\u2032 = g\u2032\u00b2/(4\u03c0) = 3.899296105751435e-10. The positive coupling g\u2032 is the same for all left-handed neutrino flavors and for the u and d quarks; no other couplings.\n\n## Derivation\n\n1. **Lorentz structure.** Vector exchange, structure `V`. No scalar entries.\n\n2. **Coupling conversion (conventions section 10).** The description gives \u03b1\u2032, not g\u2032. Inverting \u03b1\u2032 = g\u2032\u00b2/(4\u03c0):\n\n       g\u2032 = sqrt(4\u03c0 \u03b1\u2032) = sqrt(12.566370614359172 \u00d7 3.899296105751435e-10)\n          = sqrt(4.9000000e-9) = 7.0e-5.\n\n   The product 4\u03c0 \u03b1\u2032 equals 4.9e-9 to all given digits, so g\u2032 = 7e-5 exactly at double precision. The description says g\u2032 is positive, so the positive root is taken. Perturbativity: 7e-5 \u226a \u221a(4\u03c0).\n\n3. **Couplings in the framework's Lagrangian (conventions section 3).**\n   L \u2283 Z\u2032_\u03bc ( g_\u03bd^{\u03b1\u03b2} \u03bd\u0304_\u03b1 \u03b3^\u03bc P_L \u03bd_\u03b2 + g_u \u016b \u03b3^\u03bc u + g_d d\u0304 \u03b3^\u03bc d ) with\n   g_\u03bd^{ee} = g_\u03bd^{mm} = g_\u03bd^{tt} = g\u2032, off-diagonal g_\u03bd = 0, g_u = g_d = g\u2032.\n   The neutrino coupling is already to the left-handed current, so no chirality factor is needed. The quark couplings are pure vector (no chiral split given), so no axial part is dropped.\n\n4. **Matching.** Tree-level Z\u2032 exchange gives\n\n       C^{uV}_{\u03b1\u03b1}(Q\u00b2) = C^{dV}_{\u03b1\u03b1}(Q\u00b2) = g\u2032\u00b2 / ( \u221a2 G_F (Q\u00b2 + m\u00b2) ),   m = 45 MeV,\n\n   for \u03b1 = e, \u03bc, \u03c4, with the propagator 1/(Q\u00b2 + m\u00b2) and Q\u00b2 = 2 M T in MeV\u00b2. The plugin computes this from the exported `GF` and `SQRT2`, independently of the card. Note that g\u2032\u00b2 = 4\u03c0 \u03b1\u2032 = 4.9e-9, so the numerator is just 4\u03c0 \u03b1\u2032.\n\n5. **Sign and interference.** g_\u03bd g_q = +g\u2032\u00b2 > 0, so the new-physics nuclear charge shift \u0394Q = (3Z + 3N) C is positive, while the SM charge is negative (Q_SM \u2248 0.04 Z \u2212 0.5 N). The interference is destructive. This is the \"universal (toy)\" assignment of conventions section 7, for which a cancellation of the charge is possible at some coupling; the described point is kept exactly as given and not adjusted.\n\n6. **Card.** The interaction is one vector mediator with constant couplings, so the schema expresses it fully: structure V, mass 45, nu_couplings diagonal 7e-5, quark_couplings u = d = 7e-5, kinetic_mixing 0, no contacts. The card carries the converted g\u2032 = 7e-5 rather than \u03b1\u2032 because the schema has no field for \u03b1\u2032.\n\n## Parameters\n\n| parameter | value | meaning |\n|---|---|---|\n| `g_prime` | 7e-5 | g\u2032 = sqrt(4\u03c0 \u03b1\u2032), common coupling to \u03bd_L (all flavors), u and d |\n| `mass_mev` | 45.0 | Z\u2032 mass in MeV |\n\n`COUPLINGS = (\"g_prime\",)`; setting it to zero recovers the Standard Model.\n\n## Convention conversion recorded\n\nThe only conversion is \u03b1\u2032 \u2192 g\u2032 = sqrt(4\u03c0 \u03b1\u2032). No chiral, nucleon-level or kinetic-mixing conversions are needed. No parameter was tuned; the described point (\u03b1\u2032 = 3.899296105751435e-10, m = 45 MeV) is preserved exactly through g\u2032 = 7e-5.\n"
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
  "final_text": "Validation passed with no findings, and the plugin and card agree on all 66 Q\u00b2 cross-check points.\n\n**Files produced** in the task directory:\n\n- `plugin.py`: universal vector Z\u2032 with parameters g\u2032 = 7e-5 and mass 45 MeV. It sets the diagonal u and d vector coefficients for all three flavors to g\u2032\u00b2 / (\u221a2 G_F (Q\u00b2 + m\u00b2)) using the exported constants.\n- `card.json`: one V mediator, mass 45 MeV, neutrino couplings ee = mm = tt = 7e-5, quark couplings u = d = 7e-5, no kinetic mixing, no contacts.\n- `notes.md`: the derivation, including the conversion g\u2032 = sqrt(4\u03c0 \u03b1\u2032) from the given \u03b1\u2032, which gives exactly 7e-5, the matching formula from conventions section 3, and the note that g_\u03bd g_q > 0 interferes destructively with the Standard Model.\n\nNo questions.json was needed. The described point was preserved exactly and nothing was tuned.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":87340,\"stop_reason\":\"end_turn\",\"session_id\":\"8aad91ca-7318-4971-b85b-01438fa9c807\",\"total_cost_usd\":0.8509325,\"usage\":{\"input_tokens\":226,\"cache_creation_input_tokens\":24862,\"cache_read_input_tokens\":147330,\"output_tokens\":6292,\"output_tokens_details\":{\"thinking_tokens\":1354},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":24862,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":337,\"cache_read_input_tokens\":30574,\"cache_creation_input_tokens\":294,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":294},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":226,\"outputTokens\":6292,\"cacheReadInputTokens\":147330,\"cacheCreationInputTokens\":24862,\"webSearchRequests\":0,\"costUSD\":0.8509325,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":1354,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01GZV4iu4gajQS91nQP1Mmxr\",\"tool_input\":{\"command\":\"ls -R /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/02_alpha_conversion\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":20,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings, and the plugin and card agree on all 66 Q\u00b2 cross-check points.\\n\\n**Files produced** in the task directory:\\n\\n- `plugin.py`: universal vector Z\u2032 with parameters g\u2032 = 7e-5 and mass 45 MeV. It sets the diagonal u and d vector coefficients for all three flavors to g\u2032\u00b2 / (\u221a2 G_F (Q\u00b2 + m\u00b2)) using the exported constants.\\n- `card.json`: one V mediator, mass 45 MeV, neutrino couplings ee = mm = tt = 7e-5, quark couplings u = d = 7e-5, no kinetic mixing, no contacts.\\n- `notes.md`: the derivation, including the conversion g\u2032 = sqrt(4\u03c0 \u03b1\u2032) from the given \u03b1\u2032, which gives exactly 7e-5, the matching formula from conventions section 3, and the note that g_\u03bd g_q > 0 interferes destructively with the Standard Model.\\n\\nNo questions.json was needed. The described point was preserved exactly and nothing was tuned.\",\"ttft_ms\":2682,\"type\":\"result\",\"duration_ms\":91281,\"uuid\":\"aafe94ba-1ec7-4860-bca3-62c8750ced8d\",\"ttft_stream_ms\":2557,\"time_to_request_ms\":28,\"first_content_frame_ms\":2558,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/m7-conversation/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/02_alpha_conversion\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
