Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

A universal vector Z′ of mass 45 MeV couples with g′ = 7e-5 to all three left-handed neutrino flavors and to up and down quarks. There are no other couplings.

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
"""Flavor-universal vector Z' (45 MeV, g' = 7e-5) coupled to left-handed neutrinos and to u, d quarks."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_prime": 7e-5, "mass_mev": 45.0}
COUPLINGS = ("g_prime",)


def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    g_nu = params["g_prime"]
    g_u = params["g_prime"]
    g_d = params["g_prime"]
    propagator = 1.0 / (q2 + params["mass_mev"] ** 2)
    value_u = g_nu * g_u * propagator / (SQRT2 * GF)
    value_d = g_nu * g_d * propagator / (SQRT2 * GF)
    for pair in ("ee", "mm", "tt"):
        c.set("V", "u", pair, value_u)
        c.set("V", "d", pair, value_d)
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
        "ee": 7e-05,
        "mm": 7e-05,
        "tt": 7e-05,
        "em": 0,
        "et": 0,
        "mt": 0
      },
      "quark_couplings": {
        "u": 7e-05,
        "d": 7e-05
      },
      "kinetic_mixing": 0.0
    }
  ],
  "contacts": []
}

```

## notes.md

```text
# Universal vector Z′, m = 45 MeV, g′ = 7e-5

## Description

A universal vector Z′ of mass 45 MeV couples with g′ = 7e-5 to all three
left-handed neutrino flavors and to up and down quarks. No other couplings.

## Derivation

1. **Lorentz structure.** A vector boson exchanged between the left-handed
   neutrino current and the quark vector current: structure `V`. No scalar,
   axial or tensor piece is present, so the interaction lies inside the V/S
   basis.

2. **Couplings as written.** The Lagrangian is

       L ⊃ Z′_μ ( g′ Σ_α ν̄_α γ^μ P_L ν_α + g′ ū γ^μ u + g′ d̄ γ^μ d ),

   so g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = g′, all off-diagonal g_ν^{αβ} = 0,
   g_u = g_d = g′. The neutrino coupling is already to the left-handed current,
   so no factor of 2 conversion is needed (conventions section 3). The quark
   coupling is purely vector (g_L = g_R = g′), so g_q = (g_L + g_R)/2 = g′ and
   no axial part is dropped. No kinetic mixing. This is the "universal (toy)"
   charge assignment of conventions section 7 with Q′ = +1 for every field.

3. **Matching** (conventions section 3, tree-level Z′ exchange):

       C^{uV}_{αα}(Q²) = C^{dV}_{αα}(Q²) = g′² / ( √2 G_F (Q² + m²) ),
       m = 45 MeV, g′ = 7e-5.

   All off-diagonal and all scalar coefficients are zero. The plugin computes
   this directly from the exported `GF` and `SQRT2`; the card holds the same
   mediator entry so the framework can cross-check them.

4. **Numerical values** at the Cs-133 reference points (Q² = 2 M T for
   T = 5, 15, 25 keV), using g′² = 4.9e-9, m² = 2025 MeV²,
   √2 G_F = 1.6495e-11 MeV⁻²:

   | Q² (MeV²) | C^{uV} = C^{dV} |
   |---|---|
   | 1238 | 9.104e-02 |
   | 3714 | 5.176e-02 |
   | 6190 | 3.616e-02 |

5. **Sign and interference.** g_ν g_q = +g′² > 0, so C > 0 for both quarks and
   the new-physics nuclear charge shift ΔQ = (3Z + 3N) C = 3A·C is positive,
   while the SM charge Q_SM ≈ 0.04 Z − 0.5 N is negative. The interference is
   destructive. For Cs-133 at Q² = 1238 MeV², ΔQ ≈ 3·133·0.09104 ≈ 36.3,
   compared with the tree-level |Q_SM| = 36.9 (38.3 with the framework's
   radiative corrections). The described point therefore sits very close to
   the charge-cancellation coupling at low recoil (tree-level cancellation at
   this Q² occurs near g′ ≈ 7.06e-5 for m = 45 MeV), so the coherent rate is
   strongly suppressed at low recoil energies and recovers at higher Q² where
   the propagator suppresses C. This is a property of the described point, which
   is preserved exactly; no parameter has been adjusted.

6. **Perturbativity.** |g′| = 7e-5 ≪ √(4π). Mass positive. With `g_prime`
   zeroed the plugin returns all-zero coefficients, which is the SM limit.

7. **Card-expressible.** Yes: one vector mediator with flavor-diagonal
   universal neutrino couplings and equal u, d couplings. `card.json` is
   provided.

## Convention conversions

None. The description is already in the framework's normalization
(left-handed neutrino coupling, vector quark coupling, mass in MeV).

## Files

- `plugin.py`: coefficient plugin, `PARAMETERS = {"g_prime": 7e-5, "mass_mev": 45.0}`, `COUPLINGS = ("g_prime",)`.
- `card.json`: one `V` mediator, m = 45 MeV, g_ν = g_u = g_d = 7e-5.
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
  "notes": "# Universal vector Z\u2032, m = 45 MeV, g\u2032 = 7e-5\n\n## Description\n\nA universal vector Z\u2032 of mass 45 MeV couples with g\u2032 = 7e-5 to all three\nleft-handed neutrino flavors and to up and down quarks. No other couplings.\n\n## Derivation\n\n1. **Lorentz structure.** A vector boson exchanged between the left-handed\n   neutrino current and the quark vector current: structure `V`. No scalar,\n   axial or tensor piece is present, so the interaction lies inside the V/S\n   basis.\n\n2. **Couplings as written.** The Lagrangian is\n\n       L \u2283 Z\u2032_\u03bc ( g\u2032 \u03a3_\u03b1 \u03bd\u0304_\u03b1 \u03b3^\u03bc P_L \u03bd_\u03b1 + g\u2032 \u016b \u03b3^\u03bc u + g\u2032 d\u0304 \u03b3^\u03bc d ),\n\n   so g_\u03bd^{ee} = g_\u03bd^{mm} = g_\u03bd^{tt} = g\u2032, all off-diagonal g_\u03bd^{\u03b1\u03b2} = 0,\n   g_u = g_d = g\u2032. The neutrino coupling is already to the left-handed current,\n   so no factor of 2 conversion is needed (conventions section 3). The quark\n   coupling is purely vector (g_L = g_R = g\u2032), so g_q = (g_L + g_R)/2 = g\u2032 and\n   no axial part is dropped. No kinetic mixing. This is the \"universal (toy)\"\n   charge assignment of conventions section 7 with Q\u2032 = +1 for every field.\n\n3. **Matching** (conventions section 3, tree-level Z\u2032 exchange):\n\n       C^{uV}_{\u03b1\u03b1}(Q\u00b2) = C^{dV}_{\u03b1\u03b1}(Q\u00b2) = g\u2032\u00b2 / ( \u221a2 G_F (Q\u00b2 + m\u00b2) ),\n       m = 45 MeV, g\u2032 = 7e-5.\n\n   All off-diagonal and all scalar coefficients are zero. The plugin computes\n   this directly from the exported `GF` and `SQRT2`; the card holds the same\n   mediator entry so the framework can cross-check them.\n\n4. **Numerical values** at the Cs-133 reference points (Q\u00b2 = 2 M T for\n   T = 5, 15, 25 keV), using g\u2032\u00b2 = 4.9e-9, m\u00b2 = 2025 MeV\u00b2,\n   \u221a2 G_F = 1.6495e-11 MeV\u207b\u00b2:\n\n   | Q\u00b2 (MeV\u00b2) | C^{uV} = C^{dV} |\n   |---|---|\n   | 1238 | 9.104e-02 |\n   | 3714 | 5.176e-02 |\n   | 6190 | 3.616e-02 |\n\n5. **Sign and interference.** g_\u03bd g_q = +g\u2032\u00b2 > 0, so C > 0 for both quarks and\n   the new-physics nuclear charge shift \u0394Q = (3Z + 3N) C = 3A\u00b7C is positive,\n   while the SM charge Q_SM \u2248 0.04 Z \u2212 0.5 N is negative. The interference is\n   destructive. For Cs-133 at Q\u00b2 = 1238 MeV\u00b2, \u0394Q \u2248 3\u00b7133\u00b70.09104 \u2248 36.3,\n   compared with the tree-level |Q_SM| = 36.9 (38.3 with the framework's\n   radiative corrections). The described point therefore sits very close to\n   the charge-cancellation coupling at low recoil (tree-level cancellation at\n   this Q\u00b2 occurs near g\u2032 \u2248 7.06e-5 for m = 45 MeV), so the coherent rate is\n   strongly suppressed at low recoil energies and recovers at higher Q\u00b2 where\n   the propagator suppresses C. This is a property of the described point, which\n   is preserved exactly; no parameter has been adjusted.\n\n6. **Perturbativity.** |g\u2032| = 7e-5 \u226a \u221a(4\u03c0). Mass positive. With `g_prime`\n   zeroed the plugin returns all-zero coefficients, which is the SM limit.\n\n7. **Card-expressible.** Yes: one vector mediator with flavor-diagonal\n   universal neutrino couplings and equal u, d couplings. `card.json` is\n   provided.\n\n## Convention conversions\n\nNone. The description is already in the framework's normalization\n(left-handed neutrino coupling, vector quark coupling, mass in MeV).\n\n## Files\n\n- `plugin.py`: coefficient plugin, `PARAMETERS = {\"g_prime\": 7e-5, \"mass_mev\": 45.0}`, `COUPLINGS = (\"g_prime\",)`.\n- `card.json`: one `V` mediator, m = 45 MeV, g_\u03bd = g_u = g_d = 7e-5.\n- `notes.md`: this derivation.\n"
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
      0.13184754856554984,
      0.6905458030784078,
      1.6873322305501945,
      2.5566407278909957,
      2.7460294999704398,
      2.510977126785431,
      2.2899859868416885,
      1.9756864395317417,
      1.635344615134781,
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
  "final_text": "Validation passed with no findings, including the plugin/card cross-check on 66 Q\u00b2 points.\n\n**Files produced** in the task directory:\n\n- `plugin.py`: vector coefficients C^{uV} = C^{dV} = g\u2032\u00b2/(\u221a2 G_F (Q\u00b2 + m\u00b2)) for ee, mm, tt, with defaults g\u2032 = 7e-5 and m = 45 MeV, and `COUPLINGS = (\"g_prime\",)`.\n- `card.json`: one `V` mediator at 45 MeV with g_\u03bd = g_u = g_d = 7e-5, no kinetic mixing, no contacts.\n- `notes.md`: derivation, reference values, and sign discussion.\n\n**Physics summary.** The description maps directly onto the framework's vector matching with no convention conversion. The coupling product is positive, so the interference with the Standard Model is destructive. For Cs-133 at low recoil the new-physics charge shift is nearly equal and opposite to the SM charge, so this point lies close to the charge-cancellation region. That is a feature of the described point, which was preserved exactly.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":50422,\"stop_reason\":\"end_turn\",\"session_id\":\"053d52f1-c768-4c4e-b35a-b48eff2548b5\",\"total_cost_usd\":0.7641257499999999,\"usage\":{\"input_tokens\":194,\"cache_creation_input_tokens\":25787,\"cache_read_input_tokens\":102583,\"output_tokens\":4416,\"output_tokens_details\":{\"thinking_tokens\":682},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":25787,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":361,\"cache_read_input_tokens\":25492,\"cache_creation_input_tokens\":295,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":295},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":194,\"outputTokens\":4416,\"cacheReadInputTokens\":102583,\"cacheCreationInputTokens\":25787,\"webSearchRequests\":0,\"costUSD\":0.7641257499999999,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":682,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01Ev1vASgRxXaJL9H3eGwrPh\",\"tool_input\":{\"command\":\"ls -R /private/tmp/nuagent-issue17-eval-claude/01_universal_vector\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":14,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings, including the plugin/card cross-check on 66 Q\u00b2 points.\\n\\n**Files produced** in the task directory:\\n\\n- `plugin.py`: vector coefficients C^{uV} = C^{dV} = g\u2032\u00b2/(\u221a2 G_F (Q\u00b2 + m\u00b2)) for ee, mm, tt, with defaults g\u2032 = 7e-5 and m = 45 MeV, and `COUPLINGS = (\\\"g_prime\\\",)`.\\n- `card.json`: one `V` mediator at 45 MeV with g_\u03bd = g_u = g_d = 7e-5, no kinetic mixing, no contacts.\\n- `notes.md`: derivation, reference values, and sign discussion.\\n\\n**Physics summary.** The description maps directly onto the framework's vector matching with no convention conversion. The coupling product is positive, so the interference with the Standard Model is destructive. For Cs-133 at low recoil the new-physics charge shift is nearly equal and opposite to the SM charge, so this point lies close to the charge-cancellation region. That is a feature of the described point, which was preserved exactly.\",\"ttft_ms\":819,\"type\":\"result\",\"duration_ms\":52071,\"uuid\":\"04dd124d-84f1-457b-a9d1-efbc777edb85\",\"ttft_stream_ms\":788,\"time_to_request_ms\":22,\"first_content_frame_ms\":788,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/issue-17-agent-harness/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/nuagent-issue17-eval-claude/01_universal_vector\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
