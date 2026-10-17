Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

A protophobic vector mediator has mass 20 MeV, universal left-handed neutrino coupling g_nu = 1e-4, and vector nucleon couplings g_p = 0 and g_n = 3e-4. There are no other interactions.

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
"""Protophobic vector mediator; derivation in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_nu": 1e-4, "g_p": 0.0, "g_n": 3e-4, "mass_mev": 20.0}
COUPLINGS = ("g_nu", "g_p", "g_n")


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    g_nu = params["g_nu"]
    g_p = params["g_p"]
    g_n = params["g_n"]
    g_u = (2.0 * g_p - g_n) / 3.0
    g_d = (2.0 * g_n - g_p) / 3.0
    propagator = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark, g_q in (("u", g_u), ("d", g_d)):
        value = g_nu * g_q * propagator
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)
    return c

```

## card.json

```json
{
  "name": "protophobic_vector_20mev",
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
# Protophobic vector mediator, m = 20 MeV

## Description

A protophobic vector mediator of mass 20 MeV with a universal left-handed neutrino
coupling g_ν = 1e-4 and vector nucleon couplings g_p = 0, g_n = 3e-4. No other
interactions.

## Derivation

1. **Lorentz structure.** Vector exchange, structure `V`. The neutrino coupling is
   stated as left-handed, which is exactly the current in the framework's basis, so
   g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = 1e-4 and all off-diagonal entries are zero. No
   chirality or normalization conversion is needed. No scalar entries.

2. **Nucleon to quark couplings** (conventions section 3). The framework's nuclear
   vector charge is Q = Z g_p + N g_n with g_p = 2 g_u + g_d and g_n = g_u + 2 g_d.
   Inverting:

       g_u = (2 g_p − g_n) / 3 = (0 − 3e-4) / 3 = −1e-4
       g_d = (2 g_n − g_p) / 3 = (6e-4 − 0) / 3 = +2e-4

   Check: 2 g_u + g_d = −2e-4 + 2e-4 = 0 = g_p and g_u + 2 g_d = −1e-4 + 4e-4 = 3e-4 = g_n.
   The quark couplings are vector couplings; the description gives vector nucleon
   couplings, so there is no axial part to drop.

3. **Matching** (conventions section 3):

       C^{uV}_{αα}(Q²) = g_ν g_u / (√2 G_F (Q² + m²)) = −1e-8 / (√2 G_F (Q² + 400 MeV²))
       C^{dV}_{αα}(Q²) = g_ν g_d / (√2 G_F (Q² + m²)) = +2e-8 / (√2 G_F (Q² + 400 MeV²))

   for α = e, μ, τ. The plugin computes g_u and g_d from the parameters g_p and g_n
   with the same formulas and uses the exported GF and SQRT2.

4. **Sign and interference.** The nuclear charge shift is
   ΔQ = Z (2 C^u + C^d) + N (C^u + 2 C^d) = N g_ν g_n / (√2 G_F (Q² + m²)), with no
   proton contribution, as required for a protophobic mediator. Since g_ν g_n > 0, ΔQ is
   positive while Q_SM ≈ 0.04 Z − 0.5 N is negative: destructive interference with the
   Standard Model. At the first reference point (Q² = 1238 MeV², Cs-133 with N = 78),
   C^{uV} ≈ −0.370, C^{dV} ≈ +0.740, and ΔQ ≈ +86.6, which exceeds |Q_SM| ≈ 36.9, so at
   this point the total charge has flipped sign and the coherent rate is enhanced
   rather than cancelled.

5. **Perturbativity.** All couplings are of order 1e-4, far below √(4π).

6. **Card.** The interaction is a single vector mediator with constant couplings, so
   the schema expresses it exactly. `card.json` carries the converted quark couplings
   g_u = −1e-4, g_d = +2e-4, the universal neutrino coupling, mass 20 MeV, and zero
   kinetic mixing.

## Parameters

| parameter | value | role |
|---|---|---|
| g_nu | 1e-4 | universal left-handed neutrino coupling (coupling) |
| g_p | 0 | vector proton coupling (coupling) |
| g_n | 3e-4 | vector neutron coupling (coupling) |
| mass_mev | 20 | mediator mass (not a coupling) |

The described point is preserved exactly; nothing was tuned.

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
  "notes": "# Protophobic vector mediator, m = 20 MeV\n\n## Description\n\nA protophobic vector mediator of mass 20 MeV with a universal left-handed neutrino\ncoupling g_\u03bd = 1e-4 and vector nucleon couplings g_p = 0, g_n = 3e-4. No other\ninteractions.\n\n## Derivation\n\n1. **Lorentz structure.** Vector exchange, structure `V`. The neutrino coupling is\n   stated as left-handed, which is exactly the current in the framework's basis, so\n   g_\u03bd^{ee} = g_\u03bd^{mm} = g_\u03bd^{tt} = 1e-4 and all off-diagonal entries are zero. No\n   chirality or normalization conversion is needed. No scalar entries.\n\n2. **Nucleon to quark couplings** (conventions section 3). The framework's nuclear\n   vector charge is Q = Z g_p + N g_n with g_p = 2 g_u + g_d and g_n = g_u + 2 g_d.\n   Inverting:\n\n       g_u = (2 g_p \u2212 g_n) / 3 = (0 \u2212 3e-4) / 3 = \u22121e-4\n       g_d = (2 g_n \u2212 g_p) / 3 = (6e-4 \u2212 0) / 3 = +2e-4\n\n   Check: 2 g_u + g_d = \u22122e-4 + 2e-4 = 0 = g_p and g_u + 2 g_d = \u22121e-4 + 4e-4 = 3e-4 = g_n.\n   The quark couplings are vector couplings; the description gives vector nucleon\n   couplings, so there is no axial part to drop.\n\n3. **Matching** (conventions section 3):\n\n       C^{uV}_{\u03b1\u03b1}(Q\u00b2) = g_\u03bd g_u / (\u221a2 G_F (Q\u00b2 + m\u00b2)) = \u22121e-8 / (\u221a2 G_F (Q\u00b2 + 400 MeV\u00b2))\n       C^{dV}_{\u03b1\u03b1}(Q\u00b2) = g_\u03bd g_d / (\u221a2 G_F (Q\u00b2 + m\u00b2)) = +2e-8 / (\u221a2 G_F (Q\u00b2 + 400 MeV\u00b2))\n\n   for \u03b1 = e, \u03bc, \u03c4. The plugin computes g_u and g_d from the parameters g_p and g_n\n   with the same formulas and uses the exported GF and SQRT2.\n\n4. **Sign and interference.** The nuclear charge shift is\n   \u0394Q = Z (2 C^u + C^d) + N (C^u + 2 C^d) = N g_\u03bd g_n / (\u221a2 G_F (Q\u00b2 + m\u00b2)), with no\n   proton contribution, as required for a protophobic mediator. Since g_\u03bd g_n > 0, \u0394Q is\n   positive while Q_SM \u2248 0.04 Z \u2212 0.5 N is negative: destructive interference with the\n   Standard Model. At the first reference point (Q\u00b2 = 1238 MeV\u00b2, Cs-133 with N = 78),\n   C^{uV} \u2248 \u22120.370, C^{dV} \u2248 +0.740, and \u0394Q \u2248 +86.6, which exceeds |Q_SM| \u2248 36.9, so at\n   this point the total charge has flipped sign and the coherent rate is enhanced\n   rather than cancelled.\n\n5. **Perturbativity.** All couplings are of order 1e-4, far below \u221a(4\u03c0).\n\n6. **Card.** The interaction is a single vector mediator with constant couplings, so\n   the schema expresses it exactly. `card.json` carries the converted quark couplings\n   g_u = \u22121e-4, g_d = +2e-4, the universal neutrino coupling, mass 20 MeV, and zero\n   kinetic mixing.\n\n## Parameters\n\n| parameter | value | role |\n|---|---|---|\n| g_nu | 1e-4 | universal left-handed neutrino coupling (coupling) |\n| g_p | 0 | vector proton coupling (coupling) |\n| g_n | 3e-4 | vector neutron coupling (coupling) |\n| mass_mev | 20 | mediator mass (not a coupling) |\n\nThe described point is preserved exactly; nothing was tuned.\n"
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

### Attempt 1

```json
{
  "final_text": "Validation passed with no findings, including the plugin/card cross-check on 66 Q\u00b2 points.\n\nFiles produced in the task directory:\n\n- **plugin.py**: vector-only coefficients for a 20 MeV mediator. Parameters are g_nu, g_p, g_n and mass_mev, with the first three listed as couplings. It converts nucleon to quark couplings inside the function and uses the exported GF and SQRT2.\n- **card.json**: one vector mediator entry with universal neutrino coupling 1e-4, quark couplings g_u = -1e-4 and g_d = +2e-4, mass 20 MeV, zero kinetic mixing.\n- **notes.md**: the derivation. The key step is the exact inversion g_u = (2g_p - g_n)/3 and g_d = (2g_n - g_p)/3 from conventions section 3, checked against g_p = 0 and g_n = 3e-4. It records that the nuclear charge shift is purely neutron-sourced and positive, so the interference with the Standard Model is destructive, with the total charge already sign-flipped at the first reference Q\u00b2.\n\nThe described parameter point is preserved exactly and nothing was tuned.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":48068,\"stop_reason\":\"end_turn\",\"session_id\":\"af412216-2653-41b1-af96-54c5587bc60f\",\"total_cost_usd\":0.68235875,\"usage\":{\"input_tokens\":194,\"cache_creation_input_tokens\":21997,\"cache_read_input_tokens\":114315,\"output_tokens\":4238,\"output_tokens_details\":{\"thinking_tokens\":547},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":21997,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":390,\"cache_read_input_tokens\":27709,\"cache_creation_input_tokens\":294,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":294},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":194,\"outputTokens\":4238,\"cacheReadInputTokens\":114315,\"cacheCreationInputTokens\":21997,\"webSearchRequests\":0,\"costUSD\":0.68235875,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":547,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01RcuHohtC4yLqRm6DUJxhLd\",\"tool_input\":{\"command\":\"ls -R /private/tmp/nuagent-issue17-eval-claude/11_protophobic\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":15,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings, including the plugin/card cross-check on 66 Q\u00b2 points.\\n\\nFiles produced in the task directory:\\n\\n- **plugin.py**: vector-only coefficients for a 20 MeV mediator. Parameters are g_nu, g_p, g_n and mass_mev, with the first three listed as couplings. It converts nucleon to quark couplings inside the function and uses the exported GF and SQRT2.\\n- **card.json**: one vector mediator entry with universal neutrino coupling 1e-4, quark couplings g_u = -1e-4 and g_d = +2e-4, mass 20 MeV, zero kinetic mixing.\\n- **notes.md**: the derivation. The key step is the exact inversion g_u = (2g_p - g_n)/3 and g_d = (2g_n - g_p)/3 from conventions section 3, checked against g_p = 0 and g_n = 3e-4. It records that the nuclear charge shift is purely neutron-sourced and positive, so the interference with the Standard Model is destructive, with the total charge already sign-flipped at the first reference Q\u00b2.\\n\\nThe described parameter point is preserved exactly and nothing was tuned.\",\"ttft_ms\":734,\"type\":\"result\",\"duration_ms\":49806,\"uuid\":\"a25ef6d7-290c-431b-b592-7e03370a02b8\",\"ttft_stream_ms\":670,\"time_to_request_ms\":24,\"first_content_frame_ms\":671,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/issue-17-agent-harness/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/nuagent-issue17-eval-claude/11_protophobic\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
