Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

A universal vector Z′ has mass 30 MeV. Its neutrino interaction is g Z′_mu nu-bar gamma^mu (1 − gamma5) nu for every flavor, with g = 5e-5 and no factor 1/2. Its vector coupling to each of u and d quarks is 1e-4. All other couplings vanish.

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
"""Universal vector Z' with neutrino coupling g nu-bar gamma^mu (1 - gamma5) nu (no 1/2).

Derivation in notes.md.
"""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

# g: coefficient of nu-bar gamma^mu (1 - gamma5) nu as written in the description.
# g_q: vector coupling to each of u and d.
PARAMETERS = {"g": 5e-5, "g_q": 1e-4, "mass_mev": 30.0}
COUPLINGS = ("g", "g_q")


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    # (1 - gamma5) = 2 P_L, so the left-handed neutrino coupling is g_nu = 2 g.
    g_nu = 2.0 * params["g"]
    value = g_nu * params["g_q"] / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)
    return c

```

## card.json

```json
{
  "name": "zprime_universal_left_handed_no_half",
  "mediators": [
    {
      "structure": "V",
      "mass_mev": 30.0,
      "nu_couplings": {
        "ee": 0.0001,
        "mm": 0.0001,
        "tt": 0.0001,
        "em": 0,
        "et": 0,
        "mt": 0
      },
      "quark_couplings": {
        "u": 0.0001,
        "d": 0.0001
      },
      "kinetic_mixing": 0.0
    }
  ],
  "contacts": []
}

```

## notes.md

```text
# Notes: universal vector Z′ with g ν̄ γ^μ (1 − γ5) ν neutrino coupling

## Description

Vector Z′, mass 30 MeV. Neutrino interaction g Z′_μ ν̄ γ^μ (1 − γ5) ν for every
flavor with g = 5e-5 and no factor 1/2. Vector coupling 1e-4 to each of u and d.
All other couplings vanish.

## Derivation

1. **Lorentz structure.** Vector boson exchange between a neutrino current and a
   quark vector current: structure `V`. No scalar part.

2. **Neutrino coupling normalization.** The framework's matching formula
   (conventions section 3) is written for Z′_μ g_ν ν̄ γ^μ P_L ν with
   P_L = (1 − γ5)/2. The description writes g ν̄ γ^μ (1 − γ5) ν = 2g ν̄ γ^μ P_L ν
   without the 1/2, so

       g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = 2g = 2 × 5e-5 = 1e-4.

   This is exactly the case called out in conventions section 3: "If it writes
   g ν̄ γ^μ (1 − γ5) ν without the 1/2, then g_ν = 2g." Off-diagonal entries
   are zero (universal, flavor-diagonal).

3. **Quark couplings.** Given directly as vector couplings: g_u = g_d = 1e-4.
   No chiral decomposition needed, no axial part to drop, no kinetic mixing.

4. **Matching.** From conventions section 3,

       C^{uV}_{αα}(Q²) = C^{dV}_{αα}(Q²) = g_ν g_q / (√2 G_F (Q² + m²))
                        = (1e-4)(1e-4) / (√2 G_F (Q² + (30 MeV)²)).

   Numerically this coincides with worked example 01 (g′ = 1e-4 universal), so
   at the Cs-133 reference points Q² = 1238, 3714, 6190 MeV² the coefficients
   are approximately 0.28355, 0.13139, 0.085506 for both quarks and all three
   diagonal flavors.

5. **Sign and interference.** g_ν g_q = +1e-8 > 0, so the new-physics nuclear
   charge shift ΔQ = (3Z + 3N) C is positive while Q_SM is negative:
   destructive interference with the SM, with the charge sign already flipped at
   low Q² for this coupling (as in example 01).

6. **Perturbativity.** All couplings are of order 1e-4, far below √(4π).

## Card

The interaction is a single flavor-diagonal vector mediator with constant
couplings, so the schema expresses it exactly. The card stores the already
converted left-handed coupling g_ν = 2g = 1e-4, not the raw g = 5e-5, because
the card's `nu_couplings` field is defined as the coefficient of the P_L current.

## Plugin

`plugin.py` keeps the description's own parameter g = 5e-5 as a `PARAMETERS`
entry and performs the factor-2 conversion inside `coefficients`, so the
described point is preserved exactly while matching the card. `COUPLINGS` lists
g and g_q; zeroing either removes all new physics, reproducing the SM limit.

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
  "notes": "# Notes: universal vector Z\u2032 with g \u03bd\u0304 \u03b3^\u03bc (1 \u2212 \u03b35) \u03bd neutrino coupling\n\n## Description\n\nVector Z\u2032, mass 30 MeV. Neutrino interaction g Z\u2032_\u03bc \u03bd\u0304 \u03b3^\u03bc (1 \u2212 \u03b35) \u03bd for every\nflavor with g = 5e-5 and no factor 1/2. Vector coupling 1e-4 to each of u and d.\nAll other couplings vanish.\n\n## Derivation\n\n1. **Lorentz structure.** Vector boson exchange between a neutrino current and a\n   quark vector current: structure `V`. No scalar part.\n\n2. **Neutrino coupling normalization.** The framework's matching formula\n   (conventions section 3) is written for Z\u2032_\u03bc g_\u03bd \u03bd\u0304 \u03b3^\u03bc P_L \u03bd with\n   P_L = (1 \u2212 \u03b35)/2. The description writes g \u03bd\u0304 \u03b3^\u03bc (1 \u2212 \u03b35) \u03bd = 2g \u03bd\u0304 \u03b3^\u03bc P_L \u03bd\n   without the 1/2, so\n\n       g_\u03bd^{ee} = g_\u03bd^{mm} = g_\u03bd^{tt} = 2g = 2 \u00d7 5e-5 = 1e-4.\n\n   This is exactly the case called out in conventions section 3: \"If it writes\n   g \u03bd\u0304 \u03b3^\u03bc (1 \u2212 \u03b35) \u03bd without the 1/2, then g_\u03bd = 2g.\" Off-diagonal entries\n   are zero (universal, flavor-diagonal).\n\n3. **Quark couplings.** Given directly as vector couplings: g_u = g_d = 1e-4.\n   No chiral decomposition needed, no axial part to drop, no kinetic mixing.\n\n4. **Matching.** From conventions section 3,\n\n       C^{uV}_{\u03b1\u03b1}(Q\u00b2) = C^{dV}_{\u03b1\u03b1}(Q\u00b2) = g_\u03bd g_q / (\u221a2 G_F (Q\u00b2 + m\u00b2))\n                        = (1e-4)(1e-4) / (\u221a2 G_F (Q\u00b2 + (30 MeV)\u00b2)).\n\n   Numerically this coincides with worked example 01 (g\u2032 = 1e-4 universal), so\n   at the Cs-133 reference points Q\u00b2 = 1238, 3714, 6190 MeV\u00b2 the coefficients\n   are approximately 0.28355, 0.13139, 0.085506 for both quarks and all three\n   diagonal flavors.\n\n5. **Sign and interference.** g_\u03bd g_q = +1e-8 > 0, so the new-physics nuclear\n   charge shift \u0394Q = (3Z + 3N) C is positive while Q_SM is negative:\n   destructive interference with the SM, with the charge sign already flipped at\n   low Q\u00b2 for this coupling (as in example 01).\n\n6. **Perturbativity.** All couplings are of order 1e-4, far below \u221a(4\u03c0).\n\n## Card\n\nThe interaction is a single flavor-diagonal vector mediator with constant\ncouplings, so the schema expresses it exactly. The card stores the already\nconverted left-handed coupling g_\u03bd = 2g = 1e-4, not the raw g = 5e-5, because\nthe card's `nu_couplings` field is defined as the coefficient of the P_L current.\n\n## Plugin\n\n`plugin.py` keeps the description's own parameter g = 5e-5 as a `PARAMETERS`\nentry and performs the factor-2 conversion inside `coefficients`, so the\ndescribed point is preserved exactly while matching the card. `COUPLINGS` lists\ng and g_q; zeroing either removes all new physics, reproducing the SM limit.\n"
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
      29.095188985320853,
      27.791789054152893,
      21.87843791042505,
      13.32261483645687,
      6.334360943382066,
      2.592657554222013,
      1.0399132779511506,
      0.359896902121366,
      0.09547829119494593,
      0.014037088192537827,
      0.003392987026783072,
      0.013520844354206472
    ],
    "predicted_total": 102.54128867480074,
    "observed_total": 855.0,
    "delta": 14.196339128054575,
    "new_physics": {
      "neg2_log_likelihood": 31.573776627488595,
      "alpha": -0.29562423132273735,
      "beta": {
        "steady_state": 0.05293782876922028,
        "prompt_neutrons": 0.027776793969094805
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
  "final_text": "Validation passed with no findings, including the plugin/card cross-check on 66 Q\u00b2 points.\n\n**Result.** The description's coupling g \u03bd\u0304\u03b3^\u03bc(1\u2212\u03b35)\u03bd has no factor 1/2, so (1\u2212\u03b35) = 2P_L gives a left-handed coupling g_\u03bd = 2g = 1e-4. With g_u = g_d = 1e-4 and m = 30 MeV, the coefficient is C^{uV} = C^{dV} = g_\u03bd g_q / (\u221a2 G_F (Q\u00b2 + m\u00b2)) on all three diagonal flavors, numerically identical to worked example 01. The sign g_\u03bd g_q > 0 means destructive interference with the Standard Model.\n\nFiles produced in the task directory:\n\n- **plugin.py** keeps the description's own g = 5e-5 and g_q = 1e-4 as parameters and applies the factor 2 inside the coefficient function, so the described point is preserved exactly.\n- **card.json** encodes one vector mediator with the converted left-handed coupling 1e-4 for ee, mm, tt and quark couplings 1e-4 for u and d.\n- **notes.md** records the derivation, the normalization conversion, the sign discussion, and why the card stores 2g rather than g.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":73390,\"stop_reason\":\"end_turn\",\"session_id\":\"9f54ee85-eb61-4ea6-9efa-7f7679c68ae6\",\"total_cost_usd\":0.6381414999999999,\"usage\":{\"input_tokens\":226,\"cache_creation_input_tokens\":19829,\"cache_read_input_tokens\":129606,\"output_tokens\":4138,\"output_tokens_details\":{\"thinking_tokens\":402},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":19829,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":413,\"cache_read_input_tokens\":25513,\"cache_creation_input_tokens\":322,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":322},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":226,\"outputTokens\":4138,\"cacheReadInputTokens\":129606,\"cacheCreationInputTokens\":19829,\"webSearchRequests\":0,\"costUSD\":0.6381414999999999,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":402,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01GyTm8nqENJm65QYe7xQ236\",\"tool_input\":{\"command\":\"ls -R /private/tmp/nuagent-issue17-eval-claude/10_left_handed_normalization\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":17,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings, including the plugin/card cross-check on 66 Q\u00b2 points.\\n\\n**Result.** The description's coupling g \u03bd\u0304\u03b3^\u03bc(1\u2212\u03b35)\u03bd has no factor 1/2, so (1\u2212\u03b35) = 2P_L gives a left-handed coupling g_\u03bd = 2g = 1e-4. With g_u = g_d = 1e-4 and m = 30 MeV, the coefficient is C^{uV} = C^{dV} = g_\u03bd g_q / (\u221a2 G_F (Q\u00b2 + m\u00b2)) on all three diagonal flavors, numerically identical to worked example 01. The sign g_\u03bd g_q > 0 means destructive interference with the Standard Model.\\n\\nFiles produced in the task directory:\\n\\n- **plugin.py** keeps the description's own g = 5e-5 and g_q = 1e-4 as parameters and applies the factor 2 inside the coefficient function, so the described point is preserved exactly.\\n- **card.json** encodes one vector mediator with the converted left-handed coupling 1e-4 for ee, mm, tt and quark couplings 1e-4 for u and d.\\n- **notes.md** records the derivation, the normalization conversion, the sign discussion, and why the card stores 2g rather than g.\",\"ttft_ms\":3116,\"type\":\"result\",\"duration_ms\":75133,\"uuid\":\"e2df16c5-663a-46bb-b1e5-9489650bc905\",\"ttft_stream_ms\":2999,\"time_to_request_ms\":21,\"first_content_frame_ms\":2999,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/issue-17-agent-harness/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/nuagent-issue17-eval-claude/10_left_handed_normalization\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
