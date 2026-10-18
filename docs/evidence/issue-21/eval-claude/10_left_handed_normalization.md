Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

A universal vector Z′ has mass 30 MeV. Its neutrino interaction is g Z′_mu nu-bar gamma^mu (1 − gamma5) nu for every flavor, with g = 5e-5 and no factor 1/2. Its vector coupling to each of u and d quarks is 1e-4. All other couplings vanish.

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
"""Universal vector Z' with neutrino coupling written as g nubar gamma^mu (1 - gamma5) nu (no 1/2).

Derivation in notes.md.
"""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

# g is the coefficient of nubar gamma^mu (1 - gamma5) nu as written in the description.
# g_u and g_d are the vector quark couplings. mass_mev is the Z' mass.
PARAMETERS = {"g": 5e-5, "g_u": 1e-4, "g_d": 1e-4, "mass_mev": 30.0}
COUPLINGS = ("g", "g_u", "g_d")


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    # (1 - gamma5) = 2 P_L, so the left-handed neutrino coupling is g_nu = 2 g.
    g_nu = 2.0 * params["g"]
    propagator = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    quark_couplings = {"u": params["g_u"], "d": params["g_d"]}
    for quark, g_q in quark_couplings.items():
        value = g_nu * g_q * propagator
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)
    return c

```

## card.json

```json
{
  "name": "zprime_universal_left_handed_normalization",
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
# Universal vector Z′ with a (1 − γ5) neutrino current

## Description as given

Vector Z′, mass m = 30 MeV. Neutrino interaction g Z′_μ ν̄ γ^μ (1 − γ5) ν for every flavor, g = 5e-5, explicitly without a factor 1/2. Vector coupling to u and to d quarks 1e-4 each. All other couplings vanish.

## Derivation

1. **Lorentz structure.** Vector exchange between a neutrino current and a quark vector current: structure `V`. No scalar, axial or tensor pieces, so the V/S basis is sufficient and a card can express the model.

2. **Neutrino coupling normalization.** The framework's convention (conventions.md section 3) writes the Z′ vertex as g_ν ν̄ γ^μ P_L ν with P_L = (1 − γ5)/2. The description writes g ν̄ γ^μ (1 − γ5) ν = 2g ν̄ γ^μ P_L ν. Therefore

       g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = 2 g = 2 × 5e-5 = 1e-4.

   Off-diagonal neutrino entries are zero ("every flavor", flavor-diagonal and universal). This is the pitfall called out in conventions section 3: a (g′/2)(1 − γ5) form gives g_ν = g′, the (1 − γ5) form without the 1/2 gives g_ν = 2g. Only the left-handed current appears either way, so nothing is dropped.

3. **Quark couplings.** Given directly as vector couplings: g_u = g_d = 1e-4. No chiral split is given, so there is no axial part to drop. No kinetic mixing (ε = 0), consistent with "all other couplings vanish".

4. **Matching.** Tree-level Z′ exchange (conventions section 3):

       C^{fV}_{αα}(Q²) = g_ν g_f / (√2 G_F (Q² + m²)),   f ∈ {u, d},  α ∈ {e, μ, τ},

   with g_ν = 1e-4, g_u = g_d = 1e-4, m = 30 MeV. Numerically this coincides with the flavor-universal example 01 (g′ = 1e-4 for neutrinos and quarks), so the expected coefficients at the Cs-133 reference points Q² = 1238, 3714, 6190 MeV² are 2.8355e-01, 1.3139e-01, 8.5506e-02 for both C^{uV} and C^{dV}, flavor-diagonal.

5. **Sign and interference.** g_ν g_q = +1e-8 > 0, so the new-physics nuclear charge shift is positive while the SM charge is negative: destructive interference, and at the described point the shift exceeds |Q_SM|, so the total charge has flipped sign and the rate is enhanced relative to the SM.

6. **Perturbativity.** All couplings are of order 1e-4, far below √(4π).

## Plugin parameters

`PARAMETERS = {"g": 5e-5, "g_u": 1e-4, "g_d": 1e-4, "mass_mev": 30.0}`; the plugin forms g_ν = 2 g internally. `COUPLINGS = ("g", "g_u", "g_d")`; zeroing them returns the SM. The card stores the already-converted left-handed coupling g_ν = 1e-4, since the schema's `nu_couplings` is defined for the P_L current.

## Card

One vector mediator entry with mass 30 MeV, `nu_couplings` ee = mm = tt = 1e-4, `quark_couplings` u = d = 1e-4, no kinetic mixing, no contacts.

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
  "notes": "# Universal vector Z\u2032 with a (1 \u2212 \u03b35) neutrino current\n\n## Description as given\n\nVector Z\u2032, mass m = 30 MeV. Neutrino interaction g Z\u2032_\u03bc \u03bd\u0304 \u03b3^\u03bc (1 \u2212 \u03b35) \u03bd for every flavor, g = 5e-5, explicitly without a factor 1/2. Vector coupling to u and to d quarks 1e-4 each. All other couplings vanish.\n\n## Derivation\n\n1. **Lorentz structure.** Vector exchange between a neutrino current and a quark vector current: structure `V`. No scalar, axial or tensor pieces, so the V/S basis is sufficient and a card can express the model.\n\n2. **Neutrino coupling normalization.** The framework's convention (conventions.md section 3) writes the Z\u2032 vertex as g_\u03bd \u03bd\u0304 \u03b3^\u03bc P_L \u03bd with P_L = (1 \u2212 \u03b35)/2. The description writes g \u03bd\u0304 \u03b3^\u03bc (1 \u2212 \u03b35) \u03bd = 2g \u03bd\u0304 \u03b3^\u03bc P_L \u03bd. Therefore\n\n       g_\u03bd^{ee} = g_\u03bd^{mm} = g_\u03bd^{tt} = 2 g = 2 \u00d7 5e-5 = 1e-4.\n\n   Off-diagonal neutrino entries are zero (\"every flavor\", flavor-diagonal and universal). This is the pitfall called out in conventions section 3: a (g\u2032/2)(1 \u2212 \u03b35) form gives g_\u03bd = g\u2032, the (1 \u2212 \u03b35) form without the 1/2 gives g_\u03bd = 2g. Only the left-handed current appears either way, so nothing is dropped.\n\n3. **Quark couplings.** Given directly as vector couplings: g_u = g_d = 1e-4. No chiral split is given, so there is no axial part to drop. No kinetic mixing (\u03b5 = 0), consistent with \"all other couplings vanish\".\n\n4. **Matching.** Tree-level Z\u2032 exchange (conventions section 3):\n\n       C^{fV}_{\u03b1\u03b1}(Q\u00b2) = g_\u03bd g_f / (\u221a2 G_F (Q\u00b2 + m\u00b2)),   f \u2208 {u, d},  \u03b1 \u2208 {e, \u03bc, \u03c4},\n\n   with g_\u03bd = 1e-4, g_u = g_d = 1e-4, m = 30 MeV. Numerically this coincides with the flavor-universal example 01 (g\u2032 = 1e-4 for neutrinos and quarks), so the expected coefficients at the Cs-133 reference points Q\u00b2 = 1238, 3714, 6190 MeV\u00b2 are 2.8355e-01, 1.3139e-01, 8.5506e-02 for both C^{uV} and C^{dV}, flavor-diagonal.\n\n5. **Sign and interference.** g_\u03bd g_q = +1e-8 > 0, so the new-physics nuclear charge shift is positive while the SM charge is negative: destructive interference, and at the described point the shift exceeds |Q_SM|, so the total charge has flipped sign and the rate is enhanced relative to the SM.\n\n6. **Perturbativity.** All couplings are of order 1e-4, far below \u221a(4\u03c0).\n\n## Plugin parameters\n\n`PARAMETERS = {\"g\": 5e-5, \"g_u\": 1e-4, \"g_d\": 1e-4, \"mass_mev\": 30.0}`; the plugin forms g_\u03bd = 2 g internally. `COUPLINGS = (\"g\", \"g_u\", \"g_d\")`; zeroing them returns the SM. The card stores the already-converted left-handed coupling g_\u03bd = 1e-4, since the schema's `nu_couplings` is defined for the P_L current.\n\n## Card\n\nOne vector mediator entry with mass 30 MeV, `nu_couplings` ee = mm = tt = 1e-4, `quark_couplings` u = d = 1e-4, no kinetic mixing, no contacts.\n"
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
      21.878437910425053,
      13.322614836456868,
      6.334360943382067,
      2.592657554222013,
      1.039913277951151,
      0.359896902121366,
      0.09547829119494593,
      0.014037088192537833,
      0.003392987026783079,
      0.013520844354206467
    ],
    "predicted_total": 102.54128867480074,
    "observed_total": 855.0,
    "delta": 14.196339128054618,
    "new_physics": {
      "neg2_log_likelihood": 31.573776627488638,
      "alpha": -0.29562423132273746,
      "beta": {
        "steady_state": 0.05293782876922022,
        "prompt_neutrons": 0.02777679396909486
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
  "final_text": "Validation passed with no findings, including the plugin/card cross-check on 66 Q\u00b2 points.\n\n**Files produced**\n\n- `plugin.py`: vector-only coefficients with parameters g = 5e-5, g_u = g_d = 1e-4, mass 30 MeV. It forms the left-handed coupling g_\u03bd = 2g internally, since (1 \u2212 \u03b35) = 2 P_L and the description omits the 1/2.\n- `card.json`: one vector mediator at 30 MeV with the converted neutrino coupling ee = mm = tt = 1e-4, quark couplings u = d = 1e-4, no kinetic mixing, no contacts.\n- `notes.md`: the derivation, the normalization conversion from the (1 \u2212 \u03b35) form to the framework's P_L convention, the matching formula, and the sign of the interference with the Standard Model.\n\n**Key physics point**\n\nThe only nontrivial step is the neutrino normalization. With g_\u03bd = 1e-4 and g_u = g_d = 1e-4, the coefficients coincide with the universal example. The product g_\u03bd g_q is positive, so the new-physics charge interferes destructively with the Standard Model charge and flips its sign at this point.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":61180,\"stop_reason\":\"end_turn\",\"session_id\":\"b0c36612-a13b-4377-b5ac-79a615e8b8b4\",\"total_cost_usd\":0.76124525,\"usage\":{\"input_tokens\":226,\"cache_creation_input_tokens\":23392,\"cache_read_input_tokens\":145181,\"output_tokens\":5097,\"output_tokens_details\":{\"thinking_tokens\":250},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":23392,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":396,\"cache_read_input_tokens\":29099,\"cache_creation_input_tokens\":299,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":299},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":226,\"outputTokens\":5097,\"cacheReadInputTokens\":145181,\"cacheCreationInputTokens\":23392,\"webSearchRequests\":0,\"costUSD\":0.76124525,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":250,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01N94MQoMmohybC1vJttqQSt\",\"tool_input\":{\"command\":\"ls -R /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/10_left_handed_normalization\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":18,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings, including the plugin/card cross-check on 66 Q\u00b2 points.\\n\\n**Files produced**\\n\\n- `plugin.py`: vector-only coefficients with parameters g = 5e-5, g_u = g_d = 1e-4, mass 30 MeV. It forms the left-handed coupling g_\u03bd = 2g internally, since (1 \u2212 \u03b35) = 2 P_L and the description omits the 1/2.\\n- `card.json`: one vector mediator at 30 MeV with the converted neutrino coupling ee = mm = tt = 1e-4, quark couplings u = d = 1e-4, no kinetic mixing, no contacts.\\n- `notes.md`: the derivation, the normalization conversion from the (1 \u2212 \u03b35) form to the framework's P_L convention, the matching formula, and the sign of the interference with the Standard Model.\\n\\n**Key physics point**\\n\\nThe only nontrivial step is the neutrino normalization. With g_\u03bd = 1e-4 and g_u = g_d = 1e-4, the coefficients coincide with the universal example. The product g_\u03bd g_q is positive, so the new-physics charge interferes destructively with the Standard Model charge and flips its sign at this point.\",\"ttft_ms\":1073,\"type\":\"result\",\"duration_ms\":64641,\"uuid\":\"690835bc-f0a3-4d82-a014-2a9221d579ab\",\"ttft_stream_ms\":982,\"time_to_request_ms\":23,\"first_content_frame_ms\":983,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/m7-conversation/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/10_left_handed_normalization\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
