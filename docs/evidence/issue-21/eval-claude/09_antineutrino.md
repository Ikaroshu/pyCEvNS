Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

Muon antineutrinos scatter through a universal vector Z′ of mass 30 MeV with g′ = 8e-5. The same coupling applies to all left-handed neutrino flavors and to both u and d quarks, with no other couplings.

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
"""Flavor-universal vector Z' (m = 30 MeV, g' = 8e-5) coupled to left-handed neutrinos and to u, d quarks."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_prime": 8e-5, "mass_mev": 30.0}
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
  "name": "zprime_universal_antineutrino_g8e-5",
  "mediators": [
    {
      "structure": "V",
      "mass_mev": 30.0,
      "nu_couplings": {
        "ee": 8e-5,
        "mm": 8e-5,
        "tt": 8e-5,
        "em": 0,
        "et": 0,
        "mt": 0
      },
      "quark_couplings": {
        "u": 8e-5,
        "d": 8e-5
      },
      "kinetic_mixing": 0.0
    }
  ],
  "contacts": []
}

```

## notes.md

```text
# Notes: universal vector Z′, m = 30 MeV, g′ = 8e-5, muon antineutrinos

## Reading of the description

- Mediator: a single vector boson Z′, so Lorentz structure `V`. No scalar, axial or tensor part is described.
- Mass: m = 30 MeV, given directly.
- Neutrino coupling: "the same coupling applies to all left-handed neutrino flavors", so the coupling is to the left-handed current ν̄_α γ^μ P_L ν_α with g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = g′ = 8e-5 and no off-diagonal entries. This is the universal (toy) charge assignment of conventions section 7 with Q′_ν = +1.
- Quark coupling: the same g′ to both u and d, so g_u = g_d = g′ = 8e-5 (vector couplings; no chiral split is described, so there is no dropped axial part). "No other couplings" means kinetic mixing is zero and no contact terms.
- Antineutrinos: the beam is ν̄_μ, but the coefficient C is the same for neutrinos and antineutrinos (conventions section 3, "Antineutrinos: the framework handles them; you supply the same C"). The framework selects the ν̄_μ flux and applies the antineutrino kinematics. The ee and tt entries are filled because the description makes the coupling universal; they are inert for a pure ν̄_μ flux and matter only for the other flavor components of the experiment's flux.

## Matching (conventions section 3)

For L ⊃ Z′_μ ( g_ν ν̄_α γ^μ P_L ν_α + g_q q̄ γ^μ q ), tree-level exchange gives

    C^{qV}_{αα}(Q²) = g_ν g_q / ( √2 G_F (Q² + m²) ) = g′² / ( √2 G_F (Q² + m²) ),   q ∈ {u, d},  α ∈ {e, μ, τ}.

No convention conversion was needed: g′ is already the coefficient of the left-handed current and of the quark vector current, with no 1/2 or (1 − γ5) factors to absorb. All off-diagonal and all scalar coefficients are zero.

Reference values at the example-01 grid (Q² = 1238, 3714, 6190 MeV²), obtained from the example-01 numbers scaled by (8e-5 / 1e-4)² = 0.64:

| coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
|---|---|---|---|
| C^{uV}_αα = C^{dV}_αα | 1.8147e-01 | 8.4090e-02 | 5.4724e-02 |

## Sign and interference

g_ν g_q = +g′² > 0, so the new-physics nuclear charge shift ΔQ = 3(Z + N) C is positive while the SM charge Q_SM ≈ 0.04 Z − 0.5 N is negative: destructive interference. At g′ = 8e-5 and the first reference point, ΔQ ≈ 72.4 against Q_SM ≈ −36.9 (tree level), so the total charge has flipped sign and its magnitude is close to the SM one. This coupling sits near the degenerate point of the universal-Z′ limit band noted in example 01 (ΔQ ≈ −2 Q_SM near g′ ≈ 8.1e-5); the described point g′ = 8e-5 is kept exactly as given and was not adjusted.

## Card

Card-expressible: yes, one `V` mediator with mass 30 MeV, diagonal neutrino couplings 8e-5, quark couplings u = d = 8e-5, kinetic mixing 0 and no contacts. `card.json` is written alongside `plugin.py`; the plugin computes the same coefficients from the exported constants without reading the card.

## Sanity

- |g′| = 8e-5 ≪ √(4π): perturbative.
- With `g_prime` set to zero the plugin returns all-zero coefficients, so the SM limit is recovered.
- Anomaly cancellation for the universal U(1)′ is not addressed by the description (the universal assignment is anomalous as a gauge symmetry); this is the model author's concern and does not affect the translation.

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
  "notes": "# Notes: universal vector Z\u2032, m = 30 MeV, g\u2032 = 8e-5, muon antineutrinos\n\n## Reading of the description\n\n- Mediator: a single vector boson Z\u2032, so Lorentz structure `V`. No scalar, axial or tensor part is described.\n- Mass: m = 30 MeV, given directly.\n- Neutrino coupling: \"the same coupling applies to all left-handed neutrino flavors\", so the coupling is to the left-handed current \u03bd\u0304_\u03b1 \u03b3^\u03bc P_L \u03bd_\u03b1 with g_\u03bd^{ee} = g_\u03bd^{mm} = g_\u03bd^{tt} = g\u2032 = 8e-5 and no off-diagonal entries. This is the universal (toy) charge assignment of conventions section 7 with Q\u2032_\u03bd = +1.\n- Quark coupling: the same g\u2032 to both u and d, so g_u = g_d = g\u2032 = 8e-5 (vector couplings; no chiral split is described, so there is no dropped axial part). \"No other couplings\" means kinetic mixing is zero and no contact terms.\n- Antineutrinos: the beam is \u03bd\u0304_\u03bc, but the coefficient C is the same for neutrinos and antineutrinos (conventions section 3, \"Antineutrinos: the framework handles them; you supply the same C\"). The framework selects the \u03bd\u0304_\u03bc flux and applies the antineutrino kinematics. The ee and tt entries are filled because the description makes the coupling universal; they are inert for a pure \u03bd\u0304_\u03bc flux and matter only for the other flavor components of the experiment's flux.\n\n## Matching (conventions section 3)\n\nFor L \u2283 Z\u2032_\u03bc ( g_\u03bd \u03bd\u0304_\u03b1 \u03b3^\u03bc P_L \u03bd_\u03b1 + g_q q\u0304 \u03b3^\u03bc q ), tree-level exchange gives\n\n    C^{qV}_{\u03b1\u03b1}(Q\u00b2) = g_\u03bd g_q / ( \u221a2 G_F (Q\u00b2 + m\u00b2) ) = g\u2032\u00b2 / ( \u221a2 G_F (Q\u00b2 + m\u00b2) ),   q \u2208 {u, d},  \u03b1 \u2208 {e, \u03bc, \u03c4}.\n\nNo convention conversion was needed: g\u2032 is already the coefficient of the left-handed current and of the quark vector current, with no 1/2 or (1 \u2212 \u03b35) factors to absorb. All off-diagonal and all scalar coefficients are zero.\n\nReference values at the example-01 grid (Q\u00b2 = 1238, 3714, 6190 MeV\u00b2), obtained from the example-01 numbers scaled by (8e-5 / 1e-4)\u00b2 = 0.64:\n\n| coefficient | Q\u00b2=1238 | Q\u00b2=3714 | Q\u00b2=6190 |\n|---|---|---|---|\n| C^{uV}_\u03b1\u03b1 = C^{dV}_\u03b1\u03b1 | 1.8147e-01 | 8.4090e-02 | 5.4724e-02 |\n\n## Sign and interference\n\ng_\u03bd g_q = +g\u2032\u00b2 > 0, so the new-physics nuclear charge shift \u0394Q = 3(Z + N) C is positive while the SM charge Q_SM \u2248 0.04 Z \u2212 0.5 N is negative: destructive interference. At g\u2032 = 8e-5 and the first reference point, \u0394Q \u2248 72.4 against Q_SM \u2248 \u221236.9 (tree level), so the total charge has flipped sign and its magnitude is close to the SM one. This coupling sits near the degenerate point of the universal-Z\u2032 limit band noted in example 01 (\u0394Q \u2248 \u22122 Q_SM near g\u2032 \u2248 8.1e-5); the described point g\u2032 = 8e-5 is kept exactly as given and was not adjusted.\n\n## Card\n\nCard-expressible: yes, one `V` mediator with mass 30 MeV, diagonal neutrino couplings 8e-5, quark couplings u = d = 8e-5, kinetic mixing 0 and no contacts. `card.json` is written alongside `plugin.py`; the plugin computes the same coefficients from the exported constants without reading the card.\n\n## Sanity\n\n- |g\u2032| = 8e-5 \u226a \u221a(4\u03c0): perturbative.\n- With `g_prime` set to zero the plugin returns all-zero coefficients, so the SM limit is recovered.\n- Anomaly cancellation for the universal U(1)\u2032 is not addressed by the description (the universal assignment is anomalous as a gauge symmetry); this is the model author's concern and does not affect the translation.\n"
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
      5.340239586535451,
      3.6761425256550218,
      1.7603764852861654,
      0.44453299490920983,
      0.031368428374255304,
      0.09159504301082363,
      0.25403782481705206,
      0.3806947833056065,
      0.4403060426516218,
      0.44197953246176147,
      0.4048629386083885,
      0.3469080296517126
    ],
    "predicted_total": 13.61304421526707,
    "observed_total": 855.0,
    "delta": 16.91202661042706,
    "new_physics": {
      "neg2_log_likelihood": 34.28946410986108,
      "alpha": 0.025694011136446333,
      "beta": {
        "steady_state": 0.089551718480989,
        "prompt_neutrons": 0.033447300233050155
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
  "final_text": "Validation passed with no findings. The plugin and card agree on all 66 Q\u00b2 cross-check points, the Standard Model limit holds with the coupling zeroed, and the coupling is perturbative.\n\nFiles produced in the task directory:\n\n- **plugin.py** computes the vector coefficient g\u2032\u00b2 / (\u221a2 G_F (Q\u00b2 + m\u00b2)) for u and d quarks and all three diagonal flavor entries, with defaults g\u2032 = 8e-5 and m = 30 MeV and `g_prime` as the only coupling.\n- **card.json** holds one vector mediator at 30 MeV with diagonal neutrino couplings and u, d quark couplings of 8e-5, zero kinetic mixing and no contacts.\n- **notes.md** records the derivation: structure V, left-handed neutrino coupling taken as written, tree-level matching from conventions section 3, the statement that antineutrinos use the same coefficient and the framework applies the \u03bd\u0304_\u03bc flux, the destructive-interference sign, and reference coefficient values at the example-01 grid.\n\nOne physics note for the reader: at this coupling the new-physics charge nearly equals minus twice the SM charge, so the predicted rate is close to the SM rate despite the large coefficient. The described point was kept exactly as given.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":124189,\"stop_reason\":\"end_turn\",\"session_id\":\"23e38d14-d121-44c7-8425-874f07458db7\",\"total_cost_usd\":0.73969625,\"usage\":{\"input_tokens\":226,\"cache_creation_input_tokens\":22835,\"cache_read_input_tokens\":143145,\"output_tokens\":4899,\"output_tokens_details\":{\"thinking_tokens\":386},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":22835,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":418,\"cache_read_input_tokens\":28547,\"cache_creation_input_tokens\":294,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":294},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":226,\"outputTokens\":4899,\"cacheReadInputTokens\":143145,\"cacheCreationInputTokens\":22835,\"webSearchRequests\":0,\"costUSD\":0.73969625,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":386,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01S5NvpkdJEq8JdRtUh9Q91v\",\"tool_input\":{\"command\":\"ls -R /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/09_antineutrino\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":16,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings. The plugin and card agree on all 66 Q\u00b2 cross-check points, the Standard Model limit holds with the coupling zeroed, and the coupling is perturbative.\\n\\nFiles produced in the task directory:\\n\\n- **plugin.py** computes the vector coefficient g\u2032\u00b2 / (\u221a2 G_F (Q\u00b2 + m\u00b2)) for u and d quarks and all three diagonal flavor entries, with defaults g\u2032 = 8e-5 and m = 30 MeV and `g_prime` as the only coupling.\\n- **card.json** holds one vector mediator at 30 MeV with diagonal neutrino couplings and u, d quark couplings of 8e-5, zero kinetic mixing and no contacts.\\n- **notes.md** records the derivation: structure V, left-handed neutrino coupling taken as written, tree-level matching from conventions section 3, the statement that antineutrinos use the same coefficient and the framework applies the \u03bd\u0304_\u03bc flux, the destructive-interference sign, and reference coefficient values at the example-01 grid.\\n\\nOne physics note for the reader: at this coupling the new-physics charge nearly equals minus twice the SM charge, so the predicted rate is close to the SM rate despite the large coefficient. The described point was kept exactly as given.\",\"ttft_ms\":12257,\"type\":\"result\",\"duration_ms\":127824,\"uuid\":\"bc14c775-e0ee-4cb5-bb27-c7d145842a5f\",\"ttft_stream_ms\":10331,\"time_to_request_ms\":24,\"first_content_frame_ms\":10332,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/m7-conversation/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/09_antineutrino\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
