Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

Muon antineutrinos scatter through a universal vector Z′ of mass 30 MeV with g′ = 8e-5. The same coupling applies to all left-handed neutrino flavors and to both u and d quarks, with no other couplings.

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
"""Universal vector Z' (m = 30 MeV, g' = 8e-5) for muon antineutrino scattering; derivation in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_prime": 8e-5, "mass_mev": 30.0}
COUPLINGS = ("g_prime",)


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    g_nu = params["g_prime"]
    g_u = params["g_prime"]
    g_d = params["g_prime"]
    propagator = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    values = {"u": g_nu * g_u * propagator, "d": g_nu * g_d * propagator}
    for quark in ("u", "d"):
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, values[quark])
    return c

```

## card.json

```json
{
  "name": "zprime_universal_antineutrino_g8e-5_m30",
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
# Notes: muon antineutrinos through a universal vector Z′

## Description

Muon antineutrinos scatter through a universal vector Z′ of mass 30 MeV with g′ = 8e-5. The same coupling applies to all left-handed neutrino flavors and to both u and d quarks, with no other couplings.

## Derivation

1. **Lorentz structure.** A vector boson exchanged between the neutrino current and the quark vector current: structure `V` only. No scalar, axial or tensor pieces are described, and no kinetic mixing is mentioned ("no other couplings"), so `kinetic_mixing` = 0.

2. **Lagrangian.** Taking the description literally in the form of conventions section 3,

       L ⊃ Z′_μ ( g′ Σ_α ν̄_α γ^μ P_L ν_α + g′ ū γ^μ u + g′ d̄ γ^μ d ),   m = 30 MeV, g′ = 8e-5.

   So g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = g′, all off-diagonal g_ν zero, and g_u = g_d = g′. This is the "universal (toy)" assignment of conventions section 7 with all U(1)′ charges equal to +1. The coupling is already in the left-handed-current normalization, so no factor-2 conversion is needed.

3. **Matching** (conventions section 3, tree-level Z′ exchange):

       C^{uV}_{αα}(Q²) = C^{dV}_{αα}(Q²) = g′² / ( √2 G_F (Q² + m²) ),   α = e, μ, τ,

   with the propagator written as 1/(Q² + m²) for spacelike Q² = 2 M T. All off-diagonal and all scalar coefficients are zero.

4. **Antineutrinos.** The incoming particles are muon antineutrinos. Per conventions section 3 ("Antineutrinos: the framework handles them; you supply the same C"), the coefficient matrix is unchanged: the framework folds the antineutrino flux with the same C^{fV}. The muon-flavor entry `mm` is the one that matters for the described beam, but the description states that the coupling is universal, so `ee` and `tt` are filled with the same value. The framework's flux selection picks out the relevant flavor.

5. **Sign and interference.** g_ν g_q = +g′² > 0, so the new-physics nuclear charge shift ΔQ = 3(Z + N) C is positive while Q_SM ≈ 0.04 Z − 0.5 N is negative: destructive interference. With g′ = 8e-5 the coupling sits close to the degenerate point noted in worked example 01 (ΔQ ≈ −2 Q_SM near g′ ≈ 8.1e-5 at Q² = 1238 MeV²), where the total charge has flipped sign and the rate is close to the Standard Model rate again. At g′ = 8e-5 exactly, ΔQ at Q² = 1238 MeV² is (8e-5/1e-4)² × 113.1 ≈ 72.4 versus |Q_SM| = 36.9 (tree level), so the total charge is about +35.5, slightly below |Q_SM| in magnitude. This is a statement about the described point, which is preserved exactly; no parameter was adjusted.

6. **Numerical check** at the reference Q² of example 01 (Cs-133, T = 5, 15, 25 keV): scaling the example values by (8e-5/1e-4)² = 0.64 gives

   | coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
   |---|---|---|---|
   | C^{uV}_{αα} = C^{dV}_{αα} | 1.8147e-01 | 8.4090e-02 | 5.4724e-02 |

7. **Perturbativity.** |g′| = 8e-5 ≪ √(4π).

8. **Card-expressible.** Yes: one vector mediator entry with constant couplings, so `card.json` is provided and the plugin computes the same coefficients independently from the exported constants `GF` and `SQRT2`.

## Conventions used

No conversion was needed; the description is already in the normalization of conventions section 3. The plugin exposes `g_prime` as its single coupling (set to zero in the Standard Model-limit check) and `mass_mev` as the mediator mass.

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
  "notes": "# Notes: muon antineutrinos through a universal vector Z\u2032\n\n## Description\n\nMuon antineutrinos scatter through a universal vector Z\u2032 of mass 30 MeV with g\u2032 = 8e-5. The same coupling applies to all left-handed neutrino flavors and to both u and d quarks, with no other couplings.\n\n## Derivation\n\n1. **Lorentz structure.** A vector boson exchanged between the neutrino current and the quark vector current: structure `V` only. No scalar, axial or tensor pieces are described, and no kinetic mixing is mentioned (\"no other couplings\"), so `kinetic_mixing` = 0.\n\n2. **Lagrangian.** Taking the description literally in the form of conventions section 3,\n\n       L \u2283 Z\u2032_\u03bc ( g\u2032 \u03a3_\u03b1 \u03bd\u0304_\u03b1 \u03b3^\u03bc P_L \u03bd_\u03b1 + g\u2032 \u016b \u03b3^\u03bc u + g\u2032 d\u0304 \u03b3^\u03bc d ),   m = 30 MeV, g\u2032 = 8e-5.\n\n   So g_\u03bd^{ee} = g_\u03bd^{mm} = g_\u03bd^{tt} = g\u2032, all off-diagonal g_\u03bd zero, and g_u = g_d = g\u2032. This is the \"universal (toy)\" assignment of conventions section 7 with all U(1)\u2032 charges equal to +1. The coupling is already in the left-handed-current normalization, so no factor-2 conversion is needed.\n\n3. **Matching** (conventions section 3, tree-level Z\u2032 exchange):\n\n       C^{uV}_{\u03b1\u03b1}(Q\u00b2) = C^{dV}_{\u03b1\u03b1}(Q\u00b2) = g\u2032\u00b2 / ( \u221a2 G_F (Q\u00b2 + m\u00b2) ),   \u03b1 = e, \u03bc, \u03c4,\n\n   with the propagator written as 1/(Q\u00b2 + m\u00b2) for spacelike Q\u00b2 = 2 M T. All off-diagonal and all scalar coefficients are zero.\n\n4. **Antineutrinos.** The incoming particles are muon antineutrinos. Per conventions section 3 (\"Antineutrinos: the framework handles them; you supply the same C\"), the coefficient matrix is unchanged: the framework folds the antineutrino flux with the same C^{fV}. The muon-flavor entry `mm` is the one that matters for the described beam, but the description states that the coupling is universal, so `ee` and `tt` are filled with the same value. The framework's flux selection picks out the relevant flavor.\n\n5. **Sign and interference.** g_\u03bd g_q = +g\u2032\u00b2 > 0, so the new-physics nuclear charge shift \u0394Q = 3(Z + N) C is positive while Q_SM \u2248 0.04 Z \u2212 0.5 N is negative: destructive interference. With g\u2032 = 8e-5 the coupling sits close to the degenerate point noted in worked example 01 (\u0394Q \u2248 \u22122 Q_SM near g\u2032 \u2248 8.1e-5 at Q\u00b2 = 1238 MeV\u00b2), where the total charge has flipped sign and the rate is close to the Standard Model rate again. At g\u2032 = 8e-5 exactly, \u0394Q at Q\u00b2 = 1238 MeV\u00b2 is (8e-5/1e-4)\u00b2 \u00d7 113.1 \u2248 72.4 versus |Q_SM| = 36.9 (tree level), so the total charge is about +35.5, slightly below |Q_SM| in magnitude. This is a statement about the described point, which is preserved exactly; no parameter was adjusted.\n\n6. **Numerical check** at the reference Q\u00b2 of example 01 (Cs-133, T = 5, 15, 25 keV): scaling the example values by (8e-5/1e-4)\u00b2 = 0.64 gives\n\n   | coefficient | Q\u00b2=1238 | Q\u00b2=3714 | Q\u00b2=6190 |\n   |---|---|---|---|\n   | C^{uV}_{\u03b1\u03b1} = C^{dV}_{\u03b1\u03b1} | 1.8147e-01 | 8.4090e-02 | 5.4724e-02 |\n\n7. **Perturbativity.** |g\u2032| = 8e-5 \u226a \u221a(4\u03c0).\n\n8. **Card-expressible.** Yes: one vector mediator entry with constant couplings, so `card.json` is provided and the plugin computes the same coefficients independently from the exported constants `GF` and `SQRT2`.\n\n## Conventions used\n\nNo conversion was needed; the description is already in the normalization of conventions section 3. The plugin exposes `g_prime` as its single coupling (set to zero in the Standard Model-limit check) and `mass_mev` as the mediator mass.\n"
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
      3.6761425256550226,
      1.7603764852861654,
      0.44453299490920983,
      0.03136842837425527,
      0.09159504301082362,
      0.25403782481705195,
      0.38069478330560663,
      0.4403060426516218,
      0.44197953246176147,
      0.4048629386083885,
      0.34690802965171247
    ],
    "predicted_total": 13.613044215267074,
    "observed_total": 855.0,
    "delta": 16.912026610427052,
    "new_physics": {
      "neg2_log_likelihood": 34.28946410986107,
      "alpha": 0.02569401113644628,
      "beta": {
        "steady_state": 0.08955171848098899,
        "prompt_neutrons": 0.03344730023305016
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
  "final_text": "Validation passed with no findings, including the plugin/card cross-check on 66 Q\u00b2 points.\n\n**Files produced** in the task directory:\n\n- `plugin.py`: universal vector Z\u2032 with defaults g\u2032 = 8e-5 and mass 30 MeV. It sets C^{uV} = C^{dV} = g\u2032\u00b2 / (\u221a2 G_F (Q\u00b2 + m\u00b2)) on the ee, mm and tt entries, computed from the exported GF and SQRT2 constants.\n- `card.json`: the equivalent single vector mediator entry with all three diagonal neutrino couplings and both quark couplings equal to 8e-5, zero kinetic mixing, no contacts.\n- `notes.md`: the derivation. Structure V, couplings taken as written in the left-handed normalization, matching from conventions section 3, and the note that antineutrinos need the same coefficients because the framework handles the antineutrino flux itself.\n\nTwo physics points recorded in the notes. The product g_\u03bd g_q is positive, so the interference with the Standard Model is destructive. The coupling 8e-5 sits just below the degenerate point near 8.1e-5 from worked example 01, where the rate returns to roughly the Standard Model value. The described point was kept exactly as stated.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":47860,\"stop_reason\":\"end_turn\",\"session_id\":\"f20f1451-11d7-439f-a776-1b99160f5dfe\",\"total_cost_usd\":0.63330275,\"usage\":{\"input_tokens\":226,\"cache_creation_input_tokens\":19637,\"cache_read_input_tokens\":128811,\"output_tokens\":4122,\"output_tokens_details\":{\"thinking_tokens\":227},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":19637,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":411,\"cache_read_input_tokens\":25344,\"cache_creation_input_tokens\":299,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":299},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":226,\"outputTokens\":4122,\"cacheReadInputTokens\":128811,\"cacheCreationInputTokens\":19637,\"webSearchRequests\":0,\"costUSD\":0.63330275,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":227,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01RGnxqQRVKjxztTrqQNZasu\",\"tool_input\":{\"command\":\"ls -R /private/tmp/nuagent-issue17-eval-claude/09_antineutrino\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":16,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings, including the plugin/card cross-check on 66 Q\u00b2 points.\\n\\n**Files produced** in the task directory:\\n\\n- `plugin.py`: universal vector Z\u2032 with defaults g\u2032 = 8e-5 and mass 30 MeV. It sets C^{uV} = C^{dV} = g\u2032\u00b2 / (\u221a2 G_F (Q\u00b2 + m\u00b2)) on the ee, mm and tt entries, computed from the exported GF and SQRT2 constants.\\n- `card.json`: the equivalent single vector mediator entry with all three diagonal neutrino couplings and both quark couplings equal to 8e-5, zero kinetic mixing, no contacts.\\n- `notes.md`: the derivation. Structure V, couplings taken as written in the left-handed normalization, matching from conventions section 3, and the note that antineutrinos need the same coefficients because the framework handles the antineutrino flux itself.\\n\\nTwo physics points recorded in the notes. The product g_\u03bd g_q is positive, so the interference with the Standard Model is destructive. The coupling 8e-5 sits just below the degenerate point near 8.1e-5 from worked example 01, where the rate returns to roughly the Standard Model value. The described point was kept exactly as stated.\",\"ttft_ms\":831,\"type\":\"result\",\"duration_ms\":49526,\"uuid\":\"73c4868b-d450-43dd-b11c-06dd9f1823df\",\"ttft_stream_ms\":704,\"time_to_request_ms\":21,\"first_content_frame_ms\":704,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/issue-17-agent-harness/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/nuagent-issue17-eval-claude/09_antineutrino\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
