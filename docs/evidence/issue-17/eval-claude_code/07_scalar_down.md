Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

A scalar phi of mass 20 MeV has Dirac couplings 2e-5 phi nu-bar_alpha nu_alpha to all three neutrino flavors and 2e-5 phi d-bar d to down quarks only. All other couplings vanish.

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
"""Scalar mediator with universal Dirac neutrino coupling and a down-quark-only coupling."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_nu": 2e-5, "g_d": 2e-5, "mass_mev": 20.0}
COUPLINGS = ("g_nu", "g_d")


def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    value = params["g_nu"] * params["g_d"] / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for pair in ("ee", "mm", "tt"):
        c.set("S", "d", pair, value)
    return c

```

## card.json

```json
{
  "name": "scalar_down_only",
  "mediators": [
    {
      "structure": "S",
      "mass_mev": 20.0,
      "nu_couplings": {
        "ee": 2e-05,
        "mm": 2e-05,
        "tt": 2e-05,
        "em": 0,
        "et": 0,
        "mt": 0
      },
      "quark_couplings": {
        "u": 0.0,
        "d": 2e-05
      }
    }
  ],
  "contacts": []
}

```

## notes.md

```text
# Notes: scalar φ with universal neutrino coupling and down-quark-only coupling

## Description

"A scalar phi of mass 20 MeV has Dirac couplings 2e-5 phi nu-bar_alpha nu_alpha to all three
neutrino flavors and 2e-5 phi d-bar d to down quarks only. All other couplings vanish."

## Derivation

1. **Lorentz structure.** Scalar exchange, structure `S`. The description writes the couplings
   in exactly the form of conventions section 4, L ⊃ φ (g_ν^{αβ} ν̄_α ν_β + Σ_q g_q q̄ q), so no
   convention conversion is needed.

2. **Couplings as written.**
   - g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = 2e-5; off-diagonal entries zero.
   - g_d = 2e-5; g_u = 0 ("down quarks only"; "all other couplings vanish").
   - Mass m = 20 MeV.

3. **Neutrino nature.** The description states Dirac couplings, so ν̄_α ν_α pairs the SM
   left-handed neutrino with a right-handed Dirac partner. There is no Majorana 1/2 issue; g_ν is
   taken as written.

4. **Matching (conventions section 4).**

       C^{dS}_{αα}(Q²) = g_ν g_d / (√2 G_F (Q² + m²)),     α = e, μ, τ
       C^{uS}_{αβ}(Q²) = 0
       all vector coefficients = 0, all off-diagonal scalar entries = 0

   with g_ν g_d = 4e-10 and m² = 400 MeV².

5. **Reference values** (Cs-133, Q² = 1238, 3714, 6190 MeV², rounded to four digits; the plugin
   uses the exported constants at full precision):

   | coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
   |---|---|---|---|
   | C^{dS}_ee = C^{dS}_mm = C^{dS}_tt | 1.480e-02 | 5.894e-03 | 3.680e-03 |
   | C^{uS}_αβ, all C^{V} | 0 | 0 | 0 |

6. **Nucleon charges (formed by the framework, recorded for orientation).** With g_u = 0 only
   the down-quark scalar form factors contribute:
   c_p = m_p g_d f_d^p / m_d ≈ 8.20 g_d,  c_n = m_n g_d f_d^n / m_d ≈ 9.02 g_d,
   so the nucleus couples as roughly g_ν g_d (8.2 Z + 9.0 N)/(Q² + m²). This is about half the
   per-nucleon strength of the isoscalar case (c_N ≈ 17.1 g_q for g_u = g_d), and it is slightly
   neutron-favoring.

7. **Interference.** The scalar amplitude flips the neutrino helicity and does not interfere with
   the SM vector amplitude; the new-physics rate adds incoherently with the kinematic factor the
   framework supplies. The overall sign of g_ν g_d is therefore unobservable here.

8. **Perturbativity.** |g_ν| = |g_d| = 2e-5 ≪ √(4π).

9. **Card-expressible.** Yes: one scalar mediator entry with the three diagonal neutrino couplings,
   d = 2e-5 and u = 0. `card.json` is written accordingly.

## Files

- `plugin.py`: parameters `g_nu`, `g_d`, `mass_mev` (defaults 2e-5, 2e-5, 20.0); couplings
  `g_nu`, `g_d`. Sets only `S`/`d` diagonal entries.
- `card.json`: the equivalent single-mediator card.

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
  "notes": "# Notes: scalar \u03c6 with universal neutrino coupling and down-quark-only coupling\n\n## Description\n\n\"A scalar phi of mass 20 MeV has Dirac couplings 2e-5 phi nu-bar_alpha nu_alpha to all three\nneutrino flavors and 2e-5 phi d-bar d to down quarks only. All other couplings vanish.\"\n\n## Derivation\n\n1. **Lorentz structure.** Scalar exchange, structure `S`. The description writes the couplings\n   in exactly the form of conventions section 4, L \u2283 \u03c6 (g_\u03bd^{\u03b1\u03b2} \u03bd\u0304_\u03b1 \u03bd_\u03b2 + \u03a3_q g_q q\u0304 q), so no\n   convention conversion is needed.\n\n2. **Couplings as written.**\n   - g_\u03bd^{ee} = g_\u03bd^{mm} = g_\u03bd^{tt} = 2e-5; off-diagonal entries zero.\n   - g_d = 2e-5; g_u = 0 (\"down quarks only\"; \"all other couplings vanish\").\n   - Mass m = 20 MeV.\n\n3. **Neutrino nature.** The description states Dirac couplings, so \u03bd\u0304_\u03b1 \u03bd_\u03b1 pairs the SM\n   left-handed neutrino with a right-handed Dirac partner. There is no Majorana 1/2 issue; g_\u03bd is\n   taken as written.\n\n4. **Matching (conventions section 4).**\n\n       C^{dS}_{\u03b1\u03b1}(Q\u00b2) = g_\u03bd g_d / (\u221a2 G_F (Q\u00b2 + m\u00b2)),     \u03b1 = e, \u03bc, \u03c4\n       C^{uS}_{\u03b1\u03b2}(Q\u00b2) = 0\n       all vector coefficients = 0, all off-diagonal scalar entries = 0\n\n   with g_\u03bd g_d = 4e-10 and m\u00b2 = 400 MeV\u00b2.\n\n5. **Reference values** (Cs-133, Q\u00b2 = 1238, 3714, 6190 MeV\u00b2, rounded to four digits; the plugin\n   uses the exported constants at full precision):\n\n   | coefficient | Q\u00b2=1238 | Q\u00b2=3714 | Q\u00b2=6190 |\n   |---|---|---|---|\n   | C^{dS}_ee = C^{dS}_mm = C^{dS}_tt | 1.480e-02 | 5.894e-03 | 3.680e-03 |\n   | C^{uS}_\u03b1\u03b2, all C^{V} | 0 | 0 | 0 |\n\n6. **Nucleon charges (formed by the framework, recorded for orientation).** With g_u = 0 only\n   the down-quark scalar form factors contribute:\n   c_p = m_p g_d f_d^p / m_d \u2248 8.20 g_d,  c_n = m_n g_d f_d^n / m_d \u2248 9.02 g_d,\n   so the nucleus couples as roughly g_\u03bd g_d (8.2 Z + 9.0 N)/(Q\u00b2 + m\u00b2). This is about half the\n   per-nucleon strength of the isoscalar case (c_N \u2248 17.1 g_q for g_u = g_d), and it is slightly\n   neutron-favoring.\n\n7. **Interference.** The scalar amplitude flips the neutrino helicity and does not interfere with\n   the SM vector amplitude; the new-physics rate adds incoherently with the kinematic factor the\n   framework supplies. The overall sign of g_\u03bd g_d is therefore unobservable here.\n\n8. **Perturbativity.** |g_\u03bd| = |g_d| = 2e-5 \u226a \u221a(4\u03c0).\n\n9. **Card-expressible.** Yes: one scalar mediator entry with the three diagonal neutrino couplings,\n   d = 2e-5 and u = 0. `card.json` is written accordingly.\n\n## Files\n\n- `plugin.py`: parameters `g_nu`, `g_d`, `mass_mev` (defaults 2e-5, 2e-5, 20.0); couplings\n  `g_nu`, `g_d`. Sets only `S`/`d` diagonal entries.\n- `card.json`: the equivalent single-mediator card.\n"
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
      11.103995891424395,
      18.03754376961297,
      23.703621042218327,
      24.15913714880117,
      19.67266151668794,
      13.885171320797994,
      10.539067459425354,
      7.9126518188810735,
      5.84120976963275,
      4.252833093035968,
      3.053893192933922,
      2.1598672486145793
    ],
    "predicted_total": 144.3216532720664,
    "observed_total": 855.0,
    "delta": -0.3463187085860966,
    "new_physics": {
      "neg2_log_likelihood": 17.031118790847923,
      "alpha": 0.031119567330047778,
      "beta": {
        "steady_state": -0.0019107633089138478,
        "prompt_neutrons": 0.0019761265994407955
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
  "final_text": "Validation passed with no findings, including the plugin-versus-card cross-check on 66 momentum points.\n\n**Translation.** The interaction is a scalar mediator fully inside the V/S basis. Structure S, mass 20 MeV, flavor-diagonal neutrino coupling 2e-5 for e, \u03bc, \u03c4, down-quark coupling 2e-5, up-quark coupling zero. The only nonzero coefficient is the diagonal down-quark scalar entry, equal to g_\u03bd g_d divided by \u221a2 G_F (Q\u00b2 + m\u00b2), per conventions section 4. All vector, up-quark and off-diagonal entries are zero.\n\n**Files produced** in the task directory:\n\n- `plugin.py`: parameters g_nu, g_d, mass_mev with defaults 2e-5, 2e-5, 20.0 and both couplings listed in COUPLINGS. It sets only the S/d diagonal entries.\n- `card.json`: one scalar mediator with the same couplings, u = 0 and d = 2e-5.\n- `notes.md`: derivation, Dirac assumption as stated, reference coefficient values at the three Cs-133 Q\u00b2 points, the framework-derived nucleon charges for orientation, and the note that the scalar term does not interfere with the Standard Model.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":91264,\"stop_reason\":\"end_turn\",\"session_id\":\"a409dd41-9365-443e-81f3-a7a034b41ba3\",\"total_cost_usd\":1.07538025,\"usage\":{\"input_tokens\":322,\"cache_creation_input_tokens\":31862,\"cache_read_input_tokens\":236281,\"output_tokens\":7517,\"output_tokens_details\":{\"thinking_tokens\":913},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":31862,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":426,\"cache_read_input_tokens\":37574,\"cache_creation_input_tokens\":294,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":294},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":322,\"outputTokens\":7517,\"cacheReadInputTokens\":236281,\"cacheCreationInputTokens\":31862,\"webSearchRequests\":0,\"costUSD\":1.07538025,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":913,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_0143RjcKzsVgc1EAaTyCtd1j\",\"tool_input\":{\"command\":\"ls -R /private/tmp/nuagent-issue17-eval-claude/07_scalar_down\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":49,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings, including the plugin-versus-card cross-check on 66 momentum points.\\n\\n**Translation.** The interaction is a scalar mediator fully inside the V/S basis. Structure S, mass 20 MeV, flavor-diagonal neutrino coupling 2e-5 for e, \u03bc, \u03c4, down-quark coupling 2e-5, up-quark coupling zero. The only nonzero coefficient is the diagonal down-quark scalar entry, equal to g_\u03bd g_d divided by \u221a2 G_F (Q\u00b2 + m\u00b2), per conventions section 4. All vector, up-quark and off-diagonal entries are zero.\\n\\n**Files produced** in the task directory:\\n\\n- `plugin.py`: parameters g_nu, g_d, mass_mev with defaults 2e-5, 2e-5, 20.0 and both couplings listed in COUPLINGS. It sets only the S/d diagonal entries.\\n- `card.json`: one scalar mediator with the same couplings, u = 0 and d = 2e-5.\\n- `notes.md`: derivation, Dirac assumption as stated, reference coefficient values at the three Cs-133 Q\u00b2 points, the framework-derived nucleon charges for orientation, and the note that the scalar term does not interfere with the Standard Model.\",\"ttft_ms\":1551,\"type\":\"result\",\"duration_ms\":93339,\"uuid\":\"c2c088bb-c67a-4f28-857c-cb9eb6cb7aba\",\"ttft_stream_ms\":1436,\"time_to_request_ms\":22,\"first_content_frame_ms\":1436,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/issue-17-agent-harness/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/nuagent-issue17-eval-claude/07_scalar_down\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
