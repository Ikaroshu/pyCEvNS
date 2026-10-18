Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

A scalar phi of mass 20 MeV has Dirac couplings 2e-5 phi nu-bar_alpha nu_alpha to all three neutrino flavors and 2e-5 phi d-bar d to down quarks only. All other couplings vanish.

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
"""Flavor-universal scalar mediator coupling to down quarks only; derivation in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_nu": 2e-5, "g_d": 2e-5, "mass_mev": 20.0}
COUPLINGS = ("g_nu", "g_d")


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    value = params["g_nu"] * params["g_d"] / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for pair in ("ee", "mm", "tt"):
        c.set("S", "d", pair, value)
    return c

```

## card.json

```json
{
  "name": "scalar_down_quark_universal_nu",
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
# Scalar φ, mass 20 MeV, flavor-universal Dirac neutrino coupling, down quarks only

## Description as given

A scalar φ of mass 20 MeV with Dirac couplings 2e-5 φ ν̄_α ν_α for α = e, μ, τ and
2e-5 φ d̄ d. All other couplings (up quark, off-diagonal neutrino flavors, charged
leptons) vanish.

## Derivation

1. **Lorentz structure.** φ ν̄ ν and φ d̄ d are both scalar bilinears, so the
   structure is `S`. No vector, axial, pseudoscalar or tensor piece appears.

2. **Lagrangian in the framework's form** (conventions section 4):

       L ⊃ φ ( g_ν^{αβ} ν̄_α ν_β + Σ_q g_q q̄ q ),   m = 20 MeV,

   with g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = 2e-5, all off-diagonal g_ν = 0,
   g_d = 2e-5 and g_u = 0. The description is already in this normalization,
   so no convention conversion is needed.

3. **Dirac versus Majorana.** The description says the coupling is Dirac:
   ν̄_α ν_α = ν̄_L ν_R + ν̄_R ν_L requires a right-handed Dirac partner, which
   is what makes the helicity-flipping scalar amplitude possible. For a Dirac
   coupling written as g_ν φ ν̄ ν the vertex is g_ν as written; the factor-2
   subtlety of section 4 applies only to a Majorana mass-type term written
   without the 1/2, which is not the case here.

4. **Matching** (conventions section 4, tree-level t-channel φ exchange):

       C^{dS}_{αα}(Q²) = g_ν g_d / ( √2 G_F (Q² + m²) ),   α = e, μ, τ,
       C^{uS}_{αβ}(Q²) = 0,
       C^{dS}_{αβ}(Q²) = 0 for α ≠ β.

   Numerically g_ν g_d = 4e-10 and m² = 400 MeV². All vector coefficients are
   zero, so there is no new-physics interference with the Standard Model; the
   scalar rate adds incoherently with its own kinematic factor, supplied by
   the framework.

5. **Reference values** (Cs-133, Q² = 2 M T at T = 5, 15, 25 keV; rounded):

   | coefficient | Q² = 1238 | Q² = 3714 | Q² = 6190 |
   |---|---|---|---|
   | C^{dS}_ee = C^{dS}_mm = C^{dS}_tt | 1.480e-02 | 5.894e-03 | 3.680e-03 |
   | C^{uS} (all flavors) | 0 | 0 | 0 |

   The plugin computes these at full precision from the exported `GF` and
   `SQRT2`.

6. **Nuclear scalar charge** (for orientation only; the framework forms it).
   With only g_d non-zero, c_N = m_N g_d f_d^N / m_d, giving
   c_p ≈ 8.2 g_d and c_n ≈ 9.0 g_d, i.e. the nucleus couples as
   roughly (8.2 Z + 9.0 N) g_d, about half the "17 A g_q" of a
   u+d-universal scalar because the up-quark term is absent.

7. **Perturbativity.** |g_ν| = |g_d| = 2e-5 ≪ √(4π). Mass positive.

8. **Card.** Fully expressible: one `S` mediator with diagonal universal
   neutrino couplings, `quark_couplings` u = 0, d = 2e-5. No kinetic mixing
   (vector-only field, omitted). The plugin and card are independent
   representations of the same matching formula.

## Parameters

`PARAMETERS = {"g_nu": 2e-5, "g_d": 2e-5, "mass_mev": 20.0}`,
`COUPLINGS = ("g_nu", "g_d")`. Setting both couplings to zero returns the
Standard Model. The described point is preserved exactly.

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
  "notes": "# Scalar \u03c6, mass 20 MeV, flavor-universal Dirac neutrino coupling, down quarks only\n\n## Description as given\n\nA scalar \u03c6 of mass 20 MeV with Dirac couplings 2e-5 \u03c6 \u03bd\u0304_\u03b1 \u03bd_\u03b1 for \u03b1 = e, \u03bc, \u03c4 and\n2e-5 \u03c6 d\u0304 d. All other couplings (up quark, off-diagonal neutrino flavors, charged\nleptons) vanish.\n\n## Derivation\n\n1. **Lorentz structure.** \u03c6 \u03bd\u0304 \u03bd and \u03c6 d\u0304 d are both scalar bilinears, so the\n   structure is `S`. No vector, axial, pseudoscalar or tensor piece appears.\n\n2. **Lagrangian in the framework's form** (conventions section 4):\n\n       L \u2283 \u03c6 ( g_\u03bd^{\u03b1\u03b2} \u03bd\u0304_\u03b1 \u03bd_\u03b2 + \u03a3_q g_q q\u0304 q ),   m = 20 MeV,\n\n   with g_\u03bd^{ee} = g_\u03bd^{mm} = g_\u03bd^{tt} = 2e-5, all off-diagonal g_\u03bd = 0,\n   g_d = 2e-5 and g_u = 0. The description is already in this normalization,\n   so no convention conversion is needed.\n\n3. **Dirac versus Majorana.** The description says the coupling is Dirac:\n   \u03bd\u0304_\u03b1 \u03bd_\u03b1 = \u03bd\u0304_L \u03bd_R + \u03bd\u0304_R \u03bd_L requires a right-handed Dirac partner, which\n   is what makes the helicity-flipping scalar amplitude possible. For a Dirac\n   coupling written as g_\u03bd \u03c6 \u03bd\u0304 \u03bd the vertex is g_\u03bd as written; the factor-2\n   subtlety of section 4 applies only to a Majorana mass-type term written\n   without the 1/2, which is not the case here.\n\n4. **Matching** (conventions section 4, tree-level t-channel \u03c6 exchange):\n\n       C^{dS}_{\u03b1\u03b1}(Q\u00b2) = g_\u03bd g_d / ( \u221a2 G_F (Q\u00b2 + m\u00b2) ),   \u03b1 = e, \u03bc, \u03c4,\n       C^{uS}_{\u03b1\u03b2}(Q\u00b2) = 0,\n       C^{dS}_{\u03b1\u03b2}(Q\u00b2) = 0 for \u03b1 \u2260 \u03b2.\n\n   Numerically g_\u03bd g_d = 4e-10 and m\u00b2 = 400 MeV\u00b2. All vector coefficients are\n   zero, so there is no new-physics interference with the Standard Model; the\n   scalar rate adds incoherently with its own kinematic factor, supplied by\n   the framework.\n\n5. **Reference values** (Cs-133, Q\u00b2 = 2 M T at T = 5, 15, 25 keV; rounded):\n\n   | coefficient | Q\u00b2 = 1238 | Q\u00b2 = 3714 | Q\u00b2 = 6190 |\n   |---|---|---|---|\n   | C^{dS}_ee = C^{dS}_mm = C^{dS}_tt | 1.480e-02 | 5.894e-03 | 3.680e-03 |\n   | C^{uS} (all flavors) | 0 | 0 | 0 |\n\n   The plugin computes these at full precision from the exported `GF` and\n   `SQRT2`.\n\n6. **Nuclear scalar charge** (for orientation only; the framework forms it).\n   With only g_d non-zero, c_N = m_N g_d f_d^N / m_d, giving\n   c_p \u2248 8.2 g_d and c_n \u2248 9.0 g_d, i.e. the nucleus couples as\n   roughly (8.2 Z + 9.0 N) g_d, about half the \"17 A g_q\" of a\n   u+d-universal scalar because the up-quark term is absent.\n\n7. **Perturbativity.** |g_\u03bd| = |g_d| = 2e-5 \u226a \u221a(4\u03c0). Mass positive.\n\n8. **Card.** Fully expressible: one `S` mediator with diagonal universal\n   neutrino couplings, `quark_couplings` u = 0, d = 2e-5. No kinetic mixing\n   (vector-only field, omitted). The plugin and card are independent\n   representations of the same matching formula.\n\n## Parameters\n\n`PARAMETERS = {\"g_nu\": 2e-5, \"g_d\": 2e-5, \"mass_mev\": 20.0}`,\n`COUPLINGS = (\"g_nu\", \"g_d\")`. Setting both couplings to zero returns the\nStandard Model. The described point is preserved exactly.\n"
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

### Round 1, attempt 1

```json
{
  "final_text": "Validation passed with no findings, and the plugin and card agree on all 66 momentum-transfer points.\n\n**Files produced** in the task directory:\n\n- **plugin.py** sets the scalar down-quark coefficient for the three diagonal flavors to g_nu times g_d over \u221a2 G_F (Q\u00b2 + m\u00b2), with defaults g_nu = 2e-5, g_d = 2e-5, mass 20 MeV. Both couplings are listed in COUPLINGS. Up-quark and off-diagonal entries stay zero.\n- **card.json** holds one scalar mediator with universal diagonal neutrino couplings, up coupling 0 and down coupling 2e-5.\n- **notes.md** records the derivation: structure S, Dirac \u03bd\u0304\u03bd vertex taken as written with no factor-2 adjustment, matching from conventions section 4, reference coefficient values at the three Cs-133 Q\u00b2 points, the reduced nuclear scalar charge from the missing up-quark term, and a perturbativity check.\n\nThe scalar amplitude flips helicity, so this new physics adds incoherently to the Standard Model rate rather than interfering with it. One caveat: the worked examples directory could not be listed or opened by guessed names, so the translation rests on the normative conventions document rather than on a scalar worked example.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":106440,\"stop_reason\":\"end_turn\",\"session_id\":\"494037c7-71b1-4088-aed8-43d9d8660ad5\",\"total_cost_usd\":0.997442,\"usage\":{\"input_tokens\":258,\"cache_creation_input_tokens\":27752,\"cache_read_input_tokens\":183088,\"output_tokens\":7881,\"output_tokens_details\":{\"thinking_tokens\":1082},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":27752,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":430,\"cache_read_input_tokens\":33461,\"cache_creation_input_tokens\":297,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":297},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":258,\"outputTokens\":7881,\"cacheReadInputTokens\":183088,\"cacheCreationInputTokens\":27752,\"webSearchRequests\":0,\"costUSD\":0.997442,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":1082,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01De3JSyzRzRi6gofbf6TWbH\",\"tool_input\":{\"command\":\"ls -R /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/07_scalar_down\",\"description\":\"List task directory contents\"}},{\"tool_name\":\"Read\",\"tool_use_id\":\"toolu_01Xu7NVa6eJPBCz4QX1gq5r8\",\"tool_input\":{\"file_path\":\"/private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-5c51979abec1/scratchpad/live/eval-claude/07_scalar_down/context/examples/03_scalar_nucleon.md\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":32,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings, and the plugin and card agree on all 66 momentum-transfer points.\\n\\n**Files produced** in the task directory:\\n\\n- **plugin.py** sets the scalar down-quark coefficient for the three diagonal flavors to g_nu times g_d over \u221a2 G_F (Q\u00b2 + m\u00b2), with defaults g_nu = 2e-5, g_d = 2e-5, mass 20 MeV. Both couplings are listed in COUPLINGS. Up-quark and off-diagonal entries stay zero.\\n- **card.json** holds one scalar mediator with universal diagonal neutrino couplings, up coupling 0 and down coupling 2e-5.\\n- **notes.md** records the derivation: structure S, Dirac \u03bd\u0304\u03bd vertex taken as written with no factor-2 adjustment, matching from conventions section 4, reference coefficient values at the three Cs-133 Q\u00b2 points, the reduced nuclear scalar charge from the missing up-quark term, and a perturbativity check.\\n\\nThe scalar amplitude flips helicity, so this new physics adds incoherently to the Standard Model rate rather than interfering with it. One caveat: the worked examples directory could not be listed or opened by guessed names, so the translation rests on the normative conventions document rather than on a scalar worked example.\",\"ttft_ms\":2398,\"type\":\"result\",\"duration_ms\":110280,\"uuid\":\"b7e4839c-2770-4d38-83ab-27a6a55422b4\",\"ttft_stream_ms\":2281,\"time_to_request_ms\":22,\"first_content_frame_ms\":2282,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/m7-conversation/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/07_scalar_down\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
