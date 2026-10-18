Verification: **plugin-only**

SM-limit, perturbativity and count checks pass. There is no independent card cross-check; normalization and momentum dependence rest on the derivation.

## Description

A universal vector mediator has mass 30 MeV and coupling g′ = 1e-4 to all left-handed neutrino flavors. Each u and d quark coupling is g′ times the dipole form factor 1/(1 + Q²/Lambda²)², with Lambda = 200 MeV. All other couplings vanish.

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
"""Universal vector mediator with a dipole form factor on the quark couplings.

Derivation and the reason there is no card.json are in notes.md.
"""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_prime": 1e-4, "mass_mev": 30.0, "lambda_mev": 200.0}
COUPLINGS = ("g_prime",)


def dipole_form_factor(q2: np.ndarray, lambda_mev: float) -> np.ndarray:
    """F(Q^2) = 1 / (1 + Q^2 / Lambda^2)^2, equal to 1 at Q^2 = 0."""
    return 1.0 / (1.0 + np.asarray(q2, dtype=float) / lambda_mev**2) ** 2


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    g_nu = params["g_prime"]
    g_q = params["g_prime"] * dipole_form_factor(q2, params["lambda_mev"])
    value = g_nu * g_q / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)
    return c

```

## notes.md

```text
# Universal vector mediator with dipole quark form factor

## Description as read

- Vector mediator Z′, mass m = 30 MeV.
- Neutrino coupling g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = g′ = 1e-4 to the left-handed current, off-diagonal entries zero.
- Quark couplings g_u(Q²) = g_d(Q²) = g′ F(Q²) with the dipole form factor
  F(Q²) = 1 / (1 + Q²/Λ²)², Λ = 200 MeV.
- All other couplings vanish: no scalar structure, no kinetic mixing, no contacts.

## Derivation

1. Lorentz structure: vector exchange, structure `V`. No scalar entries.
2. Neutrino coupling taken as written for the left-handed current (conventions section 3), so g_ν = g′ with no factor-2 conversion.
3. Quark coupling: the description gives the vector coupling to each quark directly, g_q(Q²) = g′ F(Q²). There is no chiral split, so no averaging and no dropped axial part.
4. Matching (conventions section 3) with the Q²-dependent quark coupling inserted:

       C^{uV}_{αα}(Q²) = C^{dV}_{αα}(Q²) = g′² F(Q²) / (√2 G_F (Q² + m²)),   α = e, μ, τ,

   with F(Q²) = 1/(1 + Q²/Λ²)². Off-diagonal flavor entries are zero.
5. Sign: g_ν g_q = +g′² F(Q²) > 0 for all Q² (F is positive), so the new-physics nuclear charge shift is positive while the SM charge is negative. The interference is destructive, as in example 01, but suppressed by F(Q²) at finite Q².
6. Size of the form-factor suppression at the Cs-133 reference points (Q² = 1238, 3714, 6190 MeV², Λ² = 40000 MeV²):

   | Q² (MeV²) | 1 + Q²/Λ² | F(Q²) |
   |---|---|---|
   | 1238 | 1.03095 | 0.94091 |
   | 3714 | 1.09285 | 0.83733 |
   | 6190 | 1.15475 | 0.74991 |

   The coefficients are those of example 01 multiplied by F: approximately 2.668e-01, 1.100e-01, 6.413e-02 for both quarks and all three diagonal flavors. A percent-to-tens-of-percent effect across the recoil range, which no constant card coupling can reproduce.
7. Reduction check: at Q² → 0, F → 1 and the plugin reduces exactly to the flavor-universal Z′ of example 01 (g′ = 1e-4, m = 30 MeV). At Λ → ∞ the plugin reduces to example 01 at every Q².

## Why there is no card.json

The card schema stores `quark_couplings` as constant numbers. The quark coupling here is g′ F(Q²), which varies with Q² by about 25 percent over the COHERENT recoil range, so a card with constant g_u = g_d = g′ would fail the 1e-6 cross-check against the plugin at every nonzero Q². Absorbing F(Q²) into a different constant is not possible either. This is a plugin-only translation, as in example 06.

## Plugin parameters

- `g_prime` = 1e-4 (coupling; zeroed by the SM-limit check, |g′| ≪ √(4π)).
- `mass_mev` = 30.0.
- `lambda_mev` = 200.0 (form-factor scale, not a coupling, so not listed in `COUPLINGS`).

No convention conversion was needed. No scalar, axial, tensor or kinetic-mixing contributions.

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
    "proposed counts checked for finiteness and non-negativity"
  ],
  "notes": "# Universal vector mediator with dipole quark form factor\n\n## Description as read\n\n- Vector mediator Z\u2032, mass m = 30 MeV.\n- Neutrino coupling g_\u03bd^{ee} = g_\u03bd^{mm} = g_\u03bd^{tt} = g\u2032 = 1e-4 to the left-handed current, off-diagonal entries zero.\n- Quark couplings g_u(Q\u00b2) = g_d(Q\u00b2) = g\u2032 F(Q\u00b2) with the dipole form factor\n  F(Q\u00b2) = 1 / (1 + Q\u00b2/\u039b\u00b2)\u00b2, \u039b = 200 MeV.\n- All other couplings vanish: no scalar structure, no kinetic mixing, no contacts.\n\n## Derivation\n\n1. Lorentz structure: vector exchange, structure `V`. No scalar entries.\n2. Neutrino coupling taken as written for the left-handed current (conventions section 3), so g_\u03bd = g\u2032 with no factor-2 conversion.\n3. Quark coupling: the description gives the vector coupling to each quark directly, g_q(Q\u00b2) = g\u2032 F(Q\u00b2). There is no chiral split, so no averaging and no dropped axial part.\n4. Matching (conventions section 3) with the Q\u00b2-dependent quark coupling inserted:\n\n       C^{uV}_{\u03b1\u03b1}(Q\u00b2) = C^{dV}_{\u03b1\u03b1}(Q\u00b2) = g\u2032\u00b2 F(Q\u00b2) / (\u221a2 G_F (Q\u00b2 + m\u00b2)),   \u03b1 = e, \u03bc, \u03c4,\n\n   with F(Q\u00b2) = 1/(1 + Q\u00b2/\u039b\u00b2)\u00b2. Off-diagonal flavor entries are zero.\n5. Sign: g_\u03bd g_q = +g\u2032\u00b2 F(Q\u00b2) > 0 for all Q\u00b2 (F is positive), so the new-physics nuclear charge shift is positive while the SM charge is negative. The interference is destructive, as in example 01, but suppressed by F(Q\u00b2) at finite Q\u00b2.\n6. Size of the form-factor suppression at the Cs-133 reference points (Q\u00b2 = 1238, 3714, 6190 MeV\u00b2, \u039b\u00b2 = 40000 MeV\u00b2):\n\n   | Q\u00b2 (MeV\u00b2) | 1 + Q\u00b2/\u039b\u00b2 | F(Q\u00b2) |\n   |---|---|---|\n   | 1238 | 1.03095 | 0.94091 |\n   | 3714 | 1.09285 | 0.83733 |\n   | 6190 | 1.15475 | 0.74991 |\n\n   The coefficients are those of example 01 multiplied by F: approximately 2.668e-01, 1.100e-01, 6.413e-02 for both quarks and all three diagonal flavors. A percent-to-tens-of-percent effect across the recoil range, which no constant card coupling can reproduce.\n7. Reduction check: at Q\u00b2 \u2192 0, F \u2192 1 and the plugin reduces exactly to the flavor-universal Z\u2032 of example 01 (g\u2032 = 1e-4, m = 30 MeV). At \u039b \u2192 \u221e the plugin reduces to example 01 at every Q\u00b2.\n\n## Why there is no card.json\n\nThe card schema stores `quark_couplings` as constant numbers. The quark coupling here is g\u2032 F(Q\u00b2), which varies with Q\u00b2 by about 25 percent over the COHERENT recoil range, so a card with constant g_u = g_d = g\u2032 would fail the 1e-6 cross-check against the plugin at every nonzero Q\u00b2. Absorbing F(Q\u00b2) into a different constant is not possible either. This is a plugin-only translation, as in example 06.\n\n## Plugin parameters\n\n- `g_prime` = 1e-4 (coupling; zeroed by the SM-limit check, |g\u2032| \u226a \u221a(4\u03c0)).\n- `mass_mev` = 30.0.\n- `lambda_mev` = 200.0 (form-factor scale, not a coupling, so not listed in `COUPLINGS`).\n\nNo convention conversion was needed. No scalar, axial, tensor or kinetic-mixing contributions.\n"
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
      23.093851126095586,
      19.918884144534527,
      13.74168247043217,
      6.986793784785645,
      2.5467097343637133,
      0.6644992367618617,
      0.10829156564485926,
      0.01218390305718526,
      0.06245221565184091,
      0.13065940195792555,
      0.17463843762652087,
      0.18836620813980356
    ],
    "predicted_total": 67.62901222905165,
    "observed_total": 855.0,
    "delta": 15.802982721631825,
    "new_physics": {
      "neg2_log_likelihood": 33.180420221065845,
      "alpha": -0.21682778285884619,
      "beta": {
        "steady_state": 0.0654983516851931,
        "prompt_neutrons": 0.03078738982339799
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
  "final_text": "Validation passed with no findings.\n\n**Files produced**\n\n- `plugin.py`: vector coefficients for all three diagonal neutrino flavors and both quarks, equal to g\u2032\u00b2 F(Q\u00b2) / (\u221a2 G_F (Q\u00b2 + m\u00b2)) with the dipole form factor F(Q\u00b2) = 1/(1 + Q\u00b2/\u039b\u00b2)\u00b2. Defaults are g\u2032 = 1e-4, m = 30 MeV, \u039b = 200 MeV, and only g\u2032 is listed as a coupling.\n- `notes.md`: the derivation, the sign of the interference (destructive, as in the universal Z\u2032 example), the form-factor values at the Cs-133 reference points, and the reduction checks (Q\u00b2 \u2192 0 and \u039b \u2192 \u221e both recover example 01).\n\n**No card.json.** The schema only holds constant quark couplings, and the dipole form factor changes the quark coupling by roughly 25 percent across the COHERENT recoil range, so no card can match the plugin to the required tolerance. This is a plugin-only translation.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":65860,\"stop_reason\":\"end_turn\",\"session_id\":\"83cc06a5-c6e5-434c-b135-68203054d980\",\"total_cost_usd\":0.84682825,\"usage\":{\"input_tokens\":258,\"cache_creation_input_tokens\":26153,\"cache_read_input_tokens\":177553,\"output_tokens\":5536,\"output_tokens_details\":{\"thinking_tokens\":311},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":26153,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":338,\"cache_read_input_tokens\":31950,\"cache_creation_input_tokens\":209,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":209},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":258,\"outputTokens\":5536,\"cacheReadInputTokens\":177553,\"cacheCreationInputTokens\":26153,\"webSearchRequests\":0,\"costUSD\":0.84682825,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":311,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_0168JQQyqd1nPfVvxB4APge9\",\"tool_input\":{\"command\":\"ls -R /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/13_dipole_quark_coupling\",\"description\":\"List task directory contents\"}},{\"tool_name\":\"Read\",\"tool_use_id\":\"toolu_01W5UBiA1MqBq6BJcTAzdU4T\",\"tool_input\":{\"file_path\":\"/private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-5c51979abec1/scratchpad/live/eval-claude/13_dipole_quark_coupling/context/examples/06_lmu_ltau_running.md\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":22,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings.\\n\\n**Files produced**\\n\\n- `plugin.py`: vector coefficients for all three diagonal neutrino flavors and both quarks, equal to g\u2032\u00b2 F(Q\u00b2) / (\u221a2 G_F (Q\u00b2 + m\u00b2)) with the dipole form factor F(Q\u00b2) = 1/(1 + Q\u00b2/\u039b\u00b2)\u00b2. Defaults are g\u2032 = 1e-4, m = 30 MeV, \u039b = 200 MeV, and only g\u2032 is listed as a coupling.\\n- `notes.md`: the derivation, the sign of the interference (destructive, as in the universal Z\u2032 example), the form-factor values at the Cs-133 reference points, and the reduction checks (Q\u00b2 \u2192 0 and \u039b \u2192 \u221e both recover example 01).\\n\\n**No card.json.** The schema only holds constant quark couplings, and the dipole form factor changes the quark coupling by roughly 25 percent across the COHERENT recoil range, so no card can match the plugin to the required tolerance. This is a plugin-only translation.\",\"ttft_ms\":820,\"type\":\"result\",\"duration_ms\":69296,\"uuid\":\"58a635f2-48a5-44a0-aa13-8359e709c329\",\"ttft_stream_ms\":730,\"time_to_request_ms\":26,\"first_content_frame_ms\":730,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/m7-conversation/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/13_dipole_quark_coupling\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
