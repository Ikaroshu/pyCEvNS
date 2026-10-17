Verification: **plugin-only**

SM-limit, perturbativity and count checks pass. There is no independent card cross-check; normalization and momentum dependence rest on the derivation.

## Description

A universal vector mediator has mass 30 MeV and coupling g′ = 1e-4 to all left-handed neutrino flavors. Each u and d quark coupling is g′ times the dipole form factor 1/(1 + Q²/Lambda²)², with Lambda = 200 MeV. All other couplings vanish.

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
"""Universal vector mediator with dipole-form-factor quark couplings (plugin-only).

Derivation and the reason no card exists are in notes.md.
"""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_prime": 1e-4, "mass_mev": 30.0, "lambda_mev": 200.0}
COUPLINGS = ("g_prime",)


def dipole_form_factor(q2: np.ndarray, lambda_mev: float) -> np.ndarray:
    """F(Q^2) = 1 / (1 + Q^2 / Lambda^2)^2, equal to 1 at Q^2 = 0."""
    return 1.0 / (1.0 + np.asarray(q2, dtype=float) / lambda_mev**2) ** 2


def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    g_nu = g
    g_q = g * dipole_form_factor(q2, params["lambda_mev"])
    value = g_nu * g_q / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)
    return c

```

## notes.md

```text
# Universal vector mediator with dipole-form-factor quark couplings

Description: a vector mediator of mass m = 30 MeV couples with g′ = 1e-4 to all three left-handed neutrino flavors, and to u and d quarks with g′ times a dipole form factor F(Q²) = 1/(1 + Q²/Λ²)², Λ = 200 MeV. All other couplings vanish.

## Derivation

1. Lorentz structure: vector exchange, structure `V`. No scalar, axial or tensor piece.
2. Neutrino couplings as written: g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = g′ = 1e-4, left-handed current, off-diagonal entries zero. No convention conversion needed.
3. Quark couplings as written: g_u(Q²) = g_d(Q²) = g′ F(Q²) with F(Q²) = 1/(1 + Q²/Λ²)². F is a pure vector coupling (no chiral split), so no axial part is dropped. F(0) = 1 and F decreases monotonically with Q²; Q² is spacelike and non-negative, so F is never singular.
4. Matching (conventions section 3), with the Q²-dependent quark coupling inserted:

       C^{uV}_{αα}(Q²) = C^{dV}_{αα}(Q²) = g′² F(Q²) / ( √2 G_F (Q² + m²) ),   α = e, μ, τ.

   All off-diagonal and all scalar entries are zero.
5. Sign: g_ν g_q = g′² F > 0, so the new-physics nuclear charge shift ΔQ = (3Z + 3N) C is positive while Q_SM is negative. The interference is destructive, as in example 01. The form factor only reduces the magnitude of ΔQ with Q² and does not change the sign.
6. Reduction check: at Q² → 0 the plugin equals the universal Z′ of example 01 (F = 1). At the reference points the coefficients are those of example 01 multiplied by F(Q²):

   | Q² (MeV²) | F(Q²) | C^{uV} = C^{dV} (all diagonal flavors), approximate |
   |---|---|---|
   | 1238 | 0.9409 | 2.668e-01 |
   | 3714 | 0.8373 | 1.100e-01 |
   | 6190 | 0.7499 | 6.412e-02 |

   These are hand-rounded; the plugin computes them from the exported constants at full precision.
7. Perturbativity: |g′| = 1e-4 ≪ √(4π). The mass is positive. Λ is a scale parameter, not a coupling, so it is in `PARAMETERS` but not in `COUPLINGS`; with g′ = 0 every coefficient vanishes and the Standard Model limit holds.

## Why no card.json

The card schema describes a mediator by a mass and constant quark couplings (optionally plus a constant kinetic mixing). The description's quark couplings carry the momentum dependence F(Q²) = 1/(1 + Q²/Λ²)², which varies by about 25 percent over the COHERENT recoil range. No constant g_u, g_d can reproduce this, and the contact entries are also constants. This is therefore a plugin-only translation, analogous to example 06; the expected report label is `plugin-only`.

## Parameters

- `g_prime` = 1e-4 (coupling; in `COUPLINGS`)
- `mass_mev` = 30.0
- `lambda_mev` = 200.0 (dipole scale Λ)

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
    "proposed counts checked for finiteness and non-negativity"
  ],
  "notes": "# Universal vector mediator with dipole-form-factor quark couplings\n\nDescription: a vector mediator of mass m = 30 MeV couples with g\u2032 = 1e-4 to all three left-handed neutrino flavors, and to u and d quarks with g\u2032 times a dipole form factor F(Q\u00b2) = 1/(1 + Q\u00b2/\u039b\u00b2)\u00b2, \u039b = 200 MeV. All other couplings vanish.\n\n## Derivation\n\n1. Lorentz structure: vector exchange, structure `V`. No scalar, axial or tensor piece.\n2. Neutrino couplings as written: g_\u03bd^{ee} = g_\u03bd^{mm} = g_\u03bd^{tt} = g\u2032 = 1e-4, left-handed current, off-diagonal entries zero. No convention conversion needed.\n3. Quark couplings as written: g_u(Q\u00b2) = g_d(Q\u00b2) = g\u2032 F(Q\u00b2) with F(Q\u00b2) = 1/(1 + Q\u00b2/\u039b\u00b2)\u00b2. F is a pure vector coupling (no chiral split), so no axial part is dropped. F(0) = 1 and F decreases monotonically with Q\u00b2; Q\u00b2 is spacelike and non-negative, so F is never singular.\n4. Matching (conventions section 3), with the Q\u00b2-dependent quark coupling inserted:\n\n       C^{uV}_{\u03b1\u03b1}(Q\u00b2) = C^{dV}_{\u03b1\u03b1}(Q\u00b2) = g\u2032\u00b2 F(Q\u00b2) / ( \u221a2 G_F (Q\u00b2 + m\u00b2) ),   \u03b1 = e, \u03bc, \u03c4.\n\n   All off-diagonal and all scalar entries are zero.\n5. Sign: g_\u03bd g_q = g\u2032\u00b2 F > 0, so the new-physics nuclear charge shift \u0394Q = (3Z + 3N) C is positive while Q_SM is negative. The interference is destructive, as in example 01. The form factor only reduces the magnitude of \u0394Q with Q\u00b2 and does not change the sign.\n6. Reduction check: at Q\u00b2 \u2192 0 the plugin equals the universal Z\u2032 of example 01 (F = 1). At the reference points the coefficients are those of example 01 multiplied by F(Q\u00b2):\n\n   | Q\u00b2 (MeV\u00b2) | F(Q\u00b2) | C^{uV} = C^{dV} (all diagonal flavors), approximate |\n   |---|---|---|\n   | 1238 | 0.9409 | 2.668e-01 |\n   | 3714 | 0.8373 | 1.100e-01 |\n   | 6190 | 0.7499 | 6.412e-02 |\n\n   These are hand-rounded; the plugin computes them from the exported constants at full precision.\n7. Perturbativity: |g\u2032| = 1e-4 \u226a \u221a(4\u03c0). The mass is positive. \u039b is a scale parameter, not a coupling, so it is in `PARAMETERS` but not in `COUPLINGS`; with g\u2032 = 0 every coefficient vanishes and the Standard Model limit holds.\n\n## Why no card.json\n\nThe card schema describes a mediator by a mass and constant quark couplings (optionally plus a constant kinetic mixing). The description's quark couplings carry the momentum dependence F(Q\u00b2) = 1/(1 + Q\u00b2/\u039b\u00b2)\u00b2, which varies by about 25 percent over the COHERENT recoil range. No constant g_u, g_d can reproduce this, and the contact entries are also constants. This is therefore a plugin-only translation, analogous to example 06; the expected report label is `plugin-only`.\n\n## Parameters\n\n- `g_prime` = 1e-4 (coupling; in `COUPLINGS`)\n- `mass_mev` = 30.0\n- `lambda_mev` = 200.0 (dipole scale \u039b)\n"
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

### Attempt 1

```json
{
  "final_text": "Validation passed with no findings.\n\n**Outcome.** This is a plugin-only translation. The dipole form factor makes the u and d couplings depend on Q\u00b2, and the card schema only carries constant quark couplings, so no card.json was written.\n\n**Files produced**\n\n- `plugin.py`: vector coefficients C^{uV} = C^{dV} = g\u2032\u00b2 F(Q\u00b2) / (\u221a2 G_F (Q\u00b2 + m\u00b2)) for the three diagonal flavors, with F(Q\u00b2) = 1/(1 + Q\u00b2/\u039b\u00b2)\u00b2. Defaults are g\u2032 = 1e-4, m = 30 MeV, \u039b = 200 MeV. Only g\u2032 is listed as a coupling, so \u039b is not zeroed in the Standard Model-limit check.\n- `notes.md`: the derivation, the sign of the interference (destructive, as in the universal Z\u2032 example), the Q\u00b2 \u2192 0 reduction to example 01, approximate values at the three reference Q\u00b2 points, and the reason no card can express the interaction.\n\n**Validation.** The plugin passed the perturbativity, Standard Model-limit and finite-count checks. The expected harness label is plugin-only, since no card cross-check exists.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":58233,\"stop_reason\":\"end_turn\",\"session_id\":\"66a65646-8181-4aac-938b-991787b3e698\",\"total_cost_usd\":0.7357447500000001,\"usage\":{\"input_tokens\":258,\"cache_creation_input_tokens\":22506,\"cache_read_input_tokens\":164379,\"output_tokens\":4839,\"output_tokens_details\":{\"thinking_tokens\":821},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":22506,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":386,\"cache_read_input_tokens\":28303,\"cache_creation_input_tokens\":209,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":209},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":258,\"outputTokens\":4839,\"cacheReadInputTokens\":164379,\"cacheCreationInputTokens\":22506,\"webSearchRequests\":0,\"costUSD\":0.7357447500000001,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":821,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01HHZwFcNbcTvcgipPgYAEBd\",\"tool_input\":{\"command\":\"ls -R /private/tmp/nuagent-issue17-eval-claude/13_dipole_quark_coupling\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":20,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings.\\n\\n**Outcome.** This is a plugin-only translation. The dipole form factor makes the u and d couplings depend on Q\u00b2, and the card schema only carries constant quark couplings, so no card.json was written.\\n\\n**Files produced**\\n\\n- `plugin.py`: vector coefficients C^{uV} = C^{dV} = g\u2032\u00b2 F(Q\u00b2) / (\u221a2 G_F (Q\u00b2 + m\u00b2)) for the three diagonal flavors, with F(Q\u00b2) = 1/(1 + Q\u00b2/\u039b\u00b2)\u00b2. Defaults are g\u2032 = 1e-4, m = 30 MeV, \u039b = 200 MeV. Only g\u2032 is listed as a coupling, so \u039b is not zeroed in the Standard Model-limit check.\\n- `notes.md`: the derivation, the sign of the interference (destructive, as in the universal Z\u2032 example), the Q\u00b2 \u2192 0 reduction to example 01, approximate values at the three reference Q\u00b2 points, and the reason no card can express the interaction.\\n\\n**Validation.** The plugin passed the perturbativity, Standard Model-limit and finite-count checks. The expected harness label is plugin-only, since no card cross-check exists.\",\"ttft_ms\":1254,\"type\":\"result\",\"duration_ms\":59779,\"uuid\":\"6c087521-1a25-421f-9802-efb0c8fe79f1\",\"ttft_stream_ms\":1165,\"time_to_request_ms\":19,\"first_content_frame_ms\":1165,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/issue-17-agent-harness/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/nuagent-issue17-eval-claude/13_dipole_quark_coupling\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
