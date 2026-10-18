Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

a gauged L_mu minus L_tau Z prime with mass 10 MeV, gauge coupling 1e-4 and kinetic mixing 1e-3

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
"""Gauged L_mu - L_tau Z' with explicit kinetic mixing; derivation in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2, E_CHARGE, Q_EM

PARAMETERS = {"g_prime": 1e-4, "mass_mev": 10.0, "kinetic_mixing": 1e-3}
COUPLINGS = ("g_prime", "kinetic_mixing")

# L_mu - L_tau charges of the lepton doublets; quarks carry no charge.
NU_CHARGES = {"ee": 0.0, "mm": +1.0, "tt": -1.0}


def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    eps = params["kinetic_mixing"]
    prop = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        g_q = eps * E_CHARGE * Q_EM[quark]
        for pair, charge in NU_CHARGES.items():
            g_nu = g * charge
            c.set("V", quark, pair, g_nu * g_q * prop)
    return c

```

## card.json

```json
{
  "name": "zprime_lmu_ltau_m10_g1e-4_eps1e-3",
  "mediators": [
    {
      "structure": "V",
      "mass_mev": 10.0,
      "nu_couplings": {
        "ee": 0.0,
        "mm": 0.0001,
        "tt": -0.0001,
        "em": 0,
        "et": 0,
        "mt": 0
      },
      "quark_couplings": {
        "u": 0.0,
        "d": 0.0
      },
      "kinetic_mixing": 0.001
    }
  ],
  "contacts": []
}

```

## notes.md

```text
# Gauged L_μ−L_τ Z′, m = 10 MeV, g′ = 1e-4, kinetic mixing ε = 1e-3

## Inputs taken from the description

- Mediator: a gauge boson of U(1)_{L_μ−L_τ}, so the Lorentz structure is vector (`V`).
- Mass m = 10 MeV.
- Gauge coupling g′ = 1e-4.
- Kinetic mixing with the photon ε = 1e-3, given explicitly.

All three inputs are fixed by the description, so no questions are needed.

## Derivation

1. **Charges** (conventions section 7, L_μ−L_τ row): Q′_{ν_e} = 0, Q′_{ν_μ} = +1, Q′_{ν_τ} = −1, Q′_u = Q′_d = 0.
   The neutrino coupling is the left-handed doublet coupling, g_ν^{αα} = g′ Q′_{ν_α}:
   g_ν^{ee} = 0, g_ν^{mm} = +1e-4, g_ν^{tt} = −1e-4, all off-diagonal entries zero.
2. **Quark couplings** come only from kinetic mixing (conventions section 6):
   g_q = 0 + ε e Q_q^em, with e = 0.303, Q_u^em = +2/3, Q_d^em = −1/3, so
   g_u = +2.02e-4, g_d = −1.01e-4 (using e = E_CHARGE at full precision).
3. **Matching** (conventions section 3):
   C^{qV}_{αβ}(Q²) = g_ν^{αβ} g_q / (√2 G_F (Q² + m²)).
   Hence
   C^{uV}_{mm} = +g′ ε e (2/3) / (√2 G_F (Q² + m²)),
   C^{dV}_{mm} = −g′ ε e (1/3) / (√2 G_F (Q² + m²)),
   C^{qV}_{tt} = −C^{qV}_{mm}, and all ee and off-diagonal entries vanish.
   Scalar entries are zero.
4. **Sign of ε.** The description gives ε = +1e-3 with no Lagrangian convention, so it is taken as the framework's `kinetic_mixing` convention: the Z′ acquires +ε e Q f̄ γ^μ f, i.e. the same sign as the loop-induced mixing of section 6 (ε = +κ g′). With ε g′ > 0 the ν_μ nuclear-charge shift ΔQ = Z g′ ε e / (√2 G_F (Q² + m²)) is positive while Q_SM is negative, so the interference is destructive for ν_μ (and ν̄_μ) and constructive for ν_τ. If the user's convention has the opposite sign of ε, flip `kinetic_mixing` to −1e-3; the magnitude of the coefficients is unchanged.
5. **Loop-induced mixing is not added.** The explicit ε = 1e-3 is treated as the total low-energy photon mixing, as in worked example 03 A. The μ/τ-loop contribution would be κ g′ ≈ 1.44e-6, about 0.14 percent of the given value, and adding it on top of an explicitly specified ε would double-count if the stated ε already represents the full mixing. The Q² dependence of the loop piece (example 06) is likewise negligible at this level.
6. **Physics check.** The mixing couples the Z′ to electric charge, so only protons are affected: 2 g_u + g_d = ε e and g_u + 2 g_d = 0. Electron neutrinos see no new physics; the COHERENT prompt ν_μ and delayed ν̄_μ fluxes do.
7. **Perturbativity.** |g′| = 1e-4 and |ε| = 1e-3 are far below √(4π).
8. **Anomalies.** L_μ−L_τ is anomaly-free with the SM fermion content.

## Numerical values at the Cs-133 reference points (Q² = 1238, 3714, 6190 MeV²)

Prefactor g′ ε e (2/3) / (√2 G_F) = 1e-4 × 1e-3 × 0.303 × (2/3) / (1.64951e-11 MeV⁻²) ≈ 1.2246e+03 MeV².
With m² = 100 MeV²:

| coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
|---|---|---|---|
| C^{uV}_mm = −C^{uV}_tt | ≈ 9.15e-01 | ≈ 3.21e-01 | ≈ 1.95e-01 |
| C^{dV}_mm = −C^{dV}_tt | ≈ −4.58e-01 | ≈ −1.60e-01 | ≈ −9.74e-02 |

(Rounded for reading; the plugin uses the exported constants at full precision.)

## Card

Expressible: one vector mediator with `nu_couplings` mm = +1e-4, tt = −1e-4, zero direct `quark_couplings`, and `kinetic_mixing` = 1e-3 written explicitly (a card has no loop default). `card.json` is provided.

## Plugin

`plugin.py` computes the same coefficients from `GF`, `SQRT2`, `E_CHARGE` and `Q_EM` exported by `nuagent.physics.couplings`. `COUPLINGS = ("g_prime", "kinetic_mixing")`; zeroing either removes all new physics, so the SM-limit check passes.

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
  "notes": "# Gauged L_\u03bc\u2212L_\u03c4 Z\u2032, m = 10 MeV, g\u2032 = 1e-4, kinetic mixing \u03b5 = 1e-3\n\n## Inputs taken from the description\n\n- Mediator: a gauge boson of U(1)_{L_\u03bc\u2212L_\u03c4}, so the Lorentz structure is vector (`V`).\n- Mass m = 10 MeV.\n- Gauge coupling g\u2032 = 1e-4.\n- Kinetic mixing with the photon \u03b5 = 1e-3, given explicitly.\n\nAll three inputs are fixed by the description, so no questions are needed.\n\n## Derivation\n\n1. **Charges** (conventions section 7, L_\u03bc\u2212L_\u03c4 row): Q\u2032_{\u03bd_e} = 0, Q\u2032_{\u03bd_\u03bc} = +1, Q\u2032_{\u03bd_\u03c4} = \u22121, Q\u2032_u = Q\u2032_d = 0.\n   The neutrino coupling is the left-handed doublet coupling, g_\u03bd^{\u03b1\u03b1} = g\u2032 Q\u2032_{\u03bd_\u03b1}:\n   g_\u03bd^{ee} = 0, g_\u03bd^{mm} = +1e-4, g_\u03bd^{tt} = \u22121e-4, all off-diagonal entries zero.\n2. **Quark couplings** come only from kinetic mixing (conventions section 6):\n   g_q = 0 + \u03b5 e Q_q^em, with e = 0.303, Q_u^em = +2/3, Q_d^em = \u22121/3, so\n   g_u = +2.02e-4, g_d = \u22121.01e-4 (using e = E_CHARGE at full precision).\n3. **Matching** (conventions section 3):\n   C^{qV}_{\u03b1\u03b2}(Q\u00b2) = g_\u03bd^{\u03b1\u03b2} g_q / (\u221a2 G_F (Q\u00b2 + m\u00b2)).\n   Hence\n   C^{uV}_{mm} = +g\u2032 \u03b5 e (2/3) / (\u221a2 G_F (Q\u00b2 + m\u00b2)),\n   C^{dV}_{mm} = \u2212g\u2032 \u03b5 e (1/3) / (\u221a2 G_F (Q\u00b2 + m\u00b2)),\n   C^{qV}_{tt} = \u2212C^{qV}_{mm}, and all ee and off-diagonal entries vanish.\n   Scalar entries are zero.\n4. **Sign of \u03b5.** The description gives \u03b5 = +1e-3 with no Lagrangian convention, so it is taken as the framework's `kinetic_mixing` convention: the Z\u2032 acquires +\u03b5 e Q f\u0304 \u03b3^\u03bc f, i.e. the same sign as the loop-induced mixing of section 6 (\u03b5 = +\u03ba g\u2032). With \u03b5 g\u2032 > 0 the \u03bd_\u03bc nuclear-charge shift \u0394Q = Z g\u2032 \u03b5 e / (\u221a2 G_F (Q\u00b2 + m\u00b2)) is positive while Q_SM is negative, so the interference is destructive for \u03bd_\u03bc (and \u03bd\u0304_\u03bc) and constructive for \u03bd_\u03c4. If the user's convention has the opposite sign of \u03b5, flip `kinetic_mixing` to \u22121e-3; the magnitude of the coefficients is unchanged.\n5. **Loop-induced mixing is not added.** The explicit \u03b5 = 1e-3 is treated as the total low-energy photon mixing, as in worked example 03 A. The \u03bc/\u03c4-loop contribution would be \u03ba g\u2032 \u2248 1.44e-6, about 0.14 percent of the given value, and adding it on top of an explicitly specified \u03b5 would double-count if the stated \u03b5 already represents the full mixing. The Q\u00b2 dependence of the loop piece (example 06) is likewise negligible at this level.\n6. **Physics check.** The mixing couples the Z\u2032 to electric charge, so only protons are affected: 2 g_u + g_d = \u03b5 e and g_u + 2 g_d = 0. Electron neutrinos see no new physics; the COHERENT prompt \u03bd_\u03bc and delayed \u03bd\u0304_\u03bc fluxes do.\n7. **Perturbativity.** |g\u2032| = 1e-4 and |\u03b5| = 1e-3 are far below \u221a(4\u03c0).\n8. **Anomalies.** L_\u03bc\u2212L_\u03c4 is anomaly-free with the SM fermion content.\n\n## Numerical values at the Cs-133 reference points (Q\u00b2 = 1238, 3714, 6190 MeV\u00b2)\n\nPrefactor g\u2032 \u03b5 e (2/3) / (\u221a2 G_F) = 1e-4 \u00d7 1e-3 \u00d7 0.303 \u00d7 (2/3) / (1.64951e-11 MeV\u207b\u00b2) \u2248 1.2246e+03 MeV\u00b2.\nWith m\u00b2 = 100 MeV\u00b2:\n\n| coefficient | Q\u00b2=1238 | Q\u00b2=3714 | Q\u00b2=6190 |\n|---|---|---|---|\n| C^{uV}_mm = \u2212C^{uV}_tt | \u2248 9.15e-01 | \u2248 3.21e-01 | \u2248 1.95e-01 |\n| C^{dV}_mm = \u2212C^{dV}_tt | \u2248 \u22124.58e-01 | \u2248 \u22121.60e-01 | \u2248 \u22129.74e-02 |\n\n(Rounded for reading; the plugin uses the exported constants at full precision.)\n\n## Card\n\nExpressible: one vector mediator with `nu_couplings` mm = +1e-4, tt = \u22121e-4, zero direct `quark_couplings`, and `kinetic_mixing` = 1e-3 written explicitly (a card has no loop default). `card.json` is provided.\n\n## Plugin\n\n`plugin.py` computes the same coefficients from `GF`, `SQRT2`, `E_CHARGE` and `Q_EM` exported by `nuagent.physics.couplings`. `COUPLINGS = (\"g_prime\", \"kinetic_mixing\")`; zeroing either removes all new physics, so the SM-limit check passes.\n"
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
      6.787483894774342,
      6.625526223508306,
      7.137923778340959,
      7.3613210140445915,
      6.767876897710478,
      5.6461896956635655,
      4.517634795391879,
      3.520103940443097,
      2.6898983687262854,
      2.01997035226367,
      1.490334103702674,
      1.078906593551533
    ],
    "predicted_total": 55.64316965812137,
    "observed_total": 855.0,
    "delta": 7.507566092489984,
    "new_physics": {
      "neg2_log_likelihood": 24.885003591924004,
      "alpha": 0.23609648010760576,
      "beta": {
        "steady_state": 0.05109107728346943,
        "prompt_neutrons": 0.018821609061171335
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
  "final_text": "Validation passed with no findings. The plugin and card agree on all 66 Q\u00b2 points, the SM limit holds with couplings zeroed, and the couplings are perturbative.\n\n**Files produced** in the task directory:\n\n- `plugin.py`: vector coefficients for an L_\u03bc\u2212L_\u03c4 Z\u2032 with g_\u03bd^{mm} = +g\u2032, g_\u03bd^{tt} = \u2212g\u2032, g_\u03bd^{ee} = 0, and quark couplings from kinetic mixing only, g_q = \u03b5 e Q_q^em. Defaults are the described point: g\u2032 = 1e-4, m = 10 MeV, \u03b5 = 1e-3. Both g\u2032 and \u03b5 are listed in COUPLINGS.\n- `card.json`: one vector mediator with the same neutrino couplings, zero direct quark couplings, and kinetic_mixing 1e-3 written explicitly.\n- `notes.md`: derivation from conventions sections 3, 6 and 7, the sign convention for \u03b5 and its consequence of destructive interference for \u03bd_\u03bc, the decision not to add the loop-induced mixing on top of the explicitly given \u03b5, and reference coefficient values at the three Cs-133 Q\u00b2 points.\n\nNo `questions.json` was written, since the mass, coupling and mixing were all fixed by the description.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":74861,\"stop_reason\":\"end_turn\",\"session_id\":\"e51fee6d-088a-4a2d-9dd7-d65f1d287c96\",\"total_cost_usd\":1.0745565,\"usage\":{\"input_tokens\":226,\"cache_creation_input_tokens\":36254,\"cache_read_input_tokens\":152266,\"output_tokens\":6183,\"output_tokens_details\":{\"thinking_tokens\":546},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":36254,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":421,\"cache_read_input_tokens\":35651,\"cache_creation_input_tokens\":603,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":603},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":226,\"outputTokens\":6183,\"cacheReadInputTokens\":152266,\"cacheCreationInputTokens\":36254,\"webSearchRequests\":0,\"costUSD\":1.0745565,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":546,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01SzTCqjqUCshYkpoP5d5AwC\",\"tool_input\":{\"command\":\"ls -R /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/accept-claude/report.task\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":20,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings. The plugin and card agree on all 66 Q\u00b2 points, the SM limit holds with couplings zeroed, and the couplings are perturbative.\\n\\n**Files produced** in the task directory:\\n\\n- `plugin.py`: vector coefficients for an L_\u03bc\u2212L_\u03c4 Z\u2032 with g_\u03bd^{mm} = +g\u2032, g_\u03bd^{tt} = \u2212g\u2032, g_\u03bd^{ee} = 0, and quark couplings from kinetic mixing only, g_q = \u03b5 e Q_q^em. Defaults are the described point: g\u2032 = 1e-4, m = 10 MeV, \u03b5 = 1e-3. Both g\u2032 and \u03b5 are listed in COUPLINGS.\\n- `card.json`: one vector mediator with the same neutrino couplings, zero direct quark couplings, and kinetic_mixing 1e-3 written explicitly.\\n- `notes.md`: derivation from conventions sections 3, 6 and 7, the sign convention for \u03b5 and its consequence of destructive interference for \u03bd_\u03bc, the decision not to add the loop-induced mixing on top of the explicitly given \u03b5, and reference coefficient values at the three Cs-133 Q\u00b2 points.\\n\\nNo `questions.json` was written, since the mass, coupling and mixing were all fixed by the description.\",\"ttft_ms\":897,\"type\":\"result\",\"duration_ms\":78333,\"uuid\":\"b97d5599-d7ba-4038-8161-a8ed2d42e16f\",\"ttft_stream_ms\":758,\"time_to_request_ms\":41,\"first_content_frame_ms\":759,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/m7-conversation/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/accept-claude/report.task\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
