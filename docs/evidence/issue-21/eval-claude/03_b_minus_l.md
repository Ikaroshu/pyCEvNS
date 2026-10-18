Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

The U(1) B−L gauge boson has mass 25 MeV and gauge coupling g′ = 3e-5, with no kinetic mixing.

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
"""U(1)_{B-L} gauge boson, m = 25 MeV, g' = 3e-5, no kinetic mixing. Derivation in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_prime": 3e-5, "mass_mev": 25.0}
COUPLINGS = ("g_prime",)

# B-L charges: lepton doublet (hence neutrino) -1, every quark +1/3.
Q_NU = -1.0
Q_QUARK = 1.0 / 3.0


def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    g_nu = g * Q_NU
    g_q = g * Q_QUARK
    value = g_nu * g_q / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)
    return c

```

## card.json

```json
{
  "name": "zprime_b_minus_l_25mev",
  "mediators": [
    {
      "structure": "V",
      "mass_mev": 25.0,
      "nu_couplings": {
        "ee": -3e-05,
        "mm": -3e-05,
        "tt": -3e-05,
        "em": 0,
        "et": 0,
        "mt": 0
      },
      "quark_couplings": {
        "u": 1e-05,
        "d": 1e-05
      },
      "kinetic_mixing": 0.0
    }
  ],
  "contacts": []
}

```

## notes.md

```text
# Notes: U(1)_{B−L} gauge boson, m = 25 MeV, g′ = 3e-5

## Description

"The U(1) B−L gauge boson has mass 25 MeV and gauge coupling g′ = 3e-5, with no kinetic mixing."

The description fixes the Lorentz structure (a gauge boson, hence vector exchange), the mass, the single gauge coupling, and the absence of kinetic mixing. The fermion couplings follow from the B−L charge assignment, so nothing is missing and no questions are needed.

## Derivation

1. **Lorentz structure.** A U(1) gauge boson couples to the vector current of each charged fermion: L ⊃ g′ Z′_μ Σ_f Q′_f f̄ γ^μ f. Structure `V`. The neutrino coupling acts on the left-handed current, g′ Q′_ν ν̄ γ^μ P_L ν, since the lepton doublet carries the charge; a right-handed neutrino, if present, does not scatter at CEvNS experiments. There is no axial part to drop (the charge is the same for both chiralities of each quark).

2. **Charges** (conventions section 7, B−L row): Q′ = −1 for every lepton doublet, Q′ = +1/3 for every quark. Hence
   - g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = −g′ = −3e-5, off-diagonal entries zero (B−L is flavor-universal and diagonal in the mass basis);
   - g_u = g_d = +g′/3 = 1e-5.

3. **Kinetic mixing.** The description says none, so ε = 0 and no electromagnetic-charge-proportional term is added to the quark couplings. The card field `kinetic_mixing` is written explicitly as 0.0.

4. **Matching** (conventions section 3):

       C^{qV}_{αα}(Q²) = g_ν g_q / (√2 G_F (Q² + m²)) = −g′² / (3 √2 G_F (Q² + m²)),   q = u, d,  α = e, μ, τ.

   With g′ = 3e-5, m = 25 MeV: the numerator g_ν g_q = −3e-10. At the first reference point Q² = 1238 MeV² (Cs-133, T = 5 keV), Q² + m² = 1863 MeV², so C ≈ −3e-10 / (1.4142 × 1.16638e-11 × 1863) ≈ −9.76e-3. At Q² = 3714 MeV², C ≈ −4.19e-3; at Q² = 6190 MeV², C ≈ −2.68e-3. These are hand estimates; the plugin computes the values from the exported constants.

5. **Sign and interference.** g_ν g_q < 0, so the new-physics nuclear charge shift ΔQ = (3Z + 3N) C = −(Z + N) g′²/(√2 G_F (Q² + m²)) is negative, the same sign as the SM charge Q_SM ≈ 0.04 Z − 0.5 N < 0. The interference is constructive; there is no cancellation region for any g′. The relative sign between the neutrino and quark couplings is physical and is kept in both the plugin (via Q_NU = −1, Q_QUARK = +1/3) and the card (signed couplings).

6. **Perturbativity.** |g′| = 3e-5 ≪ √(4π). All couplings and the positive mass are within the framework's bounds.

7. **Card expressibility.** One vector mediator with constant couplings: fully expressible. `card.json` lists the mediator with the signed neutrino couplings −3e-5 on the diagonal, quark couplings +1e-5, mass 25 MeV, kinetic mixing 0. The plugin computes the same coefficients independently from `g_prime`, `mass_mev` and the exported `GF`, `SQRT2`.

8. **Scalar sector.** None; all `S` entries are zero. No contact terms.

## Parameter point

| parameter | value |
|---|---|
| g_prime | 3e-5 |
| mass_mev | 25.0 |

`COUPLINGS = ("g_prime",)`: with g′ → 0 all coefficients vanish and the SM is recovered.

## Anomalies

U(1)_{B−L} is anomaly-free once three right-handed neutrinos are added; this is the standard assumption and does not affect the CEvNS coefficients.

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
  "notes": "# Notes: U(1)_{B\u2212L} gauge boson, m = 25 MeV, g\u2032 = 3e-5\n\n## Description\n\n\"The U(1) B\u2212L gauge boson has mass 25 MeV and gauge coupling g\u2032 = 3e-5, with no kinetic mixing.\"\n\nThe description fixes the Lorentz structure (a gauge boson, hence vector exchange), the mass, the single gauge coupling, and the absence of kinetic mixing. The fermion couplings follow from the B\u2212L charge assignment, so nothing is missing and no questions are needed.\n\n## Derivation\n\n1. **Lorentz structure.** A U(1) gauge boson couples to the vector current of each charged fermion: L \u2283 g\u2032 Z\u2032_\u03bc \u03a3_f Q\u2032_f f\u0304 \u03b3^\u03bc f. Structure `V`. The neutrino coupling acts on the left-handed current, g\u2032 Q\u2032_\u03bd \u03bd\u0304 \u03b3^\u03bc P_L \u03bd, since the lepton doublet carries the charge; a right-handed neutrino, if present, does not scatter at CEvNS experiments. There is no axial part to drop (the charge is the same for both chiralities of each quark).\n\n2. **Charges** (conventions section 7, B\u2212L row): Q\u2032 = \u22121 for every lepton doublet, Q\u2032 = +1/3 for every quark. Hence\n   - g_\u03bd^{ee} = g_\u03bd^{mm} = g_\u03bd^{tt} = \u2212g\u2032 = \u22123e-5, off-diagonal entries zero (B\u2212L is flavor-universal and diagonal in the mass basis);\n   - g_u = g_d = +g\u2032/3 = 1e-5.\n\n3. **Kinetic mixing.** The description says none, so \u03b5 = 0 and no electromagnetic-charge-proportional term is added to the quark couplings. The card field `kinetic_mixing` is written explicitly as 0.0.\n\n4. **Matching** (conventions section 3):\n\n       C^{qV}_{\u03b1\u03b1}(Q\u00b2) = g_\u03bd g_q / (\u221a2 G_F (Q\u00b2 + m\u00b2)) = \u2212g\u2032\u00b2 / (3 \u221a2 G_F (Q\u00b2 + m\u00b2)),   q = u, d,  \u03b1 = e, \u03bc, \u03c4.\n\n   With g\u2032 = 3e-5, m = 25 MeV: the numerator g_\u03bd g_q = \u22123e-10. At the first reference point Q\u00b2 = 1238 MeV\u00b2 (Cs-133, T = 5 keV), Q\u00b2 + m\u00b2 = 1863 MeV\u00b2, so C \u2248 \u22123e-10 / (1.4142 \u00d7 1.16638e-11 \u00d7 1863) \u2248 \u22129.76e-3. At Q\u00b2 = 3714 MeV\u00b2, C \u2248 \u22124.19e-3; at Q\u00b2 = 6190 MeV\u00b2, C \u2248 \u22122.68e-3. These are hand estimates; the plugin computes the values from the exported constants.\n\n5. **Sign and interference.** g_\u03bd g_q < 0, so the new-physics nuclear charge shift \u0394Q = (3Z + 3N) C = \u2212(Z + N) g\u2032\u00b2/(\u221a2 G_F (Q\u00b2 + m\u00b2)) is negative, the same sign as the SM charge Q_SM \u2248 0.04 Z \u2212 0.5 N < 0. The interference is constructive; there is no cancellation region for any g\u2032. The relative sign between the neutrino and quark couplings is physical and is kept in both the plugin (via Q_NU = \u22121, Q_QUARK = +1/3) and the card (signed couplings).\n\n6. **Perturbativity.** |g\u2032| = 3e-5 \u226a \u221a(4\u03c0). All couplings and the positive mass are within the framework's bounds.\n\n7. **Card expressibility.** One vector mediator with constant couplings: fully expressible. `card.json` lists the mediator with the signed neutrino couplings \u22123e-5 on the diagonal, quark couplings +1e-5, mass 25 MeV, kinetic mixing 0. The plugin computes the same coefficients independently from `g_prime`, `mass_mev` and the exported `GF`, `SQRT2`.\n\n8. **Scalar sector.** None; all `S` entries are zero. No contact terms.\n\n## Parameter point\n\n| parameter | value |\n|---|---|\n| g_prime | 3e-5 |\n| mass_mev | 25.0 |\n\n`COUPLINGS = (\"g_prime\",)`: with g\u2032 \u2192 0 all coefficients vanish and the SM is recovered.\n\n## Anomalies\n\nU(1)_{B\u2212L} is anomaly-free once three right-handed neutrinos are added; this is the standard assumption and does not affect the CEvNS coefficients.\n"
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
      12.098813993945363,
      19.200013769442915,
      24.745282753049192,
      24.751138309578348,
      19.746508309841943,
      14.340462522853706,
      10.937146911629616,
      8.170924901203804,
      6.005020084609939,
      4.354133103061116,
      3.114463508748708,
      2.194358003990753
    ],
    "predicted_total": 149.6582661719554,
    "observed_total": 855.0,
    "delta": -0.06439944260580077,
    "new_physics": {
      "neg2_log_likelihood": 17.31303805682822,
      "alpha": 0.006507723371206537,
      "beta": {
        "steady_state": -0.0029748557434858968,
        "prompt_neutrons": 0.0019271002219321428
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
  "final_text": "Validation passed with no findings, including the plugin/card cross-check. The task is complete.\n\nTranslation of the U(1)_{B\u2212L} gauge boson at m = 25 MeV, g\u2032 = 3e-5, no kinetic mixing. Validation passed with no findings, including the plugin/card cross-check on 66 Q\u00b2 points.\n\nFiles produced:\n\n- **plugin.py** computes the vector coefficient for u and d quarks on the three diagonal flavor entries. It uses the B\u2212L charges \u22121 for neutrinos and +1/3 for quarks, so g_\u03bd g_q = \u2212g\u2032\u00b2/3, divided by \u221a2 G_F (Q\u00b2 + m\u00b2). Parameters are g_prime = 3e-5 and mass_mev = 25.0, with g_prime the only coupling.\n- **card.json** holds one vector mediator with the same signed couplings: neutrino diagonal entries \u22123e-5, quark couplings +1e-5, mass 25 MeV, kinetic mixing 0.\n- **notes.md** records the derivation: charge assignment, matching formula from the conventions, the physical relative sign giving constructive interference with the Standard Model, the absence of a cancellation region, and why the card is expressible.\n\nNo questions.json was written because the description fixes the structure, mass and all couplings.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":92968,\"stop_reason\":\"end_turn\",\"session_id\":\"66074bdd-e8f7-4c70-8d3b-5f1d7a26ffb8\",\"total_cost_usd\":0.7602404999999999,\"usage\":{\"input_tokens\":194,\"cache_creation_input_tokens\":24151,\"cache_read_input_tokens\":122522,\"output_tokens\":4893,\"output_tokens_details\":{\"thinking_tokens\":265},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":24151,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":423,\"cache_read_input_tokens\":29828,\"cache_creation_input_tokens\":329,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":329},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":194,\"outputTokens\":4893,\"cacheReadInputTokens\":122522,\"cacheCreationInputTokens\":24151,\"webSearchRequests\":0,\"costUSD\":0.7602404999999999,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":265,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_013FWPkBF67TC6Pf1EYgZwzQ\",\"tool_input\":{\"command\":\"ls -R /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/03_b_minus_l\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":16,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings, including the plugin/card cross-check. The task is complete.\\n\\nTranslation of the U(1)_{B\u2212L} gauge boson at m = 25 MeV, g\u2032 = 3e-5, no kinetic mixing. Validation passed with no findings, including the plugin/card cross-check on 66 Q\u00b2 points.\\n\\nFiles produced:\\n\\n- **plugin.py** computes the vector coefficient for u and d quarks on the three diagonal flavor entries. It uses the B\u2212L charges \u22121 for neutrinos and +1/3 for quarks, so g_\u03bd g_q = \u2212g\u2032\u00b2/3, divided by \u221a2 G_F (Q\u00b2 + m\u00b2). Parameters are g_prime = 3e-5 and mass_mev = 25.0, with g_prime the only coupling.\\n- **card.json** holds one vector mediator with the same signed couplings: neutrino diagonal entries \u22123e-5, quark couplings +1e-5, mass 25 MeV, kinetic mixing 0.\\n- **notes.md** records the derivation: charge assignment, matching formula from the conventions, the physical relative sign giving constructive interference with the Standard Model, the absence of a cancellation region, and why the card is expressible.\\n\\nNo questions.json was written because the description fixes the structure, mass and all couplings.\",\"ttft_ms\":3720,\"type\":\"result\",\"duration_ms\":96884,\"uuid\":\"8931b82f-4409-4d6b-99b6-60bbeab04c90\",\"ttft_stream_ms\":2696,\"time_to_request_ms\":30,\"first_content_frame_ms\":2696,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/m7-conversation/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/03_b_minus_l\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
