Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

The U(1) B−L gauge boson has mass 25 MeV and gauge coupling g′ = 3e-5, with no kinetic mixing.

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
"""U(1)_{B-L} gauge boson, m = 25 MeV, g' = 3e-5, no kinetic mixing; derivation in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2

PARAMETERS = {"g_prime": 3e-5, "mass_mev": 25.0}
COUPLINGS = ("g_prime",)

# B-L charges: lepton doublet (hence every neutrino) -1, every quark +1/3.
Q_NU = -1.0
Q_QUARK = 1.0 / 3.0


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
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
# U(1)_{B−L} gauge boson: m = 25 MeV, g′ = 3e-5, no kinetic mixing

## Description

"The U(1) B−L gauge boson has mass 25 MeV and gauge coupling g′ = 3e-5, with no kinetic mixing."

The description fixes the Lorentz structure (gauge boson, vector), the mass, the gauge
coupling and the charge assignment (B−L is a standard assignment, conventions section 7).
Nothing is missing, so a plugin and a card are produced.

## Derivation

1. **Lorentz structure.** A gauge boson couples to vector currents, so structure `V`.
   The B−L current is purely vector on quarks (g_L = g_R = g′/3), so no axial piece is dropped.
   On leptons the lepton doublet carries charge −1; only the left-handed neutrino current
   exists in the SM and it is the one that scatters, so g_ν = g′ Q′_L with no chirality factor.

2. **Charges** (conventions section 7, B−L row):
   Q′_{ν_e} = Q′_{ν_μ} = Q′_{ν_τ} = −1,  Q′_u = Q′_d = +1/3.
   Hence, with g′ = 3e-5,

       g_ν^{ee} = g_ν^{mm} = g_ν^{tt} = −g′ = −3e-5,   off-diagonal entries 0,
       g_u = g_d = +g′/3 = 1e-5.

   B−L with three right-handed neutrinos is anomaly-free; no anomaly caveat needed.

3. **Kinetic mixing.** Stated as zero, so `kinetic_mixing = 0.0` and no ε e Q_q^em shift of
   the quark couplings.

4. **Matching** (conventions section 3):

       C^{uV}_{αα}(Q²) = C^{dV}_{αα}(Q²) = g_ν g_q / (√2 G_F (Q² + m²))
                        = −g′² / (3 √2 G_F (Q² + m²)),   m = 25 MeV.

   No convention conversion is needed: the description gives the gauge coupling directly,
   not α′ or a chiral coupling.

5. **Sign and interference.** g_ν g_q = −g′²/3 < 0, so the nuclear charge shift
   ΔQ = (3Z + 3N) C = −A g′² / (√2 G_F (Q² + m²)) is negative, the same sign as
   Q_SM ≈ 0.04 Z − 0.5 N < 0. Interference with the SM is constructive; there is no
   cancellation region. The relative sign between the neutrino and quark couplings is
   physical and is kept in both plugin and card (dropping it would flip to destructive).

6. **Card.** One vector mediator entry with the signed couplings above; the schema expresses
   the interaction completely, so `card.json` is written.

## Reference values (for reading, rounded)

Cs-133 reference points Q² = 1238, 3714, 6190 MeV² (T = 5, 15, 25 keV):

| coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
|---|---|---|---|
| C^{uV}_αα = C^{dV}_αα (α = e, μ, τ) | −9.762e-03 | −4.191e-03 | −2.669e-03 |

At the first point ΔQ = 3·133·C ≈ −3.9 against Q_SM ≈ −36.9 (tree level), i.e. a modest
constructive enhancement of the coherent rate. All off-diagonal and all scalar entries are zero.

## Files

- `plugin.py`: `PARAMETERS = {"g_prime": 3e-5, "mass_mev": 25.0}`, `COUPLINGS = ("g_prime",)`;
  builds g_ν = −g′, g_q = g′/3 from the exported charge constants and the exported `GF`, `SQRT2`.
- `card.json`: the same mediator written with explicit signed couplings.
- Validation: `nuagent validate --plugin plugin.py --card card.json`.

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
  "notes": "# U(1)_{B\u2212L} gauge boson: m = 25 MeV, g\u2032 = 3e-5, no kinetic mixing\n\n## Description\n\n\"The U(1) B\u2212L gauge boson has mass 25 MeV and gauge coupling g\u2032 = 3e-5, with no kinetic mixing.\"\n\nThe description fixes the Lorentz structure (gauge boson, vector), the mass, the gauge\ncoupling and the charge assignment (B\u2212L is a standard assignment, conventions section 7).\nNothing is missing, so a plugin and a card are produced.\n\n## Derivation\n\n1. **Lorentz structure.** A gauge boson couples to vector currents, so structure `V`.\n   The B\u2212L current is purely vector on quarks (g_L = g_R = g\u2032/3), so no axial piece is dropped.\n   On leptons the lepton doublet carries charge \u22121; only the left-handed neutrino current\n   exists in the SM and it is the one that scatters, so g_\u03bd = g\u2032 Q\u2032_L with no chirality factor.\n\n2. **Charges** (conventions section 7, B\u2212L row):\n   Q\u2032_{\u03bd_e} = Q\u2032_{\u03bd_\u03bc} = Q\u2032_{\u03bd_\u03c4} = \u22121,  Q\u2032_u = Q\u2032_d = +1/3.\n   Hence, with g\u2032 = 3e-5,\n\n       g_\u03bd^{ee} = g_\u03bd^{mm} = g_\u03bd^{tt} = \u2212g\u2032 = \u22123e-5,   off-diagonal entries 0,\n       g_u = g_d = +g\u2032/3 = 1e-5.\n\n   B\u2212L with three right-handed neutrinos is anomaly-free; no anomaly caveat needed.\n\n3. **Kinetic mixing.** Stated as zero, so `kinetic_mixing = 0.0` and no \u03b5 e Q_q^em shift of\n   the quark couplings.\n\n4. **Matching** (conventions section 3):\n\n       C^{uV}_{\u03b1\u03b1}(Q\u00b2) = C^{dV}_{\u03b1\u03b1}(Q\u00b2) = g_\u03bd g_q / (\u221a2 G_F (Q\u00b2 + m\u00b2))\n                        = \u2212g\u2032\u00b2 / (3 \u221a2 G_F (Q\u00b2 + m\u00b2)),   m = 25 MeV.\n\n   No convention conversion is needed: the description gives the gauge coupling directly,\n   not \u03b1\u2032 or a chiral coupling.\n\n5. **Sign and interference.** g_\u03bd g_q = \u2212g\u2032\u00b2/3 < 0, so the nuclear charge shift\n   \u0394Q = (3Z + 3N) C = \u2212A g\u2032\u00b2 / (\u221a2 G_F (Q\u00b2 + m\u00b2)) is negative, the same sign as\n   Q_SM \u2248 0.04 Z \u2212 0.5 N < 0. Interference with the SM is constructive; there is no\n   cancellation region. The relative sign between the neutrino and quark couplings is\n   physical and is kept in both plugin and card (dropping it would flip to destructive).\n\n6. **Card.** One vector mediator entry with the signed couplings above; the schema expresses\n   the interaction completely, so `card.json` is written.\n\n## Reference values (for reading, rounded)\n\nCs-133 reference points Q\u00b2 = 1238, 3714, 6190 MeV\u00b2 (T = 5, 15, 25 keV):\n\n| coefficient | Q\u00b2=1238 | Q\u00b2=3714 | Q\u00b2=6190 |\n|---|---|---|---|\n| C^{uV}_\u03b1\u03b1 = C^{dV}_\u03b1\u03b1 (\u03b1 = e, \u03bc, \u03c4) | \u22129.762e-03 | \u22124.191e-03 | \u22122.669e-03 |\n\nAt the first point \u0394Q = 3\u00b7133\u00b7C \u2248 \u22123.9 against Q_SM \u2248 \u221236.9 (tree level), i.e. a modest\nconstructive enhancement of the coherent rate. All off-diagonal and all scalar entries are zero.\n\n## Files\n\n- `plugin.py`: `PARAMETERS = {\"g_prime\": 3e-5, \"mass_mev\": 25.0}`, `COUPLINGS = (\"g_prime\",)`;\n  builds g_\u03bd = \u2212g\u2032, g_q = g\u2032/3 from the exported charge constants and the exported `GF`, `SQRT2`.\n- `card.json`: the same mediator written with explicit signed couplings.\n- Validation: `nuagent validate --plugin plugin.py --card card.json`.\n"
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

### Attempt 1

```json
{
  "final_text": "Validation passed with no findings, including the plugin/card cross-check on 66 Q\u00b2 points.\n\n**Files produced** in the task directory:\n\n- `plugin.py`: vector coefficients for a 25 MeV B\u2212L boson. Defaults are g\u2032 = 3e-5 and mass 25 MeV, with g\u2032 as the only coupling parameter. Neutrino coupling is \u2212g\u2032 for all three flavors and quark coupling is +g\u2032/3 for u and d, matched as g_\u03bd g_q / (\u221a2 G_F (Q\u00b2 + m\u00b2)).\n- `card.json`: one vector mediator with the same signed couplings and zero kinetic mixing.\n- `notes.md`: the derivation, the charge assignment, the sign argument, and reference values.\n\n**Physics summary.** The product of neutrino and quark couplings is negative, so the new-physics nuclear charge has the same sign as the Standard Model charge. Interference is constructive with no cancellation region. At the lowest Cs-133 reference point the charge shift is about one tenth of the SM charge. The B\u2212L quark current is purely vector, so no axial coupling was dropped, and the description specified zero kinetic mixing, so no photon-mixing term enters.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":77774,\"stop_reason\":\"end_turn\",\"session_id\":\"8a637407-0da7-4ad9-8d47-e66666704be2\",\"total_cost_usd\":0.6754745,\"usage\":{\"input_tokens\":194,\"cache_creation_input_tokens\":21338,\"cache_read_input_tokens\":112098,\"output_tokens\":4375,\"output_tokens_details\":{\"thinking_tokens\":440},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":21338,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":395,\"cache_read_input_tokens\":27047,\"cache_creation_input_tokens\":297,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":297},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":194,\"outputTokens\":4375,\"cacheReadInputTokens\":112098,\"cacheCreationInputTokens\":21338,\"webSearchRequests\":0,\"costUSD\":0.6754745,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":440,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01McQR6jD2kNyt1gWV4n9zjy\",\"tool_input\":{\"command\":\"ls -R /private/tmp/nuagent-issue17-eval-claude/03_b_minus_l\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":17,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings, including the plugin/card cross-check on 66 Q\u00b2 points.\\n\\n**Files produced** in the task directory:\\n\\n- `plugin.py`: vector coefficients for a 25 MeV B\u2212L boson. Defaults are g\u2032 = 3e-5 and mass 25 MeV, with g\u2032 as the only coupling parameter. Neutrino coupling is \u2212g\u2032 for all three flavors and quark coupling is +g\u2032/3 for u and d, matched as g_\u03bd g_q / (\u221a2 G_F (Q\u00b2 + m\u00b2)).\\n- `card.json`: one vector mediator with the same signed couplings and zero kinetic mixing.\\n- `notes.md`: the derivation, the charge assignment, the sign argument, and reference values.\\n\\n**Physics summary.** The product of neutrino and quark couplings is negative, so the new-physics nuclear charge has the same sign as the Standard Model charge. Interference is constructive with no cancellation region. At the lowest Cs-133 reference point the charge shift is about one tenth of the SM charge. The B\u2212L quark current is purely vector, so no axial coupling was dropped, and the description specified zero kinetic mixing, so no photon-mixing term enters.\",\"ttft_ms\":5420,\"type\":\"result\",\"duration_ms\":80135,\"uuid\":\"b1ee470c-56d1-4d22-9d4b-0a32b85b91ad\",\"ttft_stream_ms\":2573,\"time_to_request_ms\":25,\"first_content_frame_ms\":2574,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/issue-17-agent-harness/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/nuagent-issue17-eval-claude/03_b_minus_l\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
