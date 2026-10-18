Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

A gauged L_mu−L_tau vector Z′ has mass 15 MeV, gauge coupling g′ = 6e-5, total photon kinetic mixing epsilon = −5e-4 in the pack's convention, and no direct quark couplings.

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
"""Gauged L_mu - L_tau Z' with explicit (total) photon kinetic mixing; derivation in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2, E_CHARGE, Q_EM

PARAMETERS = {"g_prime": 6e-5, "mass_mev": 15.0, "kinetic_mixing": -5e-4}
COUPLINGS = ("g_prime", "kinetic_mixing")

# L_mu - L_tau charges of the lepton doublets; quarks carry no charge.
Q_NU = {"ee": 0.0, "mm": +1.0, "tt": -1.0}


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    eps = params["kinetic_mixing"]
    prop = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        g_q = eps * E_CHARGE * Q_EM[quark]  # induced quark coupling, no direct piece
        for pair, charge in Q_NU.items():
            if charge == 0.0:
                continue
            c.set("V", quark, pair, (g * charge) * g_q * prop)
    return c

```

## card.json

```json
{
  "name": "zprime_lmu_ltau_negative_mixing",
  "mediators": [
    {
      "structure": "V",
      "mass_mev": 15.0,
      "nu_couplings": {
        "ee": 0,
        "mm": 6e-05,
        "tt": -6e-05,
        "em": 0,
        "et": 0,
        "mt": 0
      },
      "quark_couplings": {
        "u": 0.0,
        "d": 0.0
      },
      "kinetic_mixing": -0.0005
    }
  ],
  "contacts": []
}

```

## notes.md

```text
# Notes: gauged L_μ−L_τ Z′ with negative total kinetic mixing

Description: L_μ−L_τ Z′, mass 15 MeV, gauge coupling g′ = 6e-5, total photon kinetic mixing ε = −5e-4 in the pack's convention, no direct quark couplings.

## Derivation

1. **Lorentz structure.** Gauge boson exchange: structure `V`, left-handed neutrino current only (conventions section 3).

2. **Neutrino couplings** (conventions section 7, L_μ−L_τ row): Q′_{ν_e} = 0, Q′_{ν_μ} = +1, Q′_{ν_τ} = −1, so

       g_ν^{ee} = 0,   g_ν^{mm} = +g′ = +6e-5,   g_ν^{tt} = −g′ = −6e-5,

   all off-diagonal entries zero (the charges are flavor-diagonal).

3. **Quark couplings.** The description states no direct quark couplings (Q′_q = 0), so the quarks couple only through the kinetic mixing (conventions section 6):

       g_u = ε e (+2/3),   g_d = ε e (−1/3),   e = 0.303.

   The description gives the *total* mixing ε = −5e-4 in the pack's convention. "Total" means the loop-induced piece κ g′ (≈ +8.67e-7) is already included in the quoted number, so it is **not** added again; ε = −5e-4 is used as is. This is the same treatment as example 03 A (explicit mixing), with a different sign.

4. **Matching** (conventions section 3), with the propagator 1/(Q² + m²):

       C^{uV}_{mm} = +g′ · ε e (2/3) / (√2 G_F (Q² + m²)),
       C^{dV}_{mm} = −g′ · ε e (1/3) / (√2 G_F (Q² + m²)),
       C^{qV}_{tt} = −C^{qV}_{mm},   C^{qV}_{ee} = 0.

   Numerically at Q² = 1238 MeV² (m² = 225 MeV²): √2 G_F (Q² + m²) = 2.4134e-8, g′ ε e = −9.09e-9, so C^{uV}_{mm} ≈ −0.2511 and C^{dV}_{mm} ≈ +0.1256; both fall as 1/(Q² + m²).

5. **Sign and interference.** The induced nuclear charge is proton-only: 2 C^u + C^d ∝ 2(2/3) − 1/3 = 1 and C^u + 2 C^d ∝ 2/3 − 2/3 = 0. For ν_μ the shift is ΔQ^{μμ} = Z g′ ε e / (√2 G_F (Q² + m²)). Because ε g′ < 0 here, ΔQ^{μμ} is **negative**, the same sign as Q_SM ≈ 0.04 Z − 0.5 N < 0: the interference with the Standard Model is **constructive** for ν_μ and ν̄_μ (the COHERENT prompt and delayed muon-flavor fluxes), the opposite of the loop-only case in section 6 and example 03 B. For ν_τ the shift has the opposite sign, but there is no ν_τ flux at COHERENT. Electron neutrinos see no new physics.

   The sign of ε relative to g′ is part of the model and is taken from the description: ε = −5e-4 with g′ = +6e-5 in the pack's convention (g_q → g_q + ε e Q_q^em).

6. **Perturbativity.** |g′| = 6e-5 and |ε| = 5e-4 are both far below √(4π). Both are listed in `COUPLINGS`; zeroing either removes all new physics, so the SM-limit check passes.

7. **Card.** Card-expressible: one vector mediator with the signed neutrino couplings, zero direct quark couplings, and `kinetic_mixing: -0.0005` written explicitly (a card has no κ g′ default). The plugin computes the same coefficients independently from the exported constants `GF`, `SQRT2`, `E_CHARGE` and `Q_EM`.

No convention conversion was needed: the mixing is already quoted in the pack's convention and the coupling is the gauge coupling g′ directly. No axial quark couplings were dropped (there are no direct quark couplings). The Q² dependence of the loop mixing (example 06) is not applied, because the description fixes a constant total ε.

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
  "notes": "# Notes: gauged L_\u03bc\u2212L_\u03c4 Z\u2032 with negative total kinetic mixing\n\nDescription: L_\u03bc\u2212L_\u03c4 Z\u2032, mass 15 MeV, gauge coupling g\u2032 = 6e-5, total photon kinetic mixing \u03b5 = \u22125e-4 in the pack's convention, no direct quark couplings.\n\n## Derivation\n\n1. **Lorentz structure.** Gauge boson exchange: structure `V`, left-handed neutrino current only (conventions section 3).\n\n2. **Neutrino couplings** (conventions section 7, L_\u03bc\u2212L_\u03c4 row): Q\u2032_{\u03bd_e} = 0, Q\u2032_{\u03bd_\u03bc} = +1, Q\u2032_{\u03bd_\u03c4} = \u22121, so\n\n       g_\u03bd^{ee} = 0,   g_\u03bd^{mm} = +g\u2032 = +6e-5,   g_\u03bd^{tt} = \u2212g\u2032 = \u22126e-5,\n\n   all off-diagonal entries zero (the charges are flavor-diagonal).\n\n3. **Quark couplings.** The description states no direct quark couplings (Q\u2032_q = 0), so the quarks couple only through the kinetic mixing (conventions section 6):\n\n       g_u = \u03b5 e (+2/3),   g_d = \u03b5 e (\u22121/3),   e = 0.303.\n\n   The description gives the *total* mixing \u03b5 = \u22125e-4 in the pack's convention. \"Total\" means the loop-induced piece \u03ba g\u2032 (\u2248 +8.67e-7) is already included in the quoted number, so it is **not** added again; \u03b5 = \u22125e-4 is used as is. This is the same treatment as example 03 A (explicit mixing), with a different sign.\n\n4. **Matching** (conventions section 3), with the propagator 1/(Q\u00b2 + m\u00b2):\n\n       C^{uV}_{mm} = +g\u2032 \u00b7 \u03b5 e (2/3) / (\u221a2 G_F (Q\u00b2 + m\u00b2)),\n       C^{dV}_{mm} = \u2212g\u2032 \u00b7 \u03b5 e (1/3) / (\u221a2 G_F (Q\u00b2 + m\u00b2)),\n       C^{qV}_{tt} = \u2212C^{qV}_{mm},   C^{qV}_{ee} = 0.\n\n   Numerically at Q\u00b2 = 1238 MeV\u00b2 (m\u00b2 = 225 MeV\u00b2): \u221a2 G_F (Q\u00b2 + m\u00b2) = 2.4134e-8, g\u2032 \u03b5 e = \u22129.09e-9, so C^{uV}_{mm} \u2248 \u22120.2511 and C^{dV}_{mm} \u2248 +0.1256; both fall as 1/(Q\u00b2 + m\u00b2).\n\n5. **Sign and interference.** The induced nuclear charge is proton-only: 2 C^u + C^d \u221d 2(2/3) \u2212 1/3 = 1 and C^u + 2 C^d \u221d 2/3 \u2212 2/3 = 0. For \u03bd_\u03bc the shift is \u0394Q^{\u03bc\u03bc} = Z g\u2032 \u03b5 e / (\u221a2 G_F (Q\u00b2 + m\u00b2)). Because \u03b5 g\u2032 < 0 here, \u0394Q^{\u03bc\u03bc} is **negative**, the same sign as Q_SM \u2248 0.04 Z \u2212 0.5 N < 0: the interference with the Standard Model is **constructive** for \u03bd_\u03bc and \u03bd\u0304_\u03bc (the COHERENT prompt and delayed muon-flavor fluxes), the opposite of the loop-only case in section 6 and example 03 B. For \u03bd_\u03c4 the shift has the opposite sign, but there is no \u03bd_\u03c4 flux at COHERENT. Electron neutrinos see no new physics.\n\n   The sign of \u03b5 relative to g\u2032 is part of the model and is taken from the description: \u03b5 = \u22125e-4 with g\u2032 = +6e-5 in the pack's convention (g_q \u2192 g_q + \u03b5 e Q_q^em).\n\n6. **Perturbativity.** |g\u2032| = 6e-5 and |\u03b5| = 5e-4 are both far below \u221a(4\u03c0). Both are listed in `COUPLINGS`; zeroing either removes all new physics, so the SM-limit check passes.\n\n7. **Card.** Card-expressible: one vector mediator with the signed neutrino couplings, zero direct quark couplings, and `kinetic_mixing: -0.0005` written explicitly (a card has no \u03ba g\u2032 default). The plugin computes the same coefficients independently from the exported constants `GF`, `SQRT2`, `E_CHARGE` and `Q_EM`.\n\nNo convention conversion was needed: the mixing is already quoted in the pack's convention and the coupling is the gauge coupling g\u2032 directly. No axial quark couplings were dropped (there are no direct quark couplings). The Q\u00b2 dependence of the loop mixing (example 06) is not applied, because the description fixes a constant total \u03b5.\n"
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
      18.4384260893985,
      27.0117601677303,
      32.8775915771865,
      31.419415590854996,
      24.053841407426116,
      16.90720861031418,
      12.729321184733598,
      9.41985934487881,
      6.86995696109429,
      4.9502052460346855,
      3.5226972372599765,
      2.47153469393063
    ],
    "predicted_total": 190.67181811084262,
    "observed_total": 855.0,
    "delta": 0.9663233728839664,
    "new_physics": {
      "neg2_log_likelihood": 18.343760872317986,
      "alpha": -0.15901011147064845,
      "beta": {
        "steady_state": -0.008644003643201087,
        "prompt_neutrons": 0.001612140502165892
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
  "final_text": "Validation passed with no findings, including the plugin-versus-card cross-check on 66 momentum-transfer points.\n\n**Files produced** in the task directory:\n\n- `plugin.py`: vector coefficients for the L_\u03bc\u2212L_\u03c4 Z\u2032. Neutrino couplings are +g\u2032 for \u03bc\u03bc and \u2212g\u2032 for \u03c4\u03c4, zero for ee. Quark couplings come only from the mixing, g_q = \u03b5 e Q_q^em, using the exported constants. Defaults are the described point: g\u2032 = 6e-5, mass 15 MeV, \u03b5 = \u22125e-4, with both g\u2032 and \u03b5 listed as couplings.\n- `card.json`: one vector mediator with the same signed neutrino couplings, zero direct quark couplings and the explicit negative kinetic mixing.\n- `notes.md`: the derivation. The quoted \u03b5 is the total mixing, so the loop-induced \u03ba g\u2032 piece is not added on top. The coupling is proton-only, and because \u03b5 g\u2032 < 0 the shift to the \u03bd_\u03bc nuclear charge has the same sign as the Standard Model charge, so the interference is constructive for the muon-flavor fluxes, opposite to the loop-only case.\n\nNo questions.json was needed since the description fixes the structure, mass and all couplings.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":94713,\"stop_reason\":\"end_turn\",\"session_id\":\"ceaa2442-e712-4313-8b3b-eca916f021c2\",\"total_cost_usd\":1.0266454999999999,\"usage\":{\"input_tokens\":258,\"cache_creation_input_tokens\":31967,\"cache_read_input_tokens\":188102,\"output_tokens\":6754,\"output_tokens_details\":{\"thinking_tokens\":645},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":31967,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":412,\"cache_read_input_tokens\":37679,\"cache_creation_input_tokens\":294,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":294},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":258,\"outputTokens\":6754,\"cacheReadInputTokens\":188102,\"cacheCreationInputTokens\":31967,\"webSearchRequests\":0,\"costUSD\":1.0266454999999999,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":645,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01Pit9zUreWiUbXd1HD8DjFs\",\"tool_input\":{\"command\":\"ls -R /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/06_lmu_ltau_negative_mixing\",\"description\":\"List task directory contents\"}},{\"tool_name\":\"Read\",\"tool_use_id\":\"toolu_01R91MzBauuSXxafpozqEHhh\",\"tool_input\":{\"file_path\":\"/private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-5c51979abec1/scratchpad/live/eval-claude/06_lmu_ltau_negative_mixing/context/examples/02_lmu_ltau.md\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":24,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings, including the plugin-versus-card cross-check on 66 momentum-transfer points.\\n\\n**Files produced** in the task directory:\\n\\n- `plugin.py`: vector coefficients for the L_\u03bc\u2212L_\u03c4 Z\u2032. Neutrino couplings are +g\u2032 for \u03bc\u03bc and \u2212g\u2032 for \u03c4\u03c4, zero for ee. Quark couplings come only from the mixing, g_q = \u03b5 e Q_q^em, using the exported constants. Defaults are the described point: g\u2032 = 6e-5, mass 15 MeV, \u03b5 = \u22125e-4, with both g\u2032 and \u03b5 listed as couplings.\\n- `card.json`: one vector mediator with the same signed neutrino couplings, zero direct quark couplings and the explicit negative kinetic mixing.\\n- `notes.md`: the derivation. The quoted \u03b5 is the total mixing, so the loop-induced \u03ba g\u2032 piece is not added on top. The coupling is proton-only, and because \u03b5 g\u2032 < 0 the shift to the \u03bd_\u03bc nuclear charge has the same sign as the Standard Model charge, so the interference is constructive for the muon-flavor fluxes, opposite to the loop-only case.\\n\\nNo questions.json was needed since the description fixes the structure, mass and all couplings.\",\"ttft_ms\":3823,\"type\":\"result\",\"duration_ms\":99006,\"uuid\":\"83bd81e3-0738-430b-ba30-0fdbcc4d417d\",\"ttft_stream_ms\":2246,\"time_to_request_ms\":23,\"first_content_frame_ms\":2247,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/m7-conversation/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/06_lmu_ltau_negative_mixing\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
