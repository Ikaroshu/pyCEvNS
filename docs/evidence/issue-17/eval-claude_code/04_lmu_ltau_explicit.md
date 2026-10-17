Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

A gauged L_mu−L_tau vector Z′ has mass 10 MeV, gauge coupling g′ = 1e-4 and photon kinetic mixing epsilon = +1e-3 in the conventions of this pack. There are no direct quark couplings.

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
"""L_mu - L_tau Z' with explicit photon kinetic mixing; derivation in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2, E_CHARGE, Q_EM

PARAMETERS = {"g_prime": 1e-4, "mass_mev": 10.0, "kinetic_mixing": 1e-3}
COUPLINGS = ("g_prime", "kinetic_mixing")

# U(1)_{L_mu - L_tau} charges of the lepton doublets (conventions section 7).
Q_NU = {"ee": 0.0, "mm": +1.0, "tt": -1.0}


def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    eps = params["kinetic_mixing"]
    prop = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        # No direct quark charge; the only quark coupling is from kinetic mixing.
        g_q = eps * E_CHARGE * Q_EM[quark]
        for pair, charge in Q_NU.items():
            if charge == 0.0:
                continue
            c.set("V", quark, pair, (g * charge) * g_q * prop)
    return c

```

## card.json

```json
{
  "name": "zprime_lmu_ltau_explicit_mixing",
  "mediators": [
    {
      "structure": "V",
      "mass_mev": 10.0,
      "nu_couplings": {
        "ee": 0,
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
# Notes: gauged L_μ−L_τ Z′ with explicit kinetic mixing

Description: Z′ of U(1)_{L_μ−L_τ}, mass m = 10 MeV, gauge coupling g′ = 1e-4,
photon kinetic mixing ε = +1e-3 in the conventions of this pack, no direct
quark couplings.

## Derivation

1. **Lorentz structure.** A gauge boson couples through vector currents, so
   structure `V` only. No scalar, axial or tensor pieces arise.

2. **Neutrino couplings** (conventions section 7, L_μ−L_τ row). The lepton
   doublet charges are Q′_{ν_e} = 0, Q′_{ν_μ} = +1, Q′_{ν_τ} = −1, so

       g_ν^{ee} = 0,   g_ν^{mm} = +g′ = +1e-4,   g_ν^{tt} = −g′ = −1e-4,

   flavor-diagonal; all off-diagonal entries vanish. The neutrino current is
   left-handed as required.

3. **Quark couplings.** The description states there are no direct quark
   couplings (Q′_u = Q′_d = 0, consistent with L_μ−L_τ). The only quark
   coupling comes from kinetic mixing (conventions section 6):

       g_q = ε e Q_q^em,   g_u = +(2/3) ε e,   g_d = −(1/3) ε e,

   with e = E_CHARGE = 0.303 and ε = +1e-3 as given. Because the description
   gives ε explicitly "in the conventions of this pack", no sign conversion is
   needed; ε is used as written with its positive sign. The loop-induced value
   κ g′ ≈ 1.44e-6 is *not* added: the explicit ε supersedes it, exactly as in
   worked example 03, description A.

4. **Matching** (conventions section 3):

       C^{qV}_{αβ}(Q²) = g_ν^{αβ} g_q / (√2 G_F (Q² + m²)).

   Hence

       C^{uV}_{mm} = +g′ ε e (2/3) / (√2 G_F (Q² + m²)),
       C^{dV}_{mm} = −g′ ε e (1/3) / (√2 G_F (Q² + m²)),
       C^{qV}_{tt} = −C^{qV}_{mm},
       C^{qV}_{ee} = 0,   all off-diagonal and scalar entries = 0.

5. **Physics check.** The induced nucleon couplings are
   g_p = 2 g_u + g_d = ε e and g_n = g_u + 2 g_d = 0: only protons couple, as
   expected for photon-like mixing. For ν_μ the nuclear charge shift
   ΔQ^{μμ} = Z g′ ε e / (√2 G_F (Q² + m²)) is positive (g′ ε > 0), opposite in
   sign to Q_SM ≈ 0.04 Z − 0.5 N < 0, so the interference is destructive for
   ν_μ and ν̄_μ. For ν_τ the shift has the opposite sign (constructive), but
   COHERENT has no ν_τ flux. Electron neutrinos see no new physics.

   Magnitude at the first reference point (Q² = 1238 MeV², Cs-133):
   C^{uV}_{mm} = (1e-4)(1e-3)(0.303)(2/3) / (√2 · 1.16638e-11 · (1238 + 100))
   ≈ 0.915, C^{dV}_{mm} ≈ −0.458, ΔQ^{μμ} ≈ 55 · 1.37 ≈ 75, which exceeds
   |Q_SM| ≈ 37, so the ν_μ rate is enhanced at this point.

6. **Perturbativity.** |g′| = 1e-4 and |ε| = 1e-3 are far below √(4π).

7. **Card.** Expressible as a single vector mediator with the signed neutrino
   couplings, zero direct quark couplings and `kinetic_mixing` = 0.001 written
   explicitly (a card has no κ g′ default). The plugin computes the same
   coefficients from the exported constants `E_CHARGE` and `Q_EM` and does not
   read the card.

8. **SM limit.** `COUPLINGS = ("g_prime", "kinetic_mixing")`; zeroing either one
   removes all new physics, since every coefficient is proportional to g′ ε.

## Files

- `plugin.py`: coefficient plugin with `PARAMETERS`, `COUPLINGS`, `coefficients`.
- `card.json`: equivalent model card.
- `notes.md`: this derivation.

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
  "notes": "# Notes: gauged L_\u03bc\u2212L_\u03c4 Z\u2032 with explicit kinetic mixing\n\nDescription: Z\u2032 of U(1)_{L_\u03bc\u2212L_\u03c4}, mass m = 10 MeV, gauge coupling g\u2032 = 1e-4,\nphoton kinetic mixing \u03b5 = +1e-3 in the conventions of this pack, no direct\nquark couplings.\n\n## Derivation\n\n1. **Lorentz structure.** A gauge boson couples through vector currents, so\n   structure `V` only. No scalar, axial or tensor pieces arise.\n\n2. **Neutrino couplings** (conventions section 7, L_\u03bc\u2212L_\u03c4 row). The lepton\n   doublet charges are Q\u2032_{\u03bd_e} = 0, Q\u2032_{\u03bd_\u03bc} = +1, Q\u2032_{\u03bd_\u03c4} = \u22121, so\n\n       g_\u03bd^{ee} = 0,   g_\u03bd^{mm} = +g\u2032 = +1e-4,   g_\u03bd^{tt} = \u2212g\u2032 = \u22121e-4,\n\n   flavor-diagonal; all off-diagonal entries vanish. The neutrino current is\n   left-handed as required.\n\n3. **Quark couplings.** The description states there are no direct quark\n   couplings (Q\u2032_u = Q\u2032_d = 0, consistent with L_\u03bc\u2212L_\u03c4). The only quark\n   coupling comes from kinetic mixing (conventions section 6):\n\n       g_q = \u03b5 e Q_q^em,   g_u = +(2/3) \u03b5 e,   g_d = \u2212(1/3) \u03b5 e,\n\n   with e = E_CHARGE = 0.303 and \u03b5 = +1e-3 as given. Because the description\n   gives \u03b5 explicitly \"in the conventions of this pack\", no sign conversion is\n   needed; \u03b5 is used as written with its positive sign. The loop-induced value\n   \u03ba g\u2032 \u2248 1.44e-6 is *not* added: the explicit \u03b5 supersedes it, exactly as in\n   worked example 03, description A.\n\n4. **Matching** (conventions section 3):\n\n       C^{qV}_{\u03b1\u03b2}(Q\u00b2) = g_\u03bd^{\u03b1\u03b2} g_q / (\u221a2 G_F (Q\u00b2 + m\u00b2)).\n\n   Hence\n\n       C^{uV}_{mm} = +g\u2032 \u03b5 e (2/3) / (\u221a2 G_F (Q\u00b2 + m\u00b2)),\n       C^{dV}_{mm} = \u2212g\u2032 \u03b5 e (1/3) / (\u221a2 G_F (Q\u00b2 + m\u00b2)),\n       C^{qV}_{tt} = \u2212C^{qV}_{mm},\n       C^{qV}_{ee} = 0,   all off-diagonal and scalar entries = 0.\n\n5. **Physics check.** The induced nucleon couplings are\n   g_p = 2 g_u + g_d = \u03b5 e and g_n = g_u + 2 g_d = 0: only protons couple, as\n   expected for photon-like mixing. For \u03bd_\u03bc the nuclear charge shift\n   \u0394Q^{\u03bc\u03bc} = Z g\u2032 \u03b5 e / (\u221a2 G_F (Q\u00b2 + m\u00b2)) is positive (g\u2032 \u03b5 > 0), opposite in\n   sign to Q_SM \u2248 0.04 Z \u2212 0.5 N < 0, so the interference is destructive for\n   \u03bd_\u03bc and \u03bd\u0304_\u03bc. For \u03bd_\u03c4 the shift has the opposite sign (constructive), but\n   COHERENT has no \u03bd_\u03c4 flux. Electron neutrinos see no new physics.\n\n   Magnitude at the first reference point (Q\u00b2 = 1238 MeV\u00b2, Cs-133):\n   C^{uV}_{mm} = (1e-4)(1e-3)(0.303)(2/3) / (\u221a2 \u00b7 1.16638e-11 \u00b7 (1238 + 100))\n   \u2248 0.915, C^{dV}_{mm} \u2248 \u22120.458, \u0394Q^{\u03bc\u03bc} \u2248 55 \u00b7 1.37 \u2248 75, which exceeds\n   |Q_SM| \u2248 37, so the \u03bd_\u03bc rate is enhanced at this point.\n\n6. **Perturbativity.** |g\u2032| = 1e-4 and |\u03b5| = 1e-3 are far below \u221a(4\u03c0).\n\n7. **Card.** Expressible as a single vector mediator with the signed neutrino\n   couplings, zero direct quark couplings and `kinetic_mixing` = 0.001 written\n   explicitly (a card has no \u03ba g\u2032 default). The plugin computes the same\n   coefficients from the exported constants `E_CHARGE` and `Q_EM` and does not\n   read the card.\n\n8. **SM limit.** `COUPLINGS = (\"g_prime\", \"kinetic_mixing\")`; zeroing either one\n   removes all new physics, since every coefficient is proportional to g\u2032 \u03b5.\n\n## Files\n\n- `plugin.py`: coefficient plugin with `PARAMETERS`, `COUPLINGS`, `coefficients`.\n- `card.json`: equivalent model card.\n- `notes.md`: this derivation.\n"
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

### Attempt 1

```json
{
  "final_text": "Validation passed with no findings: perturbativity, SM limit, finite counts, schema, and the plugin/card cross-check on 66 Q\u00b2 points all succeeded.\n\n**Files produced** in the task directory:\n\n- `plugin.py`: vector-only coefficient plugin. Neutrino couplings follow the L_\u03bc\u2212L_\u03c4 charges (\u03bc\u03bc = +g\u2032, \u03c4\u03c4 = \u2212g\u2032, ee = 0). Quark couplings come solely from kinetic mixing, g_q = \u03b5 e Q_q^em, using the exported constants. Coefficients are g_\u03bd g_q / (\u221a2 G_F (Q\u00b2 + m\u00b2)). Both g\u2032 and \u03b5 are listed in COUPLINGS, so zeroing either recovers the SM.\n- `card.json`: one V mediator, mass 10 MeV, nu_couplings mm = 1e-4 and tt = \u22121e-4, zero direct quark couplings, kinetic_mixing = 0.001 written explicitly.\n- `notes.md`: derivation covering the charge assignment, the mixing-induced quark couplings with the sign taken as given, the matching formula, the proton-only nucleon coupling check, the destructive interference for \u03bd_\u03bc, and why the loop-induced \u03ba g\u2032 mixing is not added on top of the explicit \u03b5.\n\nThe described point (m = 10 MeV, g\u2032 = 1e-4, \u03b5 = +1e-3) is preserved exactly as the PARAMETERS defaults and in the card.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":70728,\"stop_reason\":\"end_turn\",\"session_id\":\"84e32b46-3b56-43ad-8ff6-9c3bb318fa76\",\"total_cost_usd\":0.7968212499999999,\"usage\":{\"input_tokens\":226,\"cache_creation_input_tokens\":25332,\"cache_read_input_tokens\":140485,\"output_tokens\":5056,\"output_tokens_details\":{\"thinking_tokens\":259},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":25332,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":477,\"cache_read_input_tokens\":31041,\"cache_creation_input_tokens\":297,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":297},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":226,\"outputTokens\":5056,\"cacheReadInputTokens\":140485,\"cacheCreationInputTokens\":25332,\"webSearchRequests\":0,\"costUSD\":0.7968212499999999,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":259,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_018HAfpxbnXTAjzgDJkHACCT\",\"tool_input\":{\"command\":\"ls -R /private/tmp/nuagent-issue17-eval-claude/04_lmu_ltau_explicit\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":21,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings: perturbativity, SM limit, finite counts, schema, and the plugin/card cross-check on 66 Q\u00b2 points all succeeded.\\n\\n**Files produced** in the task directory:\\n\\n- `plugin.py`: vector-only coefficient plugin. Neutrino couplings follow the L_\u03bc\u2212L_\u03c4 charges (\u03bc\u03bc = +g\u2032, \u03c4\u03c4 = \u2212g\u2032, ee = 0). Quark couplings come solely from kinetic mixing, g_q = \u03b5 e Q_q^em, using the exported constants. Coefficients are g_\u03bd g_q / (\u221a2 G_F (Q\u00b2 + m\u00b2)). Both g\u2032 and \u03b5 are listed in COUPLINGS, so zeroing either recovers the SM.\\n- `card.json`: one V mediator, mass 10 MeV, nu_couplings mm = 1e-4 and tt = \u22121e-4, zero direct quark couplings, kinetic_mixing = 0.001 written explicitly.\\n- `notes.md`: derivation covering the charge assignment, the mixing-induced quark couplings with the sign taken as given, the matching formula, the proton-only nucleon coupling check, the destructive interference for \u03bd_\u03bc, and why the loop-induced \u03ba g\u2032 mixing is not added on top of the explicit \u03b5.\\n\\nThe described point (m = 10 MeV, g\u2032 = 1e-4, \u03b5 = +1e-3) is preserved exactly as the PARAMETERS defaults and in the card.\",\"ttft_ms\":3367,\"type\":\"result\",\"duration_ms\":72670,\"uuid\":\"99619698-9a5d-470f-a454-0246434cd911\",\"ttft_stream_ms\":2413,\"time_to_request_ms\":19,\"first_content_frame_ms\":2414,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/issue-17-agent-harness/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/nuagent-issue17-eval-claude/04_lmu_ltau_explicit\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
