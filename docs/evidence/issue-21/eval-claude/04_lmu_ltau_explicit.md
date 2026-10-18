Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

A gauged L_mu−L_tau vector Z′ has mass 10 MeV, gauge coupling g′ = 1e-4 and photon kinetic mixing epsilon = +1e-3 in the conventions of this pack. There are no direct quark couplings.

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
"""L_mu - L_tau vector Z' with explicit photon kinetic mixing; derivation in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2, E_CHARGE, Q_EM

PARAMETERS = {"g_prime": 1e-4, "mass_mev": 10.0, "kinetic_mixing": 1e-3}
COUPLINGS = ("g_prime", "kinetic_mixing")

# U(1)_{L_mu - L_tau} charges of the neutrino flavors (conventions section 7).
Q_PRIME_NU = {"ee": 0.0, "mm": +1.0, "tt": -1.0}


def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    eps = params["kinetic_mixing"]
    prop = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        g_q = eps * E_CHARGE * Q_EM[quark]
        for pair, charge in Q_PRIME_NU.items():
            g_nu = g * charge
            c.set("V", quark, pair, g_nu * g_q * prop)
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
# L_μ−L_τ Z′ (m = 10 MeV, g′ = 1e-4) with explicit kinetic mixing ε = +1e-3

## Description as read

A gauged U(1)_{L_μ−L_τ} vector boson Z′ with mass 10 MeV, gauge coupling g′ = 1e-4, and photon kinetic mixing ε = +1e-3 given directly in the conventions of this pack (conventions section 6). No direct quark couplings. This is the pattern of worked example 03, description A (explicit mixing), at a different parameter point.

## Derivation

1. **Lorentz structure.** A gauge boson couples through vector currents: structure `V`. The neutrino current is the left-handed doublet current, so g_ν is used as written (conventions section 3).

2. **Neutrino couplings (conventions section 7).** L_μ−L_τ charges are Q′_{ν_e} = 0, Q′_{ν_μ} = +1, Q′_{ν_τ} = −1, flavor-diagonal in the mass basis. Hence

       g_ν^{ee} = 0,   g_ν^{mm} = +g′ = +1e-4,   g_ν^{tt} = −g′ = −1e-4,

   and all off-diagonal entries vanish.

3. **Quark couplings.** The description states there are no direct quark couplings (Q′_u = Q′_d = 0), so the `quark_couplings` entries are zero. The only quark coupling is the electromagnetic one induced by kinetic mixing (conventions section 6):

       g_q = ε e Q_q^em,   g_u = ε e (2/3),   g_d = ε e (−1/3),   e = E_CHARGE = 0.303.

   Numerically g_u = +2.02e-4 and g_d = −1.01e-4 (using e = 0.303).

4. **Mixing value and sign.** The description gives ε = +1e-3 explicitly "in the conventions of this pack", i.e. it is already the low-energy coupling ε in the sense g_q → g_q + ε e Q_q^em, with the sign relative to g′ fixed as positive. No Lagrangian-convention conversion is needed. Because the description fixes the total mixing, the loop-induced piece κ g′ ≈ 1.44e-6 (section 6) is **not** added on top: the stated ε is taken as the complete mixing at the parameter point, exactly as in example 03 A. (If instead the user intended ε to be an additional UV contribution, the loop term would shift ε by about 0.14 percent; this reading was not adopted because the description gives a single explicit value.) The mixing is treated as a Q²-independent constant, which is what the card field represents.

5. **Matching (conventions section 3).**

       C^{qV}_{αβ}(Q²) = g_ν^{αβ} g_q / (√2 G_F (Q² + m²)),

   so

       C^{uV}_{mm} = +g′ ε e (2/3) / (√2 G_F (Q² + m²)),
       C^{dV}_{mm} = −g′ ε e (1/3) / (√2 G_F (Q² + m²)),
       C^{uV}_{tt} = −C^{uV}_{mm},   C^{dV}_{tt} = −C^{dV}_{mm},
       C^{qV}_{ee} = 0,   all off-diagonal and all scalar entries = 0.

   At the Cs-133 reference points (Q² = 1238, 3714, 6190 MeV²) with m² = 100 MeV²: C^{uV}_{mm} ≈ 9.15e-1, 3.21e-1, 1.95e-1 and C^{dV}_{mm} is minus half of that. (Cross-check against example 03 A, which has the same product g′ ε = 1e-7: its first-point value 7.476e-1 at Q² + m² = 1638 MeV² rescales by 1638/1338 to 9.15e-1 here.)

6. **Physics check.** The mixing-induced coupling is proportional to electric charge, so only protons feel it: 2 C^u + C^d ∝ 2(2/3) − 1/3 = 1 and C^u + 2 C^d ∝ 2/3 − 2/3 = 0. The ν_μ nuclear charge shift ΔQ^{mm} = Z g′ ε e / (√2 G_F (Q² + m²)) is positive for ε g′ > 0, while Q_SM is negative: destructive interference for ν_μ and ν̄_μ. Electron neutrinos see no new physics. For COHERENT the prompt ν_μ and delayed ν̄_μ are affected; the delayed ν_e are not. No axial coupling is dropped, since the induced coupling is purely vector.

7. **Perturbativity.** |g′| = 1e-4 and |ε| = 1e-3 are far below √(4π).

8. **SM limit.** `COUPLINGS = ("g_prime", "kinetic_mixing")`; zeroing either removes the entire new-physics contribution, so the SM-limit check is satisfied.

## Card

Expressible: one vector mediator with the neutrino flavor matrix above, zero direct quark couplings and `kinetic_mixing: 0.001`. The card carries the mixing explicitly (a card has no κ g′ default). The plugin computes the same coefficients independently from the exported constants `GF`, `SQRT2`, `E_CHARGE` and `Q_EM`.

## Files

- `plugin.py`: coefficient plugin, parameters `g_prime = 1e-4`, `mass_mev = 10.0`, `kinetic_mixing = 1e-3`.
- `card.json`: equivalent model card.
- `notes.md`: this derivation.

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
  "notes": "# L_\u03bc\u2212L_\u03c4 Z\u2032 (m = 10 MeV, g\u2032 = 1e-4) with explicit kinetic mixing \u03b5 = +1e-3\n\n## Description as read\n\nA gauged U(1)_{L_\u03bc\u2212L_\u03c4} vector boson Z\u2032 with mass 10 MeV, gauge coupling g\u2032 = 1e-4, and photon kinetic mixing \u03b5 = +1e-3 given directly in the conventions of this pack (conventions section 6). No direct quark couplings. This is the pattern of worked example 03, description A (explicit mixing), at a different parameter point.\n\n## Derivation\n\n1. **Lorentz structure.** A gauge boson couples through vector currents: structure `V`. The neutrino current is the left-handed doublet current, so g_\u03bd is used as written (conventions section 3).\n\n2. **Neutrino couplings (conventions section 7).** L_\u03bc\u2212L_\u03c4 charges are Q\u2032_{\u03bd_e} = 0, Q\u2032_{\u03bd_\u03bc} = +1, Q\u2032_{\u03bd_\u03c4} = \u22121, flavor-diagonal in the mass basis. Hence\n\n       g_\u03bd^{ee} = 0,   g_\u03bd^{mm} = +g\u2032 = +1e-4,   g_\u03bd^{tt} = \u2212g\u2032 = \u22121e-4,\n\n   and all off-diagonal entries vanish.\n\n3. **Quark couplings.** The description states there are no direct quark couplings (Q\u2032_u = Q\u2032_d = 0), so the `quark_couplings` entries are zero. The only quark coupling is the electromagnetic one induced by kinetic mixing (conventions section 6):\n\n       g_q = \u03b5 e Q_q^em,   g_u = \u03b5 e (2/3),   g_d = \u03b5 e (\u22121/3),   e = E_CHARGE = 0.303.\n\n   Numerically g_u = +2.02e-4 and g_d = \u22121.01e-4 (using e = 0.303).\n\n4. **Mixing value and sign.** The description gives \u03b5 = +1e-3 explicitly \"in the conventions of this pack\", i.e. it is already the low-energy coupling \u03b5 in the sense g_q \u2192 g_q + \u03b5 e Q_q^em, with the sign relative to g\u2032 fixed as positive. No Lagrangian-convention conversion is needed. Because the description fixes the total mixing, the loop-induced piece \u03ba g\u2032 \u2248 1.44e-6 (section 6) is **not** added on top: the stated \u03b5 is taken as the complete mixing at the parameter point, exactly as in example 03 A. (If instead the user intended \u03b5 to be an additional UV contribution, the loop term would shift \u03b5 by about 0.14 percent; this reading was not adopted because the description gives a single explicit value.) The mixing is treated as a Q\u00b2-independent constant, which is what the card field represents.\n\n5. **Matching (conventions section 3).**\n\n       C^{qV}_{\u03b1\u03b2}(Q\u00b2) = g_\u03bd^{\u03b1\u03b2} g_q / (\u221a2 G_F (Q\u00b2 + m\u00b2)),\n\n   so\n\n       C^{uV}_{mm} = +g\u2032 \u03b5 e (2/3) / (\u221a2 G_F (Q\u00b2 + m\u00b2)),\n       C^{dV}_{mm} = \u2212g\u2032 \u03b5 e (1/3) / (\u221a2 G_F (Q\u00b2 + m\u00b2)),\n       C^{uV}_{tt} = \u2212C^{uV}_{mm},   C^{dV}_{tt} = \u2212C^{dV}_{mm},\n       C^{qV}_{ee} = 0,   all off-diagonal and all scalar entries = 0.\n\n   At the Cs-133 reference points (Q\u00b2 = 1238, 3714, 6190 MeV\u00b2) with m\u00b2 = 100 MeV\u00b2: C^{uV}_{mm} \u2248 9.15e-1, 3.21e-1, 1.95e-1 and C^{dV}_{mm} is minus half of that. (Cross-check against example 03 A, which has the same product g\u2032 \u03b5 = 1e-7: its first-point value 7.476e-1 at Q\u00b2 + m\u00b2 = 1638 MeV\u00b2 rescales by 1638/1338 to 9.15e-1 here.)\n\n6. **Physics check.** The mixing-induced coupling is proportional to electric charge, so only protons feel it: 2 C^u + C^d \u221d 2(2/3) \u2212 1/3 = 1 and C^u + 2 C^d \u221d 2/3 \u2212 2/3 = 0. The \u03bd_\u03bc nuclear charge shift \u0394Q^{mm} = Z g\u2032 \u03b5 e / (\u221a2 G_F (Q\u00b2 + m\u00b2)) is positive for \u03b5 g\u2032 > 0, while Q_SM is negative: destructive interference for \u03bd_\u03bc and \u03bd\u0304_\u03bc. Electron neutrinos see no new physics. For COHERENT the prompt \u03bd_\u03bc and delayed \u03bd\u0304_\u03bc are affected; the delayed \u03bd_e are not. No axial coupling is dropped, since the induced coupling is purely vector.\n\n7. **Perturbativity.** |g\u2032| = 1e-4 and |\u03b5| = 1e-3 are far below \u221a(4\u03c0).\n\n8. **SM limit.** `COUPLINGS = (\"g_prime\", \"kinetic_mixing\")`; zeroing either removes the entire new-physics contribution, so the SM-limit check is satisfied.\n\n## Card\n\nExpressible: one vector mediator with the neutrino flavor matrix above, zero direct quark couplings and `kinetic_mixing: 0.001`. The card carries the mixing explicitly (a card has no \u03ba g\u2032 default). The plugin computes the same coefficients independently from the exported constants `GF`, `SQRT2`, `E_CHARGE` and `Q_EM`.\n\n## Files\n\n- `plugin.py`: coefficient plugin, parameters `g_prime = 1e-4`, `mass_mev = 10.0`, `kinetic_mixing = 1e-3`.\n- `card.json`: equivalent model card.\n- `notes.md`: this derivation.\n"
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
  "final_text": "Validation passed with no findings, including the plugin-versus-card cross-check on 66 Q\u00b2 points.\n\n**Translation.** The Z\u2032 is a vector mediator with L_\u03bc\u2212L_\u03c4 charges, so the neutrino couplings are +g\u2032 for \u03bd_\u03bc, \u2212g\u2032 for \u03bd_\u03c4, and zero for \u03bd_e. Quarks couple only through the stated kinetic mixing, giving g_u = \u03b5 e (2/3) and g_d = \u03b5 e (\u22121/3). The description fixes \u03b5 = +1e-3 as the total low-energy mixing, so the loop-induced \u03ba g\u2032 term is not added on top. This is the same pattern as worked example 03 A. The induced coupling is proton-only and interferes destructively with the Standard Model for \u03bd_\u03bc and \u03bd\u0304_\u03bc.\n\n**Files produced** in the task directory:\n\n- `plugin.py`: coefficient plugin with defaults g_prime = 1e-4, mass_mev = 10.0, kinetic_mixing = 1e-3, both couplings listed in COUPLINGS.\n- `card.json`: one vector mediator at 10 MeV, neutrino matrix (0, +1e-4, \u22121e-4), zero direct quark couplings, kinetic mixing 0.001.\n- `notes.md`: derivation, sign and convention discussion, reference-point values cross-checked by rescaling example 03 A, and the reason the loop mixing is not added.\n\nNo questions.json was written, since the description is complete.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":83627,\"stop_reason\":\"end_turn\",\"session_id\":\"32c3ab26-47b8-4ac8-ac80-714d324fb3be\",\"total_cost_usd\":1.01563575,\"usage\":{\"input_tokens\":226,\"cache_creation_input_tokens\":31200,\"cache_read_input_tokens\":158703,\"output_tokens\":6994,\"output_tokens_details\":{\"thinking_tokens\":884},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":31200,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":480,\"cache_read_input_tokens\":35782,\"cache_creation_input_tokens\":1424,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":1424},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":226,\"outputTokens\":6994,\"cacheReadInputTokens\":158703,\"cacheCreationInputTokens\":31200,\"webSearchRequests\":0,\"costUSD\":1.01563575,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":884,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01CfHVvmihETa9qWAHDtBj4x\",\"tool_input\":{\"command\":\"ls -R /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/04_lmu_ltau_explicit\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":20,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings, including the plugin-versus-card cross-check on 66 Q\u00b2 points.\\n\\n**Translation.** The Z\u2032 is a vector mediator with L_\u03bc\u2212L_\u03c4 charges, so the neutrino couplings are +g\u2032 for \u03bd_\u03bc, \u2212g\u2032 for \u03bd_\u03c4, and zero for \u03bd_e. Quarks couple only through the stated kinetic mixing, giving g_u = \u03b5 e (2/3) and g_d = \u03b5 e (\u22121/3). The description fixes \u03b5 = +1e-3 as the total low-energy mixing, so the loop-induced \u03ba g\u2032 term is not added on top. This is the same pattern as worked example 03 A. The induced coupling is proton-only and interferes destructively with the Standard Model for \u03bd_\u03bc and \u03bd\u0304_\u03bc.\\n\\n**Files produced** in the task directory:\\n\\n- `plugin.py`: coefficient plugin with defaults g_prime = 1e-4, mass_mev = 10.0, kinetic_mixing = 1e-3, both couplings listed in COUPLINGS.\\n- `card.json`: one vector mediator at 10 MeV, neutrino matrix (0, +1e-4, \u22121e-4), zero direct quark couplings, kinetic mixing 0.001.\\n- `notes.md`: derivation, sign and convention discussion, reference-point values cross-checked by rescaling example 03 A, and the reason the loop mixing is not added.\\n\\nNo questions.json was written, since the description is complete.\",\"ttft_ms\":854,\"type\":\"result\",\"duration_ms\":87671,\"uuid\":\"9aa29fbf-237e-4673-8421-872b2e910aa3\",\"ttft_stream_ms\":763,\"time_to_request_ms\":24,\"first_content_frame_ms\":763,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/m7-conversation/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/04_lmu_ltau_explicit\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
