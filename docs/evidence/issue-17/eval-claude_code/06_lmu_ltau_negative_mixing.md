Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

A gauged L_mu−L_tau vector Z′ has mass 15 MeV, gauge coupling g′ = 6e-5, total photon kinetic mixing epsilon = −5e-4 in the pack's convention, and no direct quark couplings.

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
"""L_mu - L_tau Z' with an explicitly given total photon kinetic mixing; derivation in notes.md."""

import numpy as np

from nuagent.physics.couplings import Coefficients, GF, SQRT2, E_CHARGE, Q_EM

PARAMETERS = {"g_prime": 6e-5, "mass_mev": 15.0, "kinetic_mixing": -5e-4}
COUPLINGS = ("g_prime", "kinetic_mixing")

# L_mu - L_tau charges of the lepton doublets (conventions section 7).
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
# L_μ−L_τ Z′ with negative total kinetic mixing

Description: gauged L_μ−L_τ vector Z′, m = 15 MeV, g′ = 6e-5, total photon
kinetic mixing ε = −5e-4 in the pack's convention (conventions section 6), no
direct quark couplings.

## Derivation

1. **Lorentz structure.** Gauge boson exchange: vector, structure `V`. No scalar
   entries. The neutrino coupling is to the left-handed lepton doublet, so g_ν
   as written enters the matching formula directly.

2. **Neutrino couplings** (conventions section 7, L_μ−L_τ row): Q′_{ν_e} = 0,
   Q′_{ν_μ} = +1, Q′_{ν_τ} = −1. Hence
   g_ν^{ee} = 0, g_ν^{mm} = +g′ = +6e-5, g_ν^{tt} = −g′ = −6e-5, off-diagonal
   entries zero (charges are flavor-diagonal in the mass basis).

3. **Quark couplings.** The U(1)′ charges of the quarks vanish and the
   description states there are no direct quark couplings, so the only quark
   coupling is through the photon kinetic mixing (section 6):
   g_q = ε e Q_q^em with e = 0.303 (E_CHARGE), Q_u = +2/3, Q_d = −1/3.
   With ε = −5e-4: g_u = −1.01e-4, g_d = +5.05e-5.

   The description gives ε as the **total** mixing already in the pack's
   convention, so it is used as given. No loop-induced κ g′ term is added on
   top: κ g′ = +8.665e-7 is part of the stated total, which means the tree-level
   piece of the mixing in this model is about −5.0087e-4. No convention
   conversion is needed because the value is already in the section 6 sign
   convention (g_q → g_q + ε e Q_q^em). The negative sign is a model input and
   is preserved exactly.

4. **Matching** (section 3):

       C^{qV}_{αβ}(Q²) = g_ν^{αβ} ε e Q_q^em / (√2 G_F (Q² + m²)),   m = 15 MeV.

   Explicitly, with g′ ε e = 6e-5 × (−5e-4) × 0.303 = −9.09e-9:

       C^{uV}_mm = −C^{uV}_tt = (2/3) g′ ε e / (√2 G_F (Q² + m²))
       C^{dV}_mm = −C^{dV}_tt = (−1/3) g′ ε e / (√2 G_F (Q² + m²))
       C^{uV}_ee = C^{dV}_ee = 0, all off-diagonal entries zero.

   Approximate values at the Cs-133 reference points (Q² = 1238, 3714, 6190
   MeV²; m² = 225 MeV²), rounded for reading:

   | coefficient | Q²=1238 | Q²=3714 | Q²=6190 |
   |---|---|---|---|
   | C^{uV}_mm = −C^{uV}_tt | −2.511e-01 | −9.327e-02 | −5.727e-02 |
   | C^{dV}_mm = −C^{dV}_tt | +1.256e-01 | +4.663e-02 | +2.863e-02 |

5. **Physics check.** Photon-like mixing couples only to protons:
   2 C^u + C^d ∝ 2(2/3) − 1/3 = 1 and C^u + 2 C^d ∝ 2/3 − 2/3 = 0, so the
   neutron coupling vanishes. For ν_μ the nuclear charge shift is
   ΔQ^{μμ} = Z g′ ε e / (√2 G_F (Q² + m²)), which is **negative** because
   ε g′ < 0. Q_SM ≈ −36.9 for Cs-133 is also negative, so the interference is
   **constructive** for ν_μ (and ν̄_μ): at the first reference point
   ΔQ^{μμ} ≈ −20.7, giving a total charge ≈ −57.6 and a ν_μ rate about 2.4
   times the SM there. This is the opposite of the positive-ε cases of
   examples 03 and 06, where the ν_μ interference is destructive. For ν_τ the
   shift has the opposite sign (destructive), but there is no ν_τ flux at
   COHERENT. Electron neutrinos see no new physics.

6. **Card.** The card schema expresses this exactly: one vector mediator with
   the L_μ−L_τ neutrino charges times g′, zero direct quark couplings and
   `kinetic_mixing` = −5e-4. Because ε is a constant here (not the
   Q²-dependent loop function of example 06), plugin and card describe
   identical coefficients.

7. **Plugin.** PARAMETERS are the described point (g′ = 6e-5, m = 15 MeV,
   ε = −5e-4). Both `g_prime` and `kinetic_mixing` are listed in COUPLINGS;
   zeroing either removes all new physics, which satisfies the SM-limit check.
   All couplings are far below √(4π). Each coefficient entry is set exactly once.

8. **Dropped pieces.** None: the Z′ coupling to quarks via photon mixing is
   purely vector, so there is no axial part to discard. The Z′ also couples to
   charged muons and taus, which does not enter CEvNS.

## Validation

`nuagent validate --plugin plugin.py --card card.json` run in this directory;
result recorded in the final summary.

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
  "notes": "# L_\u03bc\u2212L_\u03c4 Z\u2032 with negative total kinetic mixing\n\nDescription: gauged L_\u03bc\u2212L_\u03c4 vector Z\u2032, m = 15 MeV, g\u2032 = 6e-5, total photon\nkinetic mixing \u03b5 = \u22125e-4 in the pack's convention (conventions section 6), no\ndirect quark couplings.\n\n## Derivation\n\n1. **Lorentz structure.** Gauge boson exchange: vector, structure `V`. No scalar\n   entries. The neutrino coupling is to the left-handed lepton doublet, so g_\u03bd\n   as written enters the matching formula directly.\n\n2. **Neutrino couplings** (conventions section 7, L_\u03bc\u2212L_\u03c4 row): Q\u2032_{\u03bd_e} = 0,\n   Q\u2032_{\u03bd_\u03bc} = +1, Q\u2032_{\u03bd_\u03c4} = \u22121. Hence\n   g_\u03bd^{ee} = 0, g_\u03bd^{mm} = +g\u2032 = +6e-5, g_\u03bd^{tt} = \u2212g\u2032 = \u22126e-5, off-diagonal\n   entries zero (charges are flavor-diagonal in the mass basis).\n\n3. **Quark couplings.** The U(1)\u2032 charges of the quarks vanish and the\n   description states there are no direct quark couplings, so the only quark\n   coupling is through the photon kinetic mixing (section 6):\n   g_q = \u03b5 e Q_q^em with e = 0.303 (E_CHARGE), Q_u = +2/3, Q_d = \u22121/3.\n   With \u03b5 = \u22125e-4: g_u = \u22121.01e-4, g_d = +5.05e-5.\n\n   The description gives \u03b5 as the **total** mixing already in the pack's\n   convention, so it is used as given. No loop-induced \u03ba g\u2032 term is added on\n   top: \u03ba g\u2032 = +8.665e-7 is part of the stated total, which means the tree-level\n   piece of the mixing in this model is about \u22125.0087e-4. No convention\n   conversion is needed because the value is already in the section 6 sign\n   convention (g_q \u2192 g_q + \u03b5 e Q_q^em). The negative sign is a model input and\n   is preserved exactly.\n\n4. **Matching** (section 3):\n\n       C^{qV}_{\u03b1\u03b2}(Q\u00b2) = g_\u03bd^{\u03b1\u03b2} \u03b5 e Q_q^em / (\u221a2 G_F (Q\u00b2 + m\u00b2)),   m = 15 MeV.\n\n   Explicitly, with g\u2032 \u03b5 e = 6e-5 \u00d7 (\u22125e-4) \u00d7 0.303 = \u22129.09e-9:\n\n       C^{uV}_mm = \u2212C^{uV}_tt = (2/3) g\u2032 \u03b5 e / (\u221a2 G_F (Q\u00b2 + m\u00b2))\n       C^{dV}_mm = \u2212C^{dV}_tt = (\u22121/3) g\u2032 \u03b5 e / (\u221a2 G_F (Q\u00b2 + m\u00b2))\n       C^{uV}_ee = C^{dV}_ee = 0, all off-diagonal entries zero.\n\n   Approximate values at the Cs-133 reference points (Q\u00b2 = 1238, 3714, 6190\n   MeV\u00b2; m\u00b2 = 225 MeV\u00b2), rounded for reading:\n\n   | coefficient | Q\u00b2=1238 | Q\u00b2=3714 | Q\u00b2=6190 |\n   |---|---|---|---|\n   | C^{uV}_mm = \u2212C^{uV}_tt | \u22122.511e-01 | \u22129.327e-02 | \u22125.727e-02 |\n   | C^{dV}_mm = \u2212C^{dV}_tt | +1.256e-01 | +4.663e-02 | +2.863e-02 |\n\n5. **Physics check.** Photon-like mixing couples only to protons:\n   2 C^u + C^d \u221d 2(2/3) \u2212 1/3 = 1 and C^u + 2 C^d \u221d 2/3 \u2212 2/3 = 0, so the\n   neutron coupling vanishes. For \u03bd_\u03bc the nuclear charge shift is\n   \u0394Q^{\u03bc\u03bc} = Z g\u2032 \u03b5 e / (\u221a2 G_F (Q\u00b2 + m\u00b2)), which is **negative** because\n   \u03b5 g\u2032 < 0. Q_SM \u2248 \u221236.9 for Cs-133 is also negative, so the interference is\n   **constructive** for \u03bd_\u03bc (and \u03bd\u0304_\u03bc): at the first reference point\n   \u0394Q^{\u03bc\u03bc} \u2248 \u221220.7, giving a total charge \u2248 \u221257.6 and a \u03bd_\u03bc rate about 2.4\n   times the SM there. This is the opposite of the positive-\u03b5 cases of\n   examples 03 and 06, where the \u03bd_\u03bc interference is destructive. For \u03bd_\u03c4 the\n   shift has the opposite sign (destructive), but there is no \u03bd_\u03c4 flux at\n   COHERENT. Electron neutrinos see no new physics.\n\n6. **Card.** The card schema expresses this exactly: one vector mediator with\n   the L_\u03bc\u2212L_\u03c4 neutrino charges times g\u2032, zero direct quark couplings and\n   `kinetic_mixing` = \u22125e-4. Because \u03b5 is a constant here (not the\n   Q\u00b2-dependent loop function of example 06), plugin and card describe\n   identical coefficients.\n\n7. **Plugin.** PARAMETERS are the described point (g\u2032 = 6e-5, m = 15 MeV,\n   \u03b5 = \u22125e-4). Both `g_prime` and `kinetic_mixing` are listed in COUPLINGS;\n   zeroing either removes all new physics, which satisfies the SM-limit check.\n   All couplings are far below \u221a(4\u03c0). Each coefficient entry is set exactly once.\n\n8. **Dropped pieces.** None: the Z\u2032 coupling to quarks via photon mixing is\n   purely vector, so there is no axial part to discard. The Z\u2032 also couples to\n   charged muons and taus, which does not enter CEvNS.\n\n## Validation\n\n`nuagent validate --plugin plugin.py --card card.json` run in this directory;\nresult recorded in the final summary.\n"
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

### Attempt 1

```json
{
  "final_text": "Validation passed with no findings. Plugin and card agree on all 66 cross-check points, the SM limit holds with couplings zeroed, and all couplings are perturbative.\n\nFiles produced in the task directory:\n\n- **plugin.py** computes vector coefficients for one Z\u2032 of mass 15 MeV. Neutrino couplings are the L_\u03bc\u2212L_\u03c4 charges times g\u2032 (\u03bc\u03bc: +6e-5, \u03c4\u03c4: \u22126e-5, ee: 0). Quark couplings come only from the photon mixing, g_q = \u03b5 e Q_q with \u03b5 = \u22125e-4 taken as the stated total. Both g\u2032 and \u03b5 are listed as couplings.\n- **card.json** expresses the same interaction: one vector mediator with the neutrino charges, zero direct quark couplings and a constant kinetic mixing of \u22125e-4.\n- **notes.md** records the derivation, the matching formula from the conventions, approximate coefficient values at the Cs-133 reference points, and the physics check. Because \u03b5 g\u2032 is negative, the proton-only shift has the same sign as the Standard Model charge, so the interference is constructive for \u03bd_\u03bc, opposite to the positive-mixing worked examples. The notes also state that the loop-induced \u03ba g\u2032 piece is not added separately since the description gives the total mixing.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":76908,\"stop_reason\":\"end_turn\",\"session_id\":\"ee32969c-f562-4167-b79a-ce800d351a61\",\"total_cost_usd\":0.895085,\"usage\":{\"input_tokens\":226,\"cache_creation_input_tokens\":27191,\"cache_read_input_tokens\":143020,\"output_tokens\":6265,\"output_tokens_details\":{\"thinking_tokens\":1045},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":27191,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":423,\"cache_read_input_tokens\":30332,\"cache_creation_input_tokens\":2865,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":2865},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":226,\"outputTokens\":6265,\"cacheReadInputTokens\":143020,\"cacheCreationInputTokens\":27191,\"webSearchRequests\":0,\"costUSD\":0.895085,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":1045,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_017xbp7KKTz9nnze2hVmZFSA\",\"tool_input\":{\"command\":\"ls -R /private/tmp/nuagent-issue17-eval-claude/06_lmu_ltau_negative_mixing\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":22,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings. Plugin and card agree on all 66 cross-check points, the SM limit holds with couplings zeroed, and all couplings are perturbative.\\n\\nFiles produced in the task directory:\\n\\n- **plugin.py** computes vector coefficients for one Z\u2032 of mass 15 MeV. Neutrino couplings are the L_\u03bc\u2212L_\u03c4 charges times g\u2032 (\u03bc\u03bc: +6e-5, \u03c4\u03c4: \u22126e-5, ee: 0). Quark couplings come only from the photon mixing, g_q = \u03b5 e Q_q with \u03b5 = \u22125e-4 taken as the stated total. Both g\u2032 and \u03b5 are listed as couplings.\\n- **card.json** expresses the same interaction: one vector mediator with the neutrino charges, zero direct quark couplings and a constant kinetic mixing of \u22125e-4.\\n- **notes.md** records the derivation, the matching formula from the conventions, approximate coefficient values at the Cs-133 reference points, and the physics check. Because \u03b5 g\u2032 is negative, the proton-only shift has the same sign as the Standard Model charge, so the interference is constructive for \u03bd_\u03bc, opposite to the positive-mixing worked examples. The notes also state that the loop-induced \u03ba g\u2032 piece is not added separately since the description gives the total mixing.\",\"ttft_ms\":1215,\"type\":\"result\",\"duration_ms\":78832,\"uuid\":\"a6b74bca-dbc6-424a-bf1f-62dccffbcd83\",\"ttft_stream_ms\":1104,\"time_to_request_ms\":21,\"first_content_frame_ms\":1104,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/issue-17-agent-harness/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/nuagent-issue17-eval-claude/06_lmu_ltau_negative_mixing\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
