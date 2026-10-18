Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

A gauged L_mu−L_tau vector Z′ has mass 15 MeV and gauge coupling g′ = 6e-5. Use only the positive loop-induced photon mixing in the Q² → 0 limit; there is no tree-level mixing or direct quark coupling.

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
"""L_mu - L_tau Z' with loop-induced photon mixing in the Q^2 -> 0 limit.

Derivation in notes.md.
"""

import numpy as np

from nuagent.physics.couplings import (
    Coefficients,
    GF,
    SQRT2,
    E_CHARGE,
    Q_EM,
    KAPPA_LMU_LTAU,
)

PARAMETERS = {"g_prime": 6e-5, "mass_mev": 15.0}
COUPLINGS = ("g_prime",)

# L_mu - L_tau charges of the lepton doublets (neutrinos); quarks are uncharged.
Q_PRIME_NU = {"ee": 0.0, "mm": +1.0, "tt": -1.0}


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    # Positive loop-induced kinetic mixing at Q^2 << m_mu^2 (conventions section 6).
    eps = KAPPA_LMU_LTAU * g
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
  "name": "zprime_lmu_ltau_loop_limit",
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
      "kinetic_mixing": 8.66505823932692e-07
    }
  ],
  "contacts": []
}

```

## notes.md

```text
# Notes: gauged L_μ−L_τ Z′ with loop-induced photon mixing (Q² → 0 limit)

## Description

Z′ of mass m = 15 MeV, gauge coupling g′ = 6e-5, gauging U(1)_{L_μ−L_τ}. Quark couplings
arise only from the positive loop-induced kinetic mixing with the photon, evaluated in the
Q² → 0 limit (Q² ≪ m_μ²). No tree-level kinetic mixing, no direct quark charge.

## Derivation

1. **Lorentz structure.** Z′ is a vector boson coupling to vector currents: structure `V`.
   The neutrino coupling is to the left-handed lepton doublet, so g_ν is the full
   coefficient of ν̄ γ^μ P_L ν with no factor-2 conversion needed.

2. **Charges (conventions §7).** Q′_{ν_e} = 0, Q′_{ν_μ} = +1, Q′_{ν_τ} = −1, Q′_u = Q′_d = 0.
   Hence g_ν^{ee} = 0, g_ν^{mm} = +g′ = 6e-5, g_ν^{tt} = −g′ = −6e-5, off-diagonal zero.
   Direct quark couplings vanish. (L_μ−L_τ is anomaly-free; nothing to flag.)

3. **Loop-induced mixing (conventions §6).** Muon and tau loops mix Z′ with the photon.
   At Q² ≪ m_μ² the mixing is a constant,

       ε = +κ g′,   κ = e ln(m_τ/m_μ)/(6π²) = 0.0144417637…,

   so ε = κ · 6e-5 = 8.66505823932692e-07. The sign is a physics result: the
   m-independent part of the loop cancels because Σ_ℓ Q′_ℓ Q_ℓ = 0, and the surviving
   ln(m_τ²/m_μ²) term is positive. The description's "positive loop-induced photon mixing"
   matches this. In the plugin ε is computed from the exported `KAPPA_LMU_LTAU` times
   `g_prime`; the card has no default and carries the numerical value at full precision so
   the plugin/card cross-check passes at 1e-6.

4. **Induced quark couplings.** g_q = ε e Q_q^em with e = 0.303:
   g_u = +(2/3) ε e, g_d = −(1/3) ε e. Nucleon level: g_p = 2g_u + g_d = ε e, g_n = g_u + 2g_d = 0.
   Only protons couple, as expected for photon-mediated mixing.

5. **Matching (conventions §3).**

       C^{qV}_{αβ}(Q²) = g_ν^{αβ} g_q / (√2 G_F (Q² + m²)),

   giving C^{uV}_{mm} = +g′ ε e (2/3)/(√2 G_F (Q² + m²)),
   C^{dV}_{mm} = −g′ ε e (1/3)/(√2 G_F (Q² + m²)),
   C^{qV}_{tt} = −C^{qV}_{mm}, C^{qV}_{ee} = 0, all off-diagonal and scalar entries zero.
   The propagator is 1/(Q² + m²) with Q² = 2MT spacelike.

6. **Momentum dependence of the mixing.** The description fixes the Q² → 0 limit, so ε is
   constant and no running of the loop function with Q² is included. This is why the card
   can express the interaction (a `kinetic_mixing` number); the Q²-dependent case would need
   a plugin only.

7. **Interference.** For ν_μ the nuclear charge shift is ΔQ = Z g′ ε e/(√2 G_F (Q² + m²)) > 0
   since g′ ε = κ g′² > 0, opposite in sign to Q_SM < 0: destructive interference for the
   COHERENT prompt ν_μ and delayed ν̄_μ. ν_e sees no new physics. The ττ entries do not
   contribute at COHERENT (no ν_τ flux) but are included for completeness.

8. **SM limit and perturbativity.** `g_prime` is the only coupling; ε is derived from it, so
   zeroing `COUPLINGS` removes all new physics. |g′| = 6e-5 ≪ √(4π).

## Files

- `plugin.py`: coefficients from exported constants (`KAPPA_LMU_LTAU`, `E_CHARGE`, `Q_EM`,
  `GF`, `SQRT2`); parameters `g_prime = 6e-5`, `mass_mev = 15.0`.
- `card.json`: one vector mediator, ν couplings (0, +6e-5, −6e-5), zero direct quark
  couplings, `kinetic_mixing` = κ g′ = 8.66505823932692e-07.

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
  "notes": "# Notes: gauged L_\u03bc\u2212L_\u03c4 Z\u2032 with loop-induced photon mixing (Q\u00b2 \u2192 0 limit)\n\n## Description\n\nZ\u2032 of mass m = 15 MeV, gauge coupling g\u2032 = 6e-5, gauging U(1)_{L_\u03bc\u2212L_\u03c4}. Quark couplings\narise only from the positive loop-induced kinetic mixing with the photon, evaluated in the\nQ\u00b2 \u2192 0 limit (Q\u00b2 \u226a m_\u03bc\u00b2). No tree-level kinetic mixing, no direct quark charge.\n\n## Derivation\n\n1. **Lorentz structure.** Z\u2032 is a vector boson coupling to vector currents: structure `V`.\n   The neutrino coupling is to the left-handed lepton doublet, so g_\u03bd is the full\n   coefficient of \u03bd\u0304 \u03b3^\u03bc P_L \u03bd with no factor-2 conversion needed.\n\n2. **Charges (conventions \u00a77).** Q\u2032_{\u03bd_e} = 0, Q\u2032_{\u03bd_\u03bc} = +1, Q\u2032_{\u03bd_\u03c4} = \u22121, Q\u2032_u = Q\u2032_d = 0.\n   Hence g_\u03bd^{ee} = 0, g_\u03bd^{mm} = +g\u2032 = 6e-5, g_\u03bd^{tt} = \u2212g\u2032 = \u22126e-5, off-diagonal zero.\n   Direct quark couplings vanish. (L_\u03bc\u2212L_\u03c4 is anomaly-free; nothing to flag.)\n\n3. **Loop-induced mixing (conventions \u00a76).** Muon and tau loops mix Z\u2032 with the photon.\n   At Q\u00b2 \u226a m_\u03bc\u00b2 the mixing is a constant,\n\n       \u03b5 = +\u03ba g\u2032,   \u03ba = e ln(m_\u03c4/m_\u03bc)/(6\u03c0\u00b2) = 0.0144417637\u2026,\n\n   so \u03b5 = \u03ba \u00b7 6e-5 = 8.66505823932692e-07. The sign is a physics result: the\n   m-independent part of the loop cancels because \u03a3_\u2113 Q\u2032_\u2113 Q_\u2113 = 0, and the surviving\n   ln(m_\u03c4\u00b2/m_\u03bc\u00b2) term is positive. The description's \"positive loop-induced photon mixing\"\n   matches this. In the plugin \u03b5 is computed from the exported `KAPPA_LMU_LTAU` times\n   `g_prime`; the card has no default and carries the numerical value at full precision so\n   the plugin/card cross-check passes at 1e-6.\n\n4. **Induced quark couplings.** g_q = \u03b5 e Q_q^em with e = 0.303:\n   g_u = +(2/3) \u03b5 e, g_d = \u2212(1/3) \u03b5 e. Nucleon level: g_p = 2g_u + g_d = \u03b5 e, g_n = g_u + 2g_d = 0.\n   Only protons couple, as expected for photon-mediated mixing.\n\n5. **Matching (conventions \u00a73).**\n\n       C^{qV}_{\u03b1\u03b2}(Q\u00b2) = g_\u03bd^{\u03b1\u03b2} g_q / (\u221a2 G_F (Q\u00b2 + m\u00b2)),\n\n   giving C^{uV}_{mm} = +g\u2032 \u03b5 e (2/3)/(\u221a2 G_F (Q\u00b2 + m\u00b2)),\n   C^{dV}_{mm} = \u2212g\u2032 \u03b5 e (1/3)/(\u221a2 G_F (Q\u00b2 + m\u00b2)),\n   C^{qV}_{tt} = \u2212C^{qV}_{mm}, C^{qV}_{ee} = 0, all off-diagonal and scalar entries zero.\n   The propagator is 1/(Q\u00b2 + m\u00b2) with Q\u00b2 = 2MT spacelike.\n\n6. **Momentum dependence of the mixing.** The description fixes the Q\u00b2 \u2192 0 limit, so \u03b5 is\n   constant and no running of the loop function with Q\u00b2 is included. This is why the card\n   can express the interaction (a `kinetic_mixing` number); the Q\u00b2-dependent case would need\n   a plugin only.\n\n7. **Interference.** For \u03bd_\u03bc the nuclear charge shift is \u0394Q = Z g\u2032 \u03b5 e/(\u221a2 G_F (Q\u00b2 + m\u00b2)) > 0\n   since g\u2032 \u03b5 = \u03ba g\u2032\u00b2 > 0, opposite in sign to Q_SM < 0: destructive interference for the\n   COHERENT prompt \u03bd_\u03bc and delayed \u03bd\u0304_\u03bc. \u03bd_e sees no new physics. The \u03c4\u03c4 entries do not\n   contribute at COHERENT (no \u03bd_\u03c4 flux) but are included for completeness.\n\n8. **SM limit and perturbativity.** `g_prime` is the only coupling; \u03b5 is derived from it, so\n   zeroing `COUPLINGS` removes all new physics. |g\u2032| = 6e-5 \u226a \u221a(4\u03c0).\n\n## Files\n\n- `plugin.py`: coefficients from exported constants (`KAPPA_LMU_LTAU`, `E_CHARGE`, `Q_EM`,\n  `GF`, `SQRT2`); parameters `g_prime = 6e-5`, `mass_mev = 15.0`.\n- `card.json`: one vector mediator, \u03bd couplings (0, +6e-5, \u22126e-5), zero direct quark\n  couplings, `kinetic_mixing` = \u03ba g\u2032 = 8.66505823932692e-07.\n"
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
      10.148208050781353,
      16.55852403834421,
      21.773427166836225,
      22.11332829519325,
      17.855833268941264,
      13.097201752765098,
      10.069995610354812,
      7.573949082804692,
      5.598204309854373,
      4.079141603945456,
      2.9302287025990035,
      2.0722741626848715
    ],
    "predicted_total": 133.8703160451046,
    "observed_total": 855.0,
    "delta": 0.0008733057057774829,
    "new_physics": {
      "neg2_log_likelihood": 17.378310805139797,
      "alpha": 0.07102870494376812,
      "beta": {
        "steady_state": 0.0015417276086680028,
        "prompt_neutrons": 0.002889571388950039
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
  "final_text": "Validation passed with no findings. The task is complete.\n\n**Outcome:** a vector L_\u03bc\u2212L_\u03c4 Z\u2032 translation at the described point, cross-validated between plugin and card on 66 Q\u00b2 points.\n\nFiles produced in the task directory:\n\n- **plugin.py** computes vector coefficients for the \u03bc\u03bc and \u03c4\u03c4 entries (\u03c4\u03c4 is the negative of \u03bc\u03bc, ee is zero) on u and d quarks. The quark coupling is the loop-induced mixing \u03b5 e Q_q with \u03b5 = \u03ba g\u2032 built from the exported `KAPPA_LMU_LTAU`. Defaults are g\u2032 = 6e-5 and mass 15 MeV, and g\u2032 is the only coupling, so zeroing it recovers the Standard Model.\n- **card.json** has one vector mediator with neutrino couplings (0, +6e-5, \u22126e-5), zero direct quark couplings, and kinetic mixing written explicitly at full precision as \u03ba g\u2032 = 8.66505823932692e-07.\n- **notes.md** records the derivation: L_\u03bc\u2212L_\u03c4 charges, the positive sign of the loop mixing from the ln(m_\u03c4\u00b2/m_\u03bc\u00b2) term, the proton-only induced coupling, the matching formula, the constant-\u03b5 Q\u00b2 \u2192 0 assumption that makes the card expressible, and the destructive interference for \u03bd_\u03bc at COHERENT.\n\nNo questions.json was needed since the description fixed the mass, coupling, Lorentz structure and mixing sign.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":83706,\"stop_reason\":\"end_turn\",\"session_id\":\"304cc16f-d735-457e-bc0c-1a1d7e81ab05\",\"total_cost_usd\":0.88954075,\"usage\":{\"input_tokens\":194,\"cache_creation_input_tokens\":27368,\"cache_read_input_tokens\":128563,\"output_tokens\":6162,\"output_tokens_details\":{\"thinking_tokens\":732},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":27368,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":480,\"cache_read_input_tokens\":33047,\"cache_creation_input_tokens\":327,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":327},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":194,\"outputTokens\":6162,\"cacheReadInputTokens\":128563,\"cacheCreationInputTokens\":27368,\"webSearchRequests\":0,\"costUSD\":0.88954075,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":732,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01NRrzZNnBpDxfFQf5GSYeCd\",\"tool_input\":{\"command\":\"ls -R /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/05_lmu_ltau_loop_limit\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":18,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings. The task is complete.\\n\\n**Outcome:** a vector L_\u03bc\u2212L_\u03c4 Z\u2032 translation at the described point, cross-validated between plugin and card on 66 Q\u00b2 points.\\n\\nFiles produced in the task directory:\\n\\n- **plugin.py** computes vector coefficients for the \u03bc\u03bc and \u03c4\u03c4 entries (\u03c4\u03c4 is the negative of \u03bc\u03bc, ee is zero) on u and d quarks. The quark coupling is the loop-induced mixing \u03b5 e Q_q with \u03b5 = \u03ba g\u2032 built from the exported `KAPPA_LMU_LTAU`. Defaults are g\u2032 = 6e-5 and mass 15 MeV, and g\u2032 is the only coupling, so zeroing it recovers the Standard Model.\\n- **card.json** has one vector mediator with neutrino couplings (0, +6e-5, \u22126e-5), zero direct quark couplings, and kinetic mixing written explicitly at full precision as \u03ba g\u2032 = 8.66505823932692e-07.\\n- **notes.md** records the derivation: L_\u03bc\u2212L_\u03c4 charges, the positive sign of the loop mixing from the ln(m_\u03c4\u00b2/m_\u03bc\u00b2) term, the proton-only induced coupling, the matching formula, the constant-\u03b5 Q\u00b2 \u2192 0 assumption that makes the card expressible, and the destructive interference for \u03bd_\u03bc at COHERENT.\\n\\nNo questions.json was needed since the description fixed the mass, coupling, Lorentz structure and mixing sign.\",\"ttft_ms\":4031,\"type\":\"result\",\"duration_ms\":87318,\"uuid\":\"1219ea42-393b-4d52-9feb-b08f59799e66\",\"ttft_stream_ms\":2372,\"time_to_request_ms\":31,\"first_content_frame_ms\":2373,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/m7-conversation/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/05_lmu_ltau_loop_limit\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
