Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

A gauged L_mu−L_tau vector Z′ has mass 15 MeV and gauge coupling g′ = 6e-5. Use only the positive loop-induced photon mixing in the Q² → 0 limit; there is no tree-level mixing or direct quark coupling.

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
"""U(1)_{L_mu - L_tau} Z' with loop-induced photon mixing in the Q^2 -> 0 limit.

Derivation in notes.md. Quarks couple only through the muon/tau loop mixing
epsilon = +kappa g', kappa = e ln(m_tau/m_mu) / (6 pi^2) (conventions section 6).
"""

import numpy as np

from nuagent.physics.couplings import (
    Coefficients,
    E_CHARGE,
    GF,
    KAPPA_LMU_LTAU,
    Q_EM,
    SQRT2,
)

PARAMETERS = {"g_prime": 6e-5, "mass_mev": 15.0}
COUPLINGS = ("g_prime",)

# L_mu - L_tau charges of the lepton doublets (conventions section 7).
Q_NU = {"ee": 0.0, "mm": 1.0, "tt": -1.0}


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    g = params["g_prime"]
    m = params["mass_mev"]

    # Loop-induced kinetic mixing with the photon, Q^2 << m_mu^2 limit.
    epsilon = KAPPA_LMU_LTAU * g
    g_quark = {q: epsilon * E_CHARGE * Q_EM[q] for q in ("u", "d")}

    propagator = 1.0 / (SQRT2 * GF * (q2 + m**2))
    for quark in ("u", "d"):
        for pair in ("mm", "tt"):
            g_nu = g * Q_NU[pair]
            c.set("V", quark, pair, g_nu * g_quark[quark] * propagator)
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
        "ee": 0.0,
        "mm": 6e-5,
        "tt": -6e-5,
        "em": 0,
        "et": 0,
        "mt": 0
      },
      "quark_couplings": {
        "u": 0.0,
        "d": 0.0
      },
      "kinetic_mixing": 8.66505822e-7
    }
  ],
  "contacts": []
}

```

## notes.md

```text
# Notes: gauged L_μ − L_τ Z′, loop-induced photon mixing, Q² → 0 limit

Description: a gauged U(1)_{L_μ−L_τ} vector Z′ with mass m = 15 MeV and gauge
coupling g′ = 6e-5. Quarks couple only through the positive loop-induced
mixing with the photon evaluated at Q² ≪ m_μ²; no tree-level kinetic mixing
and no direct quark charge.

## Derivation

1. **Lorentz structure.** Gauge boson exchange: structure `V`. Only the
   left-handed neutrino current enters (conventions section 3). No scalar
   contribution.

2. **Neutrino couplings** (conventions section 7, L_μ−L_τ row). The lepton
   doublets carry Q′_{L_e} = 0, Q′_{L_μ} = +1, Q′_{L_τ} = −1, so

       g_ν^{ee} = 0,   g_ν^{mm} = +g′ = +6e-5,   g_ν^{tt} = −g′ = −6e-5,

   flavor-diagonal; all off-diagonal entries vanish. The assignment is
   anomaly-free.

3. **Quark couplings.** Quarks have Q′_q = 0, so there is no tree-level
   coupling. The muon and tau loops mix Z′ with the photon. In the
   Q² ≪ m_μ² limit (conventions section 6) the mass-independent part of the
   loop cancels because Σ_ℓ Q′_ℓ Q_ℓ = 0, leaving

       ε = +κ g′,   κ = e ln(m_τ/m_μ) / (6π²) = 0.0144417637,

   so ε = 0.0144417637 × 6e-5 = 8.66505822e-7. The induced quark coupling is
   electric-charge proportional:

       g_u = ε e Q_u = ε e (+2/3),   g_d = ε e Q_d = ε e (−1/3),   e = 0.303.

   The sign is positive and is a physics result, not a convention: the lighter
   muon dominates the running of the electromagnetic coupling, and μ⁻ carries
   L_μ−L_τ charge +1 like ν_μ. The description asks explicitly for the
   positive loop-induced mixing, which agrees with this. The plugin uses the
   exported `KAPPA_LMU_LTAU`, `E_CHARGE` and `Q_EM` rather than retyped
   numbers; the card has no default for the mixing, so the value κ g′ is written
   explicitly.

4. **Matching** (conventions section 3):

       C^{qV}_{αα}(Q²) = g_ν^{αα} g_q / ( √2 G_F (Q² + m²) ),   m = 15 MeV.

   Explicitly, with g′ = 6e-5,

       C^{uV}_mm = + g′ · κ g′ e (2/3) / (√2 G_F (Q² + m²)),
       C^{dV}_mm = − g′ · κ g′ e (1/3) / (√2 G_F (Q² + m²)),
       C^{uV}_tt = − C^{uV}_mm,   C^{dV}_tt = − C^{dV}_mm,
       C^{qV}_ee = 0.

   Since the quark coupling is proportional to electric charge, the induced
   neutron charge g_n = g_u + 2 g_d = 0 vanishes identically and only the
   proton charge g_p = 2 g_u + g_d = ε e survives. The nuclear charge shift is
   therefore ΔQ^{αα} = Z g_ν^{αα} ε e / (√2 G_F (Q² + m²)).

5. **Approximate magnitude** at the Cs-133 reference point Q² = 1238 MeV²
   (for orientation only; the plugin computes at full precision):

   | quantity | value |
   |---|---|
   | g_u | ≈ +1.75e-7 |
   | g_d | ≈ −8.75e-8 |
   | C^{uV}_mm | ≈ +4.35e-4 |
   | C^{dV}_mm | ≈ −2.18e-4 |
   | ΔQ^{mm} (Cs-133, Z = 55) | ≈ +0.036 |

   Q_SM ≈ −36.9 is negative, so for ν_μ (and ν̄_μ) the shift is destructive
   but at the 0.1 percent level of the charge at this point. For ν_τ the sign
   flips (constructive), but no ν_τ flux is present at COHERENT. ν_e is
   unaffected.

6. **Validity of the Q² → 0 limit.** The description fixes the Q² → 0 form of
   the mixing, so the constant κ is used at every Q². For reference, the
   COHERENT CsI recoil range has Q² ≲ 10⁴ MeV², comparable to m_μ² ≈ 1.1e4
   MeV², where the full loop function deviates from its Q² → 0 value; that
   case would need the running-mixing plugin of example 06 and is deliberately
   not used here, as instructed.

7. **Card-expressible:** yes. One vector mediator with nu_couplings
   mm = +6e-5, tt = −6e-5, zero direct quark couplings, and
   kinetic_mixing = κ g′ = 8.66505822e-7. The framework adds ε e Q_q^em to each
   quark coupling, reproducing step 3.

## Conventions and sign checks

- No convention conversion was needed: the description gives the gauge coupling
  g′ directly and the L_μ−L_τ charges are the standard ones.
- Perturbativity: |g′| = 6e-5 ≪ √(4π).
- The axial quark coupling is zero (photon-like vector coupling), so nothing is
  dropped.
- Pitfall avoided: the pyCEvNS factor-of-2 normalization is not used; the
  matching formula of conventions section 3 is.

## Validation

`nuagent validate --plugin plugin.py --card card.json` returned valid with no
findings: perturbativity, SM limit with COUPLINGS zeroed, finite non-negative
counts, schema-valid card, and plugin/card cross-check on 66 Q² points.

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
  "notes": "# Notes: gauged L_\u03bc \u2212 L_\u03c4 Z\u2032, loop-induced photon mixing, Q\u00b2 \u2192 0 limit\n\nDescription: a gauged U(1)_{L_\u03bc\u2212L_\u03c4} vector Z\u2032 with mass m = 15 MeV and gauge\ncoupling g\u2032 = 6e-5. Quarks couple only through the positive loop-induced\nmixing with the photon evaluated at Q\u00b2 \u226a m_\u03bc\u00b2; no tree-level kinetic mixing\nand no direct quark charge.\n\n## Derivation\n\n1. **Lorentz structure.** Gauge boson exchange: structure `V`. Only the\n   left-handed neutrino current enters (conventions section 3). No scalar\n   contribution.\n\n2. **Neutrino couplings** (conventions section 7, L_\u03bc\u2212L_\u03c4 row). The lepton\n   doublets carry Q\u2032_{L_e} = 0, Q\u2032_{L_\u03bc} = +1, Q\u2032_{L_\u03c4} = \u22121, so\n\n       g_\u03bd^{ee} = 0,   g_\u03bd^{mm} = +g\u2032 = +6e-5,   g_\u03bd^{tt} = \u2212g\u2032 = \u22126e-5,\n\n   flavor-diagonal; all off-diagonal entries vanish. The assignment is\n   anomaly-free.\n\n3. **Quark couplings.** Quarks have Q\u2032_q = 0, so there is no tree-level\n   coupling. The muon and tau loops mix Z\u2032 with the photon. In the\n   Q\u00b2 \u226a m_\u03bc\u00b2 limit (conventions section 6) the mass-independent part of the\n   loop cancels because \u03a3_\u2113 Q\u2032_\u2113 Q_\u2113 = 0, leaving\n\n       \u03b5 = +\u03ba g\u2032,   \u03ba = e ln(m_\u03c4/m_\u03bc) / (6\u03c0\u00b2) = 0.0144417637,\n\n   so \u03b5 = 0.0144417637 \u00d7 6e-5 = 8.66505822e-7. The induced quark coupling is\n   electric-charge proportional:\n\n       g_u = \u03b5 e Q_u = \u03b5 e (+2/3),   g_d = \u03b5 e Q_d = \u03b5 e (\u22121/3),   e = 0.303.\n\n   The sign is positive and is a physics result, not a convention: the lighter\n   muon dominates the running of the electromagnetic coupling, and \u03bc\u207b carries\n   L_\u03bc\u2212L_\u03c4 charge +1 like \u03bd_\u03bc. The description asks explicitly for the\n   positive loop-induced mixing, which agrees with this. The plugin uses the\n   exported `KAPPA_LMU_LTAU`, `E_CHARGE` and `Q_EM` rather than retyped\n   numbers; the card has no default for the mixing, so the value \u03ba g\u2032 is written\n   explicitly.\n\n4. **Matching** (conventions section 3):\n\n       C^{qV}_{\u03b1\u03b1}(Q\u00b2) = g_\u03bd^{\u03b1\u03b1} g_q / ( \u221a2 G_F (Q\u00b2 + m\u00b2) ),   m = 15 MeV.\n\n   Explicitly, with g\u2032 = 6e-5,\n\n       C^{uV}_mm = + g\u2032 \u00b7 \u03ba g\u2032 e (2/3) / (\u221a2 G_F (Q\u00b2 + m\u00b2)),\n       C^{dV}_mm = \u2212 g\u2032 \u00b7 \u03ba g\u2032 e (1/3) / (\u221a2 G_F (Q\u00b2 + m\u00b2)),\n       C^{uV}_tt = \u2212 C^{uV}_mm,   C^{dV}_tt = \u2212 C^{dV}_mm,\n       C^{qV}_ee = 0.\n\n   Since the quark coupling is proportional to electric charge, the induced\n   neutron charge g_n = g_u + 2 g_d = 0 vanishes identically and only the\n   proton charge g_p = 2 g_u + g_d = \u03b5 e survives. The nuclear charge shift is\n   therefore \u0394Q^{\u03b1\u03b1} = Z g_\u03bd^{\u03b1\u03b1} \u03b5 e / (\u221a2 G_F (Q\u00b2 + m\u00b2)).\n\n5. **Approximate magnitude** at the Cs-133 reference point Q\u00b2 = 1238 MeV\u00b2\n   (for orientation only; the plugin computes at full precision):\n\n   | quantity | value |\n   |---|---|\n   | g_u | \u2248 +1.75e-7 |\n   | g_d | \u2248 \u22128.75e-8 |\n   | C^{uV}_mm | \u2248 +4.35e-4 |\n   | C^{dV}_mm | \u2248 \u22122.18e-4 |\n   | \u0394Q^{mm} (Cs-133, Z = 55) | \u2248 +0.036 |\n\n   Q_SM \u2248 \u221236.9 is negative, so for \u03bd_\u03bc (and \u03bd\u0304_\u03bc) the shift is destructive\n   but at the 0.1 percent level of the charge at this point. For \u03bd_\u03c4 the sign\n   flips (constructive), but no \u03bd_\u03c4 flux is present at COHERENT. \u03bd_e is\n   unaffected.\n\n6. **Validity of the Q\u00b2 \u2192 0 limit.** The description fixes the Q\u00b2 \u2192 0 form of\n   the mixing, so the constant \u03ba is used at every Q\u00b2. For reference, the\n   COHERENT CsI recoil range has Q\u00b2 \u2272 10\u2074 MeV\u00b2, comparable to m_\u03bc\u00b2 \u2248 1.1e4\n   MeV\u00b2, where the full loop function deviates from its Q\u00b2 \u2192 0 value; that\n   case would need the running-mixing plugin of example 06 and is deliberately\n   not used here, as instructed.\n\n7. **Card-expressible:** yes. One vector mediator with nu_couplings\n   mm = +6e-5, tt = \u22126e-5, zero direct quark couplings, and\n   kinetic_mixing = \u03ba g\u2032 = 8.66505822e-7. The framework adds \u03b5 e Q_q^em to each\n   quark coupling, reproducing step 3.\n\n## Conventions and sign checks\n\n- No convention conversion was needed: the description gives the gauge coupling\n  g\u2032 directly and the L_\u03bc\u2212L_\u03c4 charges are the standard ones.\n- Perturbativity: |g\u2032| = 6e-5 \u226a \u221a(4\u03c0).\n- The axial quark coupling is zero (photon-like vector coupling), so nothing is\n  dropped.\n- Pitfall avoided: the pyCEvNS factor-of-2 normalization is not used; the\n  matching formula of conventions section 3 is.\n\n## Validation\n\n`nuagent validate --plugin plugin.py --card card.json` returned valid with no\nfindings: perturbativity, SM limit with COUPLINGS zeroed, finite non-negative\ncounts, schema-valid card, and plugin/card cross-check on 66 Q\u00b2 points.\n"
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

### Attempt 1

```json
{
  "final_text": "Translation complete. Validation passed with no findings, including the plugin/card cross-check on 66 Q\u00b2 points.\n\n**What the interaction maps to**\n\n- **Neutrino couplings:** g_\u03bd^{mm} = +g\u2032, g_\u03bd^{tt} = \u2212g\u2032, g_\u03bd^{ee} = 0, from the L_\u03bc\u2212L_\u03c4 doublet charges. All off-diagonal entries are zero.\n- **Quark couplings:** purely through the loop-induced photon mixing \u03b5 = +\u03ba g\u2032, with \u03ba = e ln(m_\u03c4/m_\u03bc)/(6\u03c0\u00b2) from the conventions. Each quark gets \u03b5 e Q_q^em, so the induced neutron charge vanishes and only the proton charge survives.\n- **Matching:** vector coefficient C^{qV}_{\u03b1\u03b1} = g_\u03bd g_q / (\u221a2 G_F (Q\u00b2 + m\u00b2)) with m = 15 MeV. The \u03bd_\u03bc shift is destructive against the SM charge but only at the 0.1 percent level. The \u03bd_\u03c4 entry flips sign.\n\n**Files produced** in the task directory:\n\n- `plugin.py` computes the coefficients from the exported constants (KAPPA_LMU_LTAU, E_CHARGE, Q_EM, GF, SQRT2). Defaults are g\u2032 = 6e-5 and mass 15 MeV, with g\u2032 as the only coupling.\n- `card.json` holds one vector mediator with mm = +6e-5, tt = \u22126e-5, zero direct quark couplings, and kinetic_mixing written explicitly as \u03ba g\u2032 = 8.66505822e-7.\n- `notes.md` records the derivation, the sign argument for the positive mixing, a magnitude estimate at the Cs-133 reference point, and a note that the Q\u00b2 \u2192 0 form was used at all Q\u00b2 as instructed, rather than the running-mixing treatment of example 06.\n\nOne caveat: the example files for this task number and for the running-mixing case were not present in the context pack, so the derivation rests on conventions sections 3, 6 and 7 plus examples 01 and 02.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":143431,\"stop_reason\":\"end_turn\",\"session_id\":\"4ff7666a-e9f5-4235-8d68-122cda05cdab\",\"total_cost_usd\":1.00731375,\"usage\":{\"input_tokens\":290,\"cache_creation_input_tokens\":26520,\"cache_read_input_tokens\":196055,\"output_tokens\":8500,\"output_tokens_details\":{\"thinking_tokens\":1627},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":26520,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":679,\"cache_read_input_tokens\":29631,\"cache_creation_input_tokens\":2895,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":2895},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":290,\"outputTokens\":8500,\"cacheReadInputTokens\":196055,\"cacheCreationInputTokens\":26520,\"webSearchRequests\":0,\"costUSD\":1.00731375,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":1627,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_012ToKKhGSmTkTgH3kLVyA4V\",\"tool_input\":{\"command\":\"ls -R /private/tmp/nuagent-issue17-eval-claude/05_lmu_ltau_loop_limit\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":32,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Translation complete. Validation passed with no findings, including the plugin/card cross-check on 66 Q\u00b2 points.\\n\\n**What the interaction maps to**\\n\\n- **Neutrino couplings:** g_\u03bd^{mm} = +g\u2032, g_\u03bd^{tt} = \u2212g\u2032, g_\u03bd^{ee} = 0, from the L_\u03bc\u2212L_\u03c4 doublet charges. All off-diagonal entries are zero.\\n- **Quark couplings:** purely through the loop-induced photon mixing \u03b5 = +\u03ba g\u2032, with \u03ba = e ln(m_\u03c4/m_\u03bc)/(6\u03c0\u00b2) from the conventions. Each quark gets \u03b5 e Q_q^em, so the induced neutron charge vanishes and only the proton charge survives.\\n- **Matching:** vector coefficient C^{qV}_{\u03b1\u03b1} = g_\u03bd g_q / (\u221a2 G_F (Q\u00b2 + m\u00b2)) with m = 15 MeV. The \u03bd_\u03bc shift is destructive against the SM charge but only at the 0.1 percent level. The \u03bd_\u03c4 entry flips sign.\\n\\n**Files produced** in the task directory:\\n\\n- `plugin.py` computes the coefficients from the exported constants (KAPPA_LMU_LTAU, E_CHARGE, Q_EM, GF, SQRT2). Defaults are g\u2032 = 6e-5 and mass 15 MeV, with g\u2032 as the only coupling.\\n- `card.json` holds one vector mediator with mm = +6e-5, tt = \u22126e-5, zero direct quark couplings, and kinetic_mixing written explicitly as \u03ba g\u2032 = 8.66505822e-7.\\n- `notes.md` records the derivation, the sign argument for the positive mixing, a magnitude estimate at the Cs-133 reference point, and a note that the Q\u00b2 \u2192 0 form was used at all Q\u00b2 as instructed, rather than the running-mixing treatment of example 06.\\n\\nOne caveat: the example files for this task number and for the running-mixing case were not present in the context pack, so the derivation rests on conventions sections 3, 6 and 7 plus examples 01 and 02.\",\"ttft_ms\":5180,\"type\":\"result\",\"duration_ms\":145420,\"uuid\":\"7675cae7-29dd-4fad-ab35-4cb6264589b0\",\"ttft_stream_ms\":3261,\"time_to_request_ms\":25,\"first_content_frame_ms\":3262,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/issue-17-agent-harness/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/nuagent-issue17-eval-claude/05_lmu_ltau_loop_limit\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
