Verification: **plugin-only**

SM-limit, perturbativity and count checks pass. There is no independent card cross-check; normalization and momentum dependence rest on the derivation.

## Description

A gauged L_mu−L_tau vector Z′ has mass 12 MeV and gauge coupling g′ = 8e-5. Keep the full spacelike Q² dependence of the muon/tau one-loop photon mixing, with no tree-level mixing or direct quark coupling.

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
"""L_mu - L_tau Z' with the full one-loop, Q^2-dependent photon mixing (plugin-only).

Derivation and conventions are in notes.md.
"""

import numpy as np

from nuagent.physics.couplings import (
    Coefficients,
    GF,
    SQRT2,
    E_CHARGE,
    M_MU,
    M_TAU,
    Q_EM,
)

PARAMETERS = {"g_prime": 8e-5, "mass_mev": 12.0}
COUPLINGS = ("g_prime",)

# L_mu - L_tau charges of the lepton doublets (neutrinos and charged leptons alike).
_Q_PRIME = {"e": 0.0, "m": +1.0, "t": -1.0}
_PAIRS = {"ee": "e", "mm": "m", "tt": "t"}

# Gauss-Legendre nodes/weights mapped from [-1, 1] to the Feynman parameter x in [0, 1].
_T, _W = np.polynomial.legendre.leggauss(40)
_X = (_T + 1.0) / 2.0
_WX = _W / 2.0


def loop_mixing(q2: np.ndarray, g_prime: float) -> np.ndarray:
    """Spacelike one-loop gamma-Z' mixing eps(Q^2) from the muon and tau loops.

    eps(Q^2) = (e g' / 2 pi^2) * Int_0^1 dx x(1-x) ln[(m_tau^2 + x(1-x) Q^2) / (m_mu^2 + x(1-x) Q^2)]
    which reduces to kappa g' = e g' ln(m_tau/m_mu) / (6 pi^2) at Q^2 -> 0.
    """
    q = np.asarray(q2, dtype=float)[:, None]
    x = _X[None, :]
    xx = x * (1.0 - x)
    integrand = xx * np.log((M_TAU**2 + xx * q) / (M_MU**2 + xx * q))
    integral = (integrand * _WX[None, :]).sum(axis=1)
    return E_CHARGE * g_prime / (2.0 * np.pi**2) * integral


def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    g_prime = params["g_prime"]
    mass = params["mass_mev"]

    eps = loop_mixing(q2, g_prime)
    prop = 1.0 / (SQRT2 * GF * (q2 + mass**2))

    for quark in ("u", "d"):
        # No tree-level mixing and no direct quark charge: the quark coupling is purely loop-induced.
        g_q = eps * E_CHARGE * Q_EM[quark]
        for pair, flavor in _PAIRS.items():
            g_nu = g_prime * _Q_PRIME[flavor]
            value = g_nu * g_q * prop
            c.set("V", quark, pair, value)
    return c

```

## notes.md

```text
# Notes: gauged L_μ−L_τ Z′ with full one-loop (Q²-dependent) photon mixing

## Description point

- Mediator: vector Z′ of U(1)_{L_μ−L_τ}, mass m = 12 MeV, gauge coupling g′ = 8e-5.
- No tree-level kinetic mixing, no direct quark coupling. The only coupling to quarks is the one-loop γ–Z′ mixing from the muon and tau loops, kept with its full spacelike Q² dependence.
- Parameter point preserved exactly in `PARAMETERS = {"g_prime": 8e-5, "mass_mev": 12.0}`; `COUPLINGS = ("g_prime",)`.

## Lorentz structure and charges

Structure `V` (vector exchange). L_μ−L_τ charges (conventions section 7): Q′_{ν_e} = 0, Q′_{ν_μ} = +1, Q′_{ν_τ} = −1, Q′_u = Q′_d = 0. The neutrino coupling is left-handed, g_ν^{αα} = g′ Q′_{ν_α}, flavor-diagonal. The charged leptons carry the same charges as their doublet partners: Q′_μ = +1, Q′_τ = −1, with electric charge Q_ℓ = −1.

## Loop-induced mixing

A charged lepton ℓ with couplings e Q_ℓ (photon) and g′ Q′_ℓ (Z′) gives a vacuum-polarization-type γ–Z′ two-point function proportional to ∫₀¹ dx x(1−x) ln[(m_ℓ² + x(1−x)Q²)/μ²] for spacelike q² = −Q². Summing over μ and τ, the scheme-dependent, mass-independent piece cancels because Σ_ℓ Q′_ℓ Q_ℓ = 0, leaving a finite, scale-independent result:

    ε(Q²) = (e g′ / 2π²) ∫₀¹ dx x(1−x) ln[ (m_τ² + x(1−x) Q²) / (m_μ² + x(1−x) Q²) ].

Reduction check: at Q² → 0 the integral is (1/6) ln(m_τ²/m_μ²) = (1/3) ln(m_τ/m_μ), so ε(0) = e g′ ln(m_τ/m_μ)/(6π²) = κ g′ with κ = 0.0144417637 (conventions section 6), giving ε(0) = 1.1553411e-06 for g′ = 8e-5. The plugin reproduces the constant-mixing card of a `zprime_lmu_ltau` model with `kinetic_mixing = κ g′` in that limit.

Sign: positive throughout, as fixed in conventions section 6 (the ln(m²) term carries the sign that makes the running electromagnetic coupling grow with Q²; the lighter muon dominates and has Q′ = +1). For ν_μ this makes the new-physics proton coupling the same sign as the ν_μ coupling: destructive interference with the SM for ν_μ, constructive for ν_τ (whose charge is −1). The Q² dependence is mild over the COHERENT range (Q² ≲ 0.6 m_μ²), a slow decrease of ε at the percent level.

## Matching

With g_q(Q²) = ε(Q²) e Q_q^{em} (Q_u = +2/3, Q_d = −1/3, e = 0.303) and conventions section 3,

    C^{qV}_{αα}(Q²) = g′ Q′_{ν_α} · ε(Q²) e Q_q^{em} / ( √2 G_F (Q² + m²) ),

so C^{qV}_{mm} = −C^{qV}_{tt}, C^{qV}_{ee} = 0, all off-diagonal and scalar entries zero. The up-quark entries are positive for mm and the down-quark entries negative (ratio −1/2).

## Numerics

The Feynman-parameter integrand is smooth on [0, 1]; 40-point Gauss–Legendre on [0, 1] reproduces the exact integral to far better than 1e-6 relative (as in worked example 06). All constants (e, m_μ, m_τ, Q_EM, G_F, √2) are the exported ones from `nuagent.physics.couplings`.

## Why no card.json

The card's `kinetic_mixing` field is a constant ε. The description explicitly asks for the full Q² dependence of the loop mixing, which a card cannot carry, so this is a plugin-only translation. No tree-level mixing or direct quark coupling was requested, so none is added.

## Convention conversions

None needed: the description gives g′ and the mass directly in the framework's normalization. Perturbativity: |g′| = 8e-5 ≪ √(4π).

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
    "proposed counts checked for finiteness and non-negativity"
  ],
  "notes": "# Notes: gauged L_\u03bc\u2212L_\u03c4 Z\u2032 with full one-loop (Q\u00b2-dependent) photon mixing\n\n## Description point\n\n- Mediator: vector Z\u2032 of U(1)_{L_\u03bc\u2212L_\u03c4}, mass m = 12 MeV, gauge coupling g\u2032 = 8e-5.\n- No tree-level kinetic mixing, no direct quark coupling. The only coupling to quarks is the one-loop \u03b3\u2013Z\u2032 mixing from the muon and tau loops, kept with its full spacelike Q\u00b2 dependence.\n- Parameter point preserved exactly in `PARAMETERS = {\"g_prime\": 8e-5, \"mass_mev\": 12.0}`; `COUPLINGS = (\"g_prime\",)`.\n\n## Lorentz structure and charges\n\nStructure `V` (vector exchange). L_\u03bc\u2212L_\u03c4 charges (conventions section 7): Q\u2032_{\u03bd_e} = 0, Q\u2032_{\u03bd_\u03bc} = +1, Q\u2032_{\u03bd_\u03c4} = \u22121, Q\u2032_u = Q\u2032_d = 0. The neutrino coupling is left-handed, g_\u03bd^{\u03b1\u03b1} = g\u2032 Q\u2032_{\u03bd_\u03b1}, flavor-diagonal. The charged leptons carry the same charges as their doublet partners: Q\u2032_\u03bc = +1, Q\u2032_\u03c4 = \u22121, with electric charge Q_\u2113 = \u22121.\n\n## Loop-induced mixing\n\nA charged lepton \u2113 with couplings e Q_\u2113 (photon) and g\u2032 Q\u2032_\u2113 (Z\u2032) gives a vacuum-polarization-type \u03b3\u2013Z\u2032 two-point function proportional to \u222b\u2080\u00b9 dx x(1\u2212x) ln[(m_\u2113\u00b2 + x(1\u2212x)Q\u00b2)/\u03bc\u00b2] for spacelike q\u00b2 = \u2212Q\u00b2. Summing over \u03bc and \u03c4, the scheme-dependent, mass-independent piece cancels because \u03a3_\u2113 Q\u2032_\u2113 Q_\u2113 = 0, leaving a finite, scale-independent result:\n\n    \u03b5(Q\u00b2) = (e g\u2032 / 2\u03c0\u00b2) \u222b\u2080\u00b9 dx x(1\u2212x) ln[ (m_\u03c4\u00b2 + x(1\u2212x) Q\u00b2) / (m_\u03bc\u00b2 + x(1\u2212x) Q\u00b2) ].\n\nReduction check: at Q\u00b2 \u2192 0 the integral is (1/6) ln(m_\u03c4\u00b2/m_\u03bc\u00b2) = (1/3) ln(m_\u03c4/m_\u03bc), so \u03b5(0) = e g\u2032 ln(m_\u03c4/m_\u03bc)/(6\u03c0\u00b2) = \u03ba g\u2032 with \u03ba = 0.0144417637 (conventions section 6), giving \u03b5(0) = 1.1553411e-06 for g\u2032 = 8e-5. The plugin reproduces the constant-mixing card of a `zprime_lmu_ltau` model with `kinetic_mixing = \u03ba g\u2032` in that limit.\n\nSign: positive throughout, as fixed in conventions section 6 (the ln(m\u00b2) term carries the sign that makes the running electromagnetic coupling grow with Q\u00b2; the lighter muon dominates and has Q\u2032 = +1). For \u03bd_\u03bc this makes the new-physics proton coupling the same sign as the \u03bd_\u03bc coupling: destructive interference with the SM for \u03bd_\u03bc, constructive for \u03bd_\u03c4 (whose charge is \u22121). The Q\u00b2 dependence is mild over the COHERENT range (Q\u00b2 \u2272 0.6 m_\u03bc\u00b2), a slow decrease of \u03b5 at the percent level.\n\n## Matching\n\nWith g_q(Q\u00b2) = \u03b5(Q\u00b2) e Q_q^{em} (Q_u = +2/3, Q_d = \u22121/3, e = 0.303) and conventions section 3,\n\n    C^{qV}_{\u03b1\u03b1}(Q\u00b2) = g\u2032 Q\u2032_{\u03bd_\u03b1} \u00b7 \u03b5(Q\u00b2) e Q_q^{em} / ( \u221a2 G_F (Q\u00b2 + m\u00b2) ),\n\nso C^{qV}_{mm} = \u2212C^{qV}_{tt}, C^{qV}_{ee} = 0, all off-diagonal and scalar entries zero. The up-quark entries are positive for mm and the down-quark entries negative (ratio \u22121/2).\n\n## Numerics\n\nThe Feynman-parameter integrand is smooth on [0, 1]; 40-point Gauss\u2013Legendre on [0, 1] reproduces the exact integral to far better than 1e-6 relative (as in worked example 06). All constants (e, m_\u03bc, m_\u03c4, Q_EM, G_F, \u221a2) are the exported ones from `nuagent.physics.couplings`.\n\n## Why no card.json\n\nThe card's `kinetic_mixing` field is a constant \u03b5. The description explicitly asks for the full Q\u00b2 dependence of the loop mixing, which a card cannot carry, so this is a plugin-only translation. No tree-level mixing or direct quark coupling was requested, so none is added.\n\n## Convention conversions\n\nNone needed: the description gives g\u2032 and the mass directly in the framework's normalization. Perturbativity: |g\u2032| = 8e-5 \u226a \u221a(4\u03c0).\n"
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
      10.138260380578297,
      16.545799604359658,
      21.759782046395806,
      22.101819030245306,
      17.848134066105914,
      13.092454142584428,
      10.06667527315485,
      7.5716416499307115,
      5.596613693342689,
      4.078052043374353,
      2.9294878321083213,
      2.0717751889159826
    ],
    "predicted_total": 133.8004949510963,
    "observed_total": 855.0,
    "delta": 0.0015326498328072091,
    "new_physics": {
      "neg2_log_likelihood": 17.378970149266827,
      "alpha": 0.07131688687311229,
      "beta": {
        "steady_state": 0.001564404722838787,
        "prompt_neutrons": 0.0028937082601721456
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
  "final_text": "Validation passed with no findings.\n\n**Files produced** in the task directory:\n\n- `plugin.py`: vector coefficients for the L_\u03bc\u2212L_\u03c4 Z\u2032 at the described point (mass 12 MeV, g\u2032 = 8e-5). The quark coupling is purely the loop-induced \u03b3\u2013Z\u2032 mixing \u03b5(Q\u00b2) from the muon and tau loops, evaluated with its full spacelike Q\u00b2 dependence via 40-point Gauss\u2013Legendre over the Feynman parameter. Entries: C_mm = \u2212C_tt, C_ee = 0, no off-diagonal or scalar terms. No tree-level mixing or direct quark coupling is added.\n- `notes.md`: derivation, charge assignment, sign of the loop mixing (positive, per conventions section 6), the Q\u00b2 \u2192 0 reduction check to \u03ba g\u2032, and the reason no card exists.\n\n**No card.json** was written. The schema only carries a constant kinetic mixing, and the description explicitly requires the Q\u00b2-dependent loop result, so this is a plugin-only translation.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":96109,\"stop_reason\":\"end_turn\",\"session_id\":\"6e8f50e8-2594-4762-8975-7391f22ce3b1\",\"total_cost_usd\":1.0258897500000002,\"usage\":{\"input_tokens\":258,\"cache_creation_input_tokens\":29144,\"cache_read_input_tokens\":182519,\"output_tokens\":7896,\"output_tokens_details\":{\"thinking_tokens\":1015},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":29144,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":346,\"cache_read_input_tokens\":34941,\"cache_creation_input_tokens\":209,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":209},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":258,\"outputTokens\":7896,\"cacheReadInputTokens\":182519,\"cacheCreationInputTokens\":29144,\"webSearchRequests\":0,\"costUSD\":1.0258897500000002,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":1015,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_016ByfyAv1ePo3eAxt9aDe6W\",\"tool_input\":{\"command\":\"ls -R /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/12_running_mixing\",\"description\":\"List task directory contents\"}},{\"tool_name\":\"Read\",\"tool_use_id\":\"toolu_01KEfR277hXtGVTfFQVppBUS\",\"tool_input\":{\"file_path\":\"/private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-5c51979abec1/scratchpad/live/eval-claude/12_running_mixing/context/examples/06_running_kinetic_mixing.md\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":28,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings.\\n\\n**Files produced** in the task directory:\\n\\n- `plugin.py`: vector coefficients for the L_\u03bc\u2212L_\u03c4 Z\u2032 at the described point (mass 12 MeV, g\u2032 = 8e-5). The quark coupling is purely the loop-induced \u03b3\u2013Z\u2032 mixing \u03b5(Q\u00b2) from the muon and tau loops, evaluated with its full spacelike Q\u00b2 dependence via 40-point Gauss\u2013Legendre over the Feynman parameter. Entries: C_mm = \u2212C_tt, C_ee = 0, no off-diagonal or scalar terms. No tree-level mixing or direct quark coupling is added.\\n- `notes.md`: derivation, charge assignment, sign of the loop mixing (positive, per conventions section 6), the Q\u00b2 \u2192 0 reduction check to \u03ba g\u2032, and the reason no card exists.\\n\\n**No card.json** was written. The schema only carries a constant kinetic mixing, and the description explicitly requires the Q\u00b2-dependent loop result, so this is a plugin-only translation.\",\"ttft_ms\":863,\"type\":\"result\",\"duration_ms\":99753,\"uuid\":\"81e2eab1-9f42-4195-b50d-15ab5cacf3cd\",\"ttft_stream_ms\":861,\"time_to_request_ms\":23,\"first_content_frame_ms\":862,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/m7-conversation/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/12_running_mixing\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
