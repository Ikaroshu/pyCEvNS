Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

a gauged L_mu minus L_tau Z prime with mass 10 MeV, gauge coupling 1e-4 and kinetic mixing 1e-3

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
import numpy as np

from nuagent.physics.couplings import Coefficients, E_CHARGE, GF, Q_EM, SQRT2

PARAMETERS = {"g_prime": 1e-4, "kinetic_mixing": 1e-3, "mass_mev": 10.0}
COUPLINGS = ("g_prime", "kinetic_mixing")


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    g_prime = params["g_prime"]
    propagator = 1.0 / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    nu_charges = {"mm": 1.0, "tt": -1.0}
    for quark in ("u", "d"):
        g_q = params["kinetic_mixing"] * E_CHARGE * Q_EM[quark]
        for pair, charge in nu_charges.items():
            c.set("V", quark, pair, g_prime * charge * g_q * propagator)
    return c

```

## card.json

```json
{
  "name": "zprime_lmu_ltau_kinetic_mixing",
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
# Gauged U(1)_{L_μ−L_τ} Z′ with kinetic mixing

Description: "a gauged L_mu minus L_tau Z prime with mass 10 MeV, gauge coupling 1e-4 and kinetic mixing 1e-3".

## Parameter point (preserved exactly)

- m_Z′ = 10 MeV
- g′ = 1e-4
- ε = 1e-3 (kinetic mixing with the photon)

## Derivation

1. Lorentz structure: gauge boson exchange, so structure `V`.
2. Charges (conventions §7): Q′_{ν_e} = 0, Q′_{ν_μ} = +1, Q′_{ν_τ} = −1, Q′_u = Q′_d = 0. The neutrino charge is the charge of the lepton doublet, so the left-handed neutrino current couples with
   g_ν^{μμ} = +g′ = 1e-4, g_ν^{ττ} = −g′ = −1e-4, all other entries zero (flavor-diagonal).
3. Quarks have no direct L_μ−L_τ charge; they couple only through kinetic mixing (conventions §6):
   g_q = ε e Q_q^em, so g_u = ε e (2/3) = 2.02e-4, g_d = ε e (−1/3) = −1.01e-4 (e = 0.303).
4. Matching (conventions §3):
   C^{qV}_{αα}(Q²) = g_ν^{αα} g_q / (√2 G_F (Q² + m²)),
   so C^{uV}_{μμ} = −C^{uV}_{ττ} = g′ ε e (2/3) / (√2 G_F (Q² + m²)), and C^{dV}_{μμ} = −C^{dV}_{ττ} = −g′ ε e (1/3) / (√2 G_F (Q² + m²)).
   C^{qV}_{ee} = 0, all off-diagonal and scalar entries zero.
5. Nuclear charge: since 2Q_u + Q_d = 1 and Q_u + 2Q_d = 0, ΔQ^{μμ} = Z g′ ε e / (√2 G_F (Q² + m²)). Only protons respond (a dark-photon-like coupling); ΔQ^{ττ} = −ΔQ^{μμ}.

## Sign conventions

- The kinetic mixing is taken as the stated total low-energy value ε = +1e-3, entering as g_q → ε e Q_q^em with the framework's sign (kinetic term −(ε/2) F_μν Z′^μν with D_μ = ∂_μ + i e Q A_μ). The description gives the mixing as an explicit number, so the loop-induced value κ g′ = 1.444e-6 (conventions §6) is **not** added on top: the stated ε is interpreted as the full effective mixing. If the user intended 1e-3 as a tree-level mixing in addition to the μ/τ loop, the total would be ε = 1e-3 + 1.44417637e-6; please say so if that is the intent.
- With ε > 0 and g′ > 0, g_ν^{μμ} g_p > 0 has sign opposite to the (negative) SM charge: destructive interference for ν_μ and ν̄_μ; ν_τ would interfere constructively but there is no τ flux at COHERENT; ν_e is unaffected.

## Representation

- Card-expressible: one V mediator with nu_couplings mm = 1e-4, tt = −1e-4, quark_couplings zero, kinetic_mixing = 1e-3.
- The plugin computes the same coefficients independently from the exported constants `E_CHARGE`, `Q_EM`, `GF`, `SQRT2`. `COUPLINGS` = (g_prime, kinetic_mixing); zeroing them gives the SM.
- Mass is given, so the mediator (propagator) form is used, not a contact term. Q²-dependence of the loop-induced mixing (example 06) is irrelevant here because ε is given as a constant.
- Anomaly: L_μ−L_τ is anomaly-free with the SM fermion content.

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
  "notes": "# Gauged U(1)_{L_\u03bc\u2212L_\u03c4} Z\u2032 with kinetic mixing\n\nDescription: \"a gauged L_mu minus L_tau Z prime with mass 10 MeV, gauge coupling 1e-4 and kinetic mixing 1e-3\".\n\n## Parameter point (preserved exactly)\n\n- m_Z\u2032 = 10 MeV\n- g\u2032 = 1e-4\n- \u03b5 = 1e-3 (kinetic mixing with the photon)\n\n## Derivation\n\n1. Lorentz structure: gauge boson exchange, so structure `V`.\n2. Charges (conventions \u00a77): Q\u2032_{\u03bd_e} = 0, Q\u2032_{\u03bd_\u03bc} = +1, Q\u2032_{\u03bd_\u03c4} = \u22121, Q\u2032_u = Q\u2032_d = 0. The neutrino charge is the charge of the lepton doublet, so the left-handed neutrino current couples with\n   g_\u03bd^{\u03bc\u03bc} = +g\u2032 = 1e-4, g_\u03bd^{\u03c4\u03c4} = \u2212g\u2032 = \u22121e-4, all other entries zero (flavor-diagonal).\n3. Quarks have no direct L_\u03bc\u2212L_\u03c4 charge; they couple only through kinetic mixing (conventions \u00a76):\n   g_q = \u03b5 e Q_q^em, so g_u = \u03b5 e (2/3) = 2.02e-4, g_d = \u03b5 e (\u22121/3) = \u22121.01e-4 (e = 0.303).\n4. Matching (conventions \u00a73):\n   C^{qV}_{\u03b1\u03b1}(Q\u00b2) = g_\u03bd^{\u03b1\u03b1} g_q / (\u221a2 G_F (Q\u00b2 + m\u00b2)),\n   so C^{uV}_{\u03bc\u03bc} = \u2212C^{uV}_{\u03c4\u03c4} = g\u2032 \u03b5 e (2/3) / (\u221a2 G_F (Q\u00b2 + m\u00b2)), and C^{dV}_{\u03bc\u03bc} = \u2212C^{dV}_{\u03c4\u03c4} = \u2212g\u2032 \u03b5 e (1/3) / (\u221a2 G_F (Q\u00b2 + m\u00b2)).\n   C^{qV}_{ee} = 0, all off-diagonal and scalar entries zero.\n5. Nuclear charge: since 2Q_u + Q_d = 1 and Q_u + 2Q_d = 0, \u0394Q^{\u03bc\u03bc} = Z g\u2032 \u03b5 e / (\u221a2 G_F (Q\u00b2 + m\u00b2)). Only protons respond (a dark-photon-like coupling); \u0394Q^{\u03c4\u03c4} = \u2212\u0394Q^{\u03bc\u03bc}.\n\n## Sign conventions\n\n- The kinetic mixing is taken as the stated total low-energy value \u03b5 = +1e-3, entering as g_q \u2192 \u03b5 e Q_q^em with the framework's sign (kinetic term \u2212(\u03b5/2) F_\u03bc\u03bd Z\u2032^\u03bc\u03bd with D_\u03bc = \u2202_\u03bc + i e Q A_\u03bc). The description gives the mixing as an explicit number, so the loop-induced value \u03ba g\u2032 = 1.444e-6 (conventions \u00a76) is **not** added on top: the stated \u03b5 is interpreted as the full effective mixing. If the user intended 1e-3 as a tree-level mixing in addition to the \u03bc/\u03c4 loop, the total would be \u03b5 = 1e-3 + 1.44417637e-6; please say so if that is the intent.\n- With \u03b5 > 0 and g\u2032 > 0, g_\u03bd^{\u03bc\u03bc} g_p > 0 has sign opposite to the (negative) SM charge: destructive interference for \u03bd_\u03bc and \u03bd\u0304_\u03bc; \u03bd_\u03c4 would interfere constructively but there is no \u03c4 flux at COHERENT; \u03bd_e is unaffected.\n\n## Representation\n\n- Card-expressible: one V mediator with nu_couplings mm = 1e-4, tt = \u22121e-4, quark_couplings zero, kinetic_mixing = 1e-3.\n- The plugin computes the same coefficients independently from the exported constants `E_CHARGE`, `Q_EM`, `GF`, `SQRT2`. `COUPLINGS` = (g_prime, kinetic_mixing); zeroing them gives the SM.\n- Mass is given, so the mediator (propagator) form is used, not a contact term. Q\u00b2-dependence of the loop-induced mixing (example 06) is irrelevant here because \u03b5 is given as a constant.\n- Anomaly: L_\u03bc\u2212L_\u03c4 is anomaly-free with the SM fermion content.\n"
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
  "final_text": "I wrote the three files, and validation passed with no findings. The plugin and card agree on all 66 Q\u00b2 test points, so this counts as **cross-validated**.\n\n- **`plugin.py`**: a vector Z\u2032 with defaults m = 10 MeV, g\u2032 = 1e-4 and \u03b5 = 1e-3. The neutrino couplings are +g\u2032 for \u03bd_\u03bc, \u2212g\u2032 for \u03bd_\u03c4 and zero for \u03bd_e. Quarks couple only through the mixing: g_q = \u03b5\u00b7e\u00b7Q_q, built from the framework's own constants. The coefficient is C^{qV} = g_\u03bd g_q / (\u221a2 G_F (Q\u00b2 + m\u00b2)). Setting g\u2032 and \u03b5 to zero gives back the Standard Model.\n- **`card.json`**: the same model as one vector mediator: neutrino couplings of +1e-4 (\u03bc\u03bc) and \u22121e-4 (\u03c4\u03c4), no direct quark couplings, and `kinetic_mixing` set to 0.001.\n- **`notes.md`**: the derivation and sign conventions. Only protons pick up the new coupling, and it works against the Standard Model for \u03bd_\u03bc, lowering the rate.\n\n**Decision for you:** I read \u03b5 = 1e-3 as the total mixing, so I didn't add the small mixing that muon and tau loops produce on their own (about 1.44e-6). If you meant 1e-3 as a separate tree-level mixing on top of that loop term, the total would be about 1.0014e-3. The notes ask you to confirm which you meant.\n\nI couldn't find a worked L_\u03bc\u2212L_\u03c4 example in the context folder, so I worked only from the conventions file and the universal Z\u2032 example.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":49300,\"stop_reason\":\"end_turn\",\"session_id\":\"09cea5db-315c-481a-b3bf-aad19e60dbe7\",\"total_cost_usd\":0.27477140000000005,\"usage\":{\"input_tokens\":18,\"cache_creation_input_tokens\":20417,\"cache_read_input_tokens\":130717,\"output_tokens\":4261,\"output_tokens_details\":{\"thinking_tokens\":504},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":20417,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":2,\"output_tokens\":571,\"cache_read_input_tokens\":23338,\"cache_creation_input_tokens\":294,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":294},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-opus-5-5\":{\"inputTokens\":18,\"outputTokens\":4261,\"cacheReadInputTokens\":130717,\"cacheCreationInputTokens\":20417,\"webSearchRequests\":0,\"costUSD\":0.27477140000000005,\"contextWindow\":1000000,\"maxOutputTokens\":128000,\"thinkingTokens\":504,\"canonicalModel\":\"claude-opus-5-5\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01F2iZ2XNZtMbaafTXkryyRc\",\"tool_input\":{\"command\":\"ls -R\",\"description\":\"List task directory\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":15,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"I wrote the three files, and validation passed with no findings. The plugin and card agree on all 66 Q\u00b2 test points, so this counts as **cross-validated**.\\n\\n- **`plugin.py`**: a vector Z\u2032 with defaults m = 10 MeV, g\u2032 = 1e-4 and \u03b5 = 1e-3. The neutrino couplings are +g\u2032 for \u03bd_\u03bc, \u2212g\u2032 for \u03bd_\u03c4 and zero for \u03bd_e. Quarks couple only through the mixing: g_q = \u03b5\u00b7e\u00b7Q_q, built from the framework's own constants. The coefficient is C^{qV} = g_\u03bd g_q / (\u221a2 G_F (Q\u00b2 + m\u00b2)). Setting g\u2032 and \u03b5 to zero gives back the Standard Model.\\n- **`card.json`**: the same model as one vector mediator: neutrino couplings of +1e-4 (\u03bc\u03bc) and \u22121e-4 (\u03c4\u03c4), no direct quark couplings, and `kinetic_mixing` set to 0.001.\\n- **`notes.md`**: the derivation and sign conventions. Only protons pick up the new coupling, and it works against the Standard Model for \u03bd_\u03bc, lowering the rate.\\n\\n**Decision for you:** I read \u03b5 = 1e-3 as the total mixing, so I didn't add the small mixing that muon and tau loops produce on their own (about 1.44e-6). If you meant 1e-3 as a separate tree-level mixing on top of that loop term, the total would be about 1.0014e-3. The notes ask you to confirm which you meant.\\n\\nI couldn't find a worked L_\u03bc\u2212L_\u03c4 example in the context folder, so I worked only from the conventions file and the universal Z\u2032 example.\",\"ttft_ms\":2687,\"type\":\"result\",\"duration_ms\":51108,\"uuid\":\"09dae309-edf6-4457-a514-4ede3de2f698\",\"ttft_stream_ms\":1842,\"time_to_request_ms\":38,\"first_content_frame_ms\":1842,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/issue-17-agent-harness/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/nuagent-issue17-live-claude/report.task\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
