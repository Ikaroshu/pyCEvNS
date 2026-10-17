Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

Contact vector NSI on down quarks has epsilon^{dV}_{mu mu} = −0.05 and epsilon^{dV}_{e tau} = 0.02 + 0.03i, with its Hermitian partner. All other entries vanish.

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
"""Contact vector NSI on down quarks with one real diagonal and one complex off-diagonal entry."""

import numpy as np

from nuagent.physics.couplings import Coefficients

PARAMETERS = {
    "eps_dV_mm": -0.05,
    "eps_dV_et_re": 0.02,
    "eps_dV_et_im": 0.03,
}
COUPLINGS = ("eps_dV_mm", "eps_dV_et_re", "eps_dV_et_im")


def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)
    ones = np.ones_like(q2, dtype=float)
    eps_mm = params["eps_dV_mm"] * ones
    eps_et = (params["eps_dV_et_re"] + 1j * params["eps_dV_et_im"]) * ones
    c.set("V", "d", "mm", eps_mm)
    c.set("V", "d", "et", eps_et)
    return c

```

## card.json

```json
{
  "name": "contact_dV_mm_et_complex",
  "mediators": [],
  "contacts": [
    {
      "fermion": "d",
      "epsilon": {
        "ee": 0.0,
        "mm": -0.05,
        "tt": 0.0,
        "em": 0.0,
        "et": [0.02, 0.03],
        "mt": 0.0
      }
    }
  ]
}

```

## notes.md

```text
# Notes: contact vector NSI on down quarks (complex off-diagonal)

## Description

Contact vector NSI on down quarks with ε^{dV}_{μμ} = −0.05 and ε^{dV}_{eτ} = 0.02 + 0.03i,
plus its Hermitian partner ε^{dV}_{τe} = 0.02 − 0.03i. All other entries vanish.

## Derivation

1. **Lorentz structure.** The description gives vector NSI epsilons, so the structure is `V`.
   No mediator mass is given, so the contact form of conventions section 5 applies: the
   constant ε^{fV}_{αβ} is added directly to C^{fV}_{αβ}, with no Q² dependence.
2. **Quark.** Only the down quark is coupled. C^{uV} = 0 for every flavor entry.
3. **Flavor entries.** In the framework's normalization C^{fV} = ε^{fL} + ε^{fR} is the NSI
   vector combination, which is exactly what ε^{dV} denotes. So
   - C^{dV}_{μμ}(Q²) = −0.05 (real diagonal, as required),
   - C^{dV}_{eτ}(Q²) = 0.02 + 0.03i,
   - C^{dV}_{τe}(Q²) = 0.02 − 0.03i, filled by the framework as the conjugate partner when the
     `et` entry is set (conventions section 8; `set("V","d","et",…)` in the plugin API).
   - ee, tt, em, mt entries and all scalar entries are zero.
4. **Convention check.** The card and plugin key `et` means the (e, τ) entry C_{eτ}, matching the
   description's ε_{eτ}. No convention conversion was needed; the description is already in the
   framework's ε^{fV} normalization.
5. **Physics remarks.** The μμ entry interferes with the SM: ΔQ^{μμ} = (Z + 2N) ε^{dV}_{μμ} is
   negative, the same sign as Q_SM ≈ 0.04 Z − 0.5 N, so the muon-flavor rate is enhanced
   (constructive interference). The eτ entry is flavor-changing and enters incoherently through
   Σ_β |Q^{αβ}|², adding |Q^{eτ}|² = (Z + 2N)² |ε_{eτ}|² with |ε_{eτ}|² = 0.0013 to the ν_e rate
   (and the same to the ν_τ rate, which the COHERENT flux does not contain). Only the modulus of the
   complex entry affects the rate; the phase is carried faithfully but is unobservable in CEvNS.
6. **Perturbativity.** All couplings are far below √(4π).

## Plugin

`plugin.py` exposes the three real parameters `eps_dV_mm`, `eps_dV_et_re`, `eps_dV_et_im` at the
described point (−0.05, 0.02, 0.03). The complex eτ entry is built inside `coefficients` from the
separate real and imaginary parameters, as required by the plugin API. All three are listed in
`COUPLINGS`, so zeroing them reproduces the Standard Model. Each entry is set once, so no summing
before `set` was needed.

## Card

`card.json` expresses the same interaction with a single `contacts` entry on fermion `d`, with
`mm: -0.05` and `et: [0.02, 0.03]`. The schema supports contact vector entries with complex
off-diagonals, so the card is complete and should cross-validate against the plugin.

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
  "notes": "# Notes: contact vector NSI on down quarks (complex off-diagonal)\n\n## Description\n\nContact vector NSI on down quarks with \u03b5^{dV}_{\u03bc\u03bc} = \u22120.05 and \u03b5^{dV}_{e\u03c4} = 0.02 + 0.03i,\nplus its Hermitian partner \u03b5^{dV}_{\u03c4e} = 0.02 \u2212 0.03i. All other entries vanish.\n\n## Derivation\n\n1. **Lorentz structure.** The description gives vector NSI epsilons, so the structure is `V`.\n   No mediator mass is given, so the contact form of conventions section 5 applies: the\n   constant \u03b5^{fV}_{\u03b1\u03b2} is added directly to C^{fV}_{\u03b1\u03b2}, with no Q\u00b2 dependence.\n2. **Quark.** Only the down quark is coupled. C^{uV} = 0 for every flavor entry.\n3. **Flavor entries.** In the framework's normalization C^{fV} = \u03b5^{fL} + \u03b5^{fR} is the NSI\n   vector combination, which is exactly what \u03b5^{dV} denotes. So\n   - C^{dV}_{\u03bc\u03bc}(Q\u00b2) = \u22120.05 (real diagonal, as required),\n   - C^{dV}_{e\u03c4}(Q\u00b2) = 0.02 + 0.03i,\n   - C^{dV}_{\u03c4e}(Q\u00b2) = 0.02 \u2212 0.03i, filled by the framework as the conjugate partner when the\n     `et` entry is set (conventions section 8; `set(\"V\",\"d\",\"et\",\u2026)` in the plugin API).\n   - ee, tt, em, mt entries and all scalar entries are zero.\n4. **Convention check.** The card and plugin key `et` means the (e, \u03c4) entry C_{e\u03c4}, matching the\n   description's \u03b5_{e\u03c4}. No convention conversion was needed; the description is already in the\n   framework's \u03b5^{fV} normalization.\n5. **Physics remarks.** The \u03bc\u03bc entry interferes with the SM: \u0394Q^{\u03bc\u03bc} = (Z + 2N) \u03b5^{dV}_{\u03bc\u03bc} is\n   negative, the same sign as Q_SM \u2248 0.04 Z \u2212 0.5 N, so the muon-flavor rate is enhanced\n   (constructive interference). The e\u03c4 entry is flavor-changing and enters incoherently through\n   \u03a3_\u03b2 |Q^{\u03b1\u03b2}|\u00b2, adding |Q^{e\u03c4}|\u00b2 = (Z + 2N)\u00b2 |\u03b5_{e\u03c4}|\u00b2 with |\u03b5_{e\u03c4}|\u00b2 = 0.0013 to the \u03bd_e rate\n   (and the same to the \u03bd_\u03c4 rate, which the COHERENT flux does not contain). Only the modulus of the\n   complex entry affects the rate; the phase is carried faithfully but is unobservable in CEvNS.\n6. **Perturbativity.** All couplings are far below \u221a(4\u03c0).\n\n## Plugin\n\n`plugin.py` exposes the three real parameters `eps_dV_mm`, `eps_dV_et_re`, `eps_dV_et_im` at the\ndescribed point (\u22120.05, 0.02, 0.03). The complex e\u03c4 entry is built inside `coefficients` from the\nseparate real and imaginary parameters, as required by the plugin API. All three are listed in\n`COUPLINGS`, so zeroing them reproduces the Standard Model. Each entry is set once, so no summing\nbefore `set` was needed.\n\n## Card\n\n`card.json` expresses the same interaction with a single `contacts` entry on fermion `d`, with\n`mm: -0.05` and `et: [0.02, 0.03]`. The schema supports contact vector entries with complex\noff-diagonals, so the card is complete and should cross-validate against the plugin.\n"
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
      14.714847492771828,
      23.99219571054456,
      31.455695699078788,
      31.74108840896014,
      25.300882125829016,
      18.340983099164088,
      14.168934856385887,
      10.72019065150682,
      7.97248732493449,
      5.846289824480911,
      4.2275690014864535,
      3.0104786359276035
    ],
    "predicted_total": 191.49164283107058,
    "observed_total": 855.0,
    "delta": 0.2441973415699934,
    "new_physics": {
      "neg2_log_likelihood": 17.621634841004013,
      "alpha": -0.1386793774232493,
      "beta": {
        "steady_state": -0.012022556025060301,
        "prompt_neutrons": -0.0013655410821891854
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
  "final_text": "Validation passed with no findings. The plugin and card agree on all 66 Q\u00b2 points and the SM limit holds with couplings zeroed.\n\nFiles produced in the task directory:\n\n- **plugin.py** sets the down-quark vector coefficient with a real \u03bc\u03bc entry and a complex e\u03c4 entry built from separate real and imaginary parameters. Parameters default to the described point and all three are listed as couplings.\n- **card.json** expresses the same interaction as a single contact entry on the down quark, with the e\u03c4 entry written as a complex pair.\n- **notes.md** records the derivation: contact form because no mass is given, vector structure, the entry-to-key mapping, constructive interference for \u03bd_\u03bc, and the incoherent flavor-changing contribution whose phase is unobservable.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":70994,\"stop_reason\":\"end_turn\",\"session_id\":\"4d35964f-bd47-48bc-a040-d863c48d7741\",\"total_cost_usd\":0.70858175,\"usage\":{\"input_tokens\":226,\"cache_creation_input_tokens\":21258,\"cache_read_input_tokens\":135247,\"output_tokens\":4947,\"output_tokens_details\":{\"thinking_tokens\":757},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":21258,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":250,\"cache_read_input_tokens\":26969,\"cache_creation_input_tokens\":295,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":295},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":226,\"outputTokens\":4947,\"cacheReadInputTokens\":135247,\"cacheCreationInputTokens\":21258,\"webSearchRequests\":0,\"costUSD\":0.70858175,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":757,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01RqnTArc1LRUqud5HxfwCDk\",\"tool_input\":{\"command\":\"ls -R /private/tmp/nuagent-issue17-eval-claude/08_complex_contact\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":25,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings. The plugin and card agree on all 66 Q\u00b2 points and the SM limit holds with couplings zeroed.\\n\\nFiles produced in the task directory:\\n\\n- **plugin.py** sets the down-quark vector coefficient with a real \u03bc\u03bc entry and a complex e\u03c4 entry built from separate real and imaginary parameters. Parameters default to the described point and all three are listed as couplings.\\n- **card.json** expresses the same interaction as a single contact entry on the down quark, with the e\u03c4 entry written as a complex pair.\\n- **notes.md** records the derivation: contact form because no mass is given, vector structure, the entry-to-key mapping, constructive interference for \u03bd_\u03bc, and the incoherent flavor-changing contribution whose phase is unobservable.\",\"ttft_ms\":3027,\"type\":\"result\",\"duration_ms\":72815,\"uuid\":\"73a2076a-bf15-497f-b7f3-99a6c262575f\",\"ttft_stream_ms\":2217,\"time_to_request_ms\":22,\"first_content_frame_ms\":2217,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/issue-17-agent-harness/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/nuagent-issue17-eval-claude/08_complex_contact\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
