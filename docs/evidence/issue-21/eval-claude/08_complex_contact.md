Verification: **cross-validated**

Plugin and card agree on every coefficient on the reference grid. SM-limit, perturbativity and count checks pass. Shared interpretation errors can survive; this is not proof of the derivation.

## Description

Contact vector NSI on down quarks has epsilon^{dV}_{mu mu} = −0.05 and epsilon^{dV}_{e tau} = 0.02 + 0.03i, with its Hermitian partner. All other entries vanish.

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
"""Contact vector NSI on down quarks with a real mu-mu entry and a complex e-tau entry."""

import numpy as np

from nuagent.physics.couplings import Coefficients

PARAMETERS = {
    "eps_d_mm": -0.05,
    "eps_d_et_re": 0.02,
    "eps_d_et_im": 0.03,
}
COUPLINGS = ("eps_d_mm", "eps_d_et_re", "eps_d_et_im")


def coefficients(q2: np.ndarray, params: dict) -> Coefficients:
    c = Coefficients.zeros(q2)
    c.set("V", "d", "mm", params["eps_d_mm"])
    eps_et = params["eps_d_et_re"] + 1j * params["eps_d_et_im"]
    c.set("V", "d", "et", eps_et)  # te is filled as the complex conjugate
    return c

```

## card.json

```json
{
  "name": "contact_nsi_d_complex_etau",
  "mediators": [],
  "contacts": [
    {
      "fermion": "d",
      "epsilon": {
        "ee": 0,
        "mm": -0.05,
        "tt": 0,
        "em": 0,
        "et": [0.02, 0.03],
        "mt": 0
      }
    }
  ]
}

```

## notes.md

```text
# Notes: contact vector NSI on down quarks with a complex e–τ entry

Description: contact vector NSI on down quarks with ε^{dV}_{μμ} = −0.05 and
ε^{dV}_{eτ} = 0.02 + 0.03i, with its Hermitian partner. All other entries vanish.

## Derivation

1. **Lorentz structure.** The description says "vector NSI", so structure `V`.
   No mass is given and the interaction is stated as epsilons, so the contact
   form of conventions section 5 applies: the epsilons are added directly to
   C^{dV} and are Q²-independent.

2. **Coefficients.** In the framework normalization C^{fV} = ε^{fL} + ε^{fR},
   and a contact ε^{fV} is exactly that combination, so no conversion is needed:

   - C^{dV}_{μμ} = −0.05 (real, diagonal) at every Q².
   - C^{dV}_{eτ} = 0.02 + 0.03i at every Q².
   - C^{dV}_{τe} = (C^{dV}_{eτ})* = 0.02 − 0.03i by Hermiticity; the plugin and
     the framework fill this partner automatically when `et` is set.
   - All up-quark entries, all other down-quark entries and all scalar entries
     are zero.

3. **Parameters.** Plugin parameters are real, so the complex e–τ entry is
   built inside `coefficients` from `eps_d_et_re = 0.02` and
   `eps_d_et_im = 0.03`. The μμ entry is `eps_d_mm = −0.05`. All three are
   listed in `COUPLINGS`; zeroing them reproduces the Standard Model, and each
   is far below the perturbativity bound √(4π) ≈ 3.54.

4. **Interference with the SM** (Cs-133, Z = 55, N = 78, tree-level
   Q_SM ≈ −36.93). For a down-quark-only contact term the nuclear charge shift
   is ΔQ^{αβ} = (Z + 2N) C^{dV}_{αβ} = 211 C^{dV}_{αβ}.

   - μμ: ΔQ^{μμ} = 211 × (−0.05) = −10.55, the same sign as Q_SM, so the
     interference is constructive and the ν_μ and ν̄_μ coherent rate rises to
     about (47.48/36.93)² ≈ 1.65 times the SM at tree level.
   - eτ: |ΔQ^{eτ}| = 211 × |0.02 + 0.03i| = 211 × 0.03606 ≈ 7.61. This
     flavor-changing entry does not interfere with the SM (no SM
     flavor-changing neutral current); it adds incoherently as |Q^{eτ}|² to the
     ν_e rate and, with equal magnitude |Q^{τe}|², to the ν_τ rate. Only the
     modulus of the complex entry matters for the rate; the phase is kept as
     given so the coefficient matches the description exactly.

5. **Card.** The schema expresses this interaction directly through one
   `contacts` entry on fermion `d`, with the complex off-diagonal entry
   written as `"et": [0.02, 0.03]` (`[re, im]`). The plugin computes the same
   coefficients independently of the card.

No convention conversion was needed: the description already uses the
framework's ε^{fV} normalization and the e–τ ordering (`et` = C_{eτ}).

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
  "notes": "# Notes: contact vector NSI on down quarks with a complex e\u2013\u03c4 entry\n\nDescription: contact vector NSI on down quarks with \u03b5^{dV}_{\u03bc\u03bc} = \u22120.05 and\n\u03b5^{dV}_{e\u03c4} = 0.02 + 0.03i, with its Hermitian partner. All other entries vanish.\n\n## Derivation\n\n1. **Lorentz structure.** The description says \"vector NSI\", so structure `V`.\n   No mass is given and the interaction is stated as epsilons, so the contact\n   form of conventions section 5 applies: the epsilons are added directly to\n   C^{dV} and are Q\u00b2-independent.\n\n2. **Coefficients.** In the framework normalization C^{fV} = \u03b5^{fL} + \u03b5^{fR},\n   and a contact \u03b5^{fV} is exactly that combination, so no conversion is needed:\n\n   - C^{dV}_{\u03bc\u03bc} = \u22120.05 (real, diagonal) at every Q\u00b2.\n   - C^{dV}_{e\u03c4} = 0.02 + 0.03i at every Q\u00b2.\n   - C^{dV}_{\u03c4e} = (C^{dV}_{e\u03c4})* = 0.02 \u2212 0.03i by Hermiticity; the plugin and\n     the framework fill this partner automatically when `et` is set.\n   - All up-quark entries, all other down-quark entries and all scalar entries\n     are zero.\n\n3. **Parameters.** Plugin parameters are real, so the complex e\u2013\u03c4 entry is\n   built inside `coefficients` from `eps_d_et_re = 0.02` and\n   `eps_d_et_im = 0.03`. The \u03bc\u03bc entry is `eps_d_mm = \u22120.05`. All three are\n   listed in `COUPLINGS`; zeroing them reproduces the Standard Model, and each\n   is far below the perturbativity bound \u221a(4\u03c0) \u2248 3.54.\n\n4. **Interference with the SM** (Cs-133, Z = 55, N = 78, tree-level\n   Q_SM \u2248 \u221236.93). For a down-quark-only contact term the nuclear charge shift\n   is \u0394Q^{\u03b1\u03b2} = (Z + 2N) C^{dV}_{\u03b1\u03b2} = 211 C^{dV}_{\u03b1\u03b2}.\n\n   - \u03bc\u03bc: \u0394Q^{\u03bc\u03bc} = 211 \u00d7 (\u22120.05) = \u221210.55, the same sign as Q_SM, so the\n     interference is constructive and the \u03bd_\u03bc and \u03bd\u0304_\u03bc coherent rate rises to\n     about (47.48/36.93)\u00b2 \u2248 1.65 times the SM at tree level.\n   - e\u03c4: |\u0394Q^{e\u03c4}| = 211 \u00d7 |0.02 + 0.03i| = 211 \u00d7 0.03606 \u2248 7.61. This\n     flavor-changing entry does not interfere with the SM (no SM\n     flavor-changing neutral current); it adds incoherently as |Q^{e\u03c4}|\u00b2 to the\n     \u03bd_e rate and, with equal magnitude |Q^{\u03c4e}|\u00b2, to the \u03bd_\u03c4 rate. Only the\n     modulus of the complex entry matters for the rate; the phase is kept as\n     given so the coefficient matches the description exactly.\n\n5. **Card.** The schema expresses this interaction directly through one\n   `contacts` entry on fermion `d`, with the complex off-diagonal entry\n   written as `\"et\": [0.02, 0.03]` (`[re, im]`). The plugin computes the same\n   coefficients independently of the card.\n\nNo convention conversion was needed: the description already uses the\nframework's \u03b5^{fV} normalization and the e\u2013\u03c4 ordering (`et` = C_{e\u03c4}).\n"
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

### Round 1, attempt 1

```json
{
  "final_text": "Validation passed with no findings, including the plugin/card cross-check on 66 Q\u00b2 points.\n\n**Files produced**\n\n- `plugin.py` sets the contact vector coefficients on down quarks: a real \u03bc\u03bc entry of \u22120.05 and a complex e\u03c4 entry of 0.02 + 0.03i built from separate real and imaginary parameters. The \u03c4e partner is filled as the conjugate. All three parameters are listed as couplings so the Standard Model limit check works.\n- `card.json` expresses the same interaction as one `contacts` entry on fermion `d`, with the complex entry written as `[0.02, 0.03]`.\n- `notes.md` records the derivation: contact form because epsilons were given, no normalization conversion needed, Hermiticity of the e\u03c4/\u03c4e pair, constructive interference of the \u03bc\u03bc entry with the Standard Model charge, and incoherent contribution of the flavor-changing entry to the \u03bd_e and \u03bd_\u03c4 rates.\n\nThe described point is preserved exactly in both representations. No questions.json was written since the description was complete.",
  "exit_status": 0,
  "raw_output": "{\"duration_api_ms\":59811,\"stop_reason\":\"end_turn\",\"session_id\":\"7a0e0b10-610c-4d17-bfa4-5a0fe08bbb07\",\"total_cost_usd\":0.7880042500000001,\"usage\":{\"input_tokens\":226,\"cache_creation_input_tokens\":25156,\"cache_read_input_tokens\":151497,\"output_tokens\":4895,\"output_tokens_details\":{\"thinking_tokens\":626},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":25156,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"global\",\"iterations\":[{\"input_tokens\":32,\"output_tokens\":342,\"cache_read_input_tokens\":30864,\"cache_creation_input_tokens\":298,\"cache_creation\":{\"ephemeral_5m_input_tokens\":0,\"ephemeral_1h_input_tokens\":298},\"type\":\"message\"}],\"speed\":\"standard\",\"fallback_credit\":null},\"modelUsage\":{\"claude-fable-5-1\":{\"inputTokens\":226,\"outputTokens\":4895,\"cacheReadInputTokens\":151497,\"cacheCreationInputTokens\":25156,\"webSearchRequests\":0,\"costUSD\":0.7880042500000001,\"contextWindow\":1000000,\"maxOutputTokens\":64000,\"thinkingTokens\":626,\"canonicalModel\":\"claude-fable-5-1\",\"provider\":\"firstParty\",\"costBasis\":\"list\"}},\"permission_denials\":[{\"tool_name\":\"Bash\",\"tool_use_id\":\"toolu_01KAAeAmm7gAeFBASfPpFd6p\",\"tool_input\":{\"command\":\"ls -R /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/08_complex_contact\",\"description\":\"List task directory contents\"}}],\"terminal_reason\":\"completed\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":false,\"num_turns\":17,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"Validation passed with no findings, including the plugin/card cross-check on 66 Q\u00b2 points.\\n\\n**Files produced**\\n\\n- `plugin.py` sets the contact vector coefficients on down quarks: a real \u03bc\u03bc entry of \u22120.05 and a complex e\u03c4 entry of 0.02 + 0.03i built from separate real and imaginary parameters. The \u03c4e partner is filled as the conjugate. All three parameters are listed as couplings so the Standard Model limit check works.\\n- `card.json` expresses the same interaction as one `contacts` entry on fermion `d`, with the complex entry written as `[0.02, 0.03]`.\\n- `notes.md` records the derivation: contact form because epsilons were given, no normalization conversion needed, Hermiticity of the e\u03c4/\u03c4e pair, constructive interference of the \u03bc\u03bc entry with the Standard Model charge, and incoherent contribution of the flavor-changing entry to the \u03bd_e and \u03bd_\u03c4 rates.\\n\\nThe described point is preserved exactly in both representations. No questions.json was written since the description was complete.\",\"ttft_ms\":911,\"type\":\"result\",\"duration_ms\":63407,\"uuid\":\"676d35a3-e9bb-4acd-a0b2-753d9cb5e50b\",\"ttft_stream_ms\":819,\"time_to_request_ms\":23,\"first_content_frame_ms\":819,\"queued_turn_count\":0,\"result_index\":0}\n",
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
    "{\"hooks\": {\"PreToolUse\": [{\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"/Users/shu/Projects/AI4Neutrino/pyCEvNS/.worktrees/m7-conversation/.venv/bin/python -m nuagent.agent.tool_policy /private/tmp/claude-501/-Users-shu-Projects-AI4Neutrino/c19da553-5f99-4ca1-a2d1-4c51979abec1/scratchpad/live/eval-claude/08_complex_contact\"}]}]}}",
    "--strict-mcp-config",
    "--mcp-config",
    "{\"mcpServers\":{}}",
    "--disable-slash-commands",
    "--no-session-persistence"
  ],
  "error": false
}
```
