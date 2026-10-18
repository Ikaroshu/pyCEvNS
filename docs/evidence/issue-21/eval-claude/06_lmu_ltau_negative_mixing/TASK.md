# Translate a neutrino interaction

## Description

A gauged L_mu−L_tau vector Z′ has mass 15 MeV, gauge coupling g′ = 6e-5, total photon kinetic mixing epsilon = −5e-4 in the pack's convention, and no direct quark couplings.

## Output contract

Read context/conventions.md, context/plugin_api.md, context/card_schema.json,
context/framework.md and the relevant worked examples before writing outputs.
Work only in this task directory. Use your own file tools to read the context
and to write the output files (Read, Write and Edit on Claude Code; file reads
and apply_patch on Codex). The only output files are plugin.py, card.json, notes.md, questions.json.
The only shell commands needed are the exact nuagent validate commands below.
Do not inspect the source repository, evaluation cases, expected answers,
observed data or likelihood values.

For a complete description within the vector/scalar basis, write plugin.py with
coefficients, PARAMETERS defaults and COUPLINGS, plus notes.md with the derivation.
Write card.json whenever the schema can express the interaction. A plugin must
compute the coefficients independently of card.json, using exported constants;
do not read files from a plugin. If no card can express the interaction, explain
why in notes.md. Sum contributions before calling Coefficients.set (it overwrites).

If a description within the vector/scalar basis lacks information you need
(the mass, a coupling or the Lorentz structure), write questions.json, notes.md
saying what is missing, and no plugin.py or card.json. questions.json has the shape given in
context/plugin_api.md; suggest a value only when a conventional benchmark exists,
and never present a suggestion as part of the description. If the interaction is
outside the V/S basis, write notes.md explaining the basis limitation and the
user-authored raw-cross-section route, and produce no other file. Never write a
raw cross-section plugin. These question and notes-only outcomes are legitimate
and receive no predictions or physics verification label.

When a plugin is produced, run one of these commands before finishing and repair
validation findings. These commands return validation only, never likelihoods:

    nuagent validate --plugin plugin.py --card card.json
    nuagent validate --plugin plugin.py

The harness validates on: coherent_csi_2017. Do not tune any parameter
against observations, likelihoods or evaluation answers. Preserve the described
point exactly. Finish with a short description of the files produced.
