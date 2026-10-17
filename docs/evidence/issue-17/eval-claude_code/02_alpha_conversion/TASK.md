# Translate a neutrino interaction

## Description

A universal vector Z′ of mass 45 MeV has alpha′ = g′²/(4 pi) = 3.899296105751435e-10. Its positive coupling g′ is the same for all left-handed neutrino flavors and both u and d quarks, with no other couplings.

## Output contract

Read context/conventions.md, context/plugin_api.md, context/card_schema.json,
context/framework.md and the relevant worked examples before writing outputs.
Work only in this task directory. Use Read to read context and Write/Edit to
write plugin.py, card.json and notes.md. The only shell commands needed are the
exact nuagent validate commands below. Do not inspect the source repository,
evaluation cases, expected answers, observed data or likelihood values.

For a complete description within the vector/scalar basis, write plugin.py with
coefficients, PARAMETERS defaults and COUPLINGS, plus notes.md with the derivation.
Write card.json whenever the schema can express the interaction. A plugin must
compute the coefficients independently of card.json, using exported constants;
do not read files from a plugin. If no card can express the interaction, explain
why in notes.md. Sum contributions before calling Coefficients.set (it overwrites).

If information is missing, write notes.md explicitly asking for the missing
information; produce no plugin.py or card.json. If the interaction is outside
the V/S basis, write notes.md explaining the basis limitation and the
user-authored raw-cross-section route, and produce no plugin.py or card.json.
Never write a raw cross-section plugin. These notes-only outcomes are legitimate
and receive no predictions or physics verification label.

When a plugin is produced, run one of these commands before finishing and repair
validation findings. These commands return validation only, never likelihoods:

    nuagent validate --plugin plugin.py --card card.json
    nuagent validate --plugin plugin.py

The harness validates on: coherent_csi_2017. Do not tune any parameter
against observations, likelihoods or evaluation answers. Preserve the described
point exactly. Finish with a short description of the files produced.
