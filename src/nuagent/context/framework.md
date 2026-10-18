# Framework responsibilities and verification

The agent translates the interaction description into **new-physics coefficients only**. The agent owns the Lorentz structure, the conversion of charges and couplings, the normalization and signs, the momentum dependence, and the derivation in `notes.md`. For a complete interaction within the V and S basis the primary output is `plugin.py`; `card.json` is an additional representation whenever the card schema can express the interaction. A plugin with momentum-dependent couplings may be valid even when no card can express it.

From those coefficients, the framework adds the Standard Model once, forms nuclear charges for each isotope, evaluates the spin-0 vector and scalar cross sections with the Helm form factor, folds the experiment's flavor-dependent flux, and applies detector response, efficiency, exposure and binning. The experiment registry supplies the released data. The harness validates the translation before statistical comparison; the agent must not tune its translation against likelihood values.

## Verification levels

The harness report begins with one of these labels and its meaning. The context pack and offline scorer do not run these validation checks or issue these labels.

| Label | Required evidence | Limits |
|---|---|---|
| `cross-validated` | Coefficient plugin and schema-valid card pass their sanity checks and agree on every structure, quark and flavor coefficient to 1e-6 relative over the experiment's reference momentum grid. Plugin sanity checks cover the Standard Model limit with `COUPLINGS` zeroed, perturbative couplings, and finite non-negative counts. | Agreement checks two representations of a translation. A shared normalization or interpretation error can survive; it does not prove the model, derivation or description is correct. The grid is finite and the spin-0 V/S approximation still applies. |
| `plugin-only` | A coefficient plugin passes the same Standard Model-limit, perturbativity and count checks, with a derivation and explanation of why no card exists. | There is no independent card cross-check. Correct matching and momentum dependence rest on the derivation and any model-specific reduction checks, such as the running-mixing example's Q² → 0 limit. |
| `raw-cross-section` | A user-authored raw plugin returns the complete differential cross section, including the Standard Model and the appropriate form factor. | This escape hatch bypasses the coefficient/card comparison and the Standard Model-limit check. The user owns the normalization and physics outside the basis; this is a weaker verification level. |

An under-specified description within the V/S basis is a valid questions outcome: write `questions.json` asking for the missing mass, couplings or Lorentz structure, and `notes.md` saying what is missing, instead of inventing numbers, and write no plugin or card. The harness shows the questions to the user and continues the same task with the answers, for at most three rounds; a third round that still asks is reported as unresolved. A description outside the V/S basis, such as a magnetic moment, is notes-only: explain why the basis is insufficient and direct the user to the user-authored raw cross-section route in `plugin_api.md`. The agent does not author that raw plugin.

The offline evaluation scorer measures two different properties: agreement with known coefficients at three reference Q² values (or the required questions or notes-only response), and whether `card.json` is present exactly when expected. Those metrics do not validate card contents, certify the verification levels above, or establish live backend performance.
