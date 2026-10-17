# M6 Streamlit page and packaging

The current evidence was recorded on 2026-10-17 against commit
`590b779f1b48628c0164dbbecee4b1314182d2e1` (tree
`e4a80a70a58a8c7f9f0ed41278c425b14cdd0f5f`) on branch `m6-ui`, with
Streamlit 1.66.0. That commit applies the code review pass 1 repairs. The
screenshots are the built-in browser's captures of the live page at
http://127.0.0.1:8501 (1280 x 1000 viewport, downscaled by the capture tool
and converted from JPEG to PNG). The panel checks used a headless server
(`STREAMLIT_SERVER_HEADLESS=true uv run nuagent ui`), so that no browser
tab opened on the desktop.

## Launcher

`nuagent ui [--port N]` replaces its own process with Streamlit
(`os.execv`), running:

```text
python -m streamlit run <installed nuagent/ui/app.py> --server.address 127.0.0.1 --browser.gatherUsageStats false --server.showEmailPrompt false --server.port <port>
```

`--server.showEmailPrompt false` is needed because on a machine where
Streamlit has never run (no `~/.streamlit/`, as here), Streamlit first asks
for an email address. With stdin not a terminal it read end-of-file and
exited 255, which made the first background start fail. In Streamlit 1.66.0,
`check_credentials` in `streamlit/runtime/credentials.py` skips that prompt
when `server.headless` is true or `server.showEmailPrompt` is false, so the
flag removes the prompt and keeps Streamlit's automatic browser opening.

Check at `590b779`, without `STREAMLIT_SERVER_HEADLESS` and without
`~/.streamlit/`: `uv run nuagent ui --port 8604 </dev/null` started.
- `ps` showed `uv run nuagent ui --port 8604` with one child, the `nuagent`
  process now running the Streamlit command line above with port 8604.
- Streamlit logged `Uvicorn server started on 127.0.0.1:8604`, `curl` got 200,
  and `lsof -iTCP:8604 -sTCP:LISTEN` showed one listener,
  `TCP 127.0.0.1:8604`. No prompt appeared and no `~/.streamlit/` was created.
- SIGINT to `uv` and the Streamlit process (as a terminal Ctrl-C delivers to
  both) printed `Stopping...` and exited 0 with no traceback.

An earlier check at `17453f1` (before `os.execv`) showed the same listener
on port 8503. Before `--server.showEmailPrompt` was added (`efed7e8`), an
installed wheel in a fresh virtual environment outside the repository served
the page from `site-packages/nuagent/ui/app.py` on `127.0.0.1:8502` only;
that check was not repeated later.

## Panels exercised

1. Experiment selector: opened, `coherent_csi_2017` (the only registered
   experiment) chosen.
2. Model card editor: chose the `zprime_lmu_ltau` preset and pressed
   "Apply preset to card". The JSON text area was replaced by that
   template's card, and the kinetic_mixing input stayed empty, so the
   template's loop value KAPPA_LMU_LTAU · g_prime applies
   ([card-preset.png](card-preset.png)).
3. Spectrum and Δ: "Predict and compare" drew new physics + background and
   SM + background as steps over the 12 PE bins, the summed background, and
   the observed counts with √N error bars ([spectrum.png](spectrum.png)).
   At this point (10 MeV, g′ = 1e-4, loop kinetic mixing) the two curves
   nearly coincide. The page showed Δ = 0.0024282648663600526, new-physics
   `alpha` 0.07170206454158409 and `beta` steady_state 0.0015948699843009334,
   prompt_neutrons 0.0028993722580666847, and signal total
   133.70702102492538. These equal, digit for digit,
   `uv run nuagent compare --experiment coherent_csi_2017 --template zprime_lmu_ltau --set mass_mev=10 --set g_prime=1e-4`.
   (Before the repair the preset fixed kinetic_mixing at 1e-3, which gives
   Δ = 7.507566092489984.)
4. Scan: defaults `zprime_lmu_ltau`, mass_mev 1 to 1000 MeV and g_prime 1e-5
   to 1e-2, both log-spaced with 8 points ("Grid: 8 × 8 = 64 points"),
   kinetic_mixing empty (loop value), thresholds 2.71 and 4.61. 64 points
   take about 6 s.
   - Asimov data: contours on log axes with legend entries
     "Δ = 2.71 (expected)" and "Δ = 4.61 (expected)" and the caption
     "Contours of Δ at 2.71, 4.61 on asimov data (expected); Δ range 3.93e-14
     to 19.45" ([scan-expected.png](scan-expected.png)). Below about 10 MeV
     the Δ = 2.71 contour lies near g′ ≈ 1.7e-3. A separate 61-point 1D
     Asimov scan at 10 MeV first exceeds 2.71 at g′ = 2.0e-3 (Δ 2.61 at
     1.78e-3). With kinetic_mixing fixed at 1e-3 it does so already at
     g′ = 5.0e-5.
   - Observed data: "(observed)" labels, Δ range 9.693e-08 to 22.76
     ([scan-observed.png](scan-observed.png)).
   - The spectrum and Δ from step 3 stayed on the page while the scan ran,
     because each panel is a Streamlit fragment. The "Δ grid" expander below
     the caption lists the scanned values.
5. Agent panel (screenshot from `8c4f46f`, unchanged layout): entered the
   L_mu − L_tau description used for M5b and opened the backend selector
   (claude_code, codex), with experiments defaulting to `coherent_csi_2017`
   ([agent-panel.png](agent-panel.png)). "Run agent" was not pressed with a
   live backend. Page tests run it with the fake backend.

Template presets (the values the parameter inputs start with; an empty input
keeps the template default):

| Template | Preset |
| --- | --- |
| zprime_universal | mass_mev 30, g_nu 1e-4, g_q 1e-4 |
| zprime_u1 | mass_mev 10, g_prime 1e-4, q_nu_e = q_nu_mu = q_nu_tau = −1, q_u = q_d = 1/3 (B − L) |
| zprime_lmu_ltau | mass_mev 10, g_prime 1e-4; kinetic_mixing empty, so the loop value KAPPA_LMU_LTAU · g_prime |
| scalar_universal | mass_mev 10, g_nu 1e-5, g_q 1e-5 |
| contact_nsi | eps_u_ee 0.1, eps_d_ee 0.1 |

The scan panel's fixed-parameter inputs start from the same presets.

## Tests

- `uv run pytest tests/test_ui.py`: 10 passed (9 at `590b779`; review pass 2
  added the failed-run test below).
  - The launcher test checks the exact flag/value pairs and that the app file
    exists. `nuagent ui --help` runs through the installed executable.
  - The preset test checks that applying `zprime_lmu_ltau` writes
    `template_card(...).model_dump_json(indent=2)` into the card. It also
    checks that the Δ shown equals `profiled` computed directly
    (relative 1e-9).
  - For a 2 x 2 scan on Asimov and on observed data, the tests check the
    caption labels (expected, observed). They also check that the shown Δ
    grid equals `asimov.scan` (rtol 1e-9).
  - Another scan test checks that kinetic_mixing starts empty. Set to 1e-3,
    the shown grid equals `asimov.scan` with that value. Cleared again, it
    equals `asimov.scan` without kinetic_mixing (the loop value), and the two
    grids differ.
  - The agent panel runs with the fake backend from `tests/test_agent.py`, in
    a validating mode and in a mode that fails after the correction. Each
    time the test checks the success or error message and that the report
    file's text is rendered. Both still show after a rerun with an edited
    description.
  - A valid fake run followed by a run whose backend fails clears the stored
    result: after the next rerun no success message or earlier report shows.
  - The contour figure's legend labels and log axes are checked directly.
- `uv run pytest`: 332 passed (322 before this milestone plus 10). The only
  warning is the frozen legacy package's `pkg_resources` deprecation warning.
- `uv build --out-dir <scratch>`: the wheel lists `nuagent/ui/app.py`,
  `figures.py` and `launch.py`, and its metadata has
  `Requires-Dist: streamlit>=1.66`. `uv lock` changed only the streamlit
  requirement line in `uv.lock`; the locked version stays 1.66.0.
- `uv run nuagent ui --help`: exits 0 and lists `--port`.

## Not verified

- No live agent run from the page: it costs a paid backend call and minutes.
  The live propose runs are left for after the merge.
- Streamlit's automatic browser opening was not observed. The panel checks ran
  headless, and the non-headless 8604 check only verified the listener and
  the SIGINT exit.
- Fresh-clone checking is left for after the merge.
