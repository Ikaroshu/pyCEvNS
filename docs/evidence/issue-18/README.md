# M6 Streamlit page and packaging

All checks below ran on 2026-10-17 against commit
`efed7e8d1fca3d5767c68d33e858b9c66683a8e3` (tree
`5d13430a87060ef01cbeea8408046b3621ace18c`) on branch `m6-ui`, with
Streamlit 1.66.0. The screenshots are the built-in browser's captures of the
live page at http://127.0.0.1:8501 (1280 x 1000 viewport, downscaled by the
capture tool and converted from JPEG to PNG).

## Launcher

`uv run nuagent ui` ran this child process, read from `ps`:

```text
.venv/bin/python -m streamlit run .../src/nuagent/ui/app.py --server.address 127.0.0.1 --browser.gatherUsageStats false --server.port 8501
```

`lsof -iTCP:8501 -sTCP:LISTEN` showed one listener, `TCP 127.0.0.1:8501`, and
Streamlit logged `Uvicorn server started on 127.0.0.1:8501`. Stopping the
Streamlit process with SIGTERM let `nuagent ui` exit 0.

On this machine Streamlit had never run before (no `~/.streamlit/`). With
stdin not attached to a terminal, Streamlit's first-run onboarding prompt
("enter your email address") read end-of-file and Streamlit exited with
status 255, so `nuagent ui` raised `CalledProcessError`. The manual check
therefore ran `STREAMLIT_SERVER_HEADLESS=true uv run nuagent ui`. Headless
mode skips that prompt and the automatic browser opening; it does not change
the launcher's arguments, which were the ones shown above. In an interactive
terminal the prompt waits for input once (pressing Enter skips it), then
Streamlit opens the browser. This first-run prompt is Streamlit's own; it is
separate from the usage statistics, which stay disabled.

The wheel from `uv build` contains `nuagent/ui/app.py`, `figures.py` and
`launch.py`. Installed into a fresh virtual environment outside the
repository, its `streamlit_command` pointed at the installed
`site-packages/nuagent/ui/app.py`, its `nuagent ui --port 8502` (headless, as
above) listened on `127.0.0.1:8502` only, and the page rendered all panels.

## Panels exercised

1. Experiment selector: opened and chose `coherent_csi_2017`, the only
   registered experiment.
2. Model card editor: chose the `zprime_lmu_ltau` preset and pressed
   "Apply preset to card"; the JSON text area was replaced by that template's
   card ([card-preset.png](card-preset.png)).
3. Spectrum and Δ: "Predict and compare" drew new physics + background and
   SM + background as steps over the 12 PE bins, the summed background, and
   the observed counts with √N error bars ([spectrum.png](spectrum.png)). It
   showed Δ = 7.507566092489984, new-physics `alpha` 0.23609648010760576 and
   `beta` steady_state 0.05109107728346943, prompt_neutrons
   0.018821609061171335, SM `alpha` 0.07067927935625874, and signal total
   55.64316965812137. These equal, digit for digit, the output of
   `uv run nuagent compare --experiment coherent_csi_2017 --template zprime_lmu_ltau --set mass_mev=10 --set g_prime=1e-4 --set kinetic_mixing=1e-3`,
   and the M5b evidence for the same model.
4. Scan: defaults `zprime_lmu_ltau`, mass_mev 1 to 1000 MeV and g_prime 1e-5
   to 1e-2, both log-spaced with 8 points ("Grid: 8 × 8 = 64 points"),
   kinetic_mixing fixed at 1e-3, thresholds 2.71 and 4.61. On Asimov data the
   scan finished in a few seconds (64 points took about 6 s in a headless
   timing run) and drew both contours on log axes with legend entries
   "Δ = 2.71 (expected)" and "Δ = 4.61 (expected)" and the caption "Contours
   of Δ at 2.71, 4.61 on asimov data (expected); Δ range 8.062e-07 to 19.46"
   ([scan-expected.png](scan-expected.png)). Switching to observed data gave
   "(observed)" labels and Δ range 0.0006721 to 23.02
   ([scan-observed.png](scan-observed.png)). The spectrum and Δ from step 3
   stayed on the page while the scan ran, because each panel is a Streamlit
   fragment.
5. Agent panel: entered the L_mu − L_tau description used for M5b, opened the
   backend selector (claude_code, codex), with experiments defaulting to
   `coherent_csi_2017` ([agent-panel.png](agent-panel.png)). "Run agent" was
   not pressed; see below.

Template presets (the values the parameter inputs start with; an empty input
keeps the template default):

| Template | Preset |
| --- | --- |
| zprime_universal | mass_mev 30, g_nu 1e-4, g_q 1e-4 |
| zprime_u1 | mass_mev 10, g_prime 1e-4, q_nu_e = q_nu_mu = q_nu_tau = −1, q_u = q_d = 1/3 (B − L) |
| zprime_lmu_ltau | mass_mev 10, g_prime 1e-4, kinetic_mixing 1e-3 |
| scalar_universal | mass_mev 10, g_nu 1e-5, g_q 1e-5 |
| contact_nsi | eps_u_ee 0.1, eps_d_ee 0.1 |

The spectrum screenshot was taken a few minutes before the commit, from a
working tree that differed from it only in the contour legend position
(`loc="lower right"`), which does not affect the spectrum. The scan and agent
screenshots were taken from the commit itself.

## Tests

- `uv run pytest tests/test_ui.py`: 6 passed. They check the launcher
  arguments and that the app file exists, run `nuagent ui --help` through the
  installed executable, and drive the page headlessly with
  `streamlit.testing.v1.AppTest`: applying the `zprime_lmu_ltau` preset
  rewrites the card to `template_card(...).model_dump_json(indent=2)`, the Δ
  shown equals the value computed directly with `profiled` (relative 1e-9),
  and a 2 x 2 scan runs on Asimov and observed data with captions labeled
  expected and observed. A direct test of the contour figure checks the
  legend labels and log axes.
- `uv run pytest`: 328 passed (322 before this milestone plus the 6 new
  tests). The only warning is the frozen legacy package's `pkg_resources`
  deprecation warning.
- `uv build --out-dir <scratch>`: built the sdist and wheel; the wheel lists
  `nuagent/ui/app.py`.
- `uv run nuagent ui --help`: exits 0 and lists `--port`.

## Not verified

- No live agent run from the page: it costs a paid backend call and minutes.
  The panel calls `run_result` with the selected backend, exactly as tested
  offline for `propose`; the live propose runs are left for after the merge.
- The page was not opened by Streamlit's automatic browser launch, because
  the server ran headless (see Launcher).
- Fresh-clone checking is left for after the merge.
