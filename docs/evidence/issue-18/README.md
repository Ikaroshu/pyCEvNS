# M6 Streamlit page and packaging

The page was trimmed for the demo to one panel: describe an interaction, pick
the backend and experiments, run the `propose` harness, read the report, and
see the predicted spectrum. The live check below ran on 2026-10-17 with
Streamlit 1.66.0 and Claude Code 2.1.286 at commit
`fe4ab18f59948c8cbd5b4a9783f7b48d3e08cb4a` (tree
`47d137ad67e67bbb3cdec653153c167c6fc7717b`). The report records
`source_dirty: true` because the removal of the old screenshots was staged
but not yet committed; no source file differed from that commit.

At that commit the Claude Code backend was still named `claude_code`, so the
retained report says `"backend": "claude_code"`. The next commit,
`77812f7cedb071586cffca9776befdba64d6673f` ("Rename the claude_code backend
to claude"), renames it to `claude` and changes nothing else in the backend.
The same run is now started with `--backend claude` or by choosing `claude`
in the page.

## Launcher

`nuagent ui [--port N]` replaces its own process with Streamlit
(`os.execv`), running:

```text
python -m streamlit run <installed nuagent/ui/app.py> --server.address 127.0.0.1 --browser.gatherUsageStats false --server.showEmailPrompt false --server.port <port>
```

For the live check, `uv run nuagent ui </dev/null` ran in the background,
without headless mode.
- `ps` showed that command line with port 8501.
- Streamlit logged `Uvicorn server started on 127.0.0.1:8501`.
- `lsof -iTCP:8501 -sTCP:LISTEN` showed one listener, `TCP 127.0.0.1:8501`.
- There was no first-run email prompt, although no `~/.streamlit/` existed.
  `--server.showEmailPrompt false` skips it: in Streamlit 1.66.0,
  `check_credentials` in `streamlit/runtime/credentials.py` skips the prompt
  when `server.headless` is true or `server.showEmailPrompt` is false.
  Without the flag, an earlier non-interactive start read end-of-file at the
  prompt and exited 255.
- SIGINT stopped the server with `Stopping...` and no traceback.

An earlier check on port 8604 (commit `590b779`) showed the same: one
listener on `127.0.0.1:8604`, and SIGINT to `uv` and Streamlit, as a terminal
Ctrl-C delivers it, exited 0 with no traceback.

Streamlit 1.66.0 also shows its own "Help agents write better apps" toast
on the page. During the check an accidental click or keypress hit its
Install control, which ran Streamlit's skill installer in the worktree. That
created two untracked symlinks, `.agents/skills/developing-with-streamlit`
and `.claude/skills/developing-with-streamlit`, which were deleted
afterwards. They do not affect the backend, which runs in its own temporary
task directory with setting sources disabled.

## Live run from the page

In the built-in browser at http://127.0.0.1:8501, the description was
"a gauged L_mu minus L_tau Z prime with mass 10 MeV, gauge coupling 1e-4 and
kinetic mixing 1e-3". The backend was `claude_code` (now `claude`) and the
experiment `coherent_csi_2017`. "Run agent" was pressed once.

- Task directory:
  `/var/folders/m1/2x7x34rx4312bwszy_5409v00000gn/T/nuagent-ui-4gtpwq_c/task`,
  report `.../nuagent-ui-4gtpwq_c/report.md`. The run took about 97 s of
  backend time and 22 turns. Claude Code reported `total_cost_usd`
  1.0507475.
- The page showed "Outputs passed validation." and the report began
  `Verification: **cross-validated**`. Validation passed on the first attempt.
- Δ = 7.507566092489984 and predicted total 55.64316965812137 events, with
  855 observed ([agent-report.png](agent-report.png) shows the top of the
  report).
- The spectrum below the report shows new physics + background below
  SM + background in the low-PE bins, the background, and the observed counts
  with √N errors ([agent-spectrum.png](agent-spectrum.png)).
- The agent read the stated mixing ε = 1e-3 as the total photon mixing. Its
  card has a vector mediator at 10 MeV with ν couplings (0, 1e-4, −1e-4), no
  direct quark couplings, and `kinetic_mixing` 1e-3.

Cross-check:
`uv run nuagent compare --experiment coherent_csi_2017 --template zprime_lmu_ltau --set mass_mev=10 --set g_prime=1e-4 --set kinetic_mixing=1e-3`
gives `delta` 7.507566092489984 and signal total 55.64316965812137, equal
digit for digit to the page. The same values appear in the M5b live runs in
`docs/evidence/issue-17/`.

[live-run/](live-run/) keeps the unmodified `report.md` and the agent's
`plugin.py`, `card.json` and `notes.md`. The context pack, prompt and history
remain in the task directory above.

## Edits during a run

After code review, the inputs and the "Run agent" button were moved into one
`st.form`. With Streamlit's default fast reruns, a widget edit during a run
used to stop the running script before it stored the result, so the report
never appeared. Inside a form, edits send no rerun until the next submit.
The commit after `4e648f1` also refuses an empty experiment selection
("Choose at least one experiment.") before calling the backend.

To check the form without a live run, the page was served on 127.0.0.1:8605
with a fake `claude` executable first on `PATH`. The fake sleeps 12 s, then
writes a notes-only `notes.md`. In the browser the description
"a new light vector boson that couples to neutrinos" was submitted, and while
the spinner ran, the backend selector was changed to `codex`. When the fake
finished, the page showed the task directory
(`.../nuagent-ui-vqtudltj/task`), "Outputs passed validation." and the
notes-only report, whose provenance records `"backend": "claude"` (the
backend submitted with the form). The report records `source_dirty: true`
because this ran on the uncommitted repair.

## Tests and packaging

- `uv run pytest tests/test_ui.py`: 7 passed.
  - The launcher test checks the exact flag/value pairs and that the app file
    exists. `nuagent ui --help` runs through the installed executable.
  - The agent-panel tests use the fake backend from `tests/test_agent.py`.
  - In the validating mode they check: the success message; the rendered
    report text, read from `report.md`; one spectrum figure; and stored
    signal counts equal to `predicted_counts` of the fake model's card.
  - In the mode that fails after correction they check the error message,
    the report, no figure and no stored signals.
  - Both results still show after a rerun with an edited description.
  - A valid run followed by a run whose backend fails leaves no success
    message, report or figure after the next rerun.
  - The description, backend and experiment inputs and the only button,
    "Run agent", belong to the form `agent`.
  - With the experiment selection cleared, submitting shows "Choose at least
    one experiment." and never calls the backend.
- `uv run pytest`: 329 passed (322 before this milestone plus 7). The only
  warning is the frozen legacy package's `pkg_resources` deprecation warning.
- `uv build --out-dir <scratch>`: the wheel's `nuagent/ui/` contains exactly
  `app.py`, `figures.py` and `launch.py`, and its metadata requires
  `streamlit>=1.66`.
- `uv run nuagent ui --help`: exits 0 and lists `--port`.

## Not verified

- The live run was made once, with the Claude Code backend. Codex was not run
  live from the page.
- The live run used the backend name in force at `fe4ab18` (`claude_code`).
  After the rename the backend is covered by the offline tests, not by a
  second live run.
- Streamlit's automatic browser opening was not observed in the desktop
  browser.
- Fresh-clone checking is left for after the merge.
