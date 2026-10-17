# Coefficient plugin API

A coefficient plugin is one Python module. The framework imports it by path, reads three names, and calls one function.

```python
import numpy as np
from nuagent.physics.couplings import Coefficients, GF, SQRT2, E_CHARGE

PARAMETERS = {"g_nu": 1e-4, "g_q": 1e-4, "mass_mev": 30.0}   # defaults; every key is a parameter
COUPLINGS = ("g_nu", "g_q")                                   # the subset that are couplings

def coefficients(q2: np.ndarray, params: dict[str, float]) -> Coefficients:
    c = Coefficients.zeros(q2)                                 # all structures, quarks, flavors at zero
    value = params["g_nu"] * params["g_q"] / (SQRT2 * GF * (q2 + params["mass_mev"] ** 2))
    for quark in ("u", "d"):
        for pair in ("ee", "mm", "tt"):
            c.set("V", quark, pair, value)                     # value: scalar or array shaped like q2
    return c
```

Contract:

- `coefficients` returns the new-physics contribution only. Never add Standard Model values.
- `q2` is a 1D array of Q² = 2 M T in MeV², all entries ≥ 0. Return values broadcast over it.
- `Coefficients.set(structure, quark, pair, value)` with `structure` in `{"V", "S"}`, `quark` in `{"u", "d"}`, `pair` in `{"ee","mm","tt","em","et","mt"}`. Setting `em` also sets `me` to the complex conjugate. Diagonal values must be real; a complex diagonal value raises `ValueError`.
- `c.values[structure, quark]` is the complex array shaped `(3, 3, len(q2))` behind `set`, indexed by (α, β) in the order e, μ, τ. Read it to inspect values; write through `set`.
- `COUPLINGS` lists the parameter names that are couplings. The framework sets all of them to zero to check that your plugin reproduces the Standard Model, and checks each is at most √(4π) in magnitude.
- `PARAMETERS` defaults are the parameter point of the description. The harness evaluates the proposed point there, and the eval scores coefficients there.
- Parameters are real numbers. A complex off-diagonal entry is built inside `coefficients` from separate real and imaginary parameters.
- `set` overwrites. Sum every contribution to an entry (for example two mediators) before calling `set` once.
- Pure functions only: no file or network access, no randomness, no global state. Use `numpy` and the constants exported by `nuagent.physics.couplings`.
- Keep the derivation in `notes.md`, not in comments.

A raw cross-section plugin (escape hatch) instead exposes `dsigma_dt(e_nu, t, flavor, isotope, params) -> np.ndarray` returning the complete differential cross section including the Standard Model in MeV⁻³, plus the same `PARAMETERS` and `COUPLINGS`. `t` is a scalar or a 1D array of recoil energies in MeV and `e_nu` broadcasts against it; the result has the broadcast shape. `flavor` is one of `e, mu, tau, ebar, mubar, taubar`; `isotope` is a `nuagent.physics.xsec.Isotope` with attributes `z`, `n`, `mass_mev` and `helm_radius_fm` (the experiment's Helm radius), and `isotope.q2(t)` gives Q² = 2 M T. It is written by users, not by the agent: it must include the framework's SM cross section with the experiment's form factor, which it obtains as `nuagent.physics.xsec.dsigma_dt(e_nu, t, flavor, isotope, nuagent.physics.couplings.sm(isotope.q2(t)))`. It is used only for physics outside the basis (magnetic moments, inelastic channels), it skips the SM-limit check, and its report is labeled `raw-cross-section`. If a description needs physics outside the V and S basis, write `notes.md` naming the raw route and produce no plugin.

`example_plugin.py` is a standalone runnable coefficient plugin. For example, from the repository root:

```bash
uv run nuagent predict --experiment coherent_csi_2017 --plugin src/nuagent/context/example_plugin.py
```

The harness and validation command are planned in issue #17. Once implemented, validation inside a task directory will use:

```bash
nuagent validate --plugin plugin.py --card card.json
```

With both files present the command will also cross-check that the plugin and the card agree on every coefficient at a grid of Q² values. See `framework.md` for what this evidence can establish and what it cannot.
