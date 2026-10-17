"""Exercise installed CLI commands and their parseable per-bin output."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from test_modelcard import CONTEXT, example_blocks
from test_plugins import RAW_SM_PLUGIN

from nuagent.experiments.registry import get
from nuagent.physics.modelcard import ModelCard
from nuagent.rates import predicted_counts


def cli(*arguments: str) -> dict:
    result = subprocess.run([str(Path(sys.executable).with_name("nuagent")), *arguments], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def test_experiments_command():
    assert cli("experiments") == {"experiments": ["coherent_csi_2017"]}


def test_predict_templates_and_data():
    args = ("predict", "--experiment", "coherent_csi_2017", "--template", "zprime_universal", "--set", "mass_mev=30")
    standard = cli(*args, "--set", "g_nu=0", "--set", "g_q=0")
    changed = cli(*args, "--set", "g_nu=1e-4", "--set", "g_q=1e-4")
    assert len(standard["bins"]) == 12
    assert standard["bin_unit"] == "PE"
    assert standard["time_window_us"] == [0, 12]
    assert standard["data_release"] == "10.5281/zenodo.1228631"
    assert standard["totals"]["observed"] == sum(row["observed"] for row in standard["bins"]) == 855
    assert standard["totals"]["backgrounds"]["steady_state"] == 703
    assert standard["totals"]["backgrounds"]["prompt_neutrons"] == pytest.approx(5.419341500706415)
    for name, total in standard["totals"]["backgrounds"].items():
        assert total == pytest.approx(sum(row["backgrounds"][name] for row in standard["bins"]))
    assert standard["totals"]["signal"] == pytest.approx(sum(row["signal"] for row in standard["bins"]))
    np.testing.assert_allclose([row["signal"] for row in standard["bins"]], predicted_counts(get("coherent_csi_2017")), rtol=1e-12)
    assert changed["totals"]["signal"] != pytest.approx(standard["totals"]["signal"], rel=.01)
    np.testing.assert_array_equal([row["observed"] for row in standard["bins"]], [row["observed"] for row in changed["bins"]])


def test_predict_card_and_coefficient_plugin(tmp_path):
    example = CONTEXT / "examples" / "01_zprime_universal.md"
    card_text = example_blocks(example, "json")[0]
    card = tmp_path / "card.json"
    card.write_text(card_text)
    plugin = tmp_path / "plugin.py"
    plugin.write_text(example_blocks(example, "python")[0])
    from_card = cli("predict", "--experiment", "coherent_csi_2017", "--card", str(card))
    from_plugin = cli("predict", "--experiment", "coherent_csi_2017", "--plugin", str(plugin))
    expected = predicted_counts(get("coherent_csi_2017"), ModelCard.model_validate_json(card_text).coefficients)
    for output in (from_card, from_plugin):
        np.testing.assert_allclose([row["signal"] for row in output["bins"]], expected, rtol=1e-12, atol=0)
    modified = cli("predict", "--experiment", "coherent_csi_2017", "--plugin", str(plugin), "--set", "g_prime=2e-4")
    assert modified["parameters"]["g_prime"] == 2e-4
    assert modified["totals"]["signal"] != pytest.approx(from_plugin["totals"]["signal"], rel=.01)


def test_predict_raw_sm_plugin(tmp_path):
    plugin = tmp_path / "raw.py"
    plugin.write_text(RAW_SM_PLUGIN)
    output = cli("predict", "--experiment", "coherent_csi_2017", "--raw-plugin", str(plugin))
    np.testing.assert_allclose([row["signal"] for row in output["bins"]], predicted_counts(get("coherent_csi_2017")), rtol=1e-12, atol=0)
