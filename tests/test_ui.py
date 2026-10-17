"""The ui launcher command and the Streamlit page driven headlessly, without a server."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from nuagent.experiments.registry import get
from nuagent.physics.modelcard import template_card
from nuagent.rates import predicted_counts
from nuagent.stats.asimov import Scan
from nuagent.stats.likelihood import profiled
from nuagent.ui.figures import contour_figure
from nuagent.ui.launch import APP_PATH, streamlit_command


def test_streamlit_command_binds_loopback_without_usage_statistics():
    command = streamlit_command(8600)
    assert command[:5] == [sys.executable, "-m", "streamlit", "run", str(APP_PATH)]
    assert Path(command[4]).is_file()
    pairs = dict(zip(command[5::2], command[6::2]))
    assert pairs == {
        "--server.address": "127.0.0.1", "--browser.gatherUsageStats": "false",
        "--server.showEmailPrompt": "false", "--server.port": "8600",
    }


def test_ui_help():
    result = subprocess.run([str(Path(sys.executable).with_name("nuagent")), "ui", "--help"], check=True, capture_output=True, text=True)
    assert "--port" in result.stdout


def page() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception
    return app


def button(app: AppTest, label: str):
    return next(widget for widget in app.button if widget.label == label)


def test_preset_spectrum_and_delta():
    app = page()
    parameters = {"mass_mev": 10.0, "g_prime": 1e-4, "kinetic_mixing": 1e-3}
    app.selectbox(key="preset").select("zprime_lmu_ltau").run()
    button(app, "Apply preset to card").click().run()
    card = template_card("zprime_lmu_ltau", parameters)
    assert app.text_area(key="card").value == card.model_dump_json(indent=2)
    button(app, "Predict and compare").click().run()
    assert not app.exception
    experiment = get("coherent_csi_2017")
    expected = (profiled(experiment, predicted_counts(experiment, card.coefficients)).neg2_log_likelihood
                - profiled(experiment, predicted_counts(experiment)).neg2_log_likelihood)
    assert float(app.metric[0].value) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("data, label", [("asimov", "expected"), ("observed", "observed")])
def test_scan_panel_labels_contours(data, label):
    app = page()
    app.number_input(key="scan_zprime_lmu_ltau_0_mass_mev_n").set_value(2)
    app.number_input(key="scan_zprime_lmu_ltau_1_g_prime_n").set_value(2)
    app.radio(key="scan_data").set_value(data).run()
    assert "Grid: 2 × 2 = 4 points" in [markdown.value for markdown in app.markdown]
    button(app, "Run scan").click().run()
    assert not app.exception
    assert any(f"on {data} data ({label})" in caption.value for caption in app.caption)


def test_contour_figure_labels_and_log_axes():
    scan = Scan({"mass_mev": np.geomspace(1, 100, 3), "g_prime": np.linspace(0, 1, 3)}, np.arange(9.0).reshape(3, 3))
    axes = contour_figure(scan, [4.61, 2.71], "observed", (True, False)).axes[0]
    assert [text.get_text() for text in axes.get_legend().get_texts()] == ["Δ = 2.71 (observed)", "Δ = 4.61 (observed)"]
    assert (axes.get_xscale(), axes.get_yscale(), axes.get_xlabel(), axes.get_ylabel()) == ("log", "linear", "mass_mev", "g_prime")
