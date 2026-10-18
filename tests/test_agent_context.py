"""Packaged resources, their human mirror and the standalone example."""

import json
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from pathlib import Path

import numpy as np

from nuagent.agent.validate import check_questions
from nuagent.physics.modelcard import ModelCard, template_card
from nuagent.physics.plugin import load_coefficient_plugin

CONTEXT_FILES = {
    "README.md", "conventions.md", "card_schema.json", "plugin_api.md", "framework.md", "example_plugin.py",
    "examples/01_zprime_universal.md", "examples/02_zprime_b_minus_l.md", "examples/03_zprime_lmu_ltau.md",
    "examples/04_light_scalar.md", "examples/05_contact_nsi_offdiagonal.md", "examples/06_running_kinetic_mixing.md",
    "examples/07_under_specified_scalar.md",
}


def resource_contents(directory: Traversable) -> dict[str, bytes]:
    content = {}
    for child in directory.iterdir():
        if child.name == "__pycache__":
            continue
        if child.is_dir():
            content.update({f"{child.name}/{name}": data for name, data in resource_contents(child).items()})
        else:
            content[child.name] = child.read_bytes()
    return content


def test_installed_context_resources_and_human_mirror():
    content = resource_contents(files("nuagent.context"))
    assert set(content) == CONTEXT_FILES
    docs = Path(__file__).parent.parent / "docs" / "agent"
    assert resource_contents(docs) == content
    schema = json.loads(content["card_schema.json"])
    assert schema == ModelCard.model_json_schema()
    for model in (schema, *schema["$defs"].values()):
        assert all(field["description"] for field in model["properties"].values())


def test_standalone_example_plugin():
    q2 = np.array([1238.0, 3714.0, 6190.0])
    with as_file(files("nuagent.context").joinpath("example_plugin.py")) as path:
        plugin = load_coefficient_plugin(path)
    assert plugin.parameters == {"g_prime": 1e-4, "mass_mev": 30.0}
    assert plugin.couplings == ("g_prime",)
    expected = template_card("zprime_universal", {"g_nu": 1e-4, "g_q": 1e-4, "mass_mev": 30.0}).coefficients(q2)
    produced = plugin.coefficients(q2, plugin.parameters)
    for key, value in expected.values.items():
        np.testing.assert_allclose(produced.values[key], value, rtol=1e-14, atol=0)
    assert all(np.count_nonzero(value) == 0 for value in plugin.coefficients(q2, {"g_prime": 0, "mass_mev": 30}).values.values())


def test_under_specified_example_questions_are_valid(tmp_path):
    from evals.agent.cases import CASES
    text = files("nuagent.context").joinpath("examples/07_under_specified_scalar.md").read_text()
    description = text.split('Description given to the agent: "')[1].split('"')[0]
    assert description not in {case.description for case in CASES}
    path = tmp_path / "questions.json"
    path.write_text(text.split("```json\n")[1].split("```")[0])
    report, questions = check_questions(path)
    assert report.valid
    assert [question["id"] for question in questions] == ["quark_couplings"]
    assert isinstance(questions[0]["suggested"], str)
    plugin_path = tmp_path / "plugin.py"
    plugin_path.write_text(text.split("```python\n")[1].split("```")[0])
    plugin = load_coefficient_plugin(plugin_path)
    assert plugin.parameters == {"mass_mev": 10.0, "g_nu": 1e-5, "g_q": 1e-5}
    expected = template_card("scalar_universal", {"mass_mev": 10.0, "g_nu": 1e-5, "g_q": 1e-5}).coefficients(np.array([1238.0]))
    produced = plugin.coefficients(np.array([1238.0]), plugin.parameters)
    np.testing.assert_allclose(produced.values["S", "u"][0, 0], expected.values["S", "u"][0, 0], rtol=1e-14)
    assert np.count_nonzero(produced.values["S", "u"][1:]) == 0
    assert "Assumed values" not in text and "$" in text.split("## notes.md for round 2")[1]
