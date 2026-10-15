import json
import os

import pytest

from src.core.files import relative_path, write_json_atomic


@pytest.mark.unit
def test_write_json_atomic_writes_indented_json_and_leaves_no_temporary_file(tmp_path):
    target = tmp_path / "artefact.json"
    payload = {"b": [1, 2], "a": {"nested": 0.5}}

    write_json_atomic(payload, str(target))

    assert target.read_text(encoding="utf-8") == json.dumps(payload, indent=2)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["artefact.json"]


@pytest.mark.unit
def test_relative_path_strips_the_local_prefix_from_a_path_under_the_working_directory():
    assert relative_path(os.path.join(os.getcwd(), "data", "input", "limitless_model_fit.json")) == os.path.join("data", "input", "limitless_model_fit.json")


@pytest.mark.unit
def test_relative_path_keeps_a_path_it_cannot_place(monkeypatch):
    monkeypatch.setattr(os.path, "relpath", lambda path: (_ for _ in ()).throw(ValueError("different drive")))

    assert relative_path("D:/elsewhere/limitless_model_fit.json") == "D:/elsewhere/limitless_model_fit.json"
