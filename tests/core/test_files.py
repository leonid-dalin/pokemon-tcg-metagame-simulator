import json

import pytest

from src.core.files import write_json_atomic


@pytest.mark.unit
def test_write_json_atomic_writes_indented_json_and_leaves_no_temporary_file(tmp_path):
    target = tmp_path / "artefact.json"
    payload = {"b": [1, 2], "a": {"nested": 0.5}}

    write_json_atomic(payload, str(target))

    assert target.read_text(encoding="utf-8") == json.dumps(payload, indent=2)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["artefact.json"]
