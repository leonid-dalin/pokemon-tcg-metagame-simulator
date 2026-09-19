import ast
from pathlib import Path

import pytest


@pytest.mark.unit
def test_prediction_post_has_a_timeout():
    source = Path("src/ui/app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    post_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "requests"
        and node.func.attr == "post"
    ]

    assert len(post_calls) == 1
    timeout_keywords = [keyword for keyword in post_calls[0].keywords if keyword.arg == "timeout"]
    assert len(timeout_keywords) == 1
    assert isinstance(timeout_keywords[0].value, ast.Constant)
    assert timeout_keywords[0].value.value > 0