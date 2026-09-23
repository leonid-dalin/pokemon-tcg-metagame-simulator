import ast
from pathlib import Path

import pytest


@pytest.mark.unit
def test_worker_passes_the_configured_seed_to_monte_carlo():
    source = Path("src/bdif/service.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_monte_carlo_analytics"
    ]

    assert len(calls) == 1
    assert any(
        keyword.arg == "seed"
        and isinstance(keyword.value, ast.Attribute)
        and keyword.value.attr == "RNG_SEED"
        and isinstance(keyword.value.value, ast.Name)
        and keyword.value.value.id == "config"
        for keyword in calls[0].keywords
    )