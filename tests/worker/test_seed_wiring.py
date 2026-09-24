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
    seed_keyword = next(keyword for keyword in calls[0].keywords if keyword.arg == "seed")
    assert "RNG_SEED" in ast.unparse(seed_keyword.value)
    assert "config" in ast.unparse(seed_keyword.value)