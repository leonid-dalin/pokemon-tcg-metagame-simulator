import ast
from pathlib import Path

import pytest


@pytest.mark.unit
def test_worker_passes_the_configured_seed_to_monte_carlo():
    source = Path("src/worker/queue.py").read_text(encoding="utf-8")
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
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "RNG_SEED"
        for keyword in calls[0].keywords
    )