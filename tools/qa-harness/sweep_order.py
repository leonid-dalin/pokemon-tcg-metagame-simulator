"""pytest plugin for mutation_sweep.py: SWEEP_FIRST tests first, cheapest first, durations to SWEEP_RECORD."""
import json
import os

_seconds = {}


def pytest_collection_modifyitems(session, config, items):
    first = set()
    if os.environ.get("SWEEP_FIRST"):
        with open(os.environ["SWEEP_FIRST"], encoding="utf-8") as fh:
            first = {line for line in fh.read().splitlines() if line}
    cost = {}
    if os.environ.get("SWEEP_DURATIONS"):
        with open(os.environ["SWEEP_DURATIONS"], encoding="utf-8") as fh:
            cost = json.load(fh)
    items.sort(key=lambda item: (item.nodeid not in first, cost.get(item.nodeid, 0.0)))


def pytest_runtest_logreport(report):
    _seconds[report.nodeid] = _seconds.get(report.nodeid, 0.0) + report.duration


def pytest_sessionfinish(session):
    if os.environ.get("SWEEP_RECORD"):
        with open(os.environ["SWEEP_RECORD"], "w", encoding="utf-8") as fh:
            json.dump(_seconds, fh)
