import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[2] / "tools" / "qa-harness"
TARGET = "pkg/calc.py"
MODULES = ("tests/a.py", "tests/b.py", "tests/c.py", "tests/d.py")
CALC = "def double(x):\n    return x * 2\n\n\ndef triple(x):\n    return x * 3\n"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load("mutation_sweep")


def _git(repo, *args):
    return subprocess.run(
        ["git", "-c", "user.name=sweep", "-c", "user.email=sweep@example.invalid", *args],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pytest.ini").write_text("[pytest]\npythonpath = .\n")
    (root / TARGET).write_text(CALC)
    (root / "tests" / "test_calc.py").write_text(
        "from pkg.calc import double\n\n\ndef test_double():\n    assert double(2) == 4\n"
    )
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "init")
    return root


def _row(name, find, replace):
    return {"name": name, "file": TARGET, "find": find, "replace": replace}


def _sweep(repo, rows, *extra):
    manifest = repo.parent / "mutations.json"
    manifest.write_text(json.dumps(rows))
    return subprocess.run(
        [sys.executable, str(HARNESS / "mutation_sweep.py"), "--repo", str(repo), "--mutations", str(manifest),
         "--tests", "tests", "--workers", "2", *extra],
        capture_output=True, text=True, timeout=300,
    )


@pytest.mark.parametrize(("cpus", "expected"), [(32, 16), (16, 16), (4, 4), (1, 1)])
def test_workers_default_to_usable_cpus_capped_at_sixteen(monkeypatch, cpus, expected):
    monkeypatch.delenv("MUTATION_SWEEP_WORKERS", raising=False)
    monkeypatch.setattr(sweep, "usable_cpus", lambda: cpus)
    assert sweep.worker_count(0) == expected


def test_requested_workers_and_the_environment_override_stay_under_the_cap(monkeypatch):
    monkeypatch.setattr(sweep, "usable_cpus", lambda: 32)
    monkeypatch.setenv("MUTATION_SWEEP_WORKERS", "6")
    assert sweep.worker_count(0) == 6
    assert sweep.worker_count(2) == 2
    assert sweep.worker_count(64) == 16
    monkeypatch.setenv("MUTATION_SWEEP_WORKERS", "64")
    assert sweep.worker_count(0) == 16


@pytest.mark.parametrize(("shards", "expected"), [(2, [[0, 3], [1, 2]]), (8, [[0], [1], [2], [3]])])
def test_baseline_shards_split_whole_files_by_test_count(shards, expected):
    a, b, c, d = MODULES
    nodes = [f"{a}::t1", f"{a}::t2", f"{a}::t3", f"{b}::t1", f"{b}::t2", f"{c}::t1", f"{d}::t1"]

    assert sweep.shard_files(nodes, shards) == [[MODULES[i] for i in group] for group in expected]


def test_sweep_reports_each_row_and_leaves_the_checkout_untouched(repo):
    result = _sweep(repo, [
        _row("double triples", "    return x * 2\n", "    return x * 3\n"),
        _row("triple doubles", "    return x * 3\n", "    return x * 2\n"),
        _row("missing anchor", "return x * 7", "return 0"),
    ])

    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 caught, 1 survived, 1 bad anchors" in result.stdout
    assert "SURVIVED  triple doubles" in result.stdout
    assert "ANCHOR    missing anchor (matched 0 times, expected 1)" in result.stdout
    assert (repo / TARGET).read_text() == CALC
    assert _git(repo, "status", "--porcelain") == ""
    assert _git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_sweep_refuses_uncommitted_changes_and_keeps_them(repo):
    edited = CALC.replace("x * 3", "x + x + x")
    (repo / TARGET).write_text(edited)

    result = _sweep(repo, [_row("double triples", "    return x * 2\n", "    return x * 3\n")])

    assert result.returncode == 2
    assert "uncommitted changes" in result.stdout
    assert (repo / TARGET).read_text() == edited


def test_a_row_that_hangs_is_caught_at_the_row_timeout(repo):
    result = _sweep(repo, [_row("double hangs", "    return x * 2\n", "    while True:\n        pass\n")], "--timeout", "2")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 caught, 0 survived, 0 bad anchors" in result.stdout
    assert "timed out after 2s" in result.stdout


def _order(tmp_path, **environment):
    (tmp_path / "test_order.py").write_text("def test_a():\n    pass\n\n\ndef test_b():\n    pass\n\n\ndef test_c():\n    pass\n")
    env = dict(os.environ, PYTHONPATH=str(HARNESS), **environment)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "sweep_order", "-p", "no:cacheprovider", "-o", "addopts=", "-v",
         "test_order.py"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    return [line.split("::")[1].split(" ")[0] for line in result.stdout.splitlines() if "::test_" in line and "PASSED" in line]


def test_sweep_order_runs_listed_tests_first_then_cheapest_first(tmp_path):
    (tmp_path / "first.txt").write_text("test_order.py::test_c\n")
    (tmp_path / "seconds.json").write_text(json.dumps({"test_order.py::test_a": 5.0, "test_order.py::test_b": 1.0}))

    assert _order(tmp_path) == ["test_a", "test_b", "test_c"]
    assert _order(tmp_path, SWEEP_FIRST=str(tmp_path / "first.txt"),
                  SWEEP_DURATIONS=str(tmp_path / "seconds.json")) == ["test_c", "test_b", "test_a"]


def test_sweep_order_records_each_test_s_seconds(tmp_path):
    record = tmp_path / "record.json"

    _order(tmp_path, SWEEP_RECORD=str(record))

    assert sorted(json.loads(record.read_text())) == [
        "test_order.py::test_a", "test_order.py::test_b", "test_order.py::test_c",
    ]


def test_restore_failure_does_not_reuse_worker(monkeypatch):
    def fail_restore(command, **kwargs):
        return subprocess.CompletedProcess(command, 255, "", "error: unable to unlink old 'tracked.py': Invalid argument")

    monkeypatch.setattr(sweep.subprocess, "run", fail_restore)
    idle = sweep.Queue()

    with pytest.raises(RuntimeError, match="unable to unlink old"):
        sweep.restore_worker("worker", idle)

    assert idle.empty()
