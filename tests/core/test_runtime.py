import pytest

from src.core import runtime


@pytest.mark.unit
def test_v2_reads_quota_over_period(tmp_path):
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("200000 100000")
    assert runtime._read_cgroup_v2_cores(cpu_max) == 2


@pytest.mark.unit
def test_v2_floors_a_fractional_quota(tmp_path):
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("150000 100000")
    assert runtime._read_cgroup_v2_cores(cpu_max) == 1


@pytest.mark.unit
def test_v2_returns_none_when_unlimited(tmp_path):
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("max 100000")
    assert runtime._read_cgroup_v2_cores(cpu_max) is None


@pytest.mark.unit
def test_v2_returns_none_for_malformed_content(tmp_path):
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("garbage")
    assert runtime._read_cgroup_v2_cores(cpu_max) is None


@pytest.mark.unit
def test_v2_returns_none_when_file_is_absent(tmp_path):
    assert runtime._read_cgroup_v2_cores(tmp_path / "missing") is None


@pytest.mark.unit
def test_v1_reads_quota_over_period(tmp_path):
    quota = tmp_path / "quota"
    period = tmp_path / "period"
    quota.write_text("400000\n")
    period.write_text("100000\n")
    assert runtime._read_cgroup_v1_cores(quota, period) == 4


@pytest.mark.unit
def test_v1_returns_none_when_quota_is_unlimited(tmp_path):
    quota = tmp_path / "quota"
    period = tmp_path / "period"
    quota.write_text("-1\n")
    period.write_text("100000\n")
    assert runtime._read_cgroup_v1_cores(quota, period) is None


@pytest.mark.unit
def test_max_cores_env_var_wins(monkeypatch):
    monkeypatch.setenv("MAX_CORES", "3")
    assert runtime.get_container_cores() == 3


@pytest.mark.unit
def test_max_cores_env_var_is_floored_at_one(monkeypatch):
    monkeypatch.setenv("MAX_CORES", "0")
    assert runtime.get_container_cores() == 1


@pytest.mark.unit
def test_malformed_max_cores_falls_through_to_probes(monkeypatch):
    monkeypatch.setenv("MAX_CORES", "not-a-number")
    monkeypatch.setattr(runtime, "_read_cgroup_v2_cores", lambda *a, **k: 7)
    assert runtime.get_container_cores() == 7


@pytest.mark.unit
def test_falls_back_to_half_the_host_when_no_quota_is_set(monkeypatch):
    monkeypatch.delenv("MAX_CORES", raising=False)
    monkeypatch.setattr(runtime, "_read_cgroup_v2_cores", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "_read_cgroup_v1_cores", lambda *a, **k: None)
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: 16)
    assert runtime.get_container_cores() == 8


@pytest.mark.unit
def test_fallback_never_returns_zero(monkeypatch):
    monkeypatch.delenv("MAX_CORES", raising=False)
    monkeypatch.setattr(runtime, "_read_cgroup_v2_cores", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "_read_cgroup_v1_cores", lambda *a, **k: None)
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: 1)
    assert runtime.get_container_cores() == 1
