import os
from pathlib import Path

CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def _cores_from_quota(quota: int, period: int) -> int | None:
    if quota <= 0 or period <= 0:
        return None
    return max(1, quota // period)


def _read_cgroup_v2_cores(cpu_max: Path = CGROUP_V2_CPU_MAX) -> int | None:
    try:
        quota_str, period_str = cpu_max.read_text().split()
        if quota_str == "max":
            return None
        return _cores_from_quota(int(quota_str), int(period_str))
    except (OSError, ValueError):
        return None


def _read_cgroup_v1_cores(
    quota_file: Path = CGROUP_V1_QUOTA,
    period_file: Path = CGROUP_V1_PERIOD,
) -> int | None:
    try:
        return _cores_from_quota(
            int(quota_file.read_text().strip()),
            int(period_file.read_text().strip()),
        )
    except (OSError, ValueError):
        return None


def get_container_cores() -> int:
    raw = os.environ.get("MAX_CORES")
    if raw is not None:
        try:
            return max(1, int(raw))
        except ValueError:
            pass

    for probe in (_read_cgroup_v2_cores, _read_cgroup_v1_cores):
        cores = probe()
        if cores is not None:
            return cores

    return max(1, (os.cpu_count() or 2) // 2)
