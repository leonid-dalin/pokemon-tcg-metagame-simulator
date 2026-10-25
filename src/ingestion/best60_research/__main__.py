"""Command-line entry point for the offline Best-60 research pipeline."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from typing import Sequence

from .estimators import FitBudgetExceeded
from .protocol import ProtocolMismatch
from .runner import RunnerError, run_stage


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run a frozen, offline Best-60 research stage")
    result.add_argument("--database", required=True, help="read-only SQLite source database")
    result.add_argument("--output", required=True, help="ignored or external output directory")
    result.add_argument("--config", required=True, help="versioned JSON research protocol")
    result.add_argument("--stage", required=True,
                        choices=("audit", "validate", "lock", "evaluate", "simulate", "report"))
    result.add_argument("--as-of", help="optional ISO-8601 cutoff timestamp")
    result.add_argument("--field-file", help="optional JSON mapping of opponent archetype to field weight")
    result.add_argument("--archetype-id", help="focal selector benchmark ID")
    result.add_argument("--resume", help="resume an existing immutable run ID")
    result.add_argument("--max-seconds", type=int, help="stage wall-clock budget")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = run_stage(args)
    except RunnerError as exc:
        print(str(exc), file=sys.stderr)
        return exc.code
    except ProtocolMismatch as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except FitBudgetExceeded as exc:
        print(str(exc), file=sys.stderr)
        return 4
    except (sqlite3.Error, OSError) as exc:
        print(f"research stage failed: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") in {"complete", "inconclusive", "not_identified"} else 4


if __name__ == "__main__":
    raise SystemExit(main())
