"""Write or check tests/INVENTORY.txt, the sorted list of collected pytest node IDs.

    python tools/qa-harness/test_inventory.py --write
    python tools/qa-harness/test_inventory.py --check

--check exits 1 and prints the difference when collection and the committed inventory disagree.
"""
import argparse
import subprocess
import sys
from pathlib import Path

INVENTORY = Path("tests/INVENTORY.txt")


def collected() -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-qq", "-p", "no:cacheprovider"],
        capture_output=True, text=True,
    )
    nodes = sorted(line.strip() for line in proc.stdout.splitlines() if "::" in line)
    if proc.returncode != 0 or not nodes:
        sys.exit(f"collection failed (exit {proc.returncode}): {proc.stdout[-300:]}{proc.stderr[-300:]}")
    return nodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    nodes = collected()
    if args.write:
        INVENTORY.write_text("\n".join(nodes) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {len(nodes)} node IDs to {INVENTORY}")
        return 0
    recorded = INVENTORY.read_text(encoding="utf-8").splitlines() if INVENTORY.exists() else []
    removed = sorted(set(recorded) - set(nodes))
    added = sorted(set(nodes) - set(recorded))
    for node in removed:
        print(f"- {node}")
    for node in added:
        print(f"+ {node}")
    print(f"{len(nodes)} collected, {len(recorded)} recorded, {len(removed)} removed, {len(added)} added")
    return 1 if removed or added else 0


if __name__ == "__main__":
    sys.exit(main())
