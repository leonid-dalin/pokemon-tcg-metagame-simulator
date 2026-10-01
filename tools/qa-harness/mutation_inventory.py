"""Write or check tools/qa-harness/MUTATIONS.txt, the sorted list of mutation rows every sweep runs.

    python tools/qa-harness/mutation_inventory.py --write
    python tools/qa-harness/mutation_inventory.py --check

--check exits 1 and prints the difference when the mutation files and MUTATIONS.txt disagree,
so a row lost while editing a neighbouring row fails CI instead of shrinking the sweep.
"""
import argparse
import json
import sys
from pathlib import Path

HARNESS = Path("tools/qa-harness")
INVENTORY = HARNESS / "MUTATIONS.txt"


def rows() -> list[str]:
    found = []
    for path in sorted(HARNESS.glob("mutations.*.json")):
        names = [row["name"] for row in json.loads(path.read_text(encoding="utf-8"))]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            sys.exit(f"{path.name}: duplicate row names {duplicates}")
        found.extend(f"{path.name}::{name}" for name in names)
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = rows()
    if args.write:
        INVENTORY.write_text("\n".join(current) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {len(current)} mutation rows to {INVENTORY}")
        return 0
    recorded = INVENTORY.read_text(encoding="utf-8").splitlines() if INVENTORY.exists() else []
    removed = sorted(set(recorded) - set(current))
    added = sorted(set(current) - set(recorded))
    for row in removed:
        print(f"- {row}")
    for row in added:
        print(f"+ {row}")
    print(f"{len(current)} rows, {len(recorded)} recorded, {len(removed)} removed, {len(added)} added")
    return 1 if removed or added else 0


if __name__ == "__main__":
    sys.exit(main())
