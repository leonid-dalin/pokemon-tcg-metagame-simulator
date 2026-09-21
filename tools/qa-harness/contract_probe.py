"""Execute every artifact handoff end to end and fail on a degenerate result, not just on an exception.

"Round-trip verified" usually means the producer ran. The defect lives one step later,
where the consumer accepts the artifact and quietly returns nothing.

    python contract_probe.py --spec contract_spec_bdif.py

The spec module defines CASES: a list of dicts with
  name      str
  produce   () -> artifact
  consume   (artifact) -> loaded
  degenerate(loaded) -> str or None   returns why the result is empty/default, or None
"""
import argparse, importlib.util, sys


def load_spec(path):
    spec = importlib.util.spec_from_file_location("contract_spec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CASES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()

    failures = 0
    for case in load_spec(args.spec):
        name = case["name"]
        try:
            artifact = case["produce"]()
        except Exception as exc:
            print(f"PRODUCE FAILED | {name}\n               | {type(exc).__name__}: {exc}")
            failures += 1
            continue
        try:
            loaded = case["consume"](artifact)
        except Exception as exc:
            print(f"CONSUME FAILED | {name}\n               | {type(exc).__name__}: {exc}")
            failures += 1
            continue
        why = case["degenerate"](loaded)
        if why:
            print(f"*** DEGENERATE | {name}")
            print(f"               | producer succeeded, consumer returned nothing usable")
            print(f"               | {why}")
            failures += 1
        else:
            print(f"OK             | {name}")

    print("\n" + "=" * 72)
    print(f"{failures} broken handoff(s)")
    print("A handoff that raises is a bug you will find. A handoff that degenerates is one you will ship.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
