"""Run a smoke path under every combination of a module's boolean flags.

Feature flags ship defaulted off and are verified defaulted off. The code behind them is
reviewed by reading. This walks the combinations instead.

    python flag_matrix.py --spec flag_spec_bdif.py

The spec module defines
  FLAGS   list of dotted names, e.g. ["src.core.config.BDIF_USE_CARD_MODEL"]
  smoke() -> str or None   runs the path; returns a degeneracy reason, or None if healthy
"""
import argparse, importlib, importlib.util, itertools, sys


def load_spec(path):
    spec = importlib.util.spec_from_file_location("flag_spec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_flag(dotted, value):
    module_path, attr = dotted.rsplit(".", 1)
    module = importlib.import_module(module_path)
    previous = getattr(module, attr)
    setattr(module, attr, value)
    return module, attr, previous


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    spec = load_spec(args.spec)
    flags = spec.FLAGS

    failures = 0
    for combo in itertools.product([False, True], repeat=len(flags)):
        restores = [set_flag(name, value) for name, value in zip(flags, combo)]
        label = " ".join(f"{n.rsplit('.', 1)[1]}={'T' if v else 'F'}" for n, v in zip(flags, combo))
        try:
            reason = spec.smoke()
            if reason:
                print(f"*** DEGENERATE | {label}")
                print(f"               | {reason}")
                failures += 1
            else:
                print(f"OK             | {label}")
        except Exception as exc:
            print(f"*** RAISED     | {label}")
            print(f"               | {type(exc).__name__}: {exc}")
            failures += 1
        finally:
            for module, attr, previous in restores:
                setattr(module, attr, previous)

    print("\n" + "=" * 72)
    print(f"{failures} of {2 ** len(flags)} flag combination(s) broken")
    print("A flag verified only in its default state is an unreviewed branch with a switch on it.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
