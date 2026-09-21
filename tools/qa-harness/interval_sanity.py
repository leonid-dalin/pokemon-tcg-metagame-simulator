"""Check an analytic confidence interval against a bootstrap of the same estimator.

An interval is the one output nobody eyeballs. It is reported in the same units as the
estimate, so a formula error produces a plausible-looking number rather than a crash.
Resampling needs no derivation and no algebra: if the analytic half-width disagrees with
the bootstrap spread by more than the tolerance, the formula is wrong.

    python interval_sanity.py --spec interval_spec_h1.py [--draws 400] [--tolerance 2.0]

The spec module defines CASES: dicts with
  name     str
  data     sequence
  fit      (data) -> (estimate, (lower, upper))
  seed     int, optional
"""
import argparse, importlib.util, statistics, sys, random


def load_spec(path):
    spec = importlib.util.spec_from_file_location("interval_spec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CASES


def bootstrap_sd(data, fit, draws, seed):
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sample = [data[rng.randrange(len(data))] for _ in range(len(data))]
        try:
            estimate, _ = fit(sample)
        except Exception:
            continue
        estimates.append(estimate)
    if len(estimates) < 20:
        return None, len(estimates)
    return statistics.pstdev(estimates), len(estimates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--tolerance", type=float, default=2.0)
    args = ap.parse_args()

    failures = 0
    for case in load_spec(args.spec):
        name, data, fit = case["name"], case["data"], case["fit"]
        estimate, (lower, upper) = fit(data)
        analytic_se = (upper - lower) / (2 * 1.96)
        boot_se, n = bootstrap_sd(data, fit, args.draws, case.get("seed", 1312))
        print(f"--- {name}")
        print(f"    estimate            {estimate:+.4f}")
        print(f"    reported 95% CI     ({lower:+.4f}, {upper:+.4f})")
        print(f"    implied analytic SE {analytic_se:.4f}")
        if boot_se is None:
            print(f"    bootstrap           inconclusive, only {n} of {args.draws} draws fitted")
            continue
        ratio = analytic_se / boot_se if boot_se else float("inf")
        print(f"    bootstrap SE        {boot_se:.4f}  ({n} draws)")
        print(f"    ratio               {ratio:.2f}x")
        if ratio > args.tolerance or ratio < 1 / args.tolerance:
            lo, hi = estimate - 1.96 * boot_se, estimate + 1.96 * boot_se
            print(f"    *** DISAGREES      bootstrap implies ({lo:+.4f}, {hi:+.4f})")
            print(f"                       the analytic interval is {ratio:.1f}x too "
                  f"{'wide' if ratio > 1 else 'narrow'}")
            failures += 1
        else:
            print(f"    OK                 within {args.tolerance}x")
        print()

    print("=" * 72)
    print(f"{failures} interval(s) whose analytic formula disagrees with resampling")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
