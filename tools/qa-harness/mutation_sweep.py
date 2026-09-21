"""Break each decision the code makes, one at a time, and report which ones no test notices.

A green suite proves nothing until a component has been seen to fail. Every SURVIVED
row is a component a refactor can silently break.

    python mutation_sweep.py --repo PATH --mutations mutations.json [--tests tests/]

mutations.json is a list of {"name", "file", "find", "replace"}. "find" must appear
exactly once in "file"; a missing anchor is reported, never skipped silently.
"""
import argparse, json, os, subprocess, sys


def run_suite(repo, tests, env):
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", tests, "-q", "-p", "no:cacheprovider", "--tb=no", "--no-header"],
        cwd=repo, env=env, capture_output=True, text=True)
    lines = proc.stdout.splitlines()
    failed = [l.split("::")[-1].split(" ")[0] for l in lines if l.startswith(("FAILED", "ERROR"))]
    summary = next((l for l in reversed(lines) if " passed" in l or " failed" in l or " error" in l), "NO SUMMARY")
    if summary == "NO SUMMARY":
        # pytest did not run at all (missing module, broken interpreter). Counting
        # this as a green run reports every mutation as SURVIVED.
        print(f"pytest produced no summary (exit {proc.returncode}). stderr tail: {proc.stderr[-200:]!r}")
        return ["SUITE-DID-NOT-RUN"], "NO SUMMARY"
    return failed, summary.strip("= ")


def restore(repo):
    subprocess.run(["git", "checkout", "--", "."], cwd=repo, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--mutations", required=True)
    ap.add_argument("--tests", default="tests/")
    ap.add_argument("--pythonpath", default="")
    args = ap.parse_args()

    env = dict(os.environ)
    # Prepend, do not overwrite: the caller's PYTHONPATH usually carries the
    # site-packages that make pytest and the project deps importable.
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (args.pythonpath, os.environ.get("PYTHONPATH", ""), args.repo) if p)
    mutations = json.load(open(args.mutations, encoding="utf-8"))

    restore(args.repo)
    base_failed, base_summary = run_suite(args.repo, args.tests, env)
    print(f"BASELINE: {base_summary}")
    if base_failed:
        print("Baseline is not green. Fix that before trusting any row below.")
        return 2
    print()

    survivors, anchors = [], []
    for m in mutations:
        restore(args.repo)
        path = os.path.join(args.repo, m["file"].replace("/", os.sep))
        src = open(path, encoding="utf-8").read()
        if src.count(m["find"]) != 1:
            anchors.append((m["name"], src.count(m["find"])))
            print(f"{'ANCHOR x' + str(src.count(m['find'])):16} | {m['name']}")
            continue
        open(path, "w", encoding="utf-8").write(src.replace(m["find"], m["replace"], 1))
        failed, summary = run_suite(args.repo, args.tests, env)
        if failed:
            print(f"{'CAUGHT':16} | {m['name']}")
            for f in failed[:4]:
                print(f"{'':16} |   {f}")
        else:
            survivors.append(m["name"])
            print(f"{'*** SURVIVED ***':16} | {m['name']}")
    restore(args.repo)

    print("\n" + "=" * 72)
    print(f"{len(mutations) - len(survivors) - len(anchors)} caught, {len(survivors)} survived, {len(anchors)} bad anchors")
    for name in survivors:
        print(f"  SURVIVED  {name}")
    for name, n in anchors:
        print(f"  ANCHOR    {name} (matched {n} times, expected 1)")
    print("\nRead the survivors. A survivor is not a missing test, it is an unprotected decision.")
    return 1 if survivors or anchors else 0


if __name__ == "__main__":
    sys.exit(main())
