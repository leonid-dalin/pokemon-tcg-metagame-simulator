"""Break each decision the code makes, one at a time, and report which ones no test notices.

A green suite proves nothing until a component has been seen to fail. Every SURVIVED
row is a component a refactor can silently break.

    python mutation_sweep.py --repo PATH --mutations mutations.json [--tests tests/] [--workers N]

mutations.json is a list of {"name", "file", "find", "replace"}. "find" must appear
exactly once in "file"; a missing anchor is reported, never skipped silently.

Rows run on HEAD in temporary worktrees, one per worker (usable CPUs, at most 16;
MUTATION_SWEEP_WORKERS or --workers overrides).
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from coverage import CoverageData

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_WORKERS = 16
ROW_TIMEOUT = 600


def usable_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def worker_count(requested):
    wanted = requested or int(os.environ.get("MUTATION_SWEEP_WORKERS") or 0) or usable_cpus()
    return max(1, min(wanted, MAX_WORKERS))


def shard_files(node_ids, shards):
    counts = {}
    for node in node_ids:
        path = node.split("::")[0]
        counts[path] = counts.get(path, 0) + 1
    groups = [[] for _ in range(shards)]
    sizes = [0] * shards
    for path, count in sorted(counts.items(), key=lambda item: -item[1]):
        smallest = sizes.index(min(sizes))
        groups[smallest].append(path)
        sizes[smallest] += count
    return [sorted(group) for group in groups if group]


def run_suite(cwd, tests, env, extra=(), timeout=None):
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *extra, *tests, "-q", "-p", "no:cacheprovider", "--tb=no", "--no-header"],
            cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return [f"timed out after {timeout}s"], "TIMEOUT"
    lines = proc.stdout.splitlines()
    failed = [l.split("::")[-1].split(" ")[0] for l in lines if l.startswith(("FAILED", "ERROR"))]
    summary = next((l for l in reversed(lines) if " passed" in l or " failed" in l or " error" in l), "NO SUMMARY")
    if summary == "NO SUMMARY":
        # pytest did not run at all (missing module, broken interpreter). Counting
        # this as a green run reports every mutation as SURVIVED.
        print(f"pytest produced no summary (exit {proc.returncode}). stderr tail: {proc.stderr[-200:]!r}")
        return ["SUITE-DID-NOT-RUN"], "NO SUMMARY"
    return failed, summary.strip("= ")


def restore_worker(tree, idle):
    result = subprocess.run(["git", "checkout", "--", "."], cwd=tree, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Failed to restore mutation worker {tree}: {result.stderr.strip()}")
    idle.put(tree)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--mutations", required=True)
    ap.add_argument("--tests", default="tests/")
    ap.add_argument("--pythonpath", default="")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=ROW_TIMEOUT)
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    mutations = json.load(open(args.mutations, encoding="utf-8"))
    dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout
    if dirty:
        print("Tracked files have uncommitted changes. The sweep runs HEAD only; commit or stash them first.")
        print(dirty)
        return 2

    def env_for(tree, **extra):
        # Prepend, do not overwrite: the caller's PYTHONPATH usually carries the
        # site-packages that make pytest and the project deps importable.
        paths = [HERE, tree]
        for entry in (args.pythonpath, os.environ.get("PYTHONPATH", "")):
            for path in filter(None, entry.split(os.pathsep)):
                same = os.path.normcase(os.path.abspath(path)) == os.path.normcase(repo)
                paths.append(tree if same else path)
        return dict(os.environ, PYTHONPATH=os.pathsep.join(paths), **extra)

    workers = worker_count(args.workers)
    scratch = tempfile.mkdtemp(prefix="sweep-")
    trees = [os.path.join(scratch, f"w{i}") for i in range(workers)]
    try:
        def add(tree):
            subprocess.run(["git", "-C", repo, "worktree", "add", "--quiet", "--detach", tree, "HEAD"],
                           capture_output=True, text=True, check=True)

        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(add, trees))
        print(f"WORKERS: {workers}")

        collected = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-qq", "-p", "no:cacheprovider", args.tests],
            cwd=trees[0], env=env_for(trees[0]), capture_output=True, text=True).stdout
        shards = shard_files([l.strip() for l in collected.splitlines() if "::" in l], workers) or [[args.tests]]
        packages = sorted({m["file"].split("/")[0] for m in mutations})

        def baseline(shard):
            index, files = shard
            tree = trees[index]
            failed, summary = run_suite(
                tree, files,
                env_for(tree, COVERAGE_FILE=os.path.join(scratch, f"coverage-{index}"),
                        SWEEP_RECORD=os.path.join(scratch, f"durations-{index}.json")),
                ["-p", "sweep_order", *(f"--cov={p}" for p in packages), "--cov-context=test", "--cov-report="])
            return failed, summary

        start = time.time()
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(baseline, enumerate(shards)))
        totals = {}
        for _, summary in results:
            for count, outcome in re.findall(r"(\d+) ([a-z]+)", summary):
                totals[outcome] = totals.get(outcome, 0) + int(count)
        counts = ", ".join(f"{n} {outcome}" for outcome, n in totals.items())
        print(f"BASELINE: {counts} in {time.time() - start:.1f}s across {len(shards)} shards")
        base_failed = [f for failed, _ in results for f in failed]
        if base_failed:
            for f in base_failed[:8]:
                print(f"{'':16} |   {f}")
            print("Baseline is not green. Fix that before trusting any row below.")
            return 2
        print()

        covered, seconds = {}, {}
        for index in range(len(shards)):
            data = CoverageData(os.path.join(scratch, f"coverage-{index}"))
            data.read()
            for measured in data.measured_files():
                rel = os.path.relpath(measured, trees[index]).replace(os.sep, "/")
                for line, contexts in data.contexts_by_lineno(measured).items():
                    covered.setdefault((rel, line), set()).update(c.rsplit("|", 1)[0] for c in contexts if c)
            with open(os.path.join(scratch, f"durations-{index}.json"), encoding="utf-8") as fh:
                seconds.update(json.load(fh))
        durations = os.path.join(scratch, "durations.json")
        with open(durations, "w", encoding="utf-8") as fh:
            json.dump(seconds, fh)

        plans = []
        for index, m in enumerate(mutations):
            src = open(os.path.join(trees[0], *m["file"].split("/")), encoding="utf-8").read()
            count = src.count(m["find"])
            first = set()
            if count == 1:
                start_line = src[: src.index(m["find"])].count("\n") + 1
                for line in range(start_line, start_line + m["find"].count("\n") + 1):
                    first |= covered.get((m["file"], line), set())
            plans.append((index, m, count, sorted(first)))

        idle = Queue()
        for tree in trees:
            idle.put(tree)

        def run_row(plan):
            index, m, _, first = plan
            tree = idle.get()
            try:
                path = os.path.join(tree, *m["file"].split("/"))
                src = open(path, encoding="utf-8").read()
                open(path, "w", encoding="utf-8").write(src.replace(m["find"], m["replace"], 1))
                listing = os.path.join(scratch, f"first-{index}.txt")
                with open(listing, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(first))
                failed, _ = run_suite(tree, [args.tests], env_for(tree, SWEEP_FIRST=listing, SWEEP_DURATIONS=durations),
                                      ["-x", "-p", "sweep_order"], args.timeout)
                return failed
            finally:
                restore_worker(tree, idle)

        runnable = sorted((p for p in plans if p[2] == 1), key=lambda p: bool(p[3]))
        with ThreadPoolExecutor(workers) as pool:
            futures = {p[0]: pool.submit(run_row, p) for p in runnable}
            survivors, anchors = [], []
            for index, m, count, _ in plans:
                if count != 1:
                    anchors.append((m["name"], count))
                    print(f"{'ANCHOR x' + str(count):16} | {m['name']}")
                    continue
                failed = futures[index].result()
                if failed:
                    print(f"{'CAUGHT':16} | {m['name']}")
                    for f in failed[:4]:
                        print(f"{'':16} |   {f}")
                else:
                    survivors.append(m["name"])
                    print(f"{'*** SURVIVED ***':16} | {m['name']}")
    finally:
        for tree in trees:
            subprocess.run(["git", "-C", repo, "worktree", "remove", "--force", tree], capture_output=True)
        subprocess.run(["git", "-C", repo, "worktree", "prune"], capture_output=True)
        shutil.rmtree(scratch, ignore_errors=True)

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
