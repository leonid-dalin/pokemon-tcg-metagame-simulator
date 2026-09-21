"""Find symbols that nothing references, and symbols that only the test suite references.

Code reachable only from tests is not shipped behaviour, it is scaffolding that passed
review because the suite was green.

A reference is any Name, Attribute or ImportFrom occurrence anywhere under --src,
including inside the defining module. Decorated definitions are treated as entry points
(routes, tasks, fixtures) and reported separately rather than as dead; inert
decorators such as @dataclass and @property do not count as registration.

    python dead_symbol_scan.py --src src --tests tests [--show-entrypoints]
"""
import argparse, ast, os, sys
from collections import Counter


def py_files(root):
    for base, _, names in os.walk(root):
        for n in names:
            if n.endswith(".py"):
                yield os.path.join(base, n)


def parse(path):
    try:
        return ast.parse(open(path, encoding="utf-8").read())
    except (SyntaxError, UnicodeDecodeError):
        return None


INERT_DECORATORS = {"dataclass", "property", "staticmethod", "classmethod",
                    "cached_property", "total_ordering", "frozen"}


def registers(node):
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name not in INERT_DECORATORS:
            return True
    return False


def definitions(root):
    out = {}
    for path in py_files(root):
        tree = parse(path)
        if tree is None:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    out[node.name] = (path, registers(node))
    return out


def reference_counts(root):
    counts = Counter()
    for path in py_files(root):
        tree = parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                counts[node.id] += 1
            elif isinstance(node, ast.Attribute):
                counts[node.attr] += 1
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    counts[a.name] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--show-entrypoints", action="store_true")
    args = ap.parse_args()

    defs = definitions(args.src)
    src_refs = reference_counts(args.src)
    test_refs = reference_counts(args.tests)

    dead, test_only, entrypoints = [], [], []
    for name, (home, decorated) in sorted(defs.items()):
        if src_refs[name] > 0:
            continue
        if decorated:
            entrypoints.append((name, home))
        elif test_refs[name] > 0:
            test_only.append((name, home))
        else:
            dead.append((name, home))

    for name, home in dead:
        print(f"DEAD      | {name:34} | {home}")
    for name, home in test_only:
        print(f"TEST-ONLY | {name:34} | {home}")
    if args.show_entrypoints:
        for name, home in entrypoints:
            print(f"ENTRY     | {name:34} | {home}")

    print("\n" + "=" * 72)
    print(f"{len(dead)} dead, {len(test_only)} test-only, {len(entrypoints)} decorated entry points (not counted)")
    print("Each dead or test-only symbol is code a reviewer read, believed, and did not need.")
    return 1 if dead or test_only else 0


if __name__ == "__main__":
    sys.exit(main())
