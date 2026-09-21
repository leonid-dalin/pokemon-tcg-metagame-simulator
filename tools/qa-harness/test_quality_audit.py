"""Flag tests that pass without exercising the system under test.

Three shapes, all of which read as coverage and provide none:

  NO-SUT      the test never calls anything imported from the source package
  SELF-ASSERT the test loads a file and asserts literals about what it just loaded
  ECHO        the test asserts that a value it constructed came back unchanged
  SOURCE-GREP the test asserts on the text of a source file rather than its behaviour

    python test_quality_audit.py --tests tests --package src
"""
import argparse, ast, os, sys

READERS = {"read_text", "read_bytes", "load", "loads", "open", "getsource"}


def py_files(root):
    for base, _, names in os.walk(root):
        for n in names:
            if n.endswith(".py"):
                yield os.path.join(base, n)


def imported_from(tree, package):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == package:
            names.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == package:
                    names.add(a.asname or a.name.split(".")[0])
    return names


def root_name(node):
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def calls(fn):
    out = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
                root = root_name(f)
                if root:
                    out.add(root)
    return out


def literal_locals(fn):
    out = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Dict, ast.List, ast.Constant, ast.Tuple)):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
    return out


def audit(path, package):
    tree = ast.parse(open(path, encoding="utf-8").read())
    sut = imported_from(tree, package)
    findings = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name.startswith("test_")]:
        used = calls(fn)
        asserts = [n for n in ast.walk(fn) if isinstance(n, ast.Assert)]
        if not asserts:
            continue
        touches_sut = bool(used & sut)
        reads_file = bool(used & READERS)
        if any(isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.endswith(".py")
               for n in ast.walk(fn)):
            findings.append((fn.lineno, fn.name, "SOURCE-GREP",
                             "asserts on the text of a source file, not on behaviour"))
            continue

        if reads_file and all(
                isinstance(a.test, ast.Compare) and any(isinstance(c, ast.Constant) for c in a.test.comparators)
                for a in asserts):
            findings.append((fn.lineno, fn.name, "SELF-ASSERT",
                             "loads a file then asserts literals about it; no producer is involved"))
            continue

        if not touches_sut and not fn.args.args:
            findings.append((fn.lineno, fn.name, "NO-SUT",
                             f"calls nothing imported from '{package}' and takes no fixture"))
            continue

        lits = literal_locals(fn)
        echoed = [a for a in asserts if isinstance(a.test, ast.Compare)
                  and any(isinstance(c, ast.Name) and c.id in lits for c in a.test.comparators)]
        if echoed and len(echoed) == len(asserts):
            findings.append((fn.lineno, fn.name, "ECHO",
                             "asserts a locally constructed value came back unchanged"))
    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", required=True)
    ap.add_argument("--package", default="src")
    args = ap.parse_args()

    blocking = 0
    warnings = 0
    for path in sorted(py_files(args.tests)):
        for lineno, name, kind, why in audit(path, args.package):
            if kind == "SOURCE-GREP":
                warnings += 1
                label = "WARNING"
            else:
                blocking += 1
                label = kind
            print(f"{label:12} | {os.path.relpath(path)}:{lineno}")
            print(f"{'':12} | {name}")
            print(f"{'':12} | {why}")
    print("\n" + "=" * 72)
    print(f"{blocking} blocking test-quality finding(s), {warnings} SOURCE-GREP warning(s)")
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
