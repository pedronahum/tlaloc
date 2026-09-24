#!/usr/bin/env python3
"""Fails when a KDoc block in a published module's main sources cites an internal
work item instead of describing the code.

    python3 scripts/check-kdoc-internal-refs.py            # every published module
    python3 scripts/check-kdoc-internal-refs.py FILE...    # the named .kt files instead

Scans the `/** ... */` blocks of `<module>/src/*Main/**/*.kt` and
`<module>/src/main/**/*.kt` for every top-level module except benchmarks, examples,
harness and third-party. Tests, plain `//` comments and `/* */` comments are not
scanned. Exits 1 and lists each hit as `path:line: term: text`.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIPPED_DIRS = {"benchmarks", "examples", "harness", "third-party", "build", "buildSrc", "docs"}

# Each pattern names a way internal bookkeeping leaks into published API docs:
# changelog section numbers, plan phases, and the planning documents.
PATTERNS = [
    ("section sign", re.compile("§")),
    ("plan phase", re.compile(r"\bPhase [A-H][0-9]")),
    ("plan document", re.compile(r"DIFFKTX_SPEC|SPEC_with_coarsing|_PLAN\.md|_AUDIT\.md")),
    ("planning jargon", re.compile(r"\bdual-track\b|\bnamed deferral\b|\breboot incident\b")),
]


def kdoc_blocks(src):
    """Yields (start_offset, text) for every KDoc block, skipping string literals,
    character literals and line comments so that `/**` inside them is ignored."""
    i, n = 0, len(src)
    while i < n:
        if src.startswith('"""', i):
            j = src.find('"""', i + 3)
            if j == -1:
                return
            while src.startswith('""""', j):
                j += 1
            i = j + 3
        elif src[i] == '"':
            j = i + 1
            while j < n and src[j] != '"' and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            i = j + 1
        elif src[i] == "'":
            m = re.match(r"'(\\u[0-9a-fA-F]{4}|\\.|[^'\\\n])'", src[i:i + 8])
            i += len(m.group(0)) if m else 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j == -1 else j
        elif src.startswith("/*", i):
            depth, j = 0, i
            while j < n:
                if src.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                    if depth == 0:
                        break
                else:
                    j += 1
            if src.startswith("/**", i) and not src.startswith("/**/", i):
                yield i, src[i:j]
            i = j
        else:
            i += 1


def default_files():
    files = []
    for module in sorted(os.listdir(ROOT)):
        src = os.path.join(ROOT, module, "src")
        if module in SKIPPED_DIRS or module.startswith(".") or not os.path.isdir(src):
            continue
        for source_set in sorted(os.listdir(src)):
            if not (source_set == "main" or source_set.endswith("Main")):
                continue
            for dirpath, _, names in os.walk(os.path.join(src, source_set)):
                files += [os.path.join(dirpath, f) for f in sorted(names) if f.endswith(".kt")]
    return files


def check(path):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    hits = []
    for start, block in kdoc_blocks(src):
        first_line = src.count("\n", 0, start) + 1
        for offset, line in enumerate(block.split("\n")):
            for term, pattern in PATTERNS:
                if pattern.search(line):
                    rel = os.path.relpath(path, ROOT)
                    hits.append(f"{rel}:{first_line + offset}: {term}: {line.strip()}")
    return hits


def main(argv):
    files = [os.path.abspath(f) for f in argv] if argv else default_files()
    if not files:
        print("check-kdoc-internal-refs: no Kotlin sources found", file=sys.stderr)
        return 1
    hits = [h for f in files for h in check(f)]
    for h in hits:
        print(h)
    if hits:
        print(
            f"check-kdoc-internal-refs: {len(hits)} KDoc line(s) cite internal work items. "
            "Describe what the code does instead.",
            file=sys.stderr,
        )
        return 1
    print(f"check-kdoc-internal-refs: {len(files)} files, no internal references in KDoc.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
