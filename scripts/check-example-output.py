#!/usr/bin/env python3
"""Compare an example's output with the "Expected output" block of its README.

    scripts/check-example-output.py examples/differentiable-physics/README.md run.log

The block is the first fenced code block after a heading that contains "Expected
output". Lines are compared after collapsing whitespace. Within a line, numbers must
agree to a relative 1e-3 (or 1e-6 absolute), so last-digit differences between
platforms pass, and everything else must be equal. Output before the block's first
line (Gradle's) is skipped; from there the run must print the block's lines in order.
The repository's path in the output is compared as `<repo>`, as the READMEs write it.

Exit status 0 when the output matches or the README has no expected output, 1 when it
does not match.
"""
import os
import re
import sys

NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def expected_block(readme):
    lines = open(readme, encoding="utf-8").read().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("#") and "expected output" in line.lower():
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            if j == len(lines):
                return None
            k = j + 1
            block = []
            while k < len(lines) and not lines[k].startswith("```"):
                block.append(lines[k])
                k += 1
            return block
    return None


def norm(line):
    return " ".join(line.split())


def same(a, b):
    na, nb = NUMBER.findall(a), NUMBER.findall(b)
    if NUMBER.sub("#", a) != NUMBER.sub("#", b) or len(na) != len(nb):
        return False
    for x, y in zip(na, nb):
        fx, fy = float(x), float(y)
        if abs(fx - fy) > max(1e-6, 1e-3 * max(abs(fx), abs(fy))):
            return False
    return True


def main():
    readme, log = sys.argv[1], sys.argv[2]
    block = expected_block(readme)
    if block is None:
        print(f"{readme}: no expected output")
        return 0
    want = [norm(l) for l in block if l.strip()]
    # The READMEs write the checkout's path as <repo>.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(log, encoding="utf-8", errors="replace").read().replace(root, "<repo>")
    got = [norm(l) for l in text.splitlines() if l.strip()]
    start = next((i for i, g in enumerate(got) if same(g, want[0])), None)
    if start is None:
        print(f"{readme}: the output never prints the first expected line:\n  {want[0]}")
        return 1
    for n, w in enumerate(want):
        i = start + n
        if i >= len(got):
            print(f"{readme}: the output ends before expected line {n + 1}:\n  {w}")
            return 1
        if not same(got[i], w):
            print(f"{readme}: line {n + 1} differs\n  expected: {w}\n  printed:  {got[i]}")
            return 1
    print(f"{readme}: {len(want)} lines match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
