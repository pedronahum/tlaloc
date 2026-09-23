#!/usr/bin/env python3
"""Checks the user-facing Markdown: every relative link and reference-style link
points at a file that exists, every #anchor into a Markdown file matches one of its
headings (GitHub's slug rules), and every code fence is closed.

    python3 scripts/check-doc-links.py            # the user-facing docs
    python3 scripts/check-doc-links.py FILE...    # the named files instead

Exits 1 and lists each problem if there is one. Links inside code spans and fenced
code blocks are ignored; http(s) and mailto links are not fetched.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

USER_FACING = re.compile(
    r"^(README|CHANGELOG|CONTRIBUTING|SECURITY)\.md$"
    r"|^docs/(GETTING_STARTED|COMPATIBILITY|CAPABILITIES|READABLE_REVERSE|RELEASING"
    r"|SERVING_RUNBOOK|vendoring)\.md$"
    r"|^docs/papers/README\.md$"
    r"|^examples/.*README\.md$"
    r"|^\.github/.*\.md$"
)


def default_files():
    tracked = subprocess.check_output(["git", "ls-files", "*.md"], cwd=ROOT, text=True).split()
    return [f for f in tracked if USER_FACING.match(f)]


def slug(heading):
    h = heading.replace("`", "").strip().lower()
    h = re.sub(r"<[^>]+>", "", h)
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


def anchors(path):
    out, counts, in_fence = set(), {}, False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if re.match(r"^\s*(```|~~~)", line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
            if m:
                s = slug(m.group(2))
                n = counts.get(s, 0)
                counts[s] = n + 1
                out.add(s if n == 0 else f"{s}-{n}")
    return out


def check(rel):
    problems = []
    path = os.path.join(ROOT, rel)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    fences = [l for l in text.split("\n") if re.match(r"^\s*```", l)]
    if len(fences) % 2:
        problems.append(f"{rel}: unclosed code fence ({len(fences)} fence lines)")
    body = re.sub(r"```.*?```", "", text, flags=re.S)
    body = re.sub(r"`[^`\n]*`", "", body)
    links = [m.group(1) for m in re.finditer(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", body)]
    links += [m.group(1) for m in re.finditer(r"^\[[^\]]+\]:\s+(\S+)", body, flags=re.M)]
    for link in links:
        if re.match(r"^(https?|mailto):", link):
            continue
        target_part, _, anchor = link.partition("#")
        target = os.path.normpath(os.path.join(os.path.dirname(path), target_part)) if target_part else path
        if not os.path.exists(target):
            problems.append(f"{rel}: missing target -> {link}")
        elif anchor and target.endswith(".md") and anchor not in anchors(target):
            problems.append(f"{rel}: no heading for anchor -> {link}")
    return problems


def main(argv):
    files = argv or default_files()
    problems = [p for f in files for p in check(f)]
    for p in problems:
        print(p)
    print(f"check-doc-links: {len(files)} files, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
