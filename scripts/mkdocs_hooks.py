"""MkDocs hooks for the Tlaloc documentation site (mkdocs.yml).

The docs are written for GitHub, and these hooks adapt them for the site:

- Pages from outside docs/ join the site: examples/README.md as examples/index.md,
  each examples/<name>/README.md as examples/<name>.md, CHANGELOG.md as changelog.md.
- A relative link is resolved against the file's place in the repository. A link to a
  page of the site points at that page; a link to anything else (source files,
  scripts, the docs the site leaves out) points at the file on GitHub.
- A GitHub callout (`> [!NOTE]`, `> [!TIP]`, `> [!IMPORTANT]`, `> [!WARNING]`,
  `> [!CAUTION]`) becomes an admonition.
- The home page hides the side navigation; its cards are the navigation.
- Lists nested the way GitHub reads them (a child under the text of its parent item,
  often 2 or 3 spaces in) are re-indented to the 4 spaces Python-Markdown needs, and a
  list that starts right under a paragraph gets the blank line Python-Markdown needs.
- Page URLs are lower case, with hyphens: docs/HOW_IT_WORKS.md is how-it-works/.
- `<!--mkdocs: TEXT -->` becomes TEXT: markup that only the site shows, such as the
  card grid on the home page. GitHub does not render HTML comments.

A page of the site that the nav does not list is a warning, so `mkdocs build --strict`
fails on it.
"""

import logging
import os
import posixpath
import re

from mkdocs.structure.files import File

log = logging.getLogger("mkdocs.plugins.tlaloc")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "https://github.com/pedronahum/tlaloc"
BRANCH = "main"

FENCE = re.compile(r"^\s*(```+|~~~+)")
SITE_ONLY = re.compile(r"<!--mkdocs:(.*?)-->", re.DOTALL)
CALLOUT = re.compile(r"^(\s*)>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*$")
CALLOUT_TYPES = {"NOTE": "note", "TIP": "tip", "IMPORTANT": "info", "WARNING": "warning", "CAUTION": "danger"}
# [text](target) and ![alt](target), the target without spaces, with an optional title.
INLINE_LINK = re.compile(r"(!?\[(?:[^\[\]]|\[[^\]]*\])*\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")
REFERENCE_LINK = re.compile(r"^(\s*\[[^\]]+\]:\s*)(\S+)(.*)$", re.MULTILINE)
HTML_SRC = re.compile(r"(<(?:img|a)\b[^>]*?\b(?:src|href)=\")([^\"]+)(\")")
LIST_ITEM = re.compile(r"^( *)([-*+]|\d+[.)])( +)(\S.*)$")
CODE_SPAN = re.compile(r"(`+)(?:(?!\1).)+?\1")

# site path -> repository path, for the pages generated from files outside docs/
GENERATED = {}
# repository path -> site path, for every page of the site
SITE_OF = {}


def _generated_pages():
    pages = {"changelog.md": "CHANGELOG.md", "examples/index.md": "examples/README.md"}
    examples = os.path.join(ROOT, "examples")
    for name in sorted(os.listdir(examples)):
        if os.path.isfile(os.path.join(examples, name, "README.md")):
            pages[f"examples/{name}.md"] = f"examples/{name}/README.md"
    return pages


def on_files(files, config):
    GENERATED.clear()
    GENERATED.update(_generated_pages())
    for src_uri, repo_path in GENERATED.items():
        with open(os.path.join(ROOT, repo_path), encoding="utf-8") as f:
            files.append(File.generated(config, src_uri, content=f.read()))
    for f in files.documentation_pages():
        stem = f.src_uri[: -len(".md")]
        if posixpath.basename(stem) not in ("README", "index"):
            f.dest_uri = stem.lower().replace("_", "-") + "/index.html"
    docs = os.path.relpath(config["docs_dir"], ROOT).replace(os.sep, "/")
    SITE_OF.clear()
    for f in files:
        if f.inclusion.is_excluded():
            continue
        repo_path = GENERATED.get(f.src_uri, f"{docs}/{f.src_uri}")
        SITE_OF[repo_path] = f.src_uri
        if f.src_uri.endswith("README.md") or f.src_uri.endswith("index.md"):
            SITE_OF.setdefault(posixpath.dirname(repo_path) or ".", f.src_uri)
        elif f.src_uri in GENERATED:
            SITE_OF.setdefault(posixpath.dirname(repo_path), f.src_uri)
    return files


def on_nav(nav, config, files):
    listed = {p.file.src_uri for p in nav.pages}
    for f in files.documentation_pages():
        if f.src_uri not in listed:
            log.warning("%s is a page of the site that the nav in mkdocs.yml does not list", f.src_uri)
    return nav


def _rewrite(target, repo_path, src_uri):
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith(("#", "/", "<")):
        return target
    path, hash_, fragment = target.partition("#")
    if not path:
        return target
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(repo_path), path))
    if resolved.startswith(".."):
        return target
    site = SITE_OF.get(resolved) or SITE_OF.get(resolved.rstrip("/"))
    if site is not None:
        return posixpath.relpath(site, posixpath.dirname(src_uri) or ".") + hash_ + fragment
    kind = "tree" if os.path.isdir(os.path.join(ROOT, resolved)) else "blob"
    return f"{REPO}/{kind}/{BRANCH}/{resolved}" + hash_ + fragment


def _rewrite_links(text, repo_path, src_uri):
    """The links in a run of lines outside code blocks (a link's text may span lines)."""
    def links(text):
        text = INLINE_LINK.sub(lambda m: m.group(1) + _rewrite(m.group(2), repo_path, src_uri) + m.group(3), text)
        return HTML_SRC.sub(lambda m: m.group(1) + _rewrite(m.group(2), repo_path, src_uri) + m.group(3), text)

    text = REFERENCE_LINK.sub(lambda m: m.group(1) + _rewrite(m.group(2), repo_path, src_uri) + m.group(3), text)
    # Code spans are set aside, so a link inside one is left alone, while a link whose
    # text is code ([`x`](x)) is still rewritten.
    spans = []

    def hide(span):
        spans.append(span.group(0))
        return f"\0{len(spans) - 1}\0"

    text = links(CODE_SPAN.sub(hide, text))
    return re.sub(r"\0(\d+)\0", lambda m: spans[int(m.group(1))], text)


def _indent(line):
    return len(line) - len(line.lstrip(" "))


def _shift(line, delta):
    if delta >= 0:
        return " " * delta + line if line.strip() else line
    return line[min(-delta, _indent(line)) :]


def _normalize_lists(markdown):
    """Re-indents nested list content from CommonMark's columns to multiples of 4.

    Each open list item is (the column its content starts at on GitHub, the column
    Python-Markdown needs). A line belongs to the deepest item whose content column it
    reaches, and moves to that item's Python-Markdown column plus whatever indent it
    had beyond it. A fenced block moves with its opening line.
    """
    out, stack, fence, delta, blank = [], [], None, 0, True
    prev_item, loose = False, False  # loose: a blank line inside the current list
    for line in markdown.split("\n"):
        if fence is not None:
            out.append(_shift(line, delta))
            closing = FENCE.match(line)
            if closing and closing.group(1)[0] == fence[0] and len(closing.group(1)) >= len(fence) and not line.strip()[len(closing.group(1)):]:
                fence = None
            continue
        if not line.strip():
            out.append(line)
            blank, loose = True, loose or bool(stack)
            continue
        n, item = _indent(line), LIST_ITEM.match(line)
        if not item and not blank and stack and n < stack[-1][0]:
            out.append(line)  # a lazy continuation of the paragraph above
            prev_item = False
            continue
        if item and not blank and not prev_item and (not stack or loose):
            out.append("")  # an item right under a paragraph
        while stack and n < stack[-1][0]:
            stack.pop()
        if not stack:
            loose = loose and bool(item)
        new = stack[-1][1] + (n - stack[-1][0]) if stack and not item else (stack[-1][1] if stack else 0)
        if item:
            marker, spaces = item.group(2), item.group(3)
            stack.append((n + len(marker) + min(len(spaces), 4), new + 4))
            line = " " * new + marker + " " + item.group(4)
        elif stack:
            line = " " * new + line[n:]
        opening = FENCE.match(line)
        if opening:
            fence, delta = opening.group(1), _indent(line) - n
        out.append(line)
        blank, prev_item = False, bool(item)
    return "\n".join(out)


def on_page_markdown(markdown, page, config, files):
    src_uri = page.file.src_uri
    docs = os.path.relpath(config["docs_dir"], ROOT).replace(os.sep, "/")
    repo_path = GENERATED.get(src_uri, f"{docs}/{src_uri}")
    if src_uri == "README.md":
        page.meta.setdefault("hide", ["navigation"])
    if src_uri in GENERATED:
        page.edit_url = f"{REPO}/edit/{BRANCH}/{repo_path}"
    markdown = _normalize_lists(SITE_ONLY.sub(lambda m: m.group(1), markdown))

    out, prose, fence, lines, i = [], [], None, markdown.split("\n"), 0

    def flush():
        if prose:
            out.append(_rewrite_links("\n".join(prose), repo_path, src_uri))
            prose.clear()

    while i < len(lines):
        line = lines[i]
        opening = FENCE.match(line)
        if fence is not None:
            if opening and opening.group(1)[0] == fence[0] and len(opening.group(1)) >= len(fence) and not line.strip()[len(opening.group(1)):]:
                fence = None
            out.append(line)
            i += 1
            continue
        if opening:
            flush()
            fence = opening.group(1)
            out.append(line)
            i += 1
            continue
        callout = CALLOUT.match(line)
        if callout:
            flush()
            indent, kind = callout.group(1), CALLOUT_TYPES[callout.group(2)]
            body = []
            i += 1
            while i < len(lines) and re.match(r"^\s*>", lines[i]):
                body.append(re.sub(r"^\s*>\s?", "", lines[i], count=1))
                i += 1
            body = _rewrite_links("\n".join(body), repo_path, src_uri).split("\n")
            out.append(f"{indent}!!! {kind}")
            out.append("")
            out.extend((f"{indent}    {b}" if b.strip() else "") for b in body)
            out.append("")
            continue
        prose.append(line)
        i += 1
    flush()
    return "\n".join(out)
