#!/usr/bin/env python3
"""Tests of scripts/mkdocs_hooks.py's Markdown rewrites (no MkDocs needed to run them).

    python3 scripts/test_mkdocs_hooks.py
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The hooks import mkdocs only for File; a stand-in lets these tests run without it.
if "mkdocs" not in sys.modules:
    files = types.ModuleType("mkdocs.structure.files")
    files.File = object
    for name in ("mkdocs", "mkdocs.structure"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["mkdocs.structure.files"] = files
import mkdocs_hooks as hooks  # noqa: E402


class Page:
    def __init__(self, src_uri):
        self.file = types.SimpleNamespace(src_uri=src_uri)
        self.meta = {}
        self.edit_url = None


def render(markdown, src_uri="GUIDE.md"):
    return hooks.on_page_markdown(markdown, Page(src_uri), {"docs_dir": os.path.join(hooks.ROOT, "docs")}, None)


class Links(unittest.TestCase):
    def setUp(self):
        hooks.GENERATED.clear()
        hooks.GENERATED.update({"examples/quickstart.md": "examples/quickstart/README.md", "changelog.md": "CHANGELOG.md"})
        hooks.SITE_OF.clear()
        hooks.SITE_OF.update({
            "docs/GUIDE.md": "GUIDE.md",
            "docs/case-studies/step.md": "case-studies/step.md",
            "examples/quickstart/README.md": "examples/quickstart.md",
            "examples/quickstart": "examples/quickstart.md",
            "CHANGELOG.md": "changelog.md",
        })

    def test_a_page_of_the_site_is_linked_relative_to_the_page(self):
        self.assertEqual(render("[x](case-studies/step.md#run)"), "[x](case-studies/step.md#run)")
        self.assertEqual(render("[x](../examples/quickstart/)"), "[x](examples/quickstart.md)")
        self.assertEqual(render("[x](../GUIDE.md)", "case-studies/step.md"), "[x](../GUIDE.md)")

    def test_a_file_outside_the_site_is_linked_on_github(self):
        blob = f"{hooks.REPO}/blob/{hooks.BRANCH}"
        self.assertEqual(render("[x](../scripts/docs.sh)"), f"[x]({blob}/scripts/docs.sh)")
        self.assertEqual(render("[x](src/Main.kt)", "examples/quickstart.md"), f"[x]({blob}/examples/quickstart/src/Main.kt)")
        self.assertEqual(render("[x](plans/)"), f"[x]({hooks.REPO}/tree/{hooks.BRANCH}/docs/plans)")

    def test_a_link_with_code_text_or_over_two_lines_is_rewritten(self):
        self.assertEqual(render("[`quickstart/`](../examples/quickstart/)"), "[`quickstart/`](examples/quickstart.md)")
        self.assertEqual(render("a [training\nstep](case-studies/step.md)"), "a [training\nstep](case-studies/step.md)")
        self.assertEqual(render('<a href="../CHANGELOG.md">c</a>'), '<a href="changelog.md">c</a>')

    def test_code_and_absolute_links_are_left_alone(self):
        for text in ["`[x](../scripts/docs.sh)`", "[x](https://example.com/a.md)", "[x](#anchor)",
                     "```\n[x](../scripts/docs.sh)\n```"]:
            self.assertEqual(render(text), text)


class Callouts(unittest.TestCase):
    def test_a_github_callout_becomes_an_admonition(self):
        self.assertEqual(render("> [!WARNING]\n> Alpha.\n> Really.\n\nNext."),
                         "!!! warning\n\n    Alpha.\n    Really.\n\n\nNext.")

    def test_a_plain_quote_stays_a_quote(self):
        self.assertEqual(render("> Quoted."), "> Quoted.")

    def test_site_only_markup_is_uncommented(self):
        self.assertEqual(render('<!--mkdocs:<div class="grid cards" markdown>-->'), '<div class="grid cards" markdown>')


class Lists(unittest.TestCase):
    def test_a_child_under_its_parents_text_moves_to_four_spaces(self):
        self.assertEqual(hooks._normalize_lists("1. Parent\n   - child\n   - child\n2. Next"),
                         "1. Parent\n    - child\n    - child\n2. Next")
        self.assertEqual(hooks._normalize_lists("- a\n  - b\n    - c"), "- a\n    - b\n        - c")

    def test_a_fence_in_an_item_moves_with_its_opening_line(self):
        self.assertEqual(hooks._normalize_lists("1. Run:\n\n   ```\n   x\n     y\n   ```"),
                         "1. Run:\n\n    ```\n    x\n      y\n    ```")

    def test_a_list_right_under_a_paragraph_gets_a_blank_line(self):
        self.assertEqual(hooks._normalize_lists("Text:\n- a\n- b"), "Text:\n\n- a\n- b")

    def test_an_item_after_a_second_paragraph_gets_a_blank_line(self):
        self.assertEqual(hooks._normalize_lists("1. a\n\n   more\n2. b"), "1. a\n\n    more\n\n2. b")

    def test_a_tight_list_stays_tight(self):
        self.assertEqual(hooks._normalize_lists("- a\n  wrapped\n- b"), "- a\n    wrapped\n- b")


if __name__ == "__main__":
    unittest.main()
