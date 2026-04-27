"""Tests for harness/python/aggregate.py (§0.4.237).

Runs `aggregate.py` as a subprocess against synthetic JSON fixtures in
`harness/python/test_data/` and asserts on the output. Stdlib-only —
no external dependencies, runs in any environment Python is installed.

The aggregator is critical to M9 closure: it produces the Markdown table
that becomes the body of the §0.4 entry titled "Phase 1 closed —
coarsening at M9 parity." Without test coverage, a future regression in
`aggregate.py`'s parsing, formatting, or numerical-agreement logic could
silently produce wrong results when the user runs the full four-command
workflow post-toolchain-install.

Usage:
    python -m unittest harness/python/aggregate_test.py

Or via the project root's existing `python3` interpreter:
    python3 harness/python/aggregate_test.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

# Locate the script under test and the test fixtures relative to this file.
_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
AGGREGATE_PY = os.path.join(_SELF_DIR, "aggregate.py")
TEST_DATA_DIR = os.path.join(_SELF_DIR, "test_data")

PYTHON = sys.executable  # use the same interpreter that's running the tests


def setup_input_dir(tmpdir: str, fixtures: dict) -> None:
    """Copy named test fixtures into tmpdir under their canonical names.

    `fixtures` is a dict mapping target filename → fixture path under TEST_DATA_DIR.
    Example: {'harness-results-tlaloc.json': 'tlaloc_baseline.json'} copies
    `test_data/tlaloc_baseline.json` to `tmpdir/harness-results-tlaloc.json`.
    """
    for target_name, fixture_name in fixtures.items():
        src = os.path.join(TEST_DATA_DIR, fixture_name)
        dst = os.path.join(tmpdir, target_name)
        shutil.copyfile(src, dst)


def run_aggregate(input_dir: str, *extra_args: str) -> subprocess.CompletedProcess:
    """Invoke aggregate.py with --input=<input_dir> and capture output."""
    args = [PYTHON, AGGREGATE_PY, "--input", input_dir, *extra_args]
    return subprocess.run(args, capture_output=True, text=True)


class AggregateTest(unittest.TestCase):

    def test_tlaloc_only_produces_table_with_dash_cells(self):
        """With only tlaloc.json, table shows `—` in cross-framework cells."""
        with tempfile.TemporaryDirectory() as tmp:
            setup_input_dir(tmp, {
                "harness-results-tlaloc.json": "tlaloc_baseline.json",
            })
            result = run_aggregate(tmp)
            self.assertEqual(0, result.returncode, msg=f"aggregator failed: {result.stderr}")
            # Tlaloc-side benchmarks present.
            self.assertIn("brachistochrone-compound-velocity-N5", result.stdout)
            self.assertIn("hookean-spring-scalar-N10", result.stdout)
            # Cross-framework cells show `—`.
            self.assertIn("—", result.stdout)
            # Numerical-agreement section explicitly notes PyTorch+JAX absence.
            self.assertIn("PyTorch and JAX results not available", result.stdout)

    def test_with_pytorch_computes_speedup_and_m9_verdict(self):
        """With agreeing pytorch.json, table shows speedup ratio + M9 ✓."""
        with tempfile.TemporaryDirectory() as tmp:
            setup_input_dir(tmp, {
                "harness-results-tlaloc.json": "tlaloc_baseline.json",
                "harness-results-pytorch.json": "pytorch_agreeing.json",
            })
            result = run_aggregate(tmp)
            self.assertEqual(0, result.returncode, msg=f"aggregator failed: {result.stderr}")
            # Brachistochrone: tlaloc=3000ns, pytorch=24000ns → 8.00× speedup.
            # Paper range 4-11×; 8.00 within ±20% band [3.2, 13.2] → M9 ✓.
            self.assertIn("8.00×", result.stdout)
            self.assertIn("✓", result.stdout)
            # Numerical agreement passes (forward + gradient values match).
            self.assertIn("PyTorch: ✓", result.stdout)

    def test_strict_mode_passes_when_values_agree(self):
        """--strict mode exits 0 when forward + gradient values agree within f32 tolerance."""
        with tempfile.TemporaryDirectory() as tmp:
            setup_input_dir(tmp, {
                "harness-results-tlaloc.json": "tlaloc_baseline.json",
                "harness-results-pytorch.json": "pytorch_agreeing.json",
            })
            result = run_aggregate(tmp, "--strict")
            self.assertEqual(
                0, result.returncode,
                msg=f"--strict mode should pass on agreeing fixtures, got stderr: {result.stderr}",
            )

    def test_strict_mode_fails_when_values_disagree(self):
        """--strict mode exits non-zero when forward or gradient breach f32 tolerance."""
        with tempfile.TemporaryDirectory() as tmp:
            setup_input_dir(tmp, {
                "harness-results-tlaloc.json": "tlaloc_baseline.json",
                "harness-results-pytorch.json": "pytorch_disagreeing.json",
            })
            result = run_aggregate(tmp, "--strict")
            self.assertNotEqual(
                0, result.returncode,
                msg="--strict mode should fail on disagreeing fixtures",
            )
            # Stderr should mention the failing benchmark.
            self.assertIn("brachistochrone", result.stderr.lower())

    def test_missing_tlaloc_file_errors_out(self):
        """Required file: aggregator exits non-zero if tlaloc.json missing."""
        with tempfile.TemporaryDirectory() as tmp:
            # Don't copy any fixtures.
            result = run_aggregate(tmp)
            self.assertNotEqual(0, result.returncode)
            self.assertIn("not found", result.stderr.lower())

    def test_output_file_written_when_specified(self):
        """--output flag persists Markdown to a file."""
        with tempfile.TemporaryDirectory() as tmp:
            setup_input_dir(tmp, {
                "harness-results-tlaloc.json": "tlaloc_baseline.json",
            })
            output_path = os.path.join(tmp, "comparison.md")
            result = run_aggregate(tmp, "--output", output_path)
            self.assertEqual(0, result.returncode, msg=f"aggregator failed: {result.stderr}")
            self.assertTrue(
                os.path.exists(output_path),
                msg=f"output file should exist at {output_path}",
            )
            # File contents should match the Markdown table.
            with open(output_path) as f:
                content = f.read()
            self.assertIn("Head-to-head harness comparison", content)
            self.assertIn("brachistochrone-compound-velocity-N5", content)


if __name__ == "__main__":
    unittest.main()
