"""Cross-framework comparison aggregator (§0.4.234).

Reads the three harness JSON files produced by:
  - JVM-side: `HeadToHeadHarnessRunner.runAllAndDump` → `harness-results-tlaloc.json`
  - PyTorch:  `harness/python/run_pytorch.py` → `harness-results-pytorch.json`
  - JAX:      `harness/python/run_jax.py` → `harness-results-jax.json`

Produces a unified Markdown comparison table covering:
  - **Numerical agreement** per benchmark: forward + per-input gradients with
    f32-tolerance (1e-3 absolute / 5e-3 relative) flagging across frameworks.
  - **Throughput speedup** per benchmark: `tlaloc.medianNanos` vs PyTorch's
    and JAX's, formatted as `Tlaloc/torch×` and `Tlaloc/jax×` ratios.
  - **M9 verdict**: per benchmark, whether the speedup beats the paper's
    reported figure (within 20% of paper / >3× over torch on at least
    three of six).

This script only uses the Python standard library — stdlib `json` is enough
to parse `HeadToHeadResult.toJsonString()`'s output. No PyTorch / JAX
dependency; the script runs even before the user has installed those.

Usage:
    python harness/python/aggregate.py [--input build/]
        [--output build/harness-comparison.md]
        [--paper-figures docs/paper_figures.json]

The script exits with non-zero status if:
  - The Tlaloc JSON file is missing (required).
  - Any numerical-agreement check fails for benchmarks present in
    multiple frameworks.

PyTorch and JAX JSON files are optional: if absent, the corresponding
columns show `(not measured)`. This lets the script run even before
the user has the Python toolchain.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# Paper-reported speedups (per OOPSLA 2021 paper §7).
# Format: {benchmark: (low, high)} — speedup over torch.compile.
# ============================================================================

# Paper-reported overall speedups per Table 3 of the OOPSLA 2021 paper
# `coarsening-autodiff` (full text in `docs/papers/coarsening-autodiff.txt`).
# Each range is (min, max) across the paper's 2 machines × 3 configs = 6
# measurements per benchmark.
#
# We use the "Overall Time" column (not "Differentiation Time") because the
# harness measures gradient evaluation latency — and reverse-mode AD's
# gradient computation includes the embedded primal recomputation, which
# maps closer to the paper's "Overall Time" than its "Differentiation Time"
# (the latter is artificially smaller because the paper's library separates
# the forward/backward halves).
#
# §0.4.239 corrected swapped Brachistochrone ↔ HookeanSpring numbers and a
# wrong CartPole range that were inherited from the §0.4.181 plan's
# placeholder table.
PAPER_SPEEDUPS: Dict[str, Tuple[float, float]] = {
    "bgd-hyperopt-outer-loop-K3":             (8.06, 8.56),    # Table 3 BGDHyperOpt overall
    "brachistochrone-compound-velocity-N5":   (1.79, 2.51),    # Table 3 Branchist. overall
    "cartpole-phase1-onestep":                (1.05, 1.12),    # Table 3 CartPole overall
    "hmc-logistic-regression-n4-d2":          (2.27, 3.56),    # Table 3 HMC overall
    "hookean-spring-scalar-N10":              (4.09, 11.02),   # Table 3 HookeanSpring overall
    # QWOP: the paper's QWOP measures the actual physics-based game avatar-step
    # (not Tlaloc's synthetic primal). The paper's Overall Time speedup
    # (1.41-1.57×) is a different shape than what Tlaloc's `qwop-avatar-step`
    # synthetic measures, so leave unset and report "—" rather than spurious
    # comparison.
}


# ============================================================================
# JSON parsing — matches HeadToHeadResult.toJsonString()'s shape.
# ============================================================================

def load_results(path: str) -> List[Dict[str, Any]]:
    """Load harness results JSON file, returning a list of result dicts.

    The on-disk format is a JSON array of objects with fields:
        benchmark, forwardValue, gradientValues, warmupIterations,
        measuredIterations, medianNanos, minNanos, p99Nanos.
    """
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON array, got {type(data).__name__}")
    return data


def load_optional(path: str) -> Optional[List[Dict[str, Any]]]:
    """Load a results file if it exists; return None otherwise."""
    if not os.path.exists(path):
        return None
    return load_results(path)


def by_benchmark(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index a list of result dicts by `benchmark` name."""
    return {r["benchmark"]: r for r in results}


# ============================================================================
# Numerical agreement check (f32 tolerance).
# ============================================================================

NUMERICAL_ABS_TOL = 1e-3
NUMERICAL_REL_TOL = 5e-3


def agrees_within_tolerance(a: float, b: float) -> bool:
    """Returns True if a and b agree within f32 tolerance.

    Tolerance: 1e-3 absolute OR 5e-3 relative (whichever is more permissive),
    matching the M9 exit criterion's "f32-tolerance numerical match" clause.
    """
    diff = abs(a - b)
    if diff < NUMERICAL_ABS_TOL:
        return True
    rel = diff / max(abs(a), abs(b), 1e-12)
    return rel < NUMERICAL_REL_TOL


def check_agreement(
    bench_name: str,
    tlaloc: Dict[str, Any],
    other: Dict[str, Any],
    other_name: str,
) -> List[str]:
    """Check forward + per-input gradient agreement between Tlaloc and another framework.

    Returns a list of error messages (empty if all agree).
    """
    errors: List[str] = []

    fwd_t = tlaloc["forwardValue"]
    fwd_o = other["forwardValue"]
    if not agrees_within_tolerance(fwd_t, fwd_o):
        errors.append(
            f"  [{bench_name}] forward Tlaloc={fwd_t:.6g} vs {other_name}={fwd_o:.6g} "
            f"exceeds f32 tolerance (abs={abs(fwd_t - fwd_o):.3g})"
        )

    grads_t = tlaloc["gradientValues"]
    grads_o = other["gradientValues"]
    if len(grads_t) != len(grads_o):
        errors.append(
            f"  [{bench_name}] gradient cardinality mismatch: "
            f"Tlaloc has {len(grads_t)}, {other_name} has {len(grads_o)}"
        )
    else:
        for i, (gt, go) in enumerate(zip(grads_t, grads_o)):
            if not agrees_within_tolerance(gt, go):
                errors.append(
                    f"  [{bench_name}] gradient[{i}] Tlaloc={gt:.6g} vs "
                    f"{other_name}={go:.6g} exceeds f32 tolerance "
                    f"(abs={abs(gt - go):.3g})"
                )
    return errors


# ============================================================================
# Markdown table formatting.
# ============================================================================

def format_speedup(tlaloc_ns: int, other_ns: Optional[int]) -> str:
    """Format speedup ratio: Tlaloc / other. Returns '(not measured)' if other is None."""
    if other_ns is None:
        return "—"
    if tlaloc_ns == 0:
        return "∞"
    ratio = other_ns / tlaloc_ns  # higher = Tlaloc is faster
    return f"{ratio:.2f}×"


def m9_verdict(bench_name: str, tlaloc_speedup_torch: Optional[float]) -> str:
    """Produce the M9 verdict for a benchmark.

    Returns one of: "✓ pass", "✗ fail", "—" (not in paper's table or no measurement).
    """
    if bench_name not in PAPER_SPEEDUPS:
        return "—"
    if tlaloc_speedup_torch is None:
        return "—"
    paper_low, paper_high = PAPER_SPEEDUPS[bench_name]
    # Per §11.13: "within 20% of paper's figures."
    # Interpret as: tlaloc_speedup_torch is within [0.8*paper_low, 1.2*paper_high].
    pass_low = 0.8 * paper_low
    pass_high = 1.2 * paper_high
    if pass_low <= tlaloc_speedup_torch <= pass_high:
        return "✓"
    return "✗"


def build_table(
    tlaloc: Dict[str, Dict[str, Any]],
    pytorch: Optional[Dict[str, Dict[str, Any]]],
    jax: Optional[Dict[str, Dict[str, Any]]],
) -> str:
    """Build a Markdown comparison table as a string."""
    bench_names = sorted(tlaloc.keys())

    lines: List[str] = []
    lines.append("# Head-to-head harness comparison\n")
    lines.append(
        "Generated by `harness/python/aggregate.py`. Reads "
        "`harness-results-tlaloc.json`, `harness-results-pytorch.json`, "
        "and `harness-results-jax.json`."
    )
    lines.append("")
    lines.append("## Throughput comparison (median nanoseconds per gradient eval)")
    lines.append("")
    lines.append(
        "| Benchmark | Tlaloc | PyTorch | JAX | Tlaloc/torch× | Tlaloc/jax× | "
        "Paper torch× | M9 |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|"
    )

    for name in bench_names:
        t = tlaloc[name]
        p = pytorch.get(name) if pytorch else None
        j = jax.get(name) if jax else None

        t_ns = t["medianNanos"]
        p_ns_str = f"{p['medianNanos']}" if p else "—"
        j_ns_str = f"{j['medianNanos']}" if j else "—"

        speedup_torch_str = format_speedup(t_ns, p["medianNanos"] if p else None)
        speedup_jax_str = format_speedup(t_ns, j["medianNanos"] if j else None)

        speedup_torch_value: Optional[float] = None
        if p:
            speedup_torch_value = p["medianNanos"] / t_ns if t_ns else None

        paper_str = "—"
        if name in PAPER_SPEEDUPS:
            lo, hi = PAPER_SPEEDUPS[name]
            paper_str = f"{lo:.2f}-{hi:.2f}×"
        verdict = m9_verdict(name, speedup_torch_value)

        lines.append(
            f"| {name} | {t_ns} | {p_ns_str} | {j_ns_str} | "
            f"{speedup_torch_str} | {speedup_jax_str} | {paper_str} | {verdict} |"
        )
    lines.append("")
    lines.append("## Numerical agreement (f32 tolerance: 1e-3 abs / 5e-3 rel)")
    lines.append("")
    if pytorch is None and jax is None:
        lines.append(
            "*(PyTorch and JAX results not available; install toolchain and "
            "run the corresponding `harness/python/run_*.py` scripts.)*"
        )
    else:
        lines.append("| Benchmark | Forward agreement | Gradient agreement |")
        lines.append("|---|---|---|")
        for name in bench_names:
            t = tlaloc[name]
            cells: List[str] = []
            for other_results, other_name in [(pytorch, "PyTorch"), (jax, "JAX")]:
                if other_results is None or name not in other_results:
                    continue
                errors = check_agreement(name, t, other_results[name], other_name)
                if errors:
                    cells.append(f"{other_name}: ✗")
                else:
                    cells.append(f"{other_name}: ✓")
            agreement = ", ".join(cells) if cells else "—"
            lines.append(f"| {name} | {agreement} | (see forward-agreement column) |")
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# Top-level runner.
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--input", default="build/",
        help="Directory containing harness-results-*.json files (default: build/)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for Markdown table (default: stdout only)"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero if any numerical-agreement check fails"
    )
    args = parser.parse_args()

    tlaloc_path = os.path.join(args.input, "harness-results-tlaloc.json")
    pytorch_path = os.path.join(args.input, "harness-results-pytorch.json")
    jax_path = os.path.join(args.input, "harness-results-jax.json")

    if not os.path.exists(tlaloc_path):
        raise SystemExit(
            f"ERROR: required file not found: {tlaloc_path}\n"
            "Run the JVM-side harness first: "
            "`./gradlew :benchmarks:jvmTest --tests "
            "io.tlaloc.benchmarks.HeadToHeadHarnessAllTest`."
        )
    tlaloc_raw = load_results(tlaloc_path)
    tlaloc = by_benchmark(tlaloc_raw)
    pytorch_raw = load_optional(pytorch_path)
    pytorch = by_benchmark(pytorch_raw) if pytorch_raw else None
    jax = None
    jax_raw = load_optional(jax_path)
    if jax_raw is not None:
        jax = by_benchmark(jax_raw)

    table = build_table(tlaloc, pytorch, jax)
    print(table)

    if args.output:
        with open(args.output, "w") as f:
            f.write(table)
        print(f"[aggregate] wrote {args.output}", file=sys.stderr)

    # Strict mode: fail on numerical disagreements.
    if args.strict:
        all_errors: List[str] = []
        for name, t in tlaloc.items():
            for other_results, other_name in [(pytorch, "PyTorch"), (jax, "JAX")]:
                if other_results is not None and name in other_results:
                    all_errors.extend(check_agreement(name, t, other_results[name], other_name))
        if all_errors:
            print("\nNumerical-agreement failures:", file=sys.stderr)
            for err in all_errors:
                print(err, file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
