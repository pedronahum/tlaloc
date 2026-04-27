"""Head-to-head harness Phase 2 — PyTorch reference implementations (§0.4.232).

Mirrors Tlaloc's `:benchmarks/HeadToHeadHarness` inhabitants in PyTorch.
Each benchmark uses the same fixed input set and the same primal structural
shape as the corresponding JVM-side inhabitant, then times `torch.func.grad`
evaluation under `torch.compile` over a warmup + measured iteration window.

Output JSON shape matches Tlaloc's `HeadToHeadResult.toJsonString()` so that
the JVM-side aggregator can read both Tlaloc and PyTorch results from the
same file format.

Usage (post-toolchain-install):
    python harness/python/run_pytorch.py [--output build/]
        [--warmup 200] [--measured 800]

The script writes:
    <output>/harness-results-pytorch.json
    <output>/harness-results-pytorch.csv

Both formats mirror Tlaloc's runner output. The CSV uses
`framework=pytorch-compile` rows so cross-framework aggregation can
distinguish them from Tlaloc's `framework=tlaloc` rows.

Toolchain pre-requisites:
    pip install torch>=2.1   # the `torch.func.grad` API + `torch.compile`

This script is **shipped into the repo** as part of harness Phase 2's
preparation; running it is gated on user-side PyTorch availability per
the /loop's no-toolchain-install rule.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from typing import Callable, List, Tuple

# Lazy import so the file is parseable + linter-friendly without torch installed.
# Real execution requires the import; see the "Toolchain pre-requisites" doc.
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================================
# Benchmark primal implementations — mirroring Tlaloc's `BenchmarkPrimals`
# ============================================================================

def bgd_hyperopt_primal(r: "torch.Tensor", Sxy: "torch.Tensor",
                         Sx2: "torch.Tensor", M: "torch.Tensor",
                         K: int = 3) -> "torch.Tensor":
    """BGDHyperOpt outer-loop affine recurrence.

    Mirrors `BenchmarkPrimals.bgdHyperOptOuterLoopPrimal(K)`:
        a = 1 + 2*r*Sx2/M
        b = -2*r*Sxy/M
        w = 0
        for k in 0..K-1: w = a*w + b
        return w
    """
    a = 1.0 + 2.0 * r * Sx2 / M
    b = -2.0 * r * Sxy / M
    w = torch.zeros_like(r)
    for _ in range(K):
        w = a * w + b
    return w


def hookean_spring_primal(p_init: "torch.Tensor", v_init: "torch.Tensor",
                           k_spring: "torch.Tensor",
                           N: int = 10, dt: float = 0.1) -> "torch.Tensor":
    """HookeanSpring scalar 1D oscillator (semi-implicit Euler).

    Mirrors `BenchmarkPrimals.hookeanSpringPrimal(N, dt)`:
        pos, vel = p_init, v_init
        for i in 0..N-1:
            force = -k_spring * pos
            vel = vel + dt * force
            pos = pos + dt * vel
        return pos
    """
    pos, vel = p_init, v_init
    for _ in range(N):
        force = -k_spring * pos
        vel = vel + dt * force
        pos = pos + dt * vel
    return pos


def brachistochrone_primal(y: "torch.Tensor", N: int = 5) -> "torch.Tensor":
    """Brachistochrone compound-velocity sub-primal.

    Mirrors `BenchmarkPrimals.brachistochroneCompoundVelocityPrimal(N)`:
        v = 1
        for i in 0..N-1: v = v + v * y
        return v
    Closed-form: v = (1 + y)^N.
    """
    v = torch.ones_like(y)
    for _ in range(N):
        v = v + v * y
    return v


def hmc_logistic_regression_primal(b0: "torch.Tensor",
                                    b1: "torch.Tensor") -> "torch.Tensor":
    """HMC logistic-regression log-posterior (negated) at fixed n=4, d=2.

    Mirrors `BenchmarkPrimals.hmcLogisticRegressionPrimal()` with the
    same hardcoded dataset:
        X = [[1.0, 0.5], [0.5, 1.0], [-0.5, 1.5], [1.5, -0.5]]
        y = [1, 0, 1, 0]
    """
    X = torch.tensor(
        [[1.0, 0.5], [0.5, 1.0], [-0.5, 1.5], [1.5, -0.5]],
        dtype=b0.dtype,
    )
    y = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=b0.dtype)
    sum1 = torch.zeros_like(b0)
    sum2 = torch.zeros_like(b0)
    for i in range(4):
        xb = X[i, 0] * b0 + X[i, 1] * b1
        sum1 = sum1 + (y[i] - 1.0) * xb
        sum2 = sum2 + torch.log(1.0 + torch.exp(-xb))
    term3 = (b0 * b0 + b1 * b1) / 2000.0
    return sum1 - sum2 - term3


def cartpole_phase1_primal(at: "torch.Tensor", x0: "torch.Tensor",
                            x1: "torch.Tensor", x2: "torch.Tensor",
                            x3: "torch.Tensor") -> "torch.Tensor":
    """CartPole Phase 1 one-timestep reward with clip-at-zero IF.

    Mirrors `BenchmarkPrimals.cartPolePhase1Primal()`. Note the `pt`
    computation is included for parity (it's vestigial; df/dat = 0 because
    `pt` doesn't reach the return value).
    """
    sin_x2 = torch.sin(x2)
    cos_x2 = torch.cos(x2)
    rt = 9.0 * at + 0.045 * x3 * x3 * sin_x2
    qt = (9.8 * sin_x2 - rt * cos_x2) / (0.65 - 0.4 * cos_x2 * cos_x2)
    pt = rt - 0.045 * qt * cos_x2  # noqa: F841 — vestigial DCE-test piece
    xn0 = x0 + 0.02 * x1
    xn2 = x2 + 0.02 * x3
    max_arg = (2.4 - torch.abs(xn0)) * (0.21 - torch.abs(xn2))
    clipped = torch.where(max_arg > 0.0, max_arg, torch.zeros_like(max_arg))
    term = 0.5 - clipped
    return term * term


# ============================================================================
# Harness runner — mirrors the JVM `HeadToHeadBenchmark.runBaseline`
# ============================================================================

class HeadToHeadResult:
    def __init__(self, benchmark: str, forward_value: float,
                 gradient_values: List[float],
                 warmup: int, measured: int,
                 median_ns: int, min_ns: int, p99_ns: int):
        self.benchmark = benchmark
        self.forward_value = forward_value
        self.gradient_values = gradient_values
        self.warmup_iterations = warmup
        self.measured_iterations = measured
        self.median_nanos = median_ns
        self.min_nanos = min_ns
        self.p99_nanos = p99_ns

    def to_json_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "forwardValue": self.forward_value,
            "gradientValues": self.gradient_values,
            "warmupIterations": self.warmup_iterations,
            "measuredIterations": self.measured_iterations,
            "medianNanos": self.median_nanos,
            "minNanos": self.min_nanos,
            "p99Nanos": self.p99_nanos,
        }


def run_baseline(name: str,
                 primal: Callable[..., "torch.Tensor"],
                 inputs: List[float],
                 warmup: int = 200, measured: int = 800,
                 use_compile: bool = True) -> HeadToHeadResult:
    """Run a primal through `torch.func.grad` + `torch.compile` and time it."""
    if not HAS_TORCH:
        raise RuntimeError(
            "torch is not installed. Install with: pip install torch>=2.1",
        )

    # Wrap primal so torch.func.grad sees the first arg as the differentiation
    # target. For multi-input primals, take grad w.r.t. all inputs sequentially.
    n_inputs = len(inputs)

    def wrapped(*args):
        return primal(*args)

    grad_fn = torch.func.grad(wrapped, argnums=tuple(range(n_inputs)))
    if use_compile:
        # `torch.compile` traces + compiles; first call is slow, subsequent
        # calls hit the compiled cache.
        grad_fn = torch.compile(grad_fn)

    tensor_inputs = [torch.tensor(v, dtype=torch.float32, requires_grad=False)
                     for v in inputs]
    forward_out = wrapped(*tensor_inputs)
    grads = grad_fn(*tensor_inputs)
    if not isinstance(grads, tuple):
        grads = (grads,)

    forward_value = float(forward_out.detach().item())
    gradient_values = [float(g.detach().item()) for g in grads]

    # Warmup loop — the first `torch.compile` invocation traces; subsequent
    # ones hit the cache. Discard the first `warmup` iterations.
    for _ in range(warmup):
        grad_fn(*tensor_inputs)

    # Measurement loop.
    timings: List[int] = []
    for _ in range(measured):
        t0 = time.perf_counter_ns()
        grad_fn(*tensor_inputs)
        timings.append(time.perf_counter_ns() - t0)
    timings.sort()
    return HeadToHeadResult(
        benchmark=name,
        forward_value=forward_value,
        gradient_values=gradient_values,
        warmup=warmup, measured=measured,
        median_ns=timings[measured // 2],
        min_ns=timings[0],
        p99_ns=timings[(measured * 99) // 100],
    )


# ============================================================================
# Top-level runner — mirrors `HeadToHeadHarnessRunner.runAllAndDump`
# ============================================================================

# Each entry: (benchmark name, callable, fixed inputs).
# Names match the JVM-side inhabitants for easy CSV/JSON merging.
INHABITANTS: List[Tuple[str, Callable[..., "torch.Tensor"], List[float]]] = [
    ("bgd-hyperopt-outer-loop-K3",
     lambda r, Sxy, Sx2, M: bgd_hyperopt_primal(r, Sxy, Sx2, M, K=3),
     [0.01, 111.2, 55.0, 5.0]),
    ("hookean-spring-scalar-N10",
     lambda p, v, k: hookean_spring_primal(p, v, k, N=10, dt=0.1),
     [1.0, 0.0, 1.0]),
    ("brachistochrone-compound-velocity-N5",
     lambda y: brachistochrone_primal(y, N=5),
     [0.5]),
    ("hmc-logistic-regression-n4-d2",
     hmc_logistic_regression_primal,
     [0.5, 0.3]),
    ("cartpole-phase1-onestep",
     cartpole_phase1_primal,
     [0.5, 0.0, 0.1, 0.05, 0.02]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", default="build/",
                        help="Output directory for JSON+CSV (default: build/)")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--measured", type=int, default=800)
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile; use eager grad only")
    args = parser.parse_args()

    if not HAS_TORCH:
        raise SystemExit(
            "ERROR: torch is not installed. Install with: pip install torch>=2.1\n"
            "This script is part of Tlaloc's harness Phase 2 reference implementations\n"
            "and requires user-side toolchain availability per the /loop rules.",
        )

    results: List[HeadToHeadResult] = []
    for name, primal, inputs in INHABITANTS:
        print(f"[pytorch harness] running {name}...", flush=True)
        result = run_baseline(
            name=name, primal=primal, inputs=inputs,
            warmup=args.warmup, measured=args.measured,
            use_compile=not args.no_compile,
        )
        print(f"  forward={result.forward_value:.6g}, "
              f"grads={result.gradient_values}, "
              f"median={result.median_nanos}ns")
        results.append(result)

    os.makedirs(args.output, exist_ok=True)
    json_path = os.path.join(args.output, "harness-results-pytorch.json")
    csv_path = os.path.join(args.output, "harness-results-pytorch.csv")

    with open(json_path, "w") as f:
        json.dump([r.to_json_dict() for r in results], f, indent=2)

    with open(csv_path, "w") as f:
        f.write("benchmark,framework,n_iterations,median_ns,min_ns,p99_ns\n")
        framework = "pytorch-compile" if not args.no_compile else "pytorch-eager"
        for r in results:
            f.write(f"{r.benchmark},{framework},{r.measured_iterations},"
                    f"{r.median_nanos},{r.min_nanos},{r.p99_nanos}\n")

    print(f"[pytorch harness] wrote {json_path}")
    print(f"[pytorch harness] wrote {csv_path}")


if __name__ == "__main__":
    main()
