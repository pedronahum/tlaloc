"""Head-to-head harness Phase 2 — JAX reference implementations (§0.4.233).

Mirrors `run_pytorch.py` with JAX idioms (`jax.numpy` instead of `torch`,
`jax.grad` + `jax.jit` instead of `torch.func.grad` + `torch.compile`).
Same fixed input set, same primal structural shape, same JSON output format.

Output JSON is structurally identical to `harness-results-pytorch.json` and
`harness-results-tlaloc.json`, modulo the `framework` column in the CSV.

Usage (post-toolchain-install):
    python harness/python/run_jax.py [--output build/]
        [--warmup 200] [--measured 800]

The script writes:
    <output>/harness-results-jax.json
    <output>/harness-results-jax.csv

CSV uses `framework=jax-jit` rows (or `jax-eager` with --no-jit).

Toolchain pre-requisites:
    pip install jax jaxlib   # JAX with CPU backend

This script is **shipped into the repo** as part of harness Phase 2's
preparation; running it is gated on user-side JAX availability per
the /loop's no-toolchain-install rule.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, List, Tuple

# GB10 is unified-memory: JAX's default 75% preallocation would pin ~90 GB
# of system RAM per process (§0.4.333 reboot incident). Must be set before
# `import jax` (first CUDA client init reads it).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Lazy import — file parses without JAX installed (try/except + ast.parse).
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


# ============================================================================
# Benchmark primal implementations — mirroring Tlaloc's `BenchmarkPrimals`
# (and `run_pytorch.py`'s structural shape, with jax.numpy substitutions).
# ============================================================================

def bgd_hyperopt_primal(r, Sxy, Sx2, M, K: int = 3):
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
    w = jnp.zeros_like(r)
    for _ in range(K):
        w = a * w + b
    return w


def hookean_spring_primal(p_init, v_init, k_spring, N: int = 10, dt: float = 0.1):
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


def brachistochrone_primal(y, N: int = 5):
    """Brachistochrone compound-velocity sub-primal.

    Mirrors `BenchmarkPrimals.brachistochroneCompoundVelocityPrimal(N)`:
        v = 1
        for i in 0..N-1: v = v + v * y
        return v
    Closed-form: v = (1 + y)^N.
    """
    v = jnp.ones_like(y)
    for _ in range(N):
        v = v + v * y
    return v


def hmc_logistic_regression_primal(b0, b1):
    """HMC logistic-regression log-posterior (negated) at fixed n=4, d=2.

    Mirrors `BenchmarkPrimals.hmcLogisticRegressionPrimal()` with the
    same hardcoded dataset:
        X = [[1.0, 0.5], [0.5, 1.0], [-0.5, 1.5], [1.5, -0.5]]
        y = [1, 0, 1, 0]
    """
    X = jnp.array(
        [[1.0, 0.5], [0.5, 1.0], [-0.5, 1.5], [1.5, -0.5]],
        dtype=jnp.float32,
    )
    y = jnp.array([1.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
    sum1 = jnp.zeros_like(b0)
    sum2 = jnp.zeros_like(b0)
    for i in range(4):
        xb = X[i, 0] * b0 + X[i, 1] * b1
        sum1 = sum1 + (y[i] - 1.0) * xb
        sum2 = sum2 + jnp.log(1.0 + jnp.exp(-xb))
    term3 = (b0 * b0 + b1 * b1) / 2000.0
    return sum1 - sum2 - term3


def cartpole_phase1_primal(at, x0, x1, x2, x3):
    """CartPole Phase 1 one-timestep reward with clip-at-zero IF.

    Mirrors `BenchmarkPrimals.cartPolePhase1Primal()`. Note the `pt`
    computation is included for parity (it's vestigial; df/dat = 0 because
    `pt` doesn't reach the return value).
    """
    sin_x2 = jnp.sin(x2)
    cos_x2 = jnp.cos(x2)
    rt = 9.0 * at + 0.045 * x3 * x3 * sin_x2
    qt = (9.8 * sin_x2 - rt * cos_x2) / (0.65 - 0.4 * cos_x2 * cos_x2)
    pt = rt - 0.045 * qt * cos_x2  # noqa: F841 — vestigial DCE-test piece
    xn0 = x0 + 0.02 * x1
    xn2 = x2 + 0.02 * x3
    max_arg = (2.4 - jnp.abs(xn0)) * (0.21 - jnp.abs(xn2))
    clipped = jnp.where(max_arg > 0.0, max_arg, jnp.zeros_like(max_arg))
    term = 0.5 - clipped
    return term * term


# ============================================================================
# Harness runner — mirrors `run_pytorch.py`'s `run_baseline`.
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
                 primal: Callable,
                 inputs: List[float],
                 warmup: int = 200, measured: int = 800,
                 use_jit: bool = True) -> HeadToHeadResult:
    """Run a primal through `jax.grad` + `jax.jit` and time it.

    Note: JAX's `jax.grad(fn, argnums=(0, 1, ...))` returns a function whose
    output is a tuple of gradients matching `argnums`. The wrapper handles
    both single-input (returns scalar) and multi-input (returns tuple) cases.
    """
    if not HAS_JAX:
        raise RuntimeError(
            "jax is not installed. Install with: pip install jax jaxlib",
        )

    n_inputs = len(inputs)
    grad_fn = jax.grad(primal, argnums=tuple(range(n_inputs)))
    if use_jit:
        # `jax.jit` traces + compiles to XLA HLO. First call traces (slow);
        # subsequent calls hit the compiled cache.
        grad_fn = jax.jit(grad_fn)

    # JAX prefers float32 by default in modern releases; ensure all inputs
    # are jnp.float32 for numerical-comparison parity with PyTorch.
    jax_inputs = [jnp.asarray(v, dtype=jnp.float32) for v in inputs]
    forward_out = primal(*jax_inputs)
    grads = grad_fn(*jax_inputs)
    if not isinstance(grads, tuple):
        grads = (grads,)

    forward_value = float(forward_out)
    gradient_values = [float(g) for g in grads]

    # Warmup loop. JAX's `block_until_ready` ensures the timing measures
    # actual computation, not just queuing onto the device.
    for _ in range(warmup):
        out = grad_fn(*jax_inputs)
        if isinstance(out, tuple):
            jax.block_until_ready(out[0])
        else:
            jax.block_until_ready(out)

    # Measurement loop.
    timings: List[int] = []
    for _ in range(measured):
        t0 = time.perf_counter_ns()
        out = grad_fn(*jax_inputs)
        if isinstance(out, tuple):
            jax.block_until_ready(out[0])
        else:
            jax.block_until_ready(out)
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
# Top-level runner — mirrors `run_pytorch.py`'s `INHABITANTS` + `main`.
# ============================================================================

INHABITANTS: List[Tuple[str, Callable, List[float]]] = [
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
    parser.add_argument("--no-jit", action="store_true",
                        help="Skip jax.jit; use eager grad only")
    args = parser.parse_args()

    if not HAS_JAX:
        raise SystemExit(
            "ERROR: jax is not installed. Install with: pip install jax jaxlib\n"
            "This script is part of Tlaloc's harness Phase 2 reference implementations\n"
            "and requires user-side toolchain availability per the /loop rules.",
        )

    results: List[HeadToHeadResult] = []
    for name, primal, inputs in INHABITANTS:
        print(f"[jax harness] running {name}...", flush=True)
        result = run_baseline(
            name=name, primal=primal, inputs=inputs,
            warmup=args.warmup, measured=args.measured,
            use_jit=not args.no_jit,
        )
        print(f"  forward={result.forward_value:.6g}, "
              f"grads={result.gradient_values}, "
              f"median={result.median_nanos}ns")
        results.append(result)

    os.makedirs(args.output, exist_ok=True)
    json_path = os.path.join(args.output, "harness-results-jax.json")
    csv_path = os.path.join(args.output, "harness-results-jax.csv")

    with open(json_path, "w") as f:
        json.dump([r.to_json_dict() for r in results], f, indent=2)

    with open(csv_path, "w") as f:
        f.write("benchmark,framework,n_iterations,median_ns,min_ns,p99_ns\n")
        framework = "jax-jit" if not args.no_jit else "jax-eager"
        for r in results:
            f.write(f"{r.benchmark},{framework},{r.measured_iterations},"
                    f"{r.median_nanos},{r.min_nanos},{r.p99_nanos}\n")

    print(f"[jax harness] wrote {json_path}")
    print(f"[jax harness] wrote {csv_path}")


if __name__ == "__main__":
    main()
