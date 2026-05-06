"""§0.4.297 — JAX-GPU benchmark for LlamaDecoderPrimal forward + backward.

Mirrors `run_pytorch_llama_bench.py`'s methodology but in JAX, so the JAX
output is apples-to-apples with both the PyTorch-CPU and Tlaloc-IREE-CUDA
rows. JAX's XLA backend compiles to LLVM IR for CPU and SASS / PTX for
NVIDIA — the closest neighbour to Tlaloc's IREE-CUDA path (both go
through the StableHLO/XLA compiler family).

Inputs: `.npy` files under `--inputs-dir` (one per param, keyed by name).

Output JSON (`--output`):

    {
      "framework": "jax-gpu",
      "config": <free-form label>,
      "forward":  {"median_ns": …, "min_ns": …, "p99_ns": …, "n_iterations": …},
      "backward": {"median_ns": …, "min_ns": …, "p99_ns": …, "n_iterations": …}
    }

Methodology:
  - `jax.jit(forward)` compiles the forward; first call eaten for compile.
  - `jax.jit(jax.grad(forward, argnums=tuple(range(13))))` for backward.
  - Per-iter `arr.block_until_ready()` ensures the GPU computation
    completes before the timing-loop measurement closes (JAX dispatches
    asynchronously; without the block, we'd time the dispatch latency,
    not the kernel runtime).
  - Warmup: 5 iterations after the JIT compile is done, discarded.
  - Measured: iterate for `min_time_seconds` budget; record per-iter ns.

Toolchain pre-requisite (installed in §0.4.297):
    pip install --upgrade "jax[cuda12]"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _expected_param_names() -> list[str]:
    return [
        "x_in", "labels", "theta",
        "q_w", "k_w", "v_w", "out_w",
        "gate_w", "up_w", "down_w", "lm_head_w",
        "eps_attn", "eps_mlp",
    ]


def _load_inputs(inputs_dir: Path):
    arrs = {}
    for name in _expected_param_names():
        path = inputs_dir / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        arrs[name] = np.load(path).astype(np.float32)
    return arrs


def _llama_forward_jax(
    x_in, labels, theta,
    q_w, k_w, v_w, out_w,
    gate_w, up_w, down_w,
    lm_head_w,
    eps_attn, eps_mlp,
):
    """Op-for-op LlamaDecoderPrimal.kt forward in jnp ops. Returns scalar loss."""
    import jax.lax
    import jax.nn
    import jax.numpy as jnp

    sq1 = x_in * x_in
    mean1 = jnp.mean(sq1, axis=1, keepdims=True)
    rsq1 = jax.lax.rsqrt(mean1 + eps_attn)
    x_norm_attn = x_in * rsq1

    q = x_norm_attn @ q_w
    k = x_norm_attn @ k_w
    v = x_norm_attn @ v_w

    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    q_rot = (q * cos_t) - (theta * sin_t)

    k_t = k.T
    s = q_rot @ k_t
    p = jax.nn.softmax(s, axis=-1)
    attn_out = p @ v

    attn_proj = attn_out @ out_w
    x_attn = x_in + attn_proj

    sq2 = x_attn * x_attn
    mean2 = jnp.mean(sq2, axis=1, keepdims=True)
    rsq2 = jax.lax.rsqrt(mean2 + eps_mlp)
    x_norm_mlp = x_attn * rsq2

    gate_proj = x_norm_mlp @ gate_w
    up_proj = x_norm_mlp @ up_w
    swiglu = jax.nn.silu(gate_proj) * up_proj

    mlp_out = swiglu @ down_w
    x_out = x_attn + mlp_out

    logits = x_out @ lm_head_w
    probs = jax.nn.softmax(logits, axis=-1)
    logp = jnp.log(probs)
    return (labels * logp).sum()


def _percentile(sorted_values: list[int], pct: float) -> int:
    if not sorted_values:
        return 0
    idx = int((len(sorted_values) - 1) * pct)
    return sorted_values[idx]


def _time_loop(thunk, warmup_iters: int, min_time_seconds: float, max_iters: int = 100_000) -> dict:
    for _ in range(warmup_iters):
        thunk()

    times_ns: list[int] = []
    deadline_ns = time.perf_counter_ns() + int(min_time_seconds * 1_000_000_000)
    while True:
        t0 = time.perf_counter_ns()
        thunk()
        t1 = time.perf_counter_ns()
        times_ns.append(t1 - t0)
        if time.perf_counter_ns() >= deadline_ns or len(times_ns) >= max_iters:
            break

    times_ns.sort()
    return {
        "n_iterations": len(times_ns),
        "median_ns": times_ns[len(times_ns) // 2],
        "min_ns": times_ns[0],
        "p99_ns": _percentile(times_ns, 0.99),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--config-label", default="medium")
    ap.add_argument("--warmup-iters", type=int, default=5)
    ap.add_argument("--min-time-seconds", type=float, default=1.5)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    if not any(d.platform in ("cuda", "gpu") for d in jax.devices()):
        # Fail loudly rather than silently fall back to CPU. The Kotlin
        # orchestrator self-skips when CUDA isn't present, so reaching here
        # means JAX itself can't see the GPU even though the host has one.
        print(
            f"[run_jax_llama_bench] no CUDA device visible to JAX; devices={jax.devices()}",
            file=sys.stderr,
        )
        return 2

    arrs = _load_inputs(args.inputs_dir)

    # Move the inputs onto the first CUDA device. `jax.device_put` returns a
    # `jax.Array` that lives on the GPU; subsequent jitted calls dispatch
    # without staging through the host every iteration.
    cuda_device = next(d for d in jax.devices() if d.platform in ("cuda", "gpu"))
    args_in_order = [
        jax.device_put(jnp.asarray(arrs[name], dtype=jnp.float32), cuda_device)
        for name in _expected_param_names()
    ]

    forward_jit = jax.jit(_llama_forward_jax)
    grad_fn = jax.jit(jax.grad(_llama_forward_jax, argnums=tuple(range(13))))

    # Trigger one-shot JIT compilation outside the timing loop. Both jit
    # caches need warm before measurement; first call materialises the
    # XLA HLO → SASS.
    _ = forward_jit(*args_in_order).block_until_ready()
    _grads = grad_fn(*args_in_order)
    _grads[0].block_until_ready()

    def forward_thunk():
        out = forward_jit(*args_in_order)
        out.block_until_ready()

    forward_stats = _time_loop(
        forward_thunk,
        warmup_iters=args.warmup_iters,
        min_time_seconds=args.min_time_seconds,
    )

    def backward_thunk():
        grads = grad_fn(*args_in_order)
        # Single JIT call → all 13 grads computed together; blocking on
        # the first is sufficient to sync.
        grads[0].block_until_ready()

    backward_stats = _time_loop(
        backward_thunk,
        warmup_iters=args.warmup_iters,
        min_time_seconds=args.min_time_seconds,
    )

    out = {
        "framework": "jax-gpu",
        "config": args.config_label,
        "forward": forward_stats,
        "backward": backward_stats,
    }
    args.output.write_text(json.dumps(out))
    print(
        f"[run_jax_llama_bench] forward median={forward_stats['median_ns']/1e6:.3f} ms "
        f"({forward_stats['n_iterations']} iters); "
        f"backward median={backward_stats['median_ns']/1e6:.3f} ms "
        f"({backward_stats['n_iterations']} iters)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
