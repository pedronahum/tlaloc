"""§0.4.296 — PyTorch-CPU benchmark for LlamaDecoderPrimal forward + backward.

Mirrors the math in `benchmarks/src/jvmTest/.../LlamaDecoderPrimal.kt` 1:1
(same op-for-op forward as `run_pytorch_llama.py`, same backward as
`run_pytorch_llama_grad.py`), but adds a per-iteration timing loop
suitable for cross-framework comparison.

Inputs: `.npy` files under `--inputs-dir` (one per param, keyed by name —
same files the §0.4.289+ tests dump via NpyWriter).

Output JSON (`--output`):

    {
      "framework": "pytorch-cpu",
      "config": <free-form label>,
      "forward":  {"median_ns": …, "min_ns": …, "p99_ns": …, "n_iterations": …},
      "backward": {"median_ns": …, "min_ns": …, "p99_ns": …, "n_iterations": …}
    }

The Kotlin-side `LlamaDecoderPytorchBenchTest` orchestrator parses this
and dumps a `HeadToHeadResult`-shaped row alongside the existing
Tlaloc-IREE rows.

Methodology:
  - `torch.set_num_threads(os.cpu_count())` — let the CPU stretch out.
  - Warmup: 5 iterations discarded (warm caches, JIT-ish kernel paths).
  - Measured: iterate for at least `min_time_seconds` wall time. Each
    iteration's `time.perf_counter_ns()` delta is recorded; stats are
    median / min / p99 over all measurements.
  - Forward and backward are timed in separate loops (same trade-off as
    the IREE-side `iree-benchmark-module` two-pass methodology). A
    fused value-and-grad would be cheaper but isn't apples-to-apples
    with Tlaloc's separately-compiled forward / backward.

Toolchain pre-requisite (already installed in §0.4.289):
    pip install --index-url https://download.pytorch.org/whl/cpu torch
"""

from __future__ import annotations

import argparse
import json
import os
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
        arrs[name] = np.load(path)
    return arrs


def _llama_forward(tensors):
    """Op-for-op LlamaDecoderPrimal.kt forward. Returns the scalar loss."""
    import torch
    import torch.nn.functional as F

    x_in = tensors["x_in"]
    labels = tensors["labels"]
    theta = tensors["theta"]
    q_w, k_w, v_w, out_w = tensors["q_w"], tensors["k_w"], tensors["v_w"], tensors["out_w"]
    gate_w, up_w, down_w = tensors["gate_w"], tensors["up_w"], tensors["down_w"]
    lm_head_w = tensors["lm_head_w"]
    eps_attn, eps_mlp = tensors["eps_attn"], tensors["eps_mlp"]

    sq1 = x_in * x_in
    mean1 = sq1.mean(dim=1, keepdim=True)
    rsq1 = torch.rsqrt(mean1 + eps_attn)
    x_norm_attn = x_in * rsq1

    q = x_norm_attn @ q_w
    k = x_norm_attn @ k_w
    v = x_norm_attn @ v_w

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    q_rot = (q * cos_t) - (theta * sin_t)

    k_t = k.transpose(0, 1)
    s = q_rot @ k_t
    p = torch.softmax(s, dim=-1)
    attn_out = p @ v

    attn_proj = attn_out @ out_w
    x_attn = x_in + attn_proj

    sq2 = x_attn * x_attn
    mean2 = sq2.mean(dim=1, keepdim=True)
    rsq2 = torch.rsqrt(mean2 + eps_mlp)
    x_norm_mlp = x_attn * rsq2

    gate_proj = x_norm_mlp @ gate_w
    up_proj = x_norm_mlp @ up_w
    swiglu = F.silu(gate_proj) * up_proj

    mlp_out = swiglu @ down_w
    x_out = x_attn + mlp_out

    logits = x_out @ lm_head_w
    probs = torch.softmax(logits, dim=-1)
    logp = torch.log(probs)
    return (labels * logp).sum()


def _percentile(sorted_values: list[int], pct: float) -> int:
    if not sorted_values:
        return 0
    idx = int((len(sorted_values) - 1) * pct)
    return sorted_values[idx]


def _time_loop(thunk, warmup_iters: int, min_time_seconds: float, max_iters: int = 100_000) -> dict:
    """Warm up `warmup_iters` times, then run until either `min_time_seconds`
    have elapsed or `max_iters` have run. Record per-iteration ns and return
    median / min / p99 / count."""
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

    import torch
    torch.set_num_threads(os.cpu_count() or 1)
    torch.set_grad_enabled(True)

    arrs = _load_inputs(args.inputs_dir)

    # Forward: requires_grad=False on all leaves so the autograd machinery
    # doesn't tax forward-only timing.
    forward_tensors_no_grad = {
        n: torch.from_numpy(a.astype(np.float32)) for n, a in arrs.items()
    }
    def forward_thunk():
        with torch.no_grad():
            loss = _llama_forward(forward_tensors_no_grad)
            _ = loss.item()  # force materialisation
    forward_stats = _time_loop(
        forward_thunk,
        warmup_iters=args.warmup_iters,
        min_time_seconds=args.min_time_seconds,
    )

    # Backward: requires_grad=True on every param; recompute grads each iter
    # via torch.autograd.grad (functional, no .grad-state accumulation across
    # iterations).
    grad_tensors = {
        n: torch.from_numpy(a.astype(np.float32)).requires_grad_(True)
        for n, a in arrs.items()
    }
    grad_param_names = _expected_param_names()
    def backward_thunk():
        loss = _llama_forward(grad_tensors)
        grads = torch.autograd.grad(
            loss,
            [grad_tensors[n] for n in grad_param_names],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        # Touch the first element of each grad to force materialisation —
        # mirrors `iree-benchmark-module`'s end-of-iteration sync.
        for g in grads:
            if g is not None:
                _ = g.detach()[..., 0].sum().item()
    backward_stats = _time_loop(
        backward_thunk,
        warmup_iters=args.warmup_iters,
        min_time_seconds=args.min_time_seconds,
    )

    out = {
        "framework": "pytorch-cpu",
        "config": args.config_label,
        "forward": forward_stats,
        "backward": backward_stats,
    }
    args.output.write_text(json.dumps(out))
    print(
        f"[run_pytorch_llama_bench] forward median={forward_stats['median_ns']/1e6:.3f} ms "
        f"({forward_stats['n_iterations']} iters); "
        f"backward median={backward_stats['median_ns']/1e6:.3f} ms "
        f"({backward_stats['n_iterations']} iters)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
