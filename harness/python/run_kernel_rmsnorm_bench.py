"""§0.4.356 — kernel-level rms_norm comparison: JAX/Pallas and Triton
rows against the KPTX kernel's exact workload (256×512 f32, eps 1e-5,
unweighted `y = x·rsqrt(mean(x²)+eps)` — the §0.4.336-claimed form).

Methodology cribbed from pyptx's benchmarks: per-framework warmup +
`block_until_ready`-style sync, median over a fixed iteration budget,
one JSON result per row. The KPTX row itself is timed Kotlin-side
(direct cuLaunchKernel loop, KptxKernelComparisonBenchTest) — this
script contributes the Python-stack rows on the same tensors.

Rows (each skipped gracefully when its stack is unavailable):
  - jax-xla:   jnp expression, XLA-fused (the decompose-path reference).
  - pallas:    jax.experimental.pallas kernel, one program per row.
  - triton:    @triton.jit kernel, one program per row.

Output: JSON list [{"row": ..., "median_us": ..., "iters": ...,
"max_abs_vs_ref": ...}] to --output.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

# GB10 is unified-memory: JAX's default 75% preallocation would pin ~90 GB
# of system RAM per process (§0.4.333 reboot incident). Must be set before
# `import jax` (first CUDA client init reads it).
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# jax 0.10 defaults pallas-GPU to the Mosaic-GPU dialect, which the
# jax-cuda12-plugin wheel on this box does not ship; the Triton-IR path
# (jaxlib gpu_triton) is present and is what we want to measure.
os.environ.setdefault("JAX_PALLAS_USE_MOSAIC_GPU", "0")

ROWS, COLS = 256, 512
EPS = 1e-5
WARMUP = 20
ITERS = 200


def _median_us(fn, sync) -> tuple[float, int]:
    for _ in range(WARMUP):
        fn()
    sync()
    gc.disable()
    try:
        times = []
        for _ in range(ITERS):
            t0 = time.perf_counter_ns()
            fn()
            sync()
            times.append((time.perf_counter_ns() - t0) / 1e3)
        times.sort()
        return times[len(times) // 2], len(times)
    finally:
        gc.enable()


def bench_jax(x_np: np.ndarray, ref: np.ndarray) -> dict | None:
    try:
        import jax
        import jax.numpy as jnp
    except Exception as e:  # noqa: BLE001
        return {"row": "jax-xla", "error": f"{type(e).__name__}: {e}"}
    if not any(d.platform in ("cuda", "gpu") for d in jax.devices()):
        return {"row": "jax-xla", "error": "no GPU visible to JAX"}

    x = jax.device_put(jnp.asarray(x_np))

    @jax.jit
    def rms(v):
        return v * jax.lax.rsqrt(jnp.mean(v * v, axis=-1, keepdims=True) + EPS)

    out = rms(x)
    out.block_until_ready()
    med, n = _median_us(lambda: rms(x), lambda: rms(x).block_until_ready())
    err = float(np.max(np.abs(np.asarray(out) - ref)))
    return {"row": "jax-xla", "median_us": med, "iters": n, "max_abs_vs_ref": err}


def bench_pallas(x_np: np.ndarray, ref: np.ndarray) -> dict | None:
    try:
        import jax
        import jax.numpy as jnp
        from jax.experimental import pallas as pl
    except Exception as e:  # noqa: BLE001
        return {"row": "pallas", "error": f"{type(e).__name__}: {e}"}
    if not any(d.platform in ("cuda", "gpu") for d in jax.devices()):
        return {"row": "pallas", "error": "no GPU visible to JAX"}

    def kernel(x_ref, o_ref):
        row = x_ref[...]
        inv = jax.lax.rsqrt(jnp.mean(row * row) + EPS)
        o_ref[...] = row * inv

    try:
        rms = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((ROWS, COLS), jnp.float32),
            grid=(ROWS,),
            in_specs=[pl.BlockSpec((1, COLS), lambda i: (i, 0))],
            out_specs=pl.BlockSpec((1, COLS), lambda i: (i, 0)),
        )
        rms = jax.jit(rms)
        x = jax.device_put(jnp.asarray(x_np))
        out = rms(x)
        out.block_until_ready()
    except Exception as e:  # noqa: BLE001
        return {"row": "pallas", "error": f"{type(e).__name__}: {e}"}
    med, n = _median_us(lambda: rms(x), lambda: rms(x).block_until_ready())
    err = float(np.max(np.abs(np.asarray(out) - ref)))
    return {"row": "pallas", "median_us": med, "iters": n, "max_abs_vs_ref": err}


def bench_triton(x_np: np.ndarray, ref: np.ndarray) -> dict | None:
    try:
        import torch
        import triton
        import triton.language as tl
    except Exception as e:  # noqa: BLE001
        return {"row": "triton", "error": f"{type(e).__name__}: {e} (torch+triton required)"}
    if not torch.cuda.is_available():
        return {"row": "triton", "error": "torch.cuda unavailable"}

    @triton.jit
    def rms_kernel(x_ptr, o_ptr, n_cols, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
        inv = 1.0 / tl.sqrt(tl.sum(x * x) / n_cols + eps)
        tl.store(o_ptr + row * n_cols + offs, x * inv, mask=mask)

    x = torch.from_numpy(x_np).cuda()
    o = torch.empty_like(x)

    def run():
        rms_kernel[(ROWS,)](x, o, COLS, EPS, BLOCK=COLS)

    try:
        run()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        return {"row": "triton", "error": f"{type(e).__name__}: {e}"}
    med, n = _median_us(run, torch.cuda.synchronize)
    err = float(np.max(np.abs(o.cpu().numpy() - ref)))
    return {"row": "triton", "median_us": med, "iters": n, "max_abs_vs_ref": err}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    x = (rng.standard_normal((ROWS, COLS)) * 0.5).astype(np.float32)
    ref = x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS))

    results = [r for r in (bench_jax(x, ref), bench_pallas(x, ref), bench_triton(x, ref)) if r]
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for r in results:
        print(r)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
