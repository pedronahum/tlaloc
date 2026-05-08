"""§0.4.299 — PJRT-XLA-CUDA spike on Tlaloc-emitted StableHLO.

The §0.4.297 / §0.4.298 numbers showed JAX-GPU is 6.6× faster than
Tlaloc-IREE-CUDA on the LlamaDecoder medium training step. The open
question: is the gap in IREE's GPU codegen, or in the IR shape Tlaloc
emits? This spike answers it — feed Tlaloc's *exact* StableHLO MLIR to
the same PJRT-XLA backend JAX uses internally, and time the result.

  - If PJRT-XLA on Tlaloc-MLIR ≈ JAX-GPU's ~670 µs forward, the gap is
    in IREE's codegen (and tuning IREE flags / using IREE's
    kernel_descriptor route can recover most of it).
  - If PJRT-XLA on Tlaloc-MLIR is closer to IREE-CUDA's 3 750 µs, the
    gap is in Tlaloc's IR shape (decomposed primitives, no fusion-friendly
    metadata) and the lever is the IR layer (re-engaging fused
    coarseners, kernel_descriptor for CUDA, etc.).

Inputs come in as `.npy` files (one per param, keyed by name — same
layout the §0.4.296 / §0.4.297 benches use). The forward and backward
MLIRs come in as separate `.mlir` paths; Tlaloc emits them as
`func.func @<sym>` without a `module { … }` wrapper, and uses a
non-`main` symbol name. We wrap and rename to `@main` for the XLA
compiler.

Output JSON:

    {
      "framework": "tlaloc-pjrt-xla-cuda",
      "config": <free-form label>,
      "forward":  {"median_ns": …, "min_ns": …, "p99_ns": …, "n_iterations": …},
      "backward": {"median_ns": …, "min_ns": …, "p99_ns": …, "n_iterations": …}
    }

Toolchain pre-requisite (installed in §0.4.297):
    pip install --upgrade "jax[cuda12]"
"""

from __future__ import annotations

import argparse
import gc
import json
import re
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


def _normalise_mlir(mlir_text: str) -> str:
    """Wrap a function-only emit in `module { … }` and rename the entry to
    `@main` (XLA's default entry-symbol assumption). Tlaloc's emit produces
    `func.func @llama_decoder_layer_loss(...)` at the top level; both
    transformations are local and idempotent."""
    text = mlir_text.strip()
    text = re.sub(r"func\.func\s+@\w+", "func.func @main", text, count=1)
    if not text.lstrip().startswith("module"):
        text = "module @tlaloc_emit {\n" + text + "\n}\n"
    return text


def _percentile(sorted_values: list[int], pct: float) -> int:
    if not sorted_values:
        return 0
    idx = int((len(sorted_values) - 1) * pct)
    return sorted_values[idx]


def _time_loop(thunk, warmup_iters: int, min_time_seconds: float, max_iters: int = 100_000) -> dict:
    for _ in range(warmup_iters):
        thunk()
    # §0.4.313 — disable GC during measurement so Python's generational
    # collector doesn't insert variable-latency pauses into p99 readings.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        times_ns: list[int] = []
        perf = time.perf_counter_ns
        deadline_ns = perf() + int(min_time_seconds * 1_000_000_000)
        while True:
            t0 = perf()
            thunk()
            t1 = perf()
            times_ns.append(t1 - t0)
            if perf() >= deadline_ns or len(times_ns) >= max_iters:
                break
    finally:
        if gc_was_enabled:
            gc.enable()
    times_ns.sort()
    return {
        "n_iterations": len(times_ns),
        "median_ns": times_ns[len(times_ns) // 2],
        "min_ns": times_ns[0],
        "p99_ns": _percentile(times_ns, 0.99),
    }


def _compile_and_load(mlir_text: str, backend, dl, compile_options):
    """Parse + compile a StableHLO MLIR string into a LoadedExecutable."""
    import jaxlib.mlir.ir as ir
    from jaxlib.mlir._mlir_libs._jax_mlir_ext import register_dialects
    import jaxlib.mlir.dialects.stablehlo as stablehlo

    reg = ir.DialectRegistry()
    register_dialects(reg)
    ctx = ir.Context()
    ctx.append_dialect_registry(reg)
    ctx.load_all_available_dialects()
    stablehlo.register_dialect(ctx)
    with ctx, ir.Location.unknown(ctx):
        module = ir.Module.parse(_normalise_mlir(mlir_text))
        return backend.compile_and_load(module, dl, compile_options)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-dir", required=True, type=Path)
    ap.add_argument("--forward-mlir", required=True, type=Path)
    ap.add_argument("--backward-mlir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--config-label", default="medium")
    ap.add_argument("--warmup-iters", type=int, default=5)
    ap.add_argument("--min-time-seconds", type=float, default=1.5)
    args = ap.parse_args()

    import jax
    import jax._src.xla_bridge as xb
    import jaxlib._jax as _jax

    if not any(d.platform in ("cuda", "gpu") for d in jax.devices()):
        print(f"[run_pjrt_xla_llama_bench] no CUDA/GPU device visible to JAX; devices={jax.devices()}", file=sys.stderr)
        return 2

    backend = xb.backends()["cuda"]
    devs = backend.local_devices()
    dl = _jax.DeviceList(tuple(devs))
    co = _jax.CompileOptions()

    arrs = _load_inputs(args.inputs_dir)
    inputs_in_order = [
        jax.device_put(jax.numpy.asarray(arrs[name], dtype=jax.numpy.float32), devs[0])
        for name in _expected_param_names()
    ]

    # Compile both MLIRs (one-shot; outside timing loop).
    fwd_loaded = _compile_and_load(args.forward_mlir.read_text(), backend, dl, co)
    bwd_loaded = _compile_and_load(args.backward_mlir.read_text(), backend, dl, co)

    # Warm up XLA: trigger one execute on each so device buffers are
    # placed and any first-execute-only work is amortised.
    _fwd0 = fwd_loaded.execute(inputs_in_order)
    np.array(_fwd0[0])  # block via host transfer
    _bwd0 = bwd_loaded.execute(inputs_in_order)
    np.array(_bwd0[0])

    # §0.4.313 — resolve sync function once outside the timing loop. The
    # previous `if hasattr(out[0], "block_until_ready")` per call bought a
    # branch + attribute lookup per iter; we know the answer is stable for
    # this PJRT plugin once the first execute has succeeded.
    def _make_sync_fn(sample_out):
        if hasattr(sample_out, "block_until_ready"):
            return lambda out: out[0].block_until_ready()
        return lambda out: np.array(out[0])

    sync_fwd = _make_sync_fn(_fwd0[0])
    sync_bwd = _make_sync_fn(_bwd0[0])

    def forward_thunk():
        out = fwd_loaded.execute(inputs_in_order)
        sync_fwd(out)

    def backward_thunk():
        out = bwd_loaded.execute(inputs_in_order)
        sync_bwd(out)

    forward_stats = _time_loop(forward_thunk, args.warmup_iters, args.min_time_seconds)
    backward_stats = _time_loop(backward_thunk, args.warmup_iters, args.min_time_seconds)

    out = {
        "framework": "tlaloc-pjrt-xla-cuda",
        "config": args.config_label,
        "forward": forward_stats,
        "backward": backward_stats,
    }
    args.output.write_text(json.dumps(out))
    print(
        f"[run_pjrt_xla_llama_bench] forward median={forward_stats['median_ns']/1e6:.3f} ms "
        f"({forward_stats['n_iterations']} iters); "
        f"backward median={backward_stats['median_ns']/1e6:.3f} ms "
        f"({backward_stats['n_iterations']} iters)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
