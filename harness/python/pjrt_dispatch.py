"""§0.4.302 — generic PJRT-XLA dispatcher driven by Tlaloc's `:runtime-pjrt`
Kotlin facade.

Reads a StableHLO MLIR module from `--mlir`, loads ordered .npy inputs from
`--inputs-list` (a text file with one .npy path per line), compiles via PJRT-XLA
(JAX's bundled backend), executes the function named `--function`, writes
ordered .npy outputs to the paths listed in `--outputs-list`.

  python pjrt_dispatch.py \
    --mlir <path/to/module.mlir> \
    --function main \
    --device cuda \
    --inputs-list <path/to/inputs.txt> \
    --outputs-list <path/to/outputs.txt>

Toolchain pre-requisite (installed in §0.4.297):
    pip install --upgrade "jax[cuda12]"

Symmetry note: this is the long-lived companion of the §0.4.299
`run_pjrt_xla_llama_bench.py` spike. The spike was llama-specific and
inlined; this script is dxir-function-agnostic so the Kotlin facade can
dispatch any DxirFunction Tlaloc emits.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


def _normalise_mlir(mlir_text: str) -> str:
    """Wrap a function-only emit in `module { … }` and rename the entry to
    `@main`. Tlaloc's emit produces `func.func @<sym>(...)` at the top level
    without a module wrapper; XLA's compile expects `@main` as the default
    entry symbol."""
    text = mlir_text.strip()
    text = re.sub(r"func\.func\s+@\w+", "func.func @main", text, count=1)
    if not text.lstrip().startswith("module"):
        text = "module @tlaloc_emit {\n" + text + "\n}\n"
    return text


def _compile(mlir_text: str, device: str):
    """Parse + compile a StableHLO MLIR string into a LoadedExecutable."""
    import jax
    import jaxlib.mlir.ir as ir
    from jaxlib.mlir._mlir_libs._jax_mlir_ext import register_dialects
    import jaxlib.mlir.dialects.stablehlo as stablehlo
    import jax._src.xla_bridge as xb
    import jaxlib._jax as _jax

    backend_key = "cuda" if device == "cuda" else "cpu"
    backend = xb.backends()[backend_key]
    devs = backend.local_devices()
    dl = _jax.DeviceList(tuple(devs))
    co = _jax.CompileOptions()

    reg = ir.DialectRegistry()
    register_dialects(reg)
    ctx = ir.Context()
    ctx.append_dialect_registry(reg)
    ctx.load_all_available_dialects()
    stablehlo.register_dialect(ctx)
    with ctx, ir.Location.unknown(ctx):
        module = ir.Module.parse(_normalise_mlir(mlir_text))
        return backend, devs, backend.compile_and_load(module, dl, co)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir", required=True, type=Path)
    ap.add_argument("--function", default="main")  # advisory; we always rename to @main
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    ap.add_argument("--inputs-list", required=True, type=Path)
    ap.add_argument("--outputs-list", required=True, type=Path)
    args = ap.parse_args()

    import jax

    if args.device == "cuda" and not any(d.platform in ("cuda", "gpu") for d in jax.devices()):
        print(
            f"[pjrt_dispatch] requested device=cuda but no CUDA/GPU device visible to JAX; "
            f"devices={jax.devices()}",
            file=sys.stderr,
        )
        return 2

    inputs_paths = [Path(p.strip()) for p in args.inputs_list.read_text().splitlines() if p.strip()]
    outputs_paths = [Path(p.strip()) for p in args.outputs_list.read_text().splitlines() if p.strip()]

    backend, devs, loaded = _compile(args.mlir.read_text(), args.device)

    # Load inputs as JAX arrays on the device.
    inputs = [
        jax.device_put(jax.numpy.asarray(np.load(p), dtype=jax.numpy.float32), devs[0])
        for p in inputs_paths
    ]
    out = loaded.execute(inputs)
    if len(out) != len(outputs_paths):
        raise RuntimeError(
            f"PJRT execute returned {len(out)} outputs but the orchestrator "
            f"requested {len(outputs_paths)} output paths"
        )

    for arr, path in zip(out, outputs_paths):
        # Pull the array back to host and persist as .npy. `np.asarray` on a
        # JAX array materialises through the device→host copy.
        host = np.asarray(arr).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, host)

    print(f"[pjrt_dispatch] wrote {len(outputs_paths)} outputs", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
