"""§0.4.325 — pin that JAX/PJRT-XLA compile rejects MLIR that contains
`stablehlo.custom_call @<vendor_kernel_name>` ops Tlaloc emits.

Reads one StableHLO MLIR file, attempts to parse + compile via JAX's
PJRT-CUDA backend, and writes a JSON result:

    {
      "compiled": <bool>,
      "stage":    "parse" | "compile" | "ok",
      "error":    "<exception message>" | null
    }

Exit code: 0 on successful compile, 1 on parse/compile failure (the
"JAX can't" outcome the test pins), 2 on environment failures (no
CUDA, no JAX, can't read inputs).

The Tlaloc-side test invokes this twice per run:
  - once with the GB10-target MLIR (containing `@flash_attn_v3`) — expects exit 1
  - once with the CPU-target MLIR (zero custom_calls) — expects exit 0

The first failure IS the evidence: same Kotlin source compiled for two
different `KernelTarget`s produces one MLIR JAX can run and one MLIR
JAX cannot, because the GB10 artifact names a kernel symbol JAX/XLA
doesn't ship.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# GB10 is unified-memory: JAX's default 75% preallocation would claim
# ~90 GB of system RAM for a compile-only check. Must be set before
# `import jax` (first CUDA client init reads it).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _normalise_mlir(mlir_text: str) -> str:
    """Same wrapper Tlaloc's PJRT-XLA bench script uses: rename func to
    `@main` and ensure the surrounding `module { … }` is present."""
    text = mlir_text.strip()
    text = re.sub(r"func\.func\s+@\w+", "func.func @main", text, count=1)
    if not text.lstrip().startswith("module"):
        text = "module @tlaloc_emit {\n" + text + "\n}\n"
    return text


def _try_compile(mlir_text: str) -> dict:
    try:
        import jax
        import jax._src.xla_bridge as xb
        import jaxlib._jax as _jax
        import jaxlib.mlir.ir as ir
        from jaxlib.mlir._mlir_libs._jax_mlir_ext import register_dialects
        import jaxlib.mlir.dialects.stablehlo as stablehlo
    except Exception as e:
        return {"compiled": False, "stage": "import", "error": f"{type(e).__name__}: {e}"}

    if not any(d.platform in ("cuda", "gpu") for d in jax.devices()):
        return {"compiled": False, "stage": "device", "error": "no CUDA/GPU device visible to JAX"}

    try:
        reg = ir.DialectRegistry()
        register_dialects(reg)
        ctx = ir.Context()
        ctx.append_dialect_registry(reg)
        ctx.load_all_available_dialects()
        stablehlo.register_dialect(ctx)
        with ctx, ir.Location.unknown(ctx):
            module = ir.Module.parse(_normalise_mlir(mlir_text))
    except Exception as e:
        return {"compiled": False, "stage": "parse", "error": f"{type(e).__name__}: {e}"}

    try:
        backend = xb.backends()["cuda"]
        devs = backend.local_devices()
        dl = _jax.DeviceList(tuple(devs))
        co = _jax.CompileOptions()
        backend.compile_and_load(module, dl, co)
    except Exception as e:
        return {"compiled": False, "stage": "compile", "error": f"{type(e).__name__}: {e}"}

    return {"compiled": True, "stage": "ok", "error": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    if not args.mlir.exists():
        result = {"compiled": False, "stage": "input", "error": f"missing MLIR file: {args.mlir}"}
        args.output.write_text(json.dumps(result) + "\n")
        return 2

    mlir_text = args.mlir.read_text()
    result = _try_compile(mlir_text)
    args.output.write_text(json.dumps(result) + "\n")

    if result["stage"] in ("import", "device", "input"):
        return 2
    return 0 if result["compiled"] else 1


if __name__ == "__main__":
    sys.exit(main())
