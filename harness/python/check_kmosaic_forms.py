"""Check the gridded Mosaic matmul that Tlaloc's Kotlin emitter writes (MosaicTiledMatmul).

Runs on a CPU host (no TPU). Reads ``DIR/tiled_matmul.mlir`` and
``DIR/tiled_matmul.serialized.mlir`` (written by
``TpuKernelFixtureCpuTest.kotlinTiledMatmulIsWrittenInBothForms``) and checks:

1. the plain module is the module Pallas lowers for the same kernel, grid and block
   specs (equal after ``cse``, debug info dropped);
2. the serialized module is what ``mosaic-serde{serialize=true}`` makes of the plain one
   (both parsed and printed by MLIR, so spelling differences do not count).

    JAX_PLATFORMS=cpu python harness/python/check_kmosaic_forms.py runtime-pjrt/build/kmosaic
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jaxlib.mlir import ir  # noqa: E402
from jaxlib.mlir.passmanager import PassManager  # noqa: E402

import export_tpu_kernels as etk  # noqa: E402

M, N, K, TM, TN, TK = 256, 256, 512, 128, 128, 256


def kernel(x_ref, w_ref, o_ref, acc_ref):
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def pallas_module() -> str:
    def f(x, w):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
            grid=(M // TM, N // TN, K // TK),
            in_specs=[pl.BlockSpec((TM, TK), lambda i, j, k: (i, k)), pl.BlockSpec((TK, TN), lambda i, j, k: (k, j))],
            out_specs=pl.BlockSpec((TM, TN), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((TM, TN), jnp.float32)],
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
        )(x, w)

    etk._captured_mosaic.clear()
    jax.export.export(jax.jit(f), platforms=["tpu"])(
        jnp.zeros((M, K), jnp.bfloat16), jnp.zeros((K, N), jnp.bfloat16)
    )
    return etk._captured_mosaic[-1]


def printed(text: str, unregistered: bool = False) -> str:
    with etk.mosaic_context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = unregistered
        return str(ir.Module.parse(text))


def serde(text: str) -> str:
    with etk.mosaic_context() as ctx, ir.Location.unknown():
        mod = ir.Module.parse(text)
        ctx.allow_unregistered_dialects = True
        PassManager.parse(
            f"builtin.module(mosaic-serde{{serialize=true target-version={etk.MOSAIC_TARGET_VERSION}}},strip-debuginfo)"
        ).run(mod.operation)
        return str(mod)


def main() -> int:
    d = Path(sys.argv[1])
    plain = (d / "tiled_matmul.mlir").read_text()
    serialized = (d / "tiled_matmul.serialized.mlir").read_text()
    ok = True
    ours, theirs = etk.canonical_text(plain), etk.canonical_text(pallas_module())
    if ours != theirs:
        ok = False
        print("FAIL: the Kotlin tiled matmul is not Pallas's module after cse")
        import difflib
        for line in difflib.unified_diff(theirs.splitlines(), ours.splitlines(), "pallas", "kotlin", lineterm=""):
            print(line)
    else:
        print(f"ok: plain module equals Pallas's after cse ({etk.op_count(ours)} ops)")
    want, got = serde(plain), printed(serialized, unregistered=True)
    if want != got:
        ok = False
        print("FAIL: the Kotlin serialized module is not mosaic-serde's output")
        import difflib
        for line in difflib.unified_diff(want.splitlines(), got.splitlines(), "serde", "kotlin", lineterm=""):
            print(line)
    else:
        print(f"ok: serialized module equals mosaic-serde's at version {etk.MOSAIC_TARGET_VERSION}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
