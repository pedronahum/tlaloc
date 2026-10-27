"""Check the Mosaic kernels Tlaloc's Kotlin kernel DSL writes (MosaicKernels) against Pallas.

Runs on a CPU host (no TPU). For each kernel in ``KERNELS``, reads ``DIR/<name>.mlir`` and
``DIR/<name>.serialized.mlir`` (written by ``MosaicKernelsTest`` in ``:stablehlo``) and checks:

1. the plain module is the module Pallas lowers for the twin below (equal after ``cse``);
2. the serialized module is what ``mosaic-serde{serialize=true}`` makes of the plain one;
3. the twin, run in Pallas's TPU interpret mode, matches a numpy reference.

1 and 2 make the Kotlin module the twin's module; 3 checks what that module computes. The
Mosaic compiler itself is left to a TPU.

    JAX_PLATFORMS=cpu python harness/python/check_mosaic_kernels.py stablehlo/build/mosaic-kernels
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402

import check_kmosaic_forms as forms  # noqa: E402
import export_tpu_kernels as etk  # noqa: E402

PARALLEL, ARBITRARY = "parallel", "arbitrary"


def params(*sem):
    return pltpu.CompilerParams(dimension_semantics=sem)


# ---- row softmax: rows in blocks of TR, f32 arithmetic on a bf16 input ------------------------
SM_R, SM_C, SM_TR = 64, 256, 16


def softmax_kernel(x_ref, o_ref):
    x = x_ref[...].astype(jnp.float32)
    m = jnp.max(x, axis=1, keepdims=True)
    e = jnp.exp(x - m)
    s = jnp.sum(e, axis=1, keepdims=True)
    o_ref[...] = (e / s).astype(o_ref.dtype)


def softmax_call(x, interpret=None):
    return pl.pallas_call(
        softmax_kernel,
        out_shape=jax.ShapeDtypeStruct((SM_R, SM_C), jnp.bfloat16),
        grid=(SM_R // SM_TR,),
        in_specs=[pl.BlockSpec((SM_TR, SM_C), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((SM_TR, SM_C), lambda i: (i, 0)),
        compiler_params=params(PARALLEL),
        interpret=interpret,
    )(x)


def softmax_ref(x):
    x = x.astype(np.float32)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ---- RMSNorm: rows in blocks of TR, a [1, H] weight -------------------------------------------
RN_R, RN_H, RN_TR, RN_EPS = 64, 512, 16, 1e-6


def rmsnorm_kernel(x_ref, w_ref, o_ref):
    x = x_ref[...].astype(jnp.float32)
    ms = jnp.sum(x * x, axis=1, keepdims=True) / RN_H
    y = x * jax.lax.rsqrt(ms + RN_EPS) * w_ref[...].astype(jnp.float32)
    o_ref[...] = y.astype(o_ref.dtype)


def rmsnorm_call(x, w, interpret=None):
    return pl.pallas_call(
        rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct((RN_R, RN_H), jnp.bfloat16),
        grid=(RN_R // RN_TR,),
        in_specs=[pl.BlockSpec((RN_TR, RN_H), lambda i: (i, 0)), pl.BlockSpec((1, RN_H), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((RN_TR, RN_H), lambda i: (i, 0)),
        compiler_params=params(PARALLEL),
        interpret=interpret,
    )(x, w)


def rmsnorm_ref(x, w):
    x = x.astype(np.float32)
    return x / np.sqrt((x * x).mean(axis=1, keepdims=True) + RN_EPS) * w.astype(np.float32)


# ---- causal flash attention, one head: S tokens, D = 128, blocks of 128 -----------------------
FA_S, FA_D, FA_T = 512, 128, 128
FA_SCALE = 1.0 / np.sqrt(FA_D)
NEG_INF = -float("inf")


def causal_mask(s):
    row = jax.lax.broadcasted_iota(jnp.int32, (FA_T, FA_T), 0) + pl.program_id(0) * FA_T
    col = jax.lax.broadcasted_iota(jnp.int32, (FA_T, FA_T), 1) + pl.program_id(1) * FA_T
    return jnp.where(col <= row, s, NEG_INF)


def flash_fwd_kernel(q_ref, k_ref, v_ref, o_ref, lse_ref, m_sc, l_sc, acc_sc):
    @pl.when(pl.program_id(1) == 0)
    def _():
        m_sc[...] = jnp.full_like(m_sc, NEG_INF)
        l_sc[...] = jnp.zeros_like(l_sc)
        acc_sc[...] = jnp.zeros_like(acc_sc)

    s = jax.lax.dot_general(q_ref[...], k_ref[...], (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    s = causal_mask(s * FA_SCALE)
    m_prev = m_sc[...]
    m_new = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
    p = jnp.exp(s - m_new)
    alpha = jnp.exp(m_prev - m_new)
    l_sc[...] = alpha * l_sc[...] + jnp.sum(p, axis=1, keepdims=True)
    acc_sc[...] = acc_sc[...] * alpha + jnp.dot(p.astype(v_ref.dtype), v_ref[...], preferred_element_type=jnp.float32)
    m_sc[...] = m_new

    @pl.when(pl.program_id(1) == FA_S // FA_T - 1)
    def _():
        o_ref[...] = (acc_sc[...] / l_sc[...]).astype(o_ref.dtype)
        lse_ref[...] = m_sc[...] + jnp.log(l_sc[...])


def flash_fwd_call(q, k, v, interpret=None):
    n = FA_S // FA_T
    return pl.pallas_call(
        flash_fwd_kernel,
        out_shape=(jax.ShapeDtypeStruct((FA_S, FA_D), jnp.bfloat16), jax.ShapeDtypeStruct((FA_S, FA_T), jnp.float32)),
        grid=(n, n),
        in_specs=[
            pl.BlockSpec((FA_T, FA_D), lambda i, j: (i, 0)),
            pl.BlockSpec((FA_T, FA_D), lambda i, j: (j, 0)),
            pl.BlockSpec((FA_T, FA_D), lambda i, j: (j, 0)),
        ],
        out_specs=[pl.BlockSpec((FA_T, FA_D), lambda i, j: (i, 0)), pl.BlockSpec((FA_T, FA_T), lambda i, j: (i, 0))],
        scratch_shapes=[pltpu.VMEM((FA_T, FA_T), jnp.float32), pltpu.VMEM((FA_T, FA_T), jnp.float32), pltpu.VMEM((FA_T, FA_D), jnp.float32)],
        compiler_params=params(PARALLEL, ARBITRARY),
        interpret=interpret,
    )(q, k, v)


def attention_ref(q, k, v):
    q, k, v = (a.astype(np.float64) for a in (q, k, v))
    s = q @ k.T * FA_SCALE
    s = np.where(np.tril(np.ones((FA_S, FA_S), bool)), s, -np.inf)
    m = s.max(axis=1, keepdims=True)
    p = np.exp(s - m)
    l = p.sum(axis=1, keepdims=True)
    return p / l @ v, (m + np.log(l))


def flash_fwd_ref(q, k, v):
    o, lse = attention_ref(q, k, v)
    return o, np.broadcast_to(lse, (FA_S, FA_T))


# ---- the backward: dK and dV over the query blocks, dQ over the key blocks --------------------
def flash_dkdv_kernel(q_ref, k_ref, v_ref, do_ref, lse_ref, di_ref, dk_ref, dv_ref, dk_sc, dv_sc):
    # grid (key block i, query block j); the mask compares query rows with key columns.
    @pl.when(pl.program_id(1) == 0)
    def _():
        dk_sc[...] = jnp.zeros_like(dk_sc)
        dv_sc[...] = jnp.zeros_like(dv_sc)

    q = q_ref[...]
    do = do_ref[...]
    s = jax.lax.dot_general(q, k_ref[...], (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32) * FA_SCALE
    row = jax.lax.broadcasted_iota(jnp.int32, (FA_T, FA_T), 0) + pl.program_id(1) * FA_T
    col = jax.lax.broadcasted_iota(jnp.int32, (FA_T, FA_T), 1) + pl.program_id(0) * FA_T
    s = jnp.where(col <= row, s, NEG_INF)
    p = jnp.exp(s - lse_ref[...])
    dv_sc[...] += jax.lax.dot_general(p.astype(do.dtype), do, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    dp = jax.lax.dot_general(do, v_ref[...], (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    ds = p * (dp - di_ref[...]) * FA_SCALE
    dk_sc[...] += jax.lax.dot_general(ds.astype(q.dtype), q, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(1) == FA_S // FA_T - 1)
    def _():
        dk_ref[...] = dk_sc[...].astype(dk_ref.dtype)
        dv_ref[...] = dv_sc[...].astype(dv_ref.dtype)


def flash_dq_kernel(q_ref, k_ref, v_ref, do_ref, lse_ref, di_ref, dq_ref, dq_sc):
    # grid (query block i, key block j).
    @pl.when(pl.program_id(1) == 0)
    def _():
        dq_sc[...] = jnp.zeros_like(dq_sc)

    k = k_ref[...]
    s = causal_mask(jax.lax.dot_general(q_ref[...], k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32) * FA_SCALE)
    p = jnp.exp(s - lse_ref[...])
    dp = jax.lax.dot_general(do_ref[...], v_ref[...], (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    ds = p * (dp - di_ref[...]) * FA_SCALE
    dq_sc[...] += jnp.dot(ds.astype(k.dtype), k, preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(1) == FA_S // FA_T - 1)
    def _():
        dq_ref[...] = dq_sc[...].astype(dq_ref.dtype)


def bwd_specs(q_block, kv_block):
    return [
        pl.BlockSpec((FA_T, FA_D), q_block),
        pl.BlockSpec((FA_T, FA_D), kv_block),
        pl.BlockSpec((FA_T, FA_D), kv_block),
        pl.BlockSpec((FA_T, FA_D), q_block),
        pl.BlockSpec((FA_T, FA_T), q_block),
        pl.BlockSpec((FA_T, FA_T), q_block),
    ]


def flash_dkdv_call(q, k, v, do, lse, di, interpret=None):
    n = FA_S // FA_T
    q_block, kv_block = (lambda i, j: (j, 0)), (lambda i, j: (i, 0))
    return pl.pallas_call(
        flash_dkdv_kernel,
        out_shape=(jax.ShapeDtypeStruct((FA_S, FA_D), jnp.bfloat16), jax.ShapeDtypeStruct((FA_S, FA_D), jnp.bfloat16)),
        grid=(n, n),
        in_specs=bwd_specs(q_block, kv_block),
        out_specs=[pl.BlockSpec((FA_T, FA_D), kv_block), pl.BlockSpec((FA_T, FA_D), kv_block)],
        scratch_shapes=[pltpu.VMEM((FA_T, FA_D), jnp.float32), pltpu.VMEM((FA_T, FA_D), jnp.float32)],
        compiler_params=params(PARALLEL, ARBITRARY),
        interpret=interpret,
    )(q, k, v, do, lse, di)


def flash_dq_call(q, k, v, do, lse, di, interpret=None):
    n = FA_S // FA_T
    q_block, kv_block = (lambda i, j: (i, 0)), (lambda i, j: (j, 0))
    return pl.pallas_call(
        flash_dq_kernel,
        out_shape=jax.ShapeDtypeStruct((FA_S, FA_D), jnp.bfloat16),
        grid=(n, n),
        in_specs=bwd_specs(q_block, kv_block),
        out_specs=pl.BlockSpec((FA_T, FA_D), q_block),
        scratch_shapes=[pltpu.VMEM((FA_T, FA_D), jnp.float32)],
        compiler_params=params(PARALLEL, ARBITRARY),
        interpret=interpret,
    )(q, k, v, do, lse, di)


def backward_ref(q, k, v, do):
    q, k, v, do = (a.astype(np.float64) for a in (q, k, v, do))
    s = q @ k.T * FA_SCALE
    s = np.where(np.tril(np.ones((FA_S, FA_S), bool)), s, -np.inf)
    p = np.exp(s - s.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    o = p @ v
    dv = p.T @ do
    dp = do @ v.T
    ds = p * (dp - (do * o).sum(axis=1, keepdims=True)) * FA_SCALE
    return ds @ k, ds.T @ q, dv


def bf16(a):
    return jnp.asarray(a, jnp.bfloat16)


def rand(shape, seed, scale=1.0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32) * scale


def flash_inputs():
    q, k, v, do = (rand((FA_S, FA_D), s) for s in (1, 2, 3, 4))
    o, lse = attention_ref(bf16(q).astype(np.float32), bf16(k).astype(np.float32), bf16(v).astype(np.float32))
    di = (bf16(do).astype(np.float64) * o).sum(axis=1, keepdims=True)
    lanes = (FA_S, FA_T)
    return q, k, v, do, np.broadcast_to(lse, lanes).astype(np.float32), np.broadcast_to(di, lanes).astype(np.float32)


def run_softmax(interpret):
    x = rand((SM_R, SM_C), 5)
    return [np.asarray(softmax_call(bf16(x), interpret), np.float32)], [softmax_ref(np.asarray(bf16(x), np.float32))], 1e-2


def run_rmsnorm(interpret):
    x, w = rand((RN_R, RN_H), 6), rand((1, RN_H), 7)
    got = rmsnorm_call(bf16(x), bf16(w), interpret)
    return [np.asarray(got, np.float32)], [rmsnorm_ref(np.asarray(bf16(x), np.float32), np.asarray(bf16(w), np.float32))], 3e-2


def run_flash_fwd(interpret):
    q, k, v = (rand((FA_S, FA_D), s) for s in (1, 2, 3))
    o, lse = flash_fwd_call(bf16(q), bf16(k), bf16(v), interpret)
    want = flash_fwd_ref(*(np.asarray(bf16(a), np.float32) for a in (q, k, v)))
    return [np.asarray(o, np.float32), np.asarray(lse)], list(want), 2e-2


def run_flash_bwd(interpret):
    q, k, v, do, lse, di = flash_inputs()
    dk, dv = flash_dkdv_call(bf16(q), bf16(k), bf16(v), bf16(do), lse, di, interpret)
    dq = flash_dq_call(bf16(q), bf16(k), bf16(v), bf16(do), lse, di, interpret)
    want = backward_ref(*(np.asarray(bf16(a), np.float32) for a in (q, k, v, do)))
    return [np.asarray(x, np.float32) for x in (dq, dk, dv)], list(want), 5e-2


S = jax.ShapeDtypeStruct
B16 = jnp.bfloat16
F32 = jnp.float32
FA_IN = S((FA_S, FA_D), B16)
FA_STAT = S((FA_S, FA_T), F32)

# name -> (pallas call, abstract inputs, value check)
KERNELS = {
    "softmax_rows": (softmax_call, [S((SM_R, SM_C), B16)], run_softmax),
    "rmsnorm_rows": (rmsnorm_call, [S((RN_R, RN_H), B16), S((1, RN_H), B16)], run_rmsnorm),
    "flash_fwd": (flash_fwd_call, [FA_IN, FA_IN, FA_IN], run_flash_fwd),
    "flash_bwd_dkdv": (flash_dkdv_call, [FA_IN, FA_IN, FA_IN, FA_IN, FA_STAT, FA_STAT], run_flash_bwd),
    "flash_bwd_dq": (flash_dq_call, [FA_IN, FA_IN, FA_IN, FA_IN, FA_STAT, FA_STAT], None),
}


def pallas_module(call, abstract) -> str:
    etk._captured_mosaic.clear()
    jax.export.export(jax.jit(call), platforms=["tpu"])(*abstract)
    return etk._captured_mosaic[-1]


def main() -> int:
    d = Path(sys.argv[1])
    only = set(sys.argv[2:])
    ok = True
    import difflib

    for name, (call, abstract, check) in KERNELS.items():
        if only and name not in only:
            continue
        plain = (d / f"{name}.mlir").read_text()
        serialized = (d / f"{name}.serialized.mlir").read_text()
        ours, theirs = etk.canonical_text(plain), etk.canonical_text(pallas_module(call, abstract))
        if ours != theirs:
            ok = False
            print(f"FAIL {name}: the Kotlin module is not Pallas's after cse")
            for line in difflib.unified_diff(theirs.splitlines(), ours.splitlines(), "pallas", "kotlin", lineterm=""):
                print(line)
        else:
            print(f"ok   {name}: equals Pallas's module after cse ({etk.op_count(ours)} ops)")
        want, got = forms.serde(plain), forms.printed(serialized, unregistered=True)
        if want != got:
            ok = False
            print(f"FAIL {name}: the serialized module is not mosaic-serde's output")
            for line in difflib.unified_diff(want.splitlines(), got.splitlines(), "serde", "kotlin", lineterm=""):
                print(line)
        else:
            print(f"ok   {name}: serialized form equals mosaic-serde's")
        if check is not None:
            outs, refs, tol = check(pltpu.InterpretParams())
            for i, (o, r) in enumerate(zip(outs, refs)):
                err = float(np.max(np.abs(o - r)))
                if not err <= tol:
                    ok = False
                    print(f"FAIL {name}: interpret-mode result {i} is {err:.3g} from numpy (bound {tol})")
                else:
                    print(f"ok   {name}: interpret-mode result {i} within {err:.3g} of numpy (bound {tol})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
