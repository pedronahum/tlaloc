"""Export Pallas TPU kernels as Mosaic payloads for Tlaloc's MOSAIC_KERNEL op.

Runs on a CPU host (no TPU needed). For each kernel it:

1. lowers the Pallas kernel with ``jax.export(..., platforms=["tpu"])``;
2. pulls the ``stablehlo.custom_call @tpu_custom_call`` out of the exported
   module and decodes its ``backend_config`` JSON (the ``body`` field is the
   Mosaic module as MLIR bytecode after ``mosaic-serde{serialize=true}``);
3. captures the Mosaic module text Pallas produced before serialization;
4. runs the same kernel in Pallas TPU interpret mode on the CPU and compares
   it to a numpy reference;
5. writes ``<out>/<name>/manifest.json``, ``<name>/mosaic.mlir`` and one
   ``<name>/outN.bin`` (little-endian f32, the numpy reference) per result.

Inputs are not stored. They come from ``gen_values`` below, an integer
formula whose values are exact in f32 and bf16; the Kotlin tests implement
the same formula (``TpuKernelFixtures.genValues``).

``--kmosaic-dir DIR`` additionally serializes the Mosaic modules written by
Tlaloc's Kotlin emitter (``MosaicRmsNorm``; ``DIR/<kernel>.mlir``, written by
``MosaicPayloadFixtureTest``) as ``<kernel>_kmosaic`` fixtures, after checking
that each is the module Pallas lowers for ``<kernel>`` (identical after
``cse``).

Use the jax/jaxlib the payloads should be pinned to; the versions are
recorded in every manifest. Run with ``JAX_PLATFORMS=cpu``.

    JAX_PLATFORMS=cpu python harness/python/export_tpu_kernels.py \\
        --out runtime-pjrt/src/jvmTest/resources/tpu-kernels \\
        --kmosaic-dir runtime-pjrt/build/kmosaic
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import re
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jaxlib  # noqa: E402
import ml_dtypes  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax._src import tpu_custom_call as tcc  # noqa: E402
from jax._src.lib import tpu as tpu_dialect  # noqa: E402
from jax._src.interpreters import mlir as jax_mlir  # noqa: E402
from jaxlib.mlir import ir  # noqa: E402
from jaxlib.mlir.passmanager import PassManager  # noqa: E402

# The Mosaic serialization version the payloads are written at. jax 0.10.0
# exports at 9 when no TPU backend is attached (tpu_custom_call.py
# _FWD_COMPAT_VERSION); the Kotlin-emitted payload is serialized at the same
# version so both kinds of payload make the same demand on libtpu.
MOSAIC_TARGET_VERSION = 9
EPS = 1e-6


def gen_values(n: int, seed: int, scale: float) -> np.ndarray:
    """``((i*37 + seed*101) mod 257 - 128) / 128 * scale`` as f32.

    Every value is k/128 * scale with |k| <= 128, which is exact in f32 and in
    bf16 for a power-of-two scale. Integer arithmetic only, so the Kotlin twin
    produces the same bits."""
    i = np.arange(n, dtype=np.int64)
    k = (i * 37 + seed * 101) % 257 - 128
    return (k.astype(np.float64) / 128.0 * scale).astype(np.float32)


# ---------------------------------------------------------------- kernels


def rmsnorm_kernel(x_ref, w_ref, o_ref):
    x = x_ref[...].astype(jnp.float32)
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    y = x * jax.lax.rsqrt(ms + EPS) * w_ref[...].astype(jnp.float32)
    o_ref[...] = y.astype(o_ref.dtype)


def matmul_kernel(x_ref, w_ref, o_ref):
    o_ref[...] = jnp.dot(
        x_ref[...], w_ref[...],
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )


def norm_swiglu_kernel(x_ref, wn_ref, wg_ref, wu_ref, h_ref, y_ref):
    x = x_ref[...]
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    h = x * jax.lax.rsqrt(ms + EPS) * wn_ref[...]
    h_ref[...] = h
    g = jnp.dot(h, wg_ref[...], preferred_element_type=jnp.float32,
                precision=jax.lax.Precision.HIGHEST)
    u = jnp.dot(h, wu_ref[...], preferred_element_type=jnp.float32,
                precision=jax.lax.Precision.HIGHEST)
    y_ref[...] = g * jax.nn.sigmoid(g) * u


KV_OFFSET = 8


def kv_update_kernel(cache_hbm, new_ref, out_hbm, sem):
    # cache_hbm is aliased to out_hbm: the kernel DMAs `new` into rows
    # [KV_OFFSET, KV_OFFSET + new_rows) of the cache in place and leaves the
    # other rows untouched.
    del cache_hbm
    rows = new_ref.shape[0]
    copy = pltpu.make_async_copy(new_ref, out_hbm.at[pl.ds(KV_OFFSET, rows)], sem)
    copy.start()
    copy.wait()


# ------------------------------------------------------ numpy references


def np_rmsnorm(x, w):
    x64 = x.astype(np.float64)
    ms = np.mean(x64 * x64, axis=-1, keepdims=True)
    return (x64 / np.sqrt(ms + EPS) * w.astype(np.float64)).astype(np.float32)


def np_silu(v):
    return v / (1.0 + np.exp(-v))


def to_bf16(a):
    return a.astype(ml_dtypes.bfloat16)


# --------------------------------------------------------------- specs

F32, BF16 = "f32", "bf16"
NP_DTYPE = {F32: np.float32, BF16: ml_dtypes.bfloat16}
JNP_DTYPE = {F32: jnp.float32, BF16: jnp.bfloat16}


def spec(name, dtype, shape, seed, scale):
    return {"name": name, "dtype": dtype, "shape": list(shape), "seed": seed, "scale": scale}


def kernel_specs():
    rows, hidden, inter = 16, 256, 256

    def rmsnorm_call(dtype):
        def f(x, w):
            return pl.pallas_call(
                rmsnorm_kernel,
                out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                name="tlaloc_rmsnorm",
            )(x, w)
        return f

    def rmsnorm_ref(dtype):
        def r(x, w):
            y = np_rmsnorm(x.astype(np.float32), w.astype(np.float32))
            if dtype == BF16:
                y = to_bf16(y).astype(np.float32)
            return [y]
        return r

    def matmul_call(x, w):
        return pl.pallas_call(
            matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((x.shape[0], w.shape[1]), jnp.float32),
            name="tlaloc_matmul",
        )(x, w)

    def matmul_ref(x, w):
        return [(x.astype(np.float64) @ w.astype(np.float64)).astype(np.float32)]

    nrows, nh, ni = 8, 128, 256

    def swiglu_call(x, wn, wg, wu):
        return pl.pallas_call(
            norm_swiglu_kernel,
            out_shape=(
                jax.ShapeDtypeStruct((x.shape[0], x.shape[1]), jnp.float32),
                jax.ShapeDtypeStruct((x.shape[0], wg.shape[1]), jnp.float32),
            ),
            name="tlaloc_norm_swiglu",
        )(x, wn, wg, wu)

    def swiglu_ref(x, wn, wg, wu):
        h = np_rmsnorm(x, wn).astype(np.float64)
        g = h @ wg.astype(np.float64)
        u = h @ wu.astype(np.float64)
        return [h.astype(np.float32), (np_silu(g) * u).astype(np.float32)]

    kv_rows, kv_width, kv_new = 16, 128, 4

    def kv_call(cache, new):
        return pl.pallas_call(
            kv_update_kernel,
            out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
            in_specs=[
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pltpu.VMEM),
            ],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            scratch_shapes=[pltpu.SemaphoreType.DMA(())],
            input_output_aliases={0: 0},
            name="tlaloc_kv_update",
        )(cache, new)

    def kv_ref(cache, new):
        out = cache.copy()
        out[KV_OFFSET:KV_OFFSET + new.shape[0]] = new
        return [out]

    return [
        {
            "name": "rmsnorm_f32",
            "description": "RMSNorm over the last axis, f32, eps 1e-6, weight [1, hidden]",
            "call": rmsnorm_call(F32),
            "ref": rmsnorm_ref(F32),
            "inputs": [spec("x", F32, (rows, hidden), 1, 2.0), spec("w", F32, (1, hidden), 2, 1.0)],
            "outputs": [{"dtype": F32, "shape": [rows, hidden]}],
            "aliases": {},
            "tolerance": 1e-5,
        },
        {
            "name": "rmsnorm_bf16",
            "description": "RMSNorm over the last axis, bf16 in/out, f32 accumulation inside the kernel",
            "call": rmsnorm_call(BF16),
            "ref": rmsnorm_ref(BF16),
            "inputs": [spec("x", BF16, (rows, hidden), 3, 2.0), spec("w", BF16, (1, hidden), 4, 1.0)],
            "outputs": [{"dtype": BF16, "shape": [rows, hidden]}],
            "aliases": {},
            "tolerance": 1.6e-2,
        },
        {
            "name": "matmul_f32",
            "description": "[8, 256] x [256, 128] f32 matmul, HIGHEST precision",
            "call": matmul_call,
            "ref": matmul_ref,
            "inputs": [spec("x", F32, (8, hidden), 5, 1.0), spec("w", F32, (hidden, 128), 6, 0.125)],
            "outputs": [{"dtype": F32, "shape": [8, 128]}],
            "aliases": {},
            "tolerance": 1e-4,
        },
        {
            "name": "norm_swiglu_f32",
            "description": "fused RMSNorm -> (h @ Wg, h @ Wu) -> silu(g) * u; two results (h, y)",
            "call": swiglu_call,
            "ref": swiglu_ref,
            "inputs": [
                spec("x", F32, (nrows, nh), 7, 2.0),
                spec("wn", F32, (1, nh), 8, 1.0),
                spec("wg", F32, (nh, ni), 9, 0.125),
                spec("wu", F32, (nh, ni), 10, 0.125),
            ],
            "outputs": [{"dtype": F32, "shape": [nrows, nh]}, {"dtype": F32, "shape": [nrows, ni]}],
            "aliases": {},
            "tolerance": 1e-4,
        },
        {
            "name": "kv_update_inplace_f32",
            "description": "writes new[4, 128] into rows 8..11 of cache[16, 128] by DMA; cache aliased to the result",
            "call": kv_call,
            "ref": kv_ref,
            "inputs": [spec("cache", F32, (kv_rows, kv_width), 11, 1.0), spec("new", F32, (kv_new, kv_width), 12, 4.0)],
            "outputs": [{"dtype": F32, "shape": [kv_rows, kv_width]}],
            "aliases": {"0": 0},
            "tolerance": 0.0,
        },
    ]


# ------------------------------------------------------------- export


_captured_mosaic: list[str] = []
_orig_lower = tcc._lower_mosaic_module_to_asm


def _spy_lower(module, **kw):
    _captured_mosaic.append(str(module))
    return _orig_lower(module, **kw)


tcc._lower_mosaic_module_to_asm = _spy_lower


def mosaic_context() -> ir.Context:
    ctx = jax_mlir.make_ir_context()
    tpu_dialect.register_dialect(ctx)
    return ctx


def deserialize_body(body: bytes) -> str:
    """Mosaic bytecode -> Mosaic MLIR text (the inverse of mosaic-serde)."""
    with mosaic_context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        mod = ir.Module.parse(body)
        PassManager.parse("builtin.module(mosaic-serde{serialize=false})").run(mod.operation)
        return str(mod)


def serialize_text(text: str) -> bytes:
    """Mosaic MLIR text -> the bytecode a tpu_custom_call body carries."""
    with mosaic_context() as ctx, ir.Location.unknown():
        mod = ir.Module.parse(text)
        if not mod.operation.verify():
            raise ValueError("Mosaic module does not verify")
        ctx.allow_unregistered_dialects = True
        PassManager.parse(
            f"builtin.module(mosaic-serde{{serialize=true target-version={MOSAIC_TARGET_VERSION}}})"
        ).run(mod.operation)
        import io
        buf = io.BytesIO()
        mod.operation.write_bytecode(buf, desired_version=0)
        return buf.getvalue()


def canonical_text(text: str) -> str:
    """The module after parsing, verifying, and `cse`, with the kernel
    function renamed to @kernel and the serialization version attribute
    (added by deserialization) dropped. CSE merges Pallas's repeated index constants
    and drops its dead `vector.load` of the output ref, so two modules that
    compute the same thing the same way print identically."""
    with mosaic_context() as ctx, ir.Location.unknown():
        mod = ir.Module.parse(text)
        if not mod.operation.verify():
            raise ValueError("Mosaic module does not verify")
        PassManager.parse("builtin.module(cse)").run(mod.operation)
        printed = str(mod)
    printed = re.sub(r"^module attributes \{stable_mosaic\.version = \d+ : i64\} \{", "module {", printed)
    return re.sub(r"func\.func @[\w$.-]+", "func.func @kernel", printed)


def op_count(canonical: str) -> int:
    return sum(1 for line in canonical.splitlines() if re.match(r"\s+(%\S+ = )?[a-z_]+\.[a-z_]+", line))


def mosaic_version_of(body: bytes) -> int | None:
    with mosaic_context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        mod = ir.Module.parse(body)
        attr = mod.operation.attributes["stable_mosaic.version"] if "stable_mosaic.version" in mod.operation.attributes else None
        return int(ir.IntegerAttr(attr).value) if attr is not None else None


def extract_custom_call(module_text: str) -> dict:
    m = re.search(
        r'stablehlo\.custom_call @tpu_custom_call\((.*?)\) \{(.*)\} : \((.*?)\) -> (.*?) loc',
        module_text,
    )
    if m is None:
        raise ValueError("no tpu_custom_call in exported module")
    attrs = m.group(2)
    cfg = re.search(r'backend_config = "((?:[^"\\]|\\.)*)"', attrs).group(1)
    backend_config = re.sub(r"\\([0-9A-Fa-f]{2})", lambda h: chr(int(h.group(1), 16)), cfg)
    kernel_name = re.search(r'kernel_name = "([^"]*)"', attrs).group(1)
    return {
        "backend_config": backend_config,
        "kernel_name": kernel_name,
        "has_side_effect": "has_side_effect = true" in attrs,
        "output_operand_aliases": re.findall(r"#stablehlo\.output_operand_alias<[^>]*>", attrs),
    }


def versions() -> dict:
    meta = {}
    try:
        from importlib.metadata import requires
        req = [r for r in (requires("jax") or []) if r.startswith("libtpu")]
        meta["libtpu_requirement"] = req[0].split(";")[0].strip() if req else None
    except Exception:  # pragma: no cover - metadata is best effort
        meta["libtpu_requirement"] = None
    return {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "python": platform.python_version(),
        "mosaic_target_version": MOSAIC_TARGET_VERSION,
        **meta,
    }


def write_fixture(out_dir: Path, name: str, manifest: dict, outputs: list[np.ndarray], mosaic_text: str):
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    for i, arr in enumerate(outputs):
        (d / f"out{i}.bin").write_bytes(np.ascontiguousarray(arr, dtype="<f4").tobytes())
    (d / "mosaic.mlir").write_text(mosaic_text)
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def make_inputs(specs):
    host = []
    for s in specs:
        n = int(np.prod(s["shape"]))
        v = gen_values(n, s["seed"], s["scale"]).reshape(s["shape"])
        host.append(v.astype(NP_DTYPE[s["dtype"]]))
    return host


def export_one(k: dict, out_dir: Path) -> dict:
    _captured_mosaic.clear()
    host_inputs = make_inputs(k["inputs"])
    shapes = [jax.ShapeDtypeStruct(tuple(s["shape"]), JNP_DTYPE[s["dtype"]]) for s in k["inputs"]]
    exported = jax.export.export(jax.jit(k["call"]), platforms=["tpu"])(*shapes)
    module_text = exported.mlir_module()
    cc = extract_custom_call(module_text)
    cfg = json.loads(cc["backend_config"])
    body_b64 = cfg["custom_call_config"].pop("body")
    body = base64.b64decode(body_b64)
    if not _captured_mosaic:
        raise RuntimeError(f"{k['name']}: Mosaic module was not captured")
    mosaic_text = _captured_mosaic[-1]

    # Reference, and the Pallas kernel itself in TPU interpret mode on CPU.
    ref = [np.asarray(r, dtype=np.float32) for r in k["ref"](*[a.copy() for a in host_inputs])]
    interp = jax.jit(k["call"])
    with pltpu.force_tpu_interpret_mode():
        got = interp(*[jnp.asarray(a) for a in host_inputs])
    got = got if isinstance(got, (tuple, list)) else [got]
    interp_diff = max(
        float(np.max(np.abs(np.asarray(g, dtype=np.float32) - r))) for g, r in zip(got, ref)
    )
    if interp_diff > max(k["tolerance"], 1e-6) * 4:
        raise RuntimeError(f"{k['name']}: interpret-mode kernel differs from numpy by {interp_diff}")

    manifest = {
        "name": k["name"],
        "description": k["description"],
        "source": "pallas",
        "kernel_name": cc["kernel_name"],
        "call_target": "tpu_custom_call",
        "body_base64": body_b64,
        "custom_call_config": cfg["custom_call_config"],
        "backend_config_extra": {kk: v for kk, v in cfg.items() if kk != "custom_call_config"},
        "exported_backend_config": cc["backend_config"],
        "has_side_effect": cc["has_side_effect"],
        "input_output_aliases": {str(a): b for a, b in k["aliases"].items()},
        "inputs": k["inputs"],
        "outputs": [
            dict(o, file=f"out{i}.bin") for i, o in enumerate(k["outputs"])
        ],
        "eps": EPS,
        "tolerance": k["tolerance"],
        "interpret_mode_max_abs_diff": interp_diff,
        "stable_mosaic_version": mosaic_version_of(body),
        "body_bytes": len(body),
        "versions": versions(),
    }
    write_fixture(out_dir, k["name"], manifest, ref, mosaic_text)
    print(f"[export] {k['name']}: body {len(body)} B, interpret-mode max|diff| {interp_diff:.3g}")
    return manifest


def export_kmosaic(text_path: Path, out_dir: Path, pallas_name: str, pallas_manifest: dict):
    """Serialize Kotlin-emitted Mosaic text as the `<pallas_name>_kmosaic`
    fixture, after checking that it is the same module Pallas lowers."""
    text = text_path.read_text()
    pallas_text = (out_dir / pallas_name / "mosaic.mlir").read_text()
    ours, theirs = canonical_text(text), canonical_text(pallas_text)
    if ours != theirs:
        import difflib
        diff = "\n".join(difflib.unified_diff(theirs.splitlines(), ours.splitlines(), "pallas", "kotlin", lineterm=""))
        raise RuntimeError(f"Kotlin-emitted Mosaic differs from Pallas's {pallas_name}:\n{diff}")
    body = serialize_text(text)
    if canonical_text(deserialize_body(body)) != ours:
        raise RuntimeError("serialized Kotlin payload does not deserialize to the same module")
    name = f"{pallas_name}_kmosaic"
    manifest = dict(pallas_manifest)
    manifest.update(
        name=name,
        description=pallas_manifest["description"] + "; Mosaic MLIR written by Tlaloc's Kotlin emitter, serialized by jaxlib",
        source="kmosaic",
        kernel_name=f"tlaloc_{pallas_name}_kmosaic",
        body_base64=base64.b64encode(body).decode("ascii"),
        exported_backend_config=None,
        interpret_mode_max_abs_diff=None,
        stable_mosaic_version=mosaic_version_of(body),
        body_bytes=len(body),
        canonical_ops=op_count(ours),
    )
    outputs = [np.frombuffer((out_dir / pallas_name / "out0.bin").read_bytes(), dtype="<f4")]
    write_fixture(out_dir, name, manifest, outputs, text)
    print(f"[export] {name}: body {len(body)} B, {op_count(ours)} ops, identical to Pallas after cse")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--kmosaic-dir", type=Path, default=None,
        help="directory of Kotlin-emitted Mosaic text files named <pallas kernel>.mlir",
    )
    ap.add_argument("--only", default=None, help="comma-separated kernel names")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    manifests = {}
    for k in kernel_specs():
        if only and k["name"] not in only:
            continue
        manifests[k["name"]] = export_one(k, args.out)
    if args.kmosaic_dir is not None:
        for text_path in sorted(args.kmosaic_dir.glob("*.mlir")):
            base = text_path.stem
            manifest = manifests.get(base) or json.loads((args.out / base / "manifest.json").read_text())
            export_kmosaic(text_path, args.out, base, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
