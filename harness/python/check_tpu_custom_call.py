"""Check a Tlaloc-emitted StableHLO program that calls a Mosaic TPU kernel.

Runs on a CPU host. Given the StableHLO text Tlaloc emitted for one fixture
(a MOSAIC_KERNEL claimed for a TPU) and the fixture's manifest:

1. parses the program with the StableHLO and TPU dialects registered;
2. finds the `stablehlo.custom_call @tpu_custom_call`, decodes its
   `backend_config` as JSON and checks the body and config against the
   manifest, and its kernel_name, layouts and aliases against the op;
3. parses the base64 body as Mosaic bytecode and deserializes it;
4. converts the program to HLO (the form the TPU compiler receives) and
   checks the custom-call instruction there;
5. for a Pallas fixture: exports the same kernel with `jax.export` and checks
   that JAX's HLO custom-call instruction equals Tlaloc's, once both Mosaic
   bodies have their debug locations stripped (JAX's carries the path of the
   checkout it ran from);
6. for a Kotlin-emitted fixture (`--mosaic-text`): serializes that text again
   and checks it yields the checked-in body byte for byte;
7. with `--reference-mlir`: compiles the program lowered to its reference
   decomposition with XLA's CPU client, runs it on the fixture inputs and
   compares with the numpy outputs.

Writes `{"ok": bool, "checks": [...], "error": str | null}` to --output.
Exit 0 when every check passes, 1 when one fails, 2 when the environment
cannot run the checks (no jax, no TPU dialect).

    JAX_PLATFORMS=cpu python harness/python/check_tpu_custom_call.py \\
        --mlir program.mlir --manifest .../manifest.json --output result.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import traceback
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def normalise(mlir_text: str) -> str:
    text = mlir_text.strip()
    text = re.sub(r"func\.func\s+@\w+", "func.func @main", text, count=1)
    if not text.lstrip().startswith("module"):
        text = "module @tlaloc_emit {\n" + text + "\n}\n"
    return text


def hlo_custom_call(module_text: str, strip_body=None) -> str:
    """The tpu_custom_call instruction of the program's HLO, with the
    instruction and operand names and metadata removed. With [strip_body],
    the Mosaic body in the backend config is replaced by
    `strip_body(body)`, so bodies that differ only in debug locations
    compare equal; everything else is compared as written."""
    import jaxlib._jax as jx
    comp = jx.mlir.mlir_module_to_xla_computation(module_text, use_tuple_args=False, return_tuple=False)
    hlo = comp.as_hlo_text()
    for line in hlo.splitlines():
        if 'custom_call_target="tpu_custom_call"' in line:
            line = re.sub(r"^\s*(ROOT\s+)?\S+ = ", "", line)
            line = re.sub(r"custom-call\([^)]*\)", "custom-call(...)", line)
            line = re.sub(r", metadata=\{[^}]*\}", "", line)
            if strip_body is not None:
                line = re.sub(
                    r'("body": ")([A-Za-z0-9+/=]+)(")',
                    lambda m: m.group(1) + strip_body(m.group(2)) + m.group(3),
                    line,
                )
            return line.strip()
    raise ValueError("no tpu_custom_call in the HLO:\n" + hlo)


def run(args) -> dict:
    checks: list[str] = []
    try:
        import jax  # noqa: F401
        from jaxlib.mlir import ir
        from jaxlib.mlir.dialects import stablehlo
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import export_tpu_kernels as ex
    except Exception as e:  # environment, not a verdict
        return {"ok": False, "stage": "import", "checks": checks, "error": f"{type(e).__name__}: {e}"}

    manifest = json.loads(args.manifest.read_text())
    text = normalise(args.mlir.read_text())

    with ex.mosaic_context() as ctx, ir.Location.unknown():
        stablehlo.register_dialect(ctx)
        module = ir.Module.parse(text)
        if not module.operation.verify():
            raise AssertionError("the StableHLO program does not verify")
        checks.append("program parses and verifies with the stablehlo and tpu dialects")

        calls = []

        def visit(op):
            if op.name == "stablehlo.custom_call":
                calls.append(op)
            return ir.WalkResult.ADVANCE

        module.operation.walk(visit)
        tpu_calls = [c for c in calls if ir.StringAttr(c.attributes["call_target_name"]).value == "tpu_custom_call"]
        if len(tpu_calls) != 1:
            raise AssertionError(f"expected one tpu_custom_call, found {len(tpu_calls)}")
        call = tpu_calls[0]
        cfg_text = ir.StringAttr(call.attributes["backend_config"]).value
        cfg = json.loads(cfg_text)
        ccc = cfg["custom_call_config"]
        if ccc.get("body") != manifest["body_base64"]:
            raise AssertionError("backend_config body differs from the manifest")
        rest = {k: v for k, v in ccc.items() if k != "body"}
        if rest != manifest["custom_call_config"]:
            raise AssertionError(f"custom_call_config {rest} != manifest {manifest['custom_call_config']}")
        if manifest.get("exported_backend_config") is not None and cfg_text != manifest["exported_backend_config"]:
            raise AssertionError("backend_config string differs from the one jax.export wrote")
        checks.append("backend_config decodes as JSON and carries the manifest's body and config")

        name = ir.StringAttr(call.attributes["kernel_name"]).value
        if name != manifest["kernel_name"]:
            raise AssertionError(f"kernel_name {name} != {manifest['kernel_name']}")
        n_in = len(call.operands)
        n_out = len(call.results)
        if len(ir.ArrayAttr(call.attributes["operand_layouts"])) != n_in:
            raise AssertionError("operand_layouts count differs from the operand count")
        if len(ir.ArrayAttr(call.attributes["result_layouts"])) != n_out:
            raise AssertionError("result_layouts count differs from the result count")
        if n_out != len(manifest["outputs"]):
            raise AssertionError(f"{n_out} results, manifest has {len(manifest['outputs'])}")
        aliases = manifest["input_output_aliases"]
        has_alias = "output_operand_aliases" in call.attributes
        if has_alias != bool(aliases) or (has_alias and len(ir.ArrayAttr(call.attributes["output_operand_aliases"])) != len(aliases)):
            raise AssertionError("output_operand_aliases do not match the manifest")
        checks.append(f"kernel_name, {n_in} operand / {n_out} result layouts and {len(aliases)} alias(es) match")

    body = base64.b64decode(manifest["body_base64"])
    version = ex.mosaic_version_of(body)
    if version != manifest["stable_mosaic_version"]:
        raise AssertionError(f"body carries stable_mosaic.version {version}, manifest says {manifest['stable_mosaic_version']}")
    kernel_text = ex.deserialize_body(body)
    checks.append(f"body is Mosaic bytecode at serialization version {version} and deserializes ({len(kernel_text)} chars)")

    def strip(b64: str) -> str:
        return base64.b64encode(ex.strip_debug_info(base64.b64decode(b64))).decode("ascii")

    ours = hlo_custom_call(text, strip)
    if ("output_to_operand_aliasing" in ours) != bool(manifest["input_output_aliases"]):
        raise AssertionError("HLO output_to_operand_aliasing does not match the manifest: " + ours[:400])
    checks.append("program converts to HLO: " + ours[:160].replace(manifest["body_base64"], "<body>") + " ...")

    if manifest["source"] == "pallas":
        import jax
        spec = next(k for k in ex.kernel_specs() if k["name"] == manifest["name"])
        shapes = [jax.ShapeDtypeStruct(tuple(s["shape"]), ex.JNP_DTYPE[s["dtype"]]) for s in spec["inputs"]]
        exported = jax.export.export(jax.jit(spec["call"]), platforms=["tpu"])(*shapes)
        theirs = hlo_custom_call(exported.mlir_module(), strip)
        if ours != theirs:
            raise AssertionError(f"HLO custom-call differs from JAX's:\n tlaloc: {ours[:600]}\n jax:    {theirs[:600]}")
        checks.append("HLO custom-call instruction is identical to the one JAX emits for the same Pallas kernel")

    if args.mosaic_text is not None:
        again = ex.serialize_text(args.mosaic_text.read_text())
        if again != body:
            raise AssertionError("re-serializing the Kotlin-emitted Mosaic text does not reproduce the checked-in body")
        checks.append("Kotlin-emitted Mosaic text re-serializes to the checked-in body byte for byte")

    if args.reference_mlir is not None:
        checks.append(run_reference_on_cpu(args.reference_mlir.read_text(), manifest, args.manifest.parent, ex))

    return {"ok": True, "stage": "ok", "checks": checks, "error": None}


def run_reference_on_cpu(mlir_text: str, manifest: dict, fixture_dir: Path, ex) -> str:
    """Compile the reference-lowered program (no custom call) with XLA's CPU
    client, run it on the fixture inputs and compare with the numpy outputs."""
    import numpy as np
    import jax
    import jax._src.xla_bridge as xb
    import jaxlib._jax as jx

    text = normalise(mlir_text)
    if "custom_call" in text:
        raise AssertionError("the reference program still contains a custom_call")
    backend = xb.get_backend("cpu")
    dev = backend.local_devices()[0]
    exe = backend.compile_and_load(text, jx.DeviceList((dev,)), jx.CompileOptions())
    args = []
    for s in manifest["inputs"]:
        v = ex.gen_values(int(np.prod(s["shape"])), s["seed"], s["scale"]).reshape(s["shape"])
        args.append(jax.device_put(v.astype(ex.NP_DTYPE[s["dtype"]]), dev))
    outs = exe.execute_sharded(args).disassemble_into_single_device_arrays()
    worst = 0.0
    for i, o in enumerate(manifest["outputs"]):
        got = np.asarray(outs[i][0]).astype(np.float32).reshape(-1)
        want = np.frombuffer((fixture_dir / o["file"]).read_bytes(), dtype="<f4")
        d = float(np.max(np.abs(got - want)))
        worst = max(worst, d)
        if d > manifest["tolerance"]:
            raise AssertionError(f"reference program result {i} differs from numpy by {d} > {manifest['tolerance']} on XLA CPU")
    return f"reference program compiles and runs on XLA CPU; max|diff| vs numpy {worst:.3g}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mlir", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--mosaic-text", type=Path, default=None)
    ap.add_argument("--reference-mlir", type=Path, default=None,
                    help="the same program lowered to its reference (no custom call), run on XLA CPU")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    try:
        result = run(args)
    except Exception as e:
        result = {
            "ok": False, "stage": "check", "checks": [],
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if result["stage"] == "import":
        return 2
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
