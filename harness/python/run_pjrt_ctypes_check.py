"""§0.4.475 (H6a) — the certification driver for the ctypes PJRT binding.

Run by `PjrtCtypesBindingTest` (`:runtime-pjrt:jvmTest`) as a subprocess, in
the pattern every cross-language lane in this repo uses: the JVM writes a job
file, this script runs it in a Python interpreter, the JVM compares what comes
back against its own oracle.

TWO MODES.

  `--layouts <out.json>`   No plugin, no GPU, no device. Reports the ctypes
                           struct sizes and the padding-sensitive field
                           offsets, so the JVM can assert that this binding
                           and `PjrtFfm.kt` lay the PJRT structs out
                           identically. This is the half that certifies on a
                           machine with no accelerator at all.

  `--run <job.json> <out.json>`
                           Loads the plugin, creates a client, compiles each
                           lane's StableHLO, stages its inputs, executes, and
                           reports the outputs as **raw bit patterns** (f32
                           and bf16) or exact integers (i32) — never decimals,
                           so nothing this claim rests on passes through a
                           float-printing round trip.

THE PIN THAT IS THE POINT OF THE SLICE.

Before anything else happens, a `sys.meta_path` finder is installed that
**raises on any attempt to import jax, jaxlib, torch or numpy**. Then the
whole PJRT path runs underneath it. If any of this — the plugin load, the
struct marshalling, the compile, the execute — secretly needed a framework,
the import would raise and the lane would fail loudly rather than quietly
succeeding for the wrong reason.

And because a guard that never fires is an untested guard (the §0.4.474
lesson, applied to this slice), the run finishes by *deliberately* trying to
import jax and reporting whether it was stopped. The JVM asserts both halves:
nothing forbidden was loaded, AND the thing that would have caught it works.
"""

from __future__ import annotations

import json
import sys

# §0.4.476 (H6b) moved the guard itself into `import_guard.py`, because that
# slice wanted a second copy and two copies of a safety net drift. The field
# names this script reports are unchanged, so the JVM assertions are too.
import import_guard

FORBIDDEN_ROOTS = import_guard.FORBIDDEN_ROOTS

_GUARD = import_guard.install(
    why="§0.4.475 (H6a) binds PJRT through ctypes precisely so the serving runtime "
        "needs no framework; an import here means the dependency-free claim is false.",
)

# Imported under the guard, deliberately: stdlib only, and if that ever stops
# being true this line is where it is found out.
import tlaloc_pjrt as P  # noqa: E402


def _guard_report() -> dict:
    return import_guard.report(_GUARD)


def _layouts(out_path: str) -> None:
    payload = {
        "sizes": P.LAYOUT_SIZES,
        "offsets": P.LAYOUT_OFFSETS,
        "compile_options_proto_hex": P.COMPILE_OPTIONS_PROTO_BYTES.hex(),
        "buffer_type_codes": {
            "s32": P.PJRT_BUFFER_TYPE_S32,
            "f32": P.PJRT_BUFFER_TYPE_F32,
            "f64": P.PJRT_BUFFER_TYPE_F64,
            "bf16": P.PJRT_BUFFER_TYPE_BF16,
        },
        "api_offsets": {
            name[len("OFFSET_"):]: getattr(P, name)
            for name in dir(P) if name.startswith("OFFSET_")
        },
        "create_options": _create_options_report(),
        "guard": _guard_report(),
    }
    _write(out_path, payload)


def _create_options_report() -> dict:
    """The §0.4.333 NamedValue array, marshalled and read back out of its own
    bytes — no plugin involved. The JVM compares this against what
    `PjrtFfm.marshalCreateOptions` writes, which is how we know the two
    bindings hand the plugin the same allocator configuration."""
    import ctypes

    opts = P.PjrtClientOptions(memory_fraction=0.5, preallocate=False)
    array, _keep = P.marshal_create_options(opts)
    entries = []
    for i in range(opts.named_value_count):
        nv = array[i]
        entries.append({
            "struct_size": nv.struct_size,
            "name": ctypes.string_at(nv.name, nv.name_size).decode(),
            "name_size": nv.name_size,
            "type": nv.type,
            "value_size": nv.value_size,
            "float_value": nv.value.float_value if nv.type == P.PJRT_NAMED_VALUE_TYPE_FLOAT else None,
            "bool_value": bool(nv.value.bool_value) if nv.type == P.PJRT_NAMED_VALUE_TYPE_BOOL else None,
        })
    return {
        "count": opts.named_value_count,
        "entries": entries,
        "cuda_refuses_none": _refuses_none(),
        "tpu_gets_none": P.PjrtClientOptions.for_platform("tpu") is None,
    }


def _refuses_none() -> bool:
    """§0.4.333's rule, as a checkable fact: a CUDA client with no
    create_options must be refused before the plugin is ever called."""
    try:
        P.PjrtApi(None, 0, "<none>").create_client(platform="cuda", options=None)
        return False
    except ValueError as exc:
        return "§0.4.333" in str(exc) or "preallocate" in str(exc)


def _run(job_path: str, out_path: str) -> None:
    with open(job_path) as fh:
        job = json.load(fh)

    api = P.PjrtApi.load(job["plugin"])
    platform = job.get("platform", "cuda")
    results = {"plugin": api.plugin_path, "lanes": []}
    with api.create_client(platform=platform) as client:
        results["platform_name"] = client.platform_name()
        devices = client.addressable_devices()
        results["num_devices"] = len(devices)
        device = devices[0]
        for lane in job["lanes"]:
            results["lanes"].append(_run_lane(client, device, lane))
    results["guard"] = _guard_report()
    _write(out_path, results)


def _run_lane(client, device, lane: dict) -> dict:
    with open(lane["mlir_path"]) as fh:
        mlir = fh.read()
    staged = []
    out = {"name": lane["name"], "outputs": []}
    try:
        for spec in lane["inputs"]:
            dtype, dims, values = spec["dtype"], spec["dims"], spec["values"]
            if dtype == "f32":
                # Values cross as raw bit patterns: the JVM wrote them with
                # toRawBits and we rebuild the same f32, so no decimal
                # printing sits between the oracle and the device.
                buf = client.buffer_from_host_f32(device, [P.bits_to_f32(v) for v in values], dims)
            elif dtype == "i32":
                buf = client.buffer_from_host_i32(device, values, dims)
            elif dtype == "bf16":
                buf = client.buffer_from_host_bf16(device, values, dims)
            else:
                raise ValueError(f"lane {lane['name']}: unsupported input dtype {dtype}")
            staged.append(buf)
            out.setdefault("staged_device_bytes", []).append(buf.device_size_in_bytes())
        with client.compile(mlir) as exe:
            out["num_outputs"] = exe.num_outputs()
            produced = exe.execute(staged, device)
            try:
                for spec, buf in zip(lane["outputs"], produced):
                    dtype, n = spec["dtype"], spec["count"]
                    if dtype == "f32":
                        out["outputs"].append({"dtype": "f32", "bits": buf.to_f32_bits(n)})
                    elif dtype == "i32":
                        out["outputs"].append({"dtype": "i32", "values": buf.to_i32(n)})
                    elif dtype == "bf16":
                        out["outputs"].append({"dtype": "bf16", "patterns": buf.to_bf16_patterns(n)})
                    else:
                        raise ValueError(f"lane {lane['name']}: unsupported output dtype {dtype}")
            finally:
                for b in produced:
                    b.close()
    finally:
        for b in staged:
            b.close()
    return out


def _write(path: str, payload: dict) -> None:
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)


def main(argv) -> int:
    if len(argv) >= 3 and argv[1] == "--layouts":
        _layouts(argv[2])
        return 0
    if len(argv) >= 4 and argv[1] == "--run":
        _run(argv[2], argv[3])
        return 0
    sys.stderr.write(
        "usage: run_pjrt_ctypes_check.py --layouts <out.json>\n"
        "       run_pjrt_ctypes_check.py --run <job.json> <out.json>\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
