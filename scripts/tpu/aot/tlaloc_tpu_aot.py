"""Compile recorded Tlaloc programs with XLA:TPU, with no TPU attached.

Run by ``scripts/tpu/aot-check.sh`` inside the amd64 image built from the
Dockerfile beside this file (libtpu is published for x86_64 only). Each
``<name>.mlir`` in PROGRAMS, with its ``<name>.options`` as
``TpuProgramRecorder`` writes them, is compiled for a compile-only TPU
topology: SINGLE for one-device programs, MULTI for the others. The program
is normalised as ``PjrtFfm`` normalises it (entry function renamed to
``@main``, wrapped in a module).

    python -I tlaloc_tpu_aot.py PROGRAMS OUT [--single v6e:1x1] [--multi v6e:2x2]

OUT receives ``summary.tsv``, the optimized HLO of each program that
compiled (``hlo/``) and the error of each that did not (``errors/``). The
exit status is 1 when any program failed to compile.
"""

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
from jax._src import compiler
from jax._src.interpreters import mlir
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jax.experimental import topologies

COLLECTIVES = ("all-reduce", "all-gather", "reduce-scatter", "all-to-all", "collective-permute")
BLANK = [""] * 4  # the memory columns of a program that did not compile


def read_options(path):
    opts = {"replicas": "1", "partitions": "1", "spmd": "false", "shardy": "false"}
    if os.path.exists(path):
        for line in open(path):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                opts[k] = v
    return opts


def normalise(text, name):
    text = re.sub(r"func\.func\s+@\w+", "func.func @main", text.strip(), count=1)
    if not text.startswith("module"):
        text = "module @tlaloc_emit {\n" + text + "\n}\n"
    module = ir.Module.parse(text)
    module.operation.attributes["sym_name"] = ir.StringAttr.get(name)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("programs")
    ap.add_argument("out")
    ap.add_argument("--single", default="v6e:1x1")
    ap.add_argument("--multi", default="v6e:2x2")
    args = ap.parse_args()

    topology = {
        "single": topologies.get_topology_desc(topology_name=args.single, platform="tpu"),
        "multi": topologies.get_topology_desc(topology_name=args.multi, platform="tpu"),
    }
    os.makedirs(os.path.join(args.out, "hlo"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "errors"), exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.programs, "*.mlir")))
    print(f"{len(paths)} programs; one device: {args.single}, more: {args.multi} "
          f"({len(topology['multi'].devices)} devices)", flush=True)

    rows, failed = [], 0
    for path in paths:
        name = os.path.basename(path)[:-5]
        o = read_options(path[:-5] + ".options")
        replicas, partitions = int(o["replicas"]), int(o["partitions"])
        n = replicas * partitions
        devices = topology["single" if n == 1 else "multi"].devices
        if n > len(devices):
            rows.append((name, n, "skip", 0.0, f"needs {n} devices", "", BLANK))
            print(f"SKIP {name}: needs {n} devices", flush=True)
            continue
        options = compiler.get_compile_options(
            num_replicas=replicas,
            num_partitions=partitions,
            device_assignment=np.arange(n).reshape(replicas, partitions),
        )
        options.executable_build_options.use_spmd_partitioning = o["spmd"] == "true"
        options.executable_build_options.use_shardy_partitioner = o["shardy"] == "true"
        t0 = time.time()
        try:
            with mlir.make_ir_context():
                module = normalise(open(path).read(), name)
                exe = compiler.backend_compile_and_load(
                    devices[0].client, module, xc.DeviceList(tuple(devices[:n])), options, [])
            seconds = time.time() - t0
            hlo = exe.hlo_modules()[0].to_string()
            open(os.path.join(args.out, "hlo", name + ".txt"), "w").write(hlo)
            stats = exe.get_compiled_memory_stats()
            # An output aliased to a donated input reuses that input's buffer:
            # alias_size_in_bytes is counted in both the arguments and the outputs.
            parts = [stats.argument_size_in_bytes, stats.output_size_in_bytes,
                     stats.alias_size_in_bytes, stats.temp_size_in_bytes,
                     stats.generated_code_size_in_bytes]
            arg, out, alias, temp, code = (b / 2**20 for b in parts)
            mib = arg + out - alias + temp + code
            counts = {c: len(re.findall(rf"\b{c}(-start)?\(", hlo)) for c in COLLECTIVES}
            coll = " ".join(f"{c}={k}" for c, k in counts.items() if k)
            rows.append((name, n, "ok", seconds, f"{mib:.1f} MiB", coll,
                         [f"{v:.1f}" for v in (arg, out, alias, temp)]))
            print(f"OK   {name} ({n} dev) {seconds:.1f} s, {mib:.1f} MiB "
                  f"(arguments {arg:.1f}, outputs {out:.1f}, aliased {alias:.1f}, temporaries {temp:.1f}) {coll}",
                  flush=True)
        except Exception as e:  # XLA reports compile failures as JaxRuntimeError
            seconds = time.time() - t0
            failed += 1
            first = (str(e).strip().splitlines() or [type(e).__name__])[0]
            open(os.path.join(args.out, "errors", name + ".txt"), "w").write(str(e))
            rows.append((name, n, "FAIL", seconds, first[:300], "", BLANK))
            print(f"FAIL {name} ({n} dev) {seconds:.1f} s: {first[:300]}", flush=True)

    with open(os.path.join(args.out, "summary.tsv"), "w") as f:
        f.write("program\tdevices\tstatus\tseconds\tmemory_or_error\tcollectives"
                "\targument_mib\toutput_mib\taliased_mib\ttemp_mib\n")
        for r in rows:
            f.write("\t".join([r[0], str(r[1]), r[2], f"{r[3]:.1f}", r[4], r[5], *r[6]]) + "\n")
    ok = sum(1 for r in rows if r[2] == "ok")
    skipped = sum(1 for r in rows if r[2] == "skip")
    print(f"{ok} compiled, {failed} failed, {skipped} skipped", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
