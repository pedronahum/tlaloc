"""Run a StableHLO module on N virtual XLA:CPU devices (jaxlib), as one SPMD program.

    python run_on_cpu_devices.py --mlir module.mlir --request request.json --output out.json

Compiles the module the way `CompileOptions.spmd` compiles for PJRT: num_replicas 1,
num_partitions N, device assignment 0..N-1, SPMD partitioning and Shardy on. The
host platform is split into N devices (`--xla_force_host_platform_device_count`).

request.json:
  {"devices": N, "mode": "per-device" | "global", "result_dims": [[...], ...],
   "args": [{"dtype": "f32" | "f64" | "i32", "dims": [...],
             "values": [...]            # mode "global": the whole array
             "per_device": [[...], ...] # mode "per-device": one buffer per device
            }, ...]}

- "per-device": each device gets its own buffer of `dims` for each argument (a program
  whose collectives combine them: all_reduce, all_gather, reduce_scatter, inside or
  outside a manual computation). Output: every device's buffer of every result.
- "global": each argument is one array, placed by the shardings the compiled program
  asks for (a module with sdy shardings). Output: every result gathered whole, at its
  `result_dims`.

out.json: {"jax": version, "results": [[[...] per device] ...]} or
          {"jax": version, "results": [[...] ...]} for "global".
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir", required=True)
    ap.add_argument("--request", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    request = json.load(open(a.request))
    n = int(request["devices"])
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={n}"
    os.environ["JAX_PLATFORMS"] = "cpu"

    import numpy as np
    import jax
    from jax._src import xla_bridge as xb
    from jax._src.lib import _jax as jx

    dtypes = {"f32": np.float32, "f64": np.float64, "i32": np.int32}
    text = open(a.mlir).read()
    backend = xb.get_backend("cpu")
    devs = backend.local_devices()[:n]
    if len(devs) != n:
        sys.exit(f"asked for {n} devices, the host platform has {len(devs)}")

    opts = jx.CompileOptions()
    opts.num_partitions = n
    opts.num_replicas = 1
    eb = opts.executable_build_options
    eb.num_partitions = n
    eb.num_replicas = 1
    eb.use_spmd_partitioning = True
    eb.use_shardy_partitioner = True
    eb.device_assignment = jx.DeviceAssignment.create(np.arange(n).reshape(1, n))
    exe = backend.compile_and_load(text, jx.DeviceList(tuple(devs)), opts)

    mesh = jax.sharding.Mesh(np.array(devs), ("d",))
    args = []
    if request["mode"] == "per-device":
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        for spec in request["args"]:
            dt = dtypes[spec["dtype"]]
            dims = tuple(spec["dims"])
            bufs = [jax.device_put(np.asarray(v, dt).reshape(dims), d) for v, d in zip(spec["per_device"], devs)]
            args.append(jax.make_array_from_single_device_arrays(dims, replicated, bufs))
        outs = exe.execute_sharded(args).disassemble_into_single_device_arrays()
        results = [[np.asarray(o).ravel().tolist() for o in per_result] for per_result in outs]
    elif request["mode"] == "global":
        from jax._src.sharding_impls import GSPMDSharding
        for spec, hlo in zip(request["args"], exe.get_parameter_shardings()):
            value = np.asarray(spec["values"], dtypes[spec["dtype"]]).reshape(tuple(spec["dims"]))
            args.append(jax.device_put(value, GSPMDSharding(tuple(devs), hlo)))
        outs = exe.execute_sharded(args).disassemble_into_single_device_arrays()
        results = []
        for bufs, hlo, dims in zip(outs, exe.get_output_shardings(), request["result_dims"]):
            whole = jax.make_array_from_single_device_arrays(tuple(dims), GSPMDSharding(tuple(devs), hlo), list(bufs))
            results.append(np.asarray(whole).ravel().tolist())
    else:
        sys.exit(f"unknown mode {request['mode']}")
    json.dump({"jax": jax.__version__, "results": results}, open(a.output, "w"))


if __name__ == "__main__":
    main()
