"""Run the Pallas kernels behind the TPU payloads through JAX, as a baseline.

On a TPU VM with ``jax[tpu]==0.10.0`` (the toolchain the payloads were
exported with), this runs each Pallas kernel from ``export_tpu_kernels.py``
on the TPU with ``jax.jit``, compares it with the checked-in numpy reference,
and prints the median time of one call. It is the JAX side of
``PjrtTpuMosaicKernelTest``: the same Mosaic kernel, dispatched by JAX
instead of by Tlaloc's PJRT session.

    python harness/python/run_tpu_kernels_jax.py \\
        --fixtures runtime-pjrt/src/jvmTest/resources/tpu-kernels

``--interpret`` runs the kernels in Pallas's TPU interpret mode on the
default device instead (to check this script on a host without a TPU; the
timings are then meaningless).

Exit 0 when every kernel matches, 1 when one does not, 2 when there is no
TPU and ``--interpret`` was not given.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fixtures", required=True, type=Path)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--interpret", action="store_true")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu
    import export_tpu_kernels as ex

    platform = jax.devices()[0].platform
    if platform != "tpu" and not args.interpret:
        print(f"[jax-tpu] default device is {platform}, not tpu; pass --interpret to check the script")
        return 2
    print(f"[jax-tpu] jax {jax.__version__}, device {jax.devices()[0].device_kind}, interpret={args.interpret}")

    failed = False
    for k in ex.kernel_specs():
        manifest = json.loads((args.fixtures / k["name"] / "manifest.json").read_text())
        inputs = [jnp.asarray(a) for a in ex.make_inputs(k["inputs"])]
        fn = jax.jit(k["call"])
        mode = pltpu.force_tpu_interpret_mode() if args.interpret else contextlib.nullcontext()
        with mode:
            got = fn(*inputs)
            got = got if isinstance(got, (tuple, list)) else [got]
            jax.block_until_ready(got)
            samples = []
            for _ in range(args.iterations):
                t0 = time.perf_counter()
                jax.block_until_ready(fn(*inputs))
                samples.append((time.perf_counter() - t0) * 1e6)
        samples.sort()
        for i, o in enumerate(manifest["outputs"]):
            want = np.frombuffer((args.fixtures / k["name"] / o["file"]).read_bytes(), dtype="<f4")
            have = np.asarray(got[i]).astype(np.float32).reshape(-1)
            diff = float(np.max(np.abs(have - want)))
            # A copy (tolerance 0) must be exact; arithmetic gets 2e-4 of the output scale.
            tol = 0.0 if manifest["tolerance"] == 0 else max(manifest["tolerance"], 2e-4 * float(np.max(np.abs(want))))
            ok = diff <= tol
            failed |= not ok
            print(f"[jax-tpu] {k['name']} result {i}: max|diff| vs numpy {diff:.3g} (tol {tol:.3g}) {'ok' if ok else 'FAIL'}")
        print(f"[jax-tpu] {k['name']} median of {args.iterations} calls: {samples[len(samples) // 2]:.1f} us")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
