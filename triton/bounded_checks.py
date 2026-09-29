#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""Checks of the backend's bounded mode through Triton (docs/design/bounded-dims.md).

Run by verify.sh against the example repository, which holds two bounded programs over a
sequence axis of at most 16 positions, compiled for buckets 4, 8 and 16:

  bounded_mean     the mean of the rows of x0 [len, 4]      -> y0 [4]
  bounded_softmax  a softmax over x0 [len]                  -> y0 [len]

Checks:

  values    every length 1..16 of both models gives the DXIR interpreter's exact-length
            result (examples/reference/bounded_*.json) within 1e-4 of the largest value,
            with the output shape of the real length, over HTTP (KServe v2 JSON) and, when
            tritonclient is installed, over gRPC
  refusals  a length of 17 is refused by name, by the backend ("outside 1..16"); a row
            of 5 values for bounded_mean is refused by name, by Triton itself, which checks
            the config's dims [-1, 4] before the backend sees the request

Exit status 0 only if every check passes. With --perturb each length is compared with the
next length's reference, so the run must fail.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ("bounded_mean", "bounded_softmax")
TOL = 1e-4


def infer_http(url, model, shape, data):
    body = {
        "inputs": [{"name": "x0", "shape": shape, "datatype": "FP32", "data": data}],
        "outputs": [{"name": "y0"}],
    }
    req = urllib.request.Request(
        f"http://{url}/v2/models/{model}/infer", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())["outputs"][0]
            return out["shape"], out["data"], None
    except urllib.error.HTTPError as e:
        return None, None, e.read().decode(errors="replace")


def infer_grpc_factory(url):
    try:
        import numpy as np
        import tritonclient.grpc as grpc
    except Exception as e:  # environment gate
        return None, f"{type(e).__name__}: {e}"
    client = grpc.InferenceServerClient(url)

    def infer(model, shape, data):
        x = grpc.InferInput("x0", shape, "FP32")
        x.set_data_from_numpy(np.asarray(data, dtype=np.float32).reshape(shape))
        try:
            r = client.infer(model, [x], outputs=[grpc.InferRequestedOutput("y0")])
        except Exception as e:
            return None, None, str(e)
        y = r.as_numpy("y0")
        return list(y.shape), [float(v) for v in y.reshape(-1)], None

    return infer, None


def close(want, got):
    scale = max(1.0, max(abs(v) for v in want))
    return len(want) == len(got) and all(abs(a - b) <= TOL * scale for a, b in zip(want, got))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", required=True)
    ap.add_argument("--grpc", required=True)
    ap.add_argument("--perturb", action="store_true")
    args = ap.parse_args()

    transports = [("http", lambda m, s, d: infer_http(args.http, m, s, d))]
    grpc_infer, why = infer_grpc_factory(args.grpc)
    if grpc_infer is None:
        print(f"SKIP grpc: tritonclient is not importable ({why})")
    else:
        transports.append(("grpc", grpc_infer))

    failures = 0
    for model in MODELS:
        ref = json.loads((HERE / "examples" / "reference" / f"{model}.json").read_text())
        cases = ref["cases"]
        for name, infer in transports:
            worst = 0.0
            before = failures
            for i, case in enumerate(cases):
                want = cases[(i + 1) % len(cases)]["y0"] if args.perturb else case["y0"]
                shape, data, err = infer(model, case["x0"]["shape"], case["x0"]["data"])
                if err is not None:
                    print(f"FAIL {model} {name} length {i + 1}: {err}")
                    failures += 1
                    continue
                if shape != want["shape"] or not close(want["data"], data):
                    print(f"FAIL {model} {name} length {i + 1}: shape {shape} data {data[:4]}..., want {want['shape']} {want['data'][:4]}...")
                    failures += 1
                    continue
                scale = max(1.0, max(abs(v) for v in want["data"]))
                worst = max(worst, max(abs(a - b) for a, b in zip(want["data"], data)) / scale)
            if failures == before:
                print(f"ok   {model} {name}: lengths 1..{len(cases)} match the interpreter (largest difference {worst:.2e})")

            over = [0.5] * (17 * (4 if model == "bounded_mean" else 1))
            shape = [17, 4] if model == "bounded_mean" else [17]
            _, _, err = infer(model, shape, over)
            if err is None or "outside 1..16" not in err:
                print(f"FAIL {model} {name}: length 17 not refused by name: {err}")
                failures += 1
            else:
                print(f"ok   {model} {name}: length 17 refused: {err.strip()[:160]}")
        if model == "bounded_mean":
            _, _, err = infer_http(args.http, model, [3, 5], [0.0] * 15)
            if err is None or "Expected [-1,4], got [3,5]" not in err:
                print(f"FAIL {model}: a row of 5 values not refused by name: {err}")
                failures += 1
            else:
                print(f"ok   {model}: a row of 5 values refused: {err.strip()[:160]}")

    print("PASS" if failures == 0 else f"{failures} FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
