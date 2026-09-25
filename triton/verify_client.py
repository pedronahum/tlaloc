#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""Inference checks against a running Triton server with the tlaloc backend.

Run by verify.sh. Three transports:

  json   KServe v2 JSON over HTTP, Python standard library only
  http   tritonclient.http (binary tensor data; needed for BF16)
  grpc   tritonclient.grpc

Expected values:

  matmul_sumsq  the DXIR interpreter's result from examples/reference/, and
                the closed form: f(A) = 54, df/dA = [[7, 11], [9, 13]] for
                A = [[1, 2], [3, 4]]. Compared bit for bit.
  dtypes        x*x + x, computed here. Every value is exactly representable
                in its dtype, so this is compared bit for bit too.

--perturb changes one expected GRAD value; the run must then fail. That is
the negative control: it shows a wrong answer is caught.

Exit status 0 when every check passes, 1 otherwise.
"""

import argparse
import json
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path

FAILURES = []


def check(label, got, want):
    """Bitwise equality of two float/int lists (NaN-safe, -0.0 != 0.0)."""
    ok = len(got) == len(want) and all(
        (struct.pack("<d", float(g)) == struct.pack("<d", float(w))) for g, w in zip(got, want)
    )
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: got {list(got)} want {list(want)}")
    if not ok:
        FAILURES.append(label)


def expect_error(label, fn, needle):
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - any client error type counts
        msg = str(e)
        ok = needle in msg
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: error '{msg.strip()[:160]}'")
        if not ok:
            FAILURES.append(label)
        return
    print(f"  FAIL {label}: the request succeeded, an error was expected")
    FAILURES.append(label)


def bf16_round(x):
    """float -> nearest bf16 value (round to nearest even), as a float."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def buckets(http, np, httpclient):
    """One model, two compiled lengths; a third length is refused by the backend."""
    print(f"buckets  {http}")
    client = httpclient.InferenceServerClient(url=http)
    for n in (4, 8):
        x = [float(i) - 1.5 for i in range(n)]
        i = httpclient.InferInput("X", [n], "FP32")
        i.set_data_from_numpy(np.array(x, dtype=np.float32), binary_data=True)
        y = client.infer("buckets", [i]).as_numpy("Y").tolist()
        check(f"buckets length {n}", y, [v * v + v for v in x])
    i = httpclient.InferInput("X", [5], "FP32")
    i.set_data_from_numpy(np.zeros(5, dtype=np.float32), binary_data=True)
    expect_error("buckets length 5 is refused by the backend",
                 lambda: client.infer("buckets", [i]), "no compiled artifact takes input shapes ([5])")
    if not client.is_server_live():
        print("  FAIL the server is not live after a refused request")
        FAILURES.append("live after refusal")


def load(http, name, config=None):
    """POST /v2/repository/models/<name>/load; returns (status, body)."""
    body = {} if config is None else {"parameters": {"config": json.dumps(config)}}
    req = urllib.request.Request(
        f"http://{http}/v2/repository/models/{name}/load", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def refusals(http):
    """Model configurations the backend must refuse at load, by name."""
    print(f"refusals  {http}")
    matmul = {
        "name": "matmul_sumsq", "backend": "tlaloc", "max_batch_size": 0,
        "input": [{"name": "A", "data_type": "TYPE_FP32", "dims": [2, 2]}],
        "output": [{"name": "VALUE", "data_type": "TYPE_FP32", "dims": [1]},
                   {"name": "GRAD", "data_type": "TYPE_FP32", "dims": [2, 2]}],
        "instance_group": [{"kind": "KIND_GPU", "count": 1, "gpus": [0]}],
    }

    def variant(params, **changes):
        c = json.loads(json.dumps(matmul))
        c.update(changes)
        c["parameters"] = {k: {"string_value": v} for k, v in params.items()}
        return c

    good = {"artifact": "model.mlir", "entry": "matmul_sumsq_grad"}
    cases = [
        ("missing artifact parameter", variant({}), "has no 'artifact' parameter"),
        ("unknown entry function", variant({**good, "entry": "nope"}),
         "entry function @nope is not in the artifact; it defines @matmul_sumsq_grad"),
        ("dtype differs from the artifact",
         variant(good, input=[{"name": "A", "data_type": "TYPE_FP64", "dims": [2, 2]}]),
         "is tensor<2x2xf32> but config.pbtxt says FP64"),
        ("shape differs from the artifact",
         variant(good, input=[{"name": "A", "data_type": "TYPE_FP32", "dims": [4]}]),
         "is tensor<2x2xf32> but config.pbtxt allows shape [4]"),
        ("unloadable PJRT plugin", variant({**good, "pjrt_plugin_path": "/nonexistent/plugin.so"}),
         "cannot load the PJRT plugin /nonexistent/plugin.so"),
        ("unsupported data type",
         variant(good, input=[{"name": "A", "data_type": "TYPE_STRING", "dims": [2, 2]}]),
         "has data_type TYPE_STRING"),
    ]
    for label, config, needle in cases:
        status, body = load(http, "matmul_sumsq", config)
        ok = status != 200 and needle in body
        print(f"  {'ok  ' if ok else 'FAIL'} refused: {label}: HTTP {status} {body.strip()[:200]}")
        if not ok:
            FAILURES.append(f"refusal: {label}")
    status, body = load(http, "matmul_sumsq")
    ok = status == 200
    print(f"  {'ok  ' if ok else 'FAIL'} reload from config.pbtxt: HTTP {status} {body.strip()[:120]}")
    if not ok:
        FAILURES.append("reload")
        return
    req = urllib.request.Request(
        f"http://{http}/v2/models/matmul_sumsq/infer",
        data=json.dumps({"inputs": [{"name": "A", "shape": [2, 2], "datatype": "FP32",
                                     "data": [1, 2, 3, 4]}]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        out = {o["name"]: o for o in json.loads(r.read())["outputs"]}
    check("after reload matmul_sumsq GRAD", out["GRAD"]["data"], [7.0, 11.0, 9.0, 13.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", default="localhost:8000")
    ap.add_argument("--grpc", default="localhost:8001")
    ap.add_argument("--reference", default=str(Path(__file__).parent / "examples/reference/matmul_sumsq.json"))
    ap.add_argument("--perturb", action="store_true", help="negative control: expect a wrong GRAD")
    args = ap.parse_args()

    ref = json.loads(Path(args.reference).read_text())
    a = ref["inputs"]["A"]["data"]
    ref_value = ref["outputs"]["VALUE"]["data"]
    ref_grad = ref["outputs"]["GRAD"]["data"]

    closed_value = [54.0]
    closed_grad = [7.0, 11.0, 9.0, 13.0]
    if args.perturb:
        closed_grad = [7.0, 11.0, 9.0, 14.0]
        ref_grad = list(ref_grad)
        ref_grad[3] += 1.0
        print("negative control: expecting GRAD[3] one larger than the truth")

    x_f64 = [1.0, 2.0, -0.5, 3.0]
    x_bf16 = [1.0, 2.0, -0.5, 3.0]
    x_i32 = [1, 2, -5, 30000]
    y_f64 = [x * x + x for x in x_f64]
    y_bf16 = [bf16_round(bf16_round(x * x) + x) for x in x_bf16]
    y_i32 = [x * x + x for x in x_i32]

    # --- KServe v2 JSON, standard library only -----------------------------
    print(f"json  http://{args.http}")
    body = json.dumps({"inputs": [{"name": "A", "shape": [2, 2], "datatype": "FP32", "data": a}]})
    req = urllib.request.Request(
        f"http://{args.http}/v2/models/matmul_sumsq/infer", data=body.encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        out = {o["name"]: o for o in json.loads(r.read())["outputs"]}
    check("json matmul_sumsq VALUE vs interpreter", out["VALUE"]["data"], ref_value)
    check("json matmul_sumsq GRAD vs interpreter", out["GRAD"]["data"], ref_grad)
    check("json matmul_sumsq GRAD vs closed form", out["GRAD"]["data"], closed_grad)
    check("json matmul_sumsq VALUE vs closed form", out["VALUE"]["data"], closed_value)
    if out["GRAD"]["shape"] != [2, 2] or out["VALUE"]["shape"] != [1]:
        print(f"  FAIL json shapes: {out['GRAD']['shape']} {out['VALUE']['shape']}")
        FAILURES.append("json shapes")

    bad = json.dumps({"inputs": [{"name": "A", "shape": [2, 2], "datatype": "FP64", "data": a}]})

    def bad_dtype():
        urllib.request.urlopen(urllib.request.Request(
            f"http://{args.http}/v2/models/matmul_sumsq/infer", data=bad.encode()), timeout=30)

    expect_error("json wrong dtype is refused", bad_dtype, "400")

    # --- tritonclient ------------------------------------------------------
    try:
        import numpy as np
        import tritonclient.grpc as grpcclient
        import tritonclient.http as httpclient
        # Recent tritonclient releases take BF16 as ml_dtypes.bfloat16; older
        # ones take float32 and truncate.
        try:
            import ml_dtypes
            bf16 = ml_dtypes.bfloat16
        except ImportError:
            bf16 = np.float32
    except ImportError as e:
        print(f"FAIL tritonclient is not importable ({e}); install tritonclient[http,grpc] "
              "into the Python that runs this script, or set TRITON_CLIENT_PYTHON in verify.sh")
        return 1

    for kind, mod, url in (("http", httpclient, args.http), ("grpc", grpcclient, args.grpc)):
        print(f"{kind}  {url}")
        client = mod.InferenceServerClient(url=url)
        if not client.is_model_ready("matmul_sumsq") or not client.is_model_ready("dtypes"):
            print(f"  FAIL {kind}: models not ready")
            FAILURES.append(f"{kind} ready")
            continue

        def inp(name, arr, dt):
            i = mod.InferInput(name, list(arr.shape), dt)
            if kind == "http":
                i.set_data_from_numpy(arr, binary_data=True)
            else:
                i.set_data_from_numpy(arr)
            return i

        res = client.infer("matmul_sumsq", [inp("A", np.array(a, dtype=np.float32).reshape(2, 2), "FP32")])
        grad = res.as_numpy("GRAD")
        value = res.as_numpy("VALUE")
        check(f"{kind} matmul_sumsq VALUE vs interpreter", value.reshape(-1).tolist(), ref_value)
        check(f"{kind} matmul_sumsq GRAD vs interpreter", grad.reshape(-1).tolist(), ref_grad)
        check(f"{kind} matmul_sumsq GRAD vs closed form", grad.reshape(-1).tolist(), closed_grad)
        if grad.dtype != np.float32 or grad.shape != (2, 2) or value.shape != (1,):
            print(f"  FAIL {kind} dtype/shape: {grad.dtype} {grad.shape} {value.shape}")
            FAILURES.append(f"{kind} dtype/shape")

        res = client.infer("dtypes", [
            inp("X_F64", np.array(x_f64, dtype=np.float64), "FP64"),
            inp("X_BF16", np.array(x_bf16, dtype=bf16), "BF16"),
            inp("X_I32", np.array(x_i32, dtype=np.int32), "INT32"),
        ])
        check(f"{kind} dtypes Y_F64", res.as_numpy("Y_F64").tolist(), y_f64)
        check(f"{kind} dtypes Y_BF16", res.as_numpy("Y_BF16").astype(np.float32).tolist(), y_bf16)
        check(f"{kind} dtypes Y_I32", res.as_numpy("Y_I32").tolist(), y_i32)
        if res.as_numpy("Y_F64").dtype != np.float64 or res.as_numpy("Y_I32").dtype != np.int32:
            print(f"  FAIL {kind} dtypes output dtypes")
            FAILURES.append(f"{kind} dtypes output dtypes")

        # Only the requested output comes back.
        out = (mod.InferRequestedOutput("Y_I32", binary_data=True) if kind == "http"
               else mod.InferRequestedOutput("Y_I32"))
        res = client.infer("dtypes", [
            inp("X_F64", np.array(x_f64, dtype=np.float64), "FP64"),
            inp("X_BF16", np.array(x_bf16, dtype=bf16), "BF16"),
            inp("X_I32", np.array(x_i32, dtype=np.int32), "INT32"),
        ], outputs=[out])
        check(f"{kind} dtypes requested output only", res.as_numpy("Y_I32").tolist(), y_i32)
        if res.as_numpy("Y_F64") is not None:
            print(f"  FAIL {kind}: an output that was not requested came back")
            FAILURES.append(f"{kind} requested outputs")

        expect_error(
            f"{kind} wrong shape is refused",
            lambda: client.infer("matmul_sumsq", [inp("A", np.zeros((3, 3), dtype=np.float32), "FP32")]),
            "unexpected shape",
        )

    buckets(args.http, np, httpclient)
    refusals(args.http)

    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
