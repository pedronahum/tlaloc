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
  reference_decode
                decode steps with the KV pools held by the backend, against
                the DXIR interpreter (examples/reference/reference_decode.json),
                within 1e-3 of the largest logit and with the same argmax.
  reference_sequence
                the same artifact in sequence mode: the backend keeps each
                sequence's pages by correlation ID. The reference steps' first
                sequence sent one token per request and all at once, two more
                sequences, and requests the backend must refuse by name.
  int64_bool    i64 and bool tensors emitted by Tlaloc, computed here exactly.
  dtypes_small  f16, i8 and u8 through a hand-written StableHLO module (Tlaloc
                has no such dtypes), computed here exactly.
  grad_batched / grad_unbatched
                the gradient of sum(A . A) per 2x2 row, against the DXIR
                interpreter (examples/reference/grad_batched.json) and the
                closed form rowsum(A)[q] + colsum(A)[p], bit for bit; many
                concurrent one-row requests, which Triton's dynamic batcher
                gathers for grad_batched, each get exactly their own row back.
  ragged_batched / ragged_consecutive
                x*x + x over rows of width 4 or 8, computed here exactly;
                requests of both widths sent at once each get exactly their
                own rows back, and grouping a batch by the shape of its rows
                (ragged_batched) runs fewer executions than grouping only
                consecutive requests of one width (ragged_consecutive).
  zero copy     matmul_sumsq and large_io with inputs and outputs in CUDA
                shared memory over gRPC: the same bits as the host path, and
                the input region is left unchanged. (verify.sh checks the
                server log for which path each tensor took.)

--perturb changes one expected GRAD value (and one decode logit and one
ragged row); the run must then fail. That is
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


def check_exact(label, got, want):
    """Equality of integer or bool lists, compared as Python ints."""
    ok = [int(g) for g in got] == [int(w) for w in want]
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: got {list(got)} want {list(want)}")
    if not ok:
        FAILURES.append(label)


def dtypes_wide(kind, mod, url, np, perturb):
    """INT64 and BOOL (int64_bool, emitted by Tlaloc) and FP16, INT8, UINT8
    (dtypes_small, hand-written StableHLO) over one transport."""
    print(f"{kind}  int64_bool, dtypes_small  {url}")
    client = mod.InferenceServerClient(url=url)

    def inp(name, arr, dt):
        i = mod.InferInput(name, list(arr.shape), dt)
        if kind == "http":
            i.set_data_from_numpy(arr, binary_data=True)
        else:
            i.set_data_from_numpy(arr)
        return i

    x = [1, 2, -5, 3_000_000_000]  # x * x + x = 9e18 + 3e9, past the i32 range
    b = [True, True, False, True]
    want_y = [v * v + v for v in x]
    if perturb:
        want_y[3] += 1
    res = client.infer("int64_bool", [inp("X", np.array(x, dtype=np.int64), "INT64"),
                                      inp("B", np.array(b, dtype=np.bool_), "BOOL")])
    y = res.as_numpy("Y")
    check_exact(f"{kind} int64_bool Y (INT64)", y.tolist(), want_y)
    check_exact(f"{kind} int64_bool ABOVE_AND_B (BOOL)", res.as_numpy("ABOVE_AND_B").tolist(),
                [v * v > v and c for v, c in zip(x, b)])
    check_exact(f"{kind} int64_bool NOT_B (BOOL)", res.as_numpy("NOT_B").tolist(), [not c for c in b])
    if y.dtype != np.int64 or res.as_numpy("NOT_B").dtype != np.bool_:
        print(f"  FAIL {kind} int64_bool output dtypes {y.dtype} {res.as_numpy('NOT_B').dtype}")
        FAILURES.append(f"{kind} int64_bool dtypes")

    xf = [1.0, 2.0, -0.5, 3.0]
    xi = [1, 2, -5, 10]
    xu = [1, 2, 5, 15]
    res = client.infer("dtypes_small", [inp("X_F16", np.array(xf, dtype=np.float16), "FP16"),
                                        inp("X_I8", np.array(xi, dtype=np.int8), "INT8"),
                                        inp("X_U8", np.array(xu, dtype=np.uint8), "UINT8")])
    yf = res.as_numpy("Y_F16")
    check(f"{kind} dtypes_small Y_F16 (FP16)", yf.astype(np.float32).tolist(), [v * v + v for v in xf])
    check_exact(f"{kind} dtypes_small Y_I8 (INT8)", res.as_numpy("Y_I8").tolist(), [v * v + v for v in xi])
    check_exact(f"{kind} dtypes_small Y_U8 (UINT8)", res.as_numpy("Y_U8").tolist(), [v * v + v for v in xu])
    if yf.dtype != np.float16 or res.as_numpy("Y_I8").dtype != np.int8 \
            or res.as_numpy("Y_U8").dtype != np.uint8:
        print(f"  FAIL {kind} dtypes_small output dtypes")
        FAILURES.append(f"{kind} dtypes_small dtypes")


def grad_rows(np, a):
    """Closed-form df/dA of sum(A . A) per row: rowsum(A)[q] + colsum(A)[p]."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 2, 2)
    return (a.sum(axis=2)[:, None, :] + a.sum(axis=1)[:, :, None]).astype(np.float32)


def model_counts(client, name):
    stats = client.get_inference_statistics(model_name=name, as_json=True)["model_stats"][0]
    return int(stats.get("inference_count", 0)), int(stats.get("execution_count", 0))


def batching(grpc, np, grpcclient, reference, perturb):
    """Dynamic batching of the gradient model: every request gets its own
    rows back, batched or not, bit for bit."""
    import concurrent.futures
    import random
    import threading

    print(f"grpc  grad_batched, grad_unbatched  {grpc}")
    ref = json.loads(Path(reference).read_text())
    a = np.array(ref["inputs"]["A"]["data"], dtype=np.float32).reshape(ref["inputs"]["A"]["shape"])
    want = np.array(ref["outputs"]["GRAD"]["data"], dtype=np.float32).reshape(a.shape)
    closed = grad_rows(np, a)
    if perturb:
        want = want.copy()
        want[5, 1, 0] += 1.0
    check("interpreter reference vs closed form", want.reshape(-1).tolist(), closed.reshape(-1).tolist())

    def infer(client, model, rows):
        i = grpcclient.InferInput("A", list(rows.shape), "FP32")
        i.set_data_from_numpy(rows)
        return client.infer(model, [i]).as_numpy("GRAD")

    client = grpcclient.InferenceServerClient(url=grpc)
    for model in ("grad_batched", "grad_unbatched"):
        got = infer(client, model, a)
        check(f"{model} 8 rows in one request vs interpreter", got.reshape(-1).tolist(),
              want.reshape(-1).tolist())
        got = infer(client, model, a[:3])  # runs on the batch-4 artifact, one row of padding
        check(f"{model} 3 rows (padded to 4) vs interpreter", got.reshape(-1).tolist(),
              want[:3].reshape(-1).tolist())

    # Concurrent one-row requests, random small integers (exact in f32).
    rng = random.Random(7)
    rows = [np.array([[rng.randint(-8, 8) for _ in range(2)] for _ in range(2)],
                     dtype=np.float32).reshape(1, 2, 2) for _ in range(512)]
    expected = [grad_rows(np, r) for r in rows]
    if perturb:
        expected[17] = expected[17].copy()
        expected[17][0, 0, 0] += 1.0
    results = {}
    for model in ("grad_batched", "grad_unbatched"):
        before = model_counts(client, model)
        local = threading.local()

        def one(k, model=model):
            if not hasattr(local, "client"):
                local.client = grpcclient.InferenceServerClient(url=grpc)
            return infer(local.client, model, rows[k])

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            got = list(pool.map(one, range(len(rows))))
        after = model_counts(client, model)
        results[model] = got
        bad = [k for k in range(len(rows)) if got[k].tobytes() != expected[k].tobytes()]
        ok = not bad
        print(f"  {'ok  ' if ok else 'FAIL'} {model}: {len(rows)} concurrent one-row requests, "
              f"{len(rows) - len(bad)} match the closed form bit for bit"
              + (f" (first mismatch: request {bad[0]})" if bad else ""))
        if not ok:
            FAILURES.append(f"{model} concurrent rows")
        inferences, executions = after[0] - before[0], after[1] - before[1]
        print(f"       {model}: {inferences} inferences in {executions} executions")
        if model == "grad_batched" and not executions < inferences:
            print("  FAIL grad_batched: no requests were batched together")
            FAILURES.append("grad_batched batched nothing")
        if model == "grad_unbatched" and executions != inferences:
            print("  FAIL grad_unbatched: requests were batched together")
            FAILURES.append("grad_unbatched batched")
    same = all(x.tobytes() == y.tobytes() for x, y in zip(results["grad_batched"], results["grad_unbatched"]))
    print(f"  {'ok  ' if same else 'FAIL'} batched and unbatched answers are identical for all "
          f"{len(rows)} requests")
    if not same:
        FAILURES.append("batched vs unbatched")


# Rows per request of the ragged check: mostly one, sometimes two.
ROWS = (1, 1, 2)


def ragged(grpc, np, grpcclient, perturb, rounds=3, per_round=72):
    """Ragged dynamic batching: requests of rows of width 4 and of width 8
    (one or two rows each, in random order) sent all at once, so that one
    batch holds both widths. Every request must get exactly its own rows of
    x*x + x back. ragged_batched runs each width of a batch as one execution;
    ragged_consecutive (group_by_shape false, the control) runs only
    consecutive requests of one width together, so it must need more
    executions for the same requests."""
    import concurrent.futures
    import random
    import threading

    print(f"grpc  ragged_batched, ragged_consecutive  {grpc}")
    rng = random.Random(11)
    requests = []
    for _ in range(rounds * per_round):
        width = rng.choice((4, 8))
        rows = rng.choice(ROWS)
        requests.append(np.array([[rng.randint(-8, 8) for _ in range(width)] for _ in range(rows)],
                                 dtype=np.float32))
    expected = [x * x + x for x in requests]
    if perturb:
        expected[29] = expected[29].copy()
        expected[29][0, 0] += 1.0
    client = grpcclient.InferenceServerClient(url=grpc)
    counts, answers = {}, {}
    for model in ("ragged_batched", "ragged_consecutive"):
        local = threading.local()

        def one(k, model=model):
            if not hasattr(local, "client"):
                local.client = grpcclient.InferenceServerClient(url=grpc)
            x = requests[k]
            i = grpcclient.InferInput("X", list(x.shape), "FP32")
            i.set_data_from_numpy(x)
            return local.client.infer(model, [i]).as_numpy("Y")

        before = model_counts(client, model)
        got = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=per_round) as pool:
            for r in range(rounds):
                got += list(pool.map(one, range(r * per_round, (r + 1) * per_round)))
        after = model_counts(client, model)
        answers[model] = got
        bad = [k for k in range(len(requests))
               if got[k].shape != expected[k].shape or got[k].tobytes() != expected[k].tobytes()]
        ok = not bad
        print(f"  {'ok  ' if ok else 'FAIL'} {model}: {len(requests)} concurrent requests of widths 4 "
              f"and 8, {len(requests) - len(bad)} get their own rows bit for bit"
              + (f" (first mismatch: request {bad[0]})" if bad else ""))
        if not ok:
            FAILURES.append(f"{model} rows")
        counts[model] = (after[0] - before[0], after[1] - before[1])
        print(f"       {model}: {len(requests)} requests ({counts[model][0]} rows) in "
              f"{counts[model][1]} executions")
    grouped, control = counts["ragged_batched"], counts["ragged_consecutive"]
    ok = grouped[1] < control[1]
    print(f"  {'ok  ' if ok else 'FAIL'} grouping by shape ran {grouped[1]} executions where running "
          f"consecutive requests of one width together ran {control[1]}")
    if not ok:
        FAILURES.append("ragged grouping by shape")
    # A batch of requests of two widths needs 2 executions grouped by shape,
    # so executions fall well below requests.
    ok = 2 * grouped[1] <= len(requests)
    print(f"  {'ok  ' if ok else 'FAIL'} ragged_batched: {len(requests)} requests in {grouped[1]} executions "
          f"(at most half as many executions as requests)")
    if not ok:
        FAILURES.append("ragged_batched batched too little")
    same = all(x.tobytes() == y.tobytes()
               for x, y in zip(answers["ragged_batched"], answers["ragged_consecutive"]))
    print(f"  {'ok  ' if same else 'FAIL'} both models' answers are identical for all {len(requests)} requests")
    if not same:
        FAILURES.append("ragged batched vs consecutive")


def zero_copy(grpc, np, grpcclient, ref_value, ref_grad):
    """Inputs and outputs in CUDA shared memory over gRPC."""
    try:
        import tritonclient.utils.cuda_shared_memory as cudashm
    except Exception as e:  # noqa: BLE001 - missing cuda-python or no GPU
        print(f"FAIL zero copy: tritonclient's CUDA shared memory is unavailable ({e}); "
              "install tritonclient[all] (it needs cuda-python for the CUDA version of the driver)")
        FAILURES.append("zero copy unavailable")
        return
    print(f"grpc  zero copy (CUDA shared memory)  {grpc}")
    client = grpcclient.InferenceServerClient(url=grpc)
    client.unregister_cuda_shared_memory()
    regions = []

    def region(name, nbytes):
        h = cudashm.create_shared_memory_region(name, nbytes, 0)
        regions.append(h)
        client.register_cuda_shared_memory(name, cudashm.get_raw_handle(h), 0, nbytes)
        return h

    try:
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        h_in = region("tl_a", a.nbytes)
        h_grad = region("tl_grad", 16)
        h_value = region("tl_value", 4)
        cudashm.set_shared_memory_region(h_in, [a])
        i = grpcclient.InferInput("A", [2, 2], "FP32")
        i.set_shared_memory("tl_a", a.nbytes)
        o_grad = grpcclient.InferRequestedOutput("GRAD")
        o_grad.set_shared_memory("tl_grad", 16)
        o_value = grpcclient.InferRequestedOutput("VALUE")
        o_value.set_shared_memory("tl_value", 4)
        client.infer("matmul_sumsq", [i], outputs=[o_grad, o_value])
        check("grpc cuda-shm matmul_sumsq GRAD vs interpreter",
              cudashm.get_contents_as_numpy(h_grad, np.float32, [4]).tolist(), ref_grad)
        check("grpc cuda-shm matmul_sumsq VALUE vs interpreter",
              cudashm.get_contents_as_numpy(h_value, np.float32, [1]).tolist(), ref_value)

        n = 4 * 1024 * 1024
        x = (np.arange(n, dtype=np.float32) % 1024) * np.float32(0.5) - np.float32(256)
        want = x * x + x  # exact: every value is a multiple of 0.25 below 2^17
        h_x = region("tl_x", x.nbytes)
        h_y = region("tl_y", x.nbytes)
        for model in ("large_io", "large_io_host"):
            cudashm.set_shared_memory_region(h_x, [x])
            cudashm.set_shared_memory_region(h_y, [np.zeros(n, dtype=np.float32)])
            i = grpcclient.InferInput("X", [n], "FP32")
            i.set_shared_memory("tl_x", x.nbytes)
            o = grpcclient.InferRequestedOutput("Y")
            o.set_shared_memory("tl_y", x.nbytes)
            client.infer(model, [i], outputs=[o])
            y = cudashm.get_contents_as_numpy(h_y, np.float32, [n])
            ok = y.tobytes() == want.tobytes()
            print(f"  {'ok  ' if ok else 'FAIL'} grpc cuda-shm {model}: all {n} values equal x * x + x "
                  f"bit for bit")
            if not ok:
                FAILURES.append(f"cuda-shm {model}")
            unchanged = cudashm.get_contents_as_numpy(h_x, np.float32, [n]).tobytes() == x.tobytes()
            print(f"  {'ok  ' if unchanged else 'FAIL'} grpc cuda-shm {model}: the input region is unchanged")
            if not unchanged:
                FAILURES.append(f"cuda-shm {model} input written")
        i = grpcclient.InferInput("X", [n], "FP32")
        i.set_data_from_numpy(x)
        y = client.infer("large_io", [i]).as_numpy("Y")
        ok = y.tobytes() == want.tobytes()
        print(f"  {'ok  ' if ok else 'FAIL'} grpc large_io without shared memory: the same bits")
        if not ok:
            FAILURES.append("large_io host bytes")
    finally:
        client.unregister_cuda_shared_memory()
        for h in regions:
            cudashm.destroy_shared_memory_region(h)


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


def close(label, got, want, rel=1e-3):
    """Every |got - want| <= rel * max|want|, and the argmax agrees."""
    scale = max(abs(w) for w in want) or 1.0
    worst = max(abs(g - w) for g, w in zip(got, want)) if len(got) == len(want) else float("inf")
    arg = max(range(len(got)), key=got.__getitem__) == max(range(len(want)), key=want.__getitem__) \
        if len(got) == len(want) and got else False
    ok = worst <= rel * scale and arg
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: max |diff| {worst:.2e} (bound {rel * scale:.2e}), "
          f"argmax {'agrees' if arg else 'differs'}")
    if not ok:
        FAILURES.append(label)


def reference_decode(kind, mod, url, np, steps):
    """Decode steps of the reference decode graph, KV pools kept by the backend.

    The expected logits are the DXIR interpreter's, with the pools carried
    from step to step. XLA's GPU f32 dot runs as TF32, so the comparison is
    within 1e-3 of the step's largest logit, plus an exact argmax, rather
    than bit for bit. Steps 2 to 4 attend over keys and values that earlier
    requests wrote, so they only agree if the backend carried the state.
    """
    print(f"reference_decode  {kind} {url}")
    client = mod.InferenceServerClient(url=url)
    for k, step in enumerate(steps):
        inputs = []
        for name, t in step["inputs"].items():
            arr = np.array(t["data"], dtype=np.int32).reshape(t["shape"])
            i = mod.InferInput(name, list(arr.shape), "INT32")
            if kind == "http":
                i.set_data_from_numpy(arr, binary_data=True)
            else:
                i.set_data_from_numpy(arr)
            inputs.append(i)
        res = client.infer("reference_decode", inputs)
        logits = res.as_numpy("logits")
        want = step["outputs"]["logits"]
        if list(logits.shape) != want["shape"]:
            print(f"  FAIL {kind} step {k}: logits shape {list(logits.shape)}, want {want['shape']}")
            FAILURES.append(f"{kind} reference_decode step {k} shape")
            continue
        batch = want["shape"][0]
        row = len(want["data"]) // batch
        for b in range(batch):
            close(f"{kind} reference_decode step {k} row {b} (batch {batch})",
                  logits.reshape(-1)[b * row:(b + 1) * row].tolist(),
                  want["data"][b * row:(b + 1) * row])

    # The same token and position as step 1, on page 5, whose slot for
    # position 0 no request has written. Its logits must NOT match step 1's:
    # that is what shows step 1's agreement depends on the earlier request.
    step = json.loads(json.dumps(steps[1]))
    step["inputs"]["blockTables"]["data"] = [5, 0]
    step["inputs"]["slotMapping"]["data"] = [5 * 2 + 1]
    inputs = []
    for name, t in step["inputs"].items():
        arr = np.array(t["data"], dtype=np.int32).reshape(t["shape"])
        i = mod.InferInput(name, list(arr.shape), "INT32")
        if kind == "http":
            i.set_data_from_numpy(arr, binary_data=True)
        else:
            i.set_data_from_numpy(arr)
        inputs.append(i)
    got = client.infer("reference_decode", inputs).as_numpy("logits").reshape(-1).tolist()
    want = steps[1]["outputs"]["logits"]["data"]
    worst = max(abs(g - w) for g, w in zip(got, want))
    bound = 1e-3 * max(abs(w) for w in want)
    ok = worst > 10 * bound
    print(f"  {'ok  ' if ok else 'FAIL'} {kind} step 1 on an unwritten page differs from step 1: "
          f"max |diff| {worst:.2e} (must exceed {10 * bound:.2e})")
    if not ok:
        FAILURES.append(f"{kind} reference_decode state dependence")


def reference_sequence(kind, mod, url, np, steps):
    """The reference decode artifact in sequence mode, against the same
    interpreter steps: the first four steps are one sequence of tokens
    3, 7, 1, 9; the fifth runs two new one-token sequences."""
    print(f"{kind}  reference_sequence  {url}")
    client = mod.InferenceServerClient(url=url)
    base = 7000 if kind == "http" else 8000

    def step(corrid, tokens, start=False, end=False):
        arr = np.asarray(tokens, dtype=np.int32).reshape(1, -1)
        inp = mod.InferInput("TOKENS", list(arr.shape), "INT32")
        inp.set_data_from_numpy(arr)
        res = client.infer("reference_sequence", [inp], sequence_id=corrid,
                           sequence_start=start, sequence_end=end)
        out = res.as_numpy("LOGITS")
        return None if out is None else [float(x) for x in out.reshape(-1)]

    want = [st["outputs"]["logits"]["data"] for st in steps[:4]]
    tokens = [st["inputs"]["tokenIds"]["data"][0] for st in steps[:4]]
    for i, t in enumerate(tokens):
        got = step(base + 1, [t], start=i == 0, end=i == len(tokens) - 1)
        close(f"{kind} reference_sequence token {i} vs interpreter step {i}", got, want[i])
    got = step(base + 2, tokens, start=True, end=True)
    close(f"{kind} reference_sequence 4 tokens in one request vs interpreter step 3", got, want[3])
    v = len(want[0])
    pair = steps[4]["outputs"]["logits"]["data"]
    for r, t in enumerate(steps[4]["inputs"]["tokenIds"]["data"]):
        got = step(base + 3 + r, [t], start=True, end=True)
        close(f"{kind} reference_sequence new sequence of token {t} vs interpreter row {r}",
              got, pair[r * v:(r + 1) * v])
    expect_error(f"{kind} reference_sequence 5 tokens exceed the context",
                 lambda: step(base + 5, [1, 2, 3, 4, 5], start=True),
                 "the largest context this model was compiled for is 4")
    expect_error(f"{kind} reference_sequence token outside the vocabulary",
                 lambda: step(base + 6, [11], start=True), "outside the vocabulary [0, 11)")
    expect_error(f"{kind} reference_sequence step without START",
                 lambda: step(base + 7, [1]), "must specify the START flag")
    step(base + 8, [3], start=True)
    ended = step(base + 8, [], end=True)
    ok = ended is None
    print(f"  {'ok  ' if ok else 'FAIL'} {kind} reference_sequence a request with no tokens ends "
          f"the sequence and returns no LOGITS")
    if not ok:
        FAILURES.append(f"{kind} reference_sequence empty END")
    expect_error(f"{kind} reference_sequence step after END",
                 lambda: step(base + 8, [1]), "must specify the START flag")
    expect_error(f"{kind} reference_sequence a request with no tokens and no END",
                 lambda: step(base + 9, [], start=True), "a request without tokens only ends a sequence")
    # The refused STARTs leave sequences Triton still counts as live; ending
    # them releases their batcher slots (the backend holds nothing for them).
    for corrid in (base + 5, base + 6, base + 9):
        step(corrid, [], end=True)


def refusals_batching(http):
    """Batching configurations the backend must refuse at load, by name."""
    print(f"refusals batching  {http}")
    with urllib.request.urlopen(f"http://{http}/v2/models/grad_batched/config") as r:
        batched = json.load(r)
    with urllib.request.urlopen(f"http://{http}/v2/models/reference_decode/config") as r:
        decode = json.load(r)
    too_big = json.loads(json.dumps(batched))
    too_big["max_batch_size"] = 16
    stateful = json.loads(json.dumps(decode))
    stateful["max_batch_size"] = 4
    cases = [
        ("grad_batched", "max_batch_size above the largest artifact", too_big,
         "max_batch_size is 16 but the largest artifact takes batch 8"),
        ("reference_decode", "state with max_batch_size > 0", stateful,
         "state 'keyCache0' cannot be combined with max_batch_size > 0"),
    ]
    for name, label, config, needle in cases:
        status, body = load(http, name, config)
        ok = status != 200 and needle in body
        print(f"  {'ok  ' if ok else 'FAIL'} refused: {label}: HTTP {status} {body.strip()[:200]}")
        if not ok:
            FAILURES.append(f"refusal: {label}")
        status, body = load(http, name)
        ok = status == 200
        print(f"  {'ok  ' if ok else 'FAIL'} reload {name} from config.pbtxt: HTTP {status}")
        if not ok:
            FAILURES.append(f"reload {name}")


def refusals_sequence(http):
    """Sequence-mode configurations the backend must refuse at load, by name."""
    print(f"refusals sequence mode  {http}")
    with urllib.request.urlopen(f"http://{http}/v2/models/reference_sequence/config") as r:
        good = json.load(r)

    def variant(fn):
        c = json.loads(json.dumps(good))
        fn(c)
        return c

    def set_param(c, key, value):
        c["parameters"][key] = {"string_value": value}

    cases = [
        ("no sequence_batching", variant(lambda c: c.pop("sequence_batching")),
         "needs sequence_batching"),
        ("direct strategy", variant(lambda c: (c["sequence_batching"].pop("oldest"),
                                              c["sequence_batching"].update({"direct": {}}))),
         "uses the direct strategy; use oldest"),
        ("no CORRID control", variant(lambda c: c["sequence_batching"].update(
            {"control_input": c["sequence_batching"]["control_input"][:2]})),
         "missing: CORRID"),
        ("max_batch_size 0", variant(lambda c: (c.update({"max_batch_size": 0}),
                                               c["sequence_batching"]["oldest"].pop("preferred_batch_size"))),
         "needs max_batch_size >= 1"),
        ("serving_manifest with artifact", variant(lambda c: set_param(c, "artifact", "x.mlir")),
         "cannot be combined with 'artifact'"),
        ("manifest outside the version directory",
         variant(lambda c: set_param(c, "serving_manifest", "../1/tlaloc-serving.json")),
         "must be a file inside the model version directory"),
        ("two inputs", variant(lambda c: c["input"].append(
            {"name": "EXTRA", "data_type": "TYPE_INT32", "dims": [-1]})),
         "declares exactly one input"),
        ("a pages output of three", variant(lambda c: c["output"][1].update({"dims": [3]})),
         "the pages output is [ 2 ]"),
    ]
    for label, config, needle in cases:
        status, body = load(http, "reference_sequence", config)
        ok = status != 200 and needle in body
        print(f"  {'ok  ' if ok else 'FAIL'} refused: {label}: HTTP {status} {body.strip()[:200]}")
        if not ok:
            FAILURES.append(f"refusal: sequence {label}")
    status, body = load(http, "reference_sequence")
    ok = status == 200
    print(f"  {'ok  ' if ok else 'FAIL'} reload reference_sequence from config.pbtxt: HTTP {status}")
    if not ok:
        FAILURES.append("reload reference_sequence")
    # A windowed KV pool whose ring is shorter than the window.
    with urllib.request.urlopen(f"http://{http}/v2/models/window_sequence/config") as r:
        window = json.load(r)
    window["parameters"]["serving_manifest"] = {"string_value": "tlaloc-serving-short-ring.json"}
    status, body = load(http, "window_sequence", window)
    needle = "so a decode step would read a position already written over"
    ok = status != 200 and needle in body
    print(f"  {'ok  ' if ok else 'FAIL'} refused: a windowed ring shorter than the window: HTTP {status} "
          f"{body.strip()[:200]}")
    if not ok:
        FAILURES.append("refusal: sequence short ring")
    status, body = load(http, "window_sequence")
    ok = status == 200
    print(f"  {'ok  ' if ok else 'FAIL'} reload window_sequence from config.pbtxt: HTTP {status}")
    if not ok:
        FAILURES.append("reload window_sequence")


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
        ("artifact outside the version directory",
         variant({**good, "artifact": "../1/model.mlir"}),
         "artifact '../1/model.mlir' leaves the model version directory"),
        ("unsupported data type",
         variant(good, input=[{"name": "A", "data_type": "TYPE_STRING", "dims": [2, 2]}]),
         "has data_type TYPE_STRING"),
        ("argument that is not input:, weight: or state:", variant({**good, "arguments": "A"}),
         "argument 'A' in the 'arguments' parameter is not input:<name>, weight:<file> or state:<name>"),
        ("argument naming an undeclared input", variant({**good, "arguments": "input:B"}),
         "the 'arguments' parameter names input 'B', which config.pbtxt does not declare"),
        ("weight file outside the version directory",
         variant({**good, "arguments": "input:A, weight:../w.bin"}),
         "weight file '../w.bin' leaves the model version directory"),
        ("result writing a state no argument reads",
         variant({**good, "results": "output:VALUE, output:GRAD, state:kv"}),
         "the 'results' parameter writes state 'kv', which no state: argument"),
        ("zero_copy that is not true or false", variant({**good, "zero_copy": "yes"}),
         "the 'zero_copy' parameter must be true or false, got 'yes'"),
        ("a GPU this machine does not have",
         variant(good, instance_group=[{"kind": "KIND_GPU", "count": 1, "gpus": [7]}]),
         "gpu id 7"),
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
    ap.add_argument("--reference-decode",
                    default=str(Path(__file__).parent / "examples/reference/reference_decode.json"))
    ap.add_argument("--reference-batched",
                    default=str(Path(__file__).parent / "examples/reference/grad_batched.json"))
    ap.add_argument("--perturb", action="store_true",
                    help="negative control: expect a wrong GRAD and a wrong decode logit")
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

    # Both transports replay the same steps. The second pass writes the same
    # keys and values into the same slots, so it expects the same logits.
    steps = json.loads(Path(args.reference_decode).read_text())["steps"]
    if args.perturb:
        steps[1]["outputs"]["logits"]["data"][0] += 1.0
    for kind, mod, url in (("http", httpclient, args.http), ("grpc", grpcclient, args.grpc)):
        reference_decode(kind, mod, url, np, steps)
    for kind, mod, url in (("http", httpclient, args.http), ("grpc", grpcclient, args.grpc)):
        reference_sequence(kind, mod, url, np, steps)

    for kind, mod, url in (("http", httpclient, args.http), ("grpc", grpcclient, args.grpc)):
        dtypes_wide(kind, mod, url, np, args.perturb)
    batching(args.grpc, np, grpcclient, args.reference_batched, args.perturb)
    ragged(args.grpc, np, grpcclient, args.perturb)
    zero_copy(args.grpc, np, grpcclient, ref_value, ref_grad)

    buckets(args.http, np, httpclient)
    refusals(args.http)
    refusals_batching(args.http)
    refusals_sequence(args.http)

    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
