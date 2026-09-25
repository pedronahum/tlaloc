#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""Measurements against a running Triton server with the example models.

Run by verify.sh after the checks pass (verify_client.py certifies the
answers; this script only times them):

  batching   grad_batched (dynamic batching) against grad_unbatched (the same
             artifacts, one request per execution): requests per second for
             one-row requests at a fixed number in flight, and how many
             executions Triton ran for them.
  zero copy  large_io (16 MiB in, 16 MiB out) three ways over gRPC: tensors in
             CUDA shared memory read and written in place (large_io), the same
             shared memory taken through the host (large_io_host), and plain
             gRPC bytes. Median latency seen by the client, and the server's
             own time per request (queue + compute input + infer + output).

Numbers depend on the machine; verify.sh prints them, it does not judge them.
"""

import argparse
import statistics
import threading
import time

import numpy as np
import tritonclient.grpc as grpcclient


def stats(client, model):
    s = client.get_inference_statistics(model_name=model, as_json=True)["model_stats"][0]
    inf = s.get("inference_stats", {})

    def ns(key):
        return int(inf.get(key, {}).get("ns", 0))

    return {
        "inferences": int(s.get("inference_count", 0)),
        "executions": int(s.get("execution_count", 0)),
        "success": int(inf.get("success", {}).get("count", 0)),
        "server_ns": ns("queue") + ns("compute_input") + ns("compute_infer") + ns("compute_output"),
        "input_ns": ns("compute_input"),
        "infer_ns": ns("compute_infer"),
        "output_ns": ns("compute_output"),
    }


def delta(a, b):
    return {k: b[k] - a[k] for k in a}


def throughput(url, model, requests, in_flight):
    """One-row requests with `in_flight` outstanding at all times."""
    client = grpcclient.InferenceServerClient(url=url)
    rng = np.random.default_rng(3)
    rows = rng.integers(-8, 9, size=(64, 1, 2, 2)).astype(np.float32)
    inputs = []
    for r in rows:
        i = grpcclient.InferInput("A", [1, 2, 2], "FP32")
        i.set_data_from_numpy(r)
        inputs.append(i)
    slots = threading.Semaphore(in_flight)
    done = threading.Event()
    lock = threading.Lock()
    finished = [0]
    errors = []

    def callback(result, error):
        if error is not None:
            errors.append(error)
        slots.release()
        with lock:
            finished[0] += 1
            if finished[0] == requests:
                done.set()

    # warm up
    for k in range(64):
        client.infer(model, [inputs[k]])
    before = stats(client, model)
    start = time.perf_counter()
    for k in range(requests):
        slots.acquire()
        client.async_infer(model, [inputs[k % len(inputs)]], callback)
    done.wait()
    seconds = time.perf_counter() - start
    d = delta(before, stats(client, model))
    client.close()
    if errors:
        raise RuntimeError(f"{model}: {len(errors)} requests failed, first: {errors[0]}")
    return requests / seconds, d


def large_io(url, repeats):
    import tritonclient.utils.cuda_shared_memory as cudashm

    client = grpcclient.InferenceServerClient(url=url)
    n = 4 * 1024 * 1024
    x = (np.arange(n, dtype=np.float32) % 1024) * np.float32(0.5) - np.float32(256)
    client.unregister_cuda_shared_memory()
    h_x = cudashm.create_shared_memory_region("perf_x", x.nbytes, 0)
    h_y = cudashm.create_shared_memory_region("perf_y", x.nbytes, 0)
    rows = []
    try:
        client.register_cuda_shared_memory("perf_x", cudashm.get_raw_handle(h_x), 0, x.nbytes)
        client.register_cuda_shared_memory("perf_y", cudashm.get_raw_handle(h_y), 0, x.nbytes)
        cudashm.set_shared_memory_region(h_x, [x])

        def shm_call(model):
            i = grpcclient.InferInput("X", [n], "FP32")
            i.set_shared_memory("perf_x", x.nbytes)
            o = grpcclient.InferRequestedOutput("Y")
            o.set_shared_memory("perf_y", x.nbytes)
            client.infer(model, [i], outputs=[o])

        def bytes_call(model):
            i = grpcclient.InferInput("X", [n], "FP32")
            i.set_data_from_numpy(x)
            client.infer(model, [i]).as_numpy("Y")

        for label, model, call in (
            ("CUDA shared memory, read and written in place", "large_io", shm_call),
            ("CUDA shared memory, through the host", "large_io_host", shm_call),
            ("gRPC bytes (no shared memory)", "large_io", bytes_call),
        ):
            for _ in range(3):
                call(model)
            before = stats(client, model)
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                call(model)
                times.append(time.perf_counter() - t0)
            d = delta(before, stats(client, model))
            rows.append((label, statistics.median(times) * 1e3, d))
    finally:
        client.unregister_cuda_shared_memory()
        cudashm.destroy_shared_memory_region(h_x)
        cudashm.destroy_shared_memory_region(h_y)
        client.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grpc", default="localhost:8001")
    ap.add_argument("--requests", type=int, default=4000)
    ap.add_argument("--in-flight", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=30)
    args = ap.parse_args()

    print(f"dynamic batching: {args.requests} one-row requests of the gradient model, "
          f"{args.in_flight} in flight, gRPC")
    for model in ("grad_unbatched", "grad_batched"):
        rate, d = throughput(args.grpc, model, args.requests, args.in_flight)
        per = d["inferences"] / max(d["executions"], 1)
        print(f"  {model:15s} {rate:8.0f} requests/s   {d['inferences']} inferences in "
              f"{d['executions']} executions ({per:.1f} per execution)")

    print(f"zero copy: large_io, 16 MiB in and 16 MiB out, median of {args.repeats}, gRPC")
    for label, ms, d in large_io(args.grpc, args.repeats):
        count = max(d["success"], 1)
        print(f"  {label:48s} client {ms:7.2f} ms   server {d['server_ns'] / count / 1e6:6.2f} ms "
              f"(input {d['input_ns'] / count / 1e6:.2f}, infer {d['infer_ns'] / count / 1e6:.2f}, "
              f"output {d['output_ns'] / count / 1e6:.2f})")


if __name__ == "__main__":
    main()
