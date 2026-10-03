#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""Timed decode and prefill workloads against a sequence-mode model, for
triton/profile.sh.

Each workload starts a fresh sequence with a seeded random prompt of PREFIX
tokens (the backend splits it into prefill calls), warms the entry the
measurement uses, then runs the measured part:

  decode:PREFIX:STEPS    STEPS single-token requests, one after the other
  prefill:PREFIX:CHUNKS  CHUNKS requests of --chunk tokens (default 512), each
                         one prefill call at the context bucket that holds
                         PREFIX + (k + 1) * chunk tokens

and repeats it --repeat times on new sequences. For each request it records
the time the client sees; from the model's statistics it takes the server's
own time per request (compute_infer: building inputs, the PJRT execution,
copying the logits back) and the time requests waited in Triton's queue.

With --window-start and --window-stop (shell commands; {name} is replaced by
the workload's name), the first repeat's measured part runs between the two,
so a profiler attached to the server captures exactly it.

  profile_client.py --grpc localhost:8021 --http localhost:8020 --model qwen3 \\
      --max-id 150000 --workload decode:8:32 --repeat 3

Prints one line per workload and, with --json FILE, writes the numbers.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request

import numpy as np

from sequence_client import SequenceClient


def stats(http, model):
    with urllib.request.urlopen(f"http://{http}/v2/models/{model}/stats") as r:
        s = json.load(r)["model_stats"][0]
    inf = s.get("inference_stats", {})

    def ns(k):
        return int(inf.get(k, {}).get("ns", 0))

    return {"count": int(inf.get("success", {}).get("count", 0)), "queue": ns("queue"),
            "input": ns("compute_input"), "infer": ns("compute_infer"), "output": ns("compute_output"),
            "executions": int(s.get("execution_count", 0))}


def next_token(client, corrid, toks, start=False):
    """The token to send next: the argmax of the logits, or for a speculative
    model (a NEXT_TOKENS output) the last token the request emits."""
    if getattr(client, "speculative", None) is None:
        cfg = client.client.get_model_config(client.model, as_json=True)
        cfg = cfg.get("config", cfg)
        client.speculative = any(o.get("name") == "NEXT_TOKENS" for o in cfg.get("output", []))
    if client.speculative:
        return client.step_tokens(corrid, toks, start=start)[-1]
    return int(np.argmax(client.step(corrid, toks, start=start)))


def prompt_ids(seed, n, max_id, refused):
    rng = np.random.default_rng(seed)
    ids = [int(t) for t in rng.integers(1, max_id, n)]
    return [t if t not in refused else t + 1 for t in ids]


def concurrent_decode(args, corrid, prefix, count, streams, refused, profiled, name):
    """`streams` sequences of `prefix` tokens each, then `count` decode steps
    each, sent from one thread per sequence so that the backend batches them."""
    import threading
    clients = [SequenceClient(args.grpc, args.model, "grpc") for _ in range(streams)]
    last = []
    for k, c in enumerate(clients):
        last.append(next_token(c, corrid + k, prompt_ids(corrid + k, prefix, args.max_id, refused), start=True))
    for _ in range(args.warmup):
        for k, c in enumerate(clients):
            last[k] = next_token(c, corrid + k, [last[k]])
    times = [[] for _ in range(streams)]

    def run(k):
        tok = last[k]
        for _ in range(count):
            t = time.perf_counter()
            tok = next_token(clients[k], corrid + k, [tok])
            times[k].append((time.perf_counter() - t) * 1e3)

    window = profiled and args.window_start
    s0 = stats(args.http, args.model)
    if window:
        subprocess.run(args.window_start.format(name=name), shell=True, check=True)
    ts = [threading.Thread(target=run, args=(k,)) for k in range(streams)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if window:
        subprocess.run(args.window_stop.format(name=name), shell=True, check=True)
    s1 = stats(args.http, args.model)
    for k, c in enumerate(clients):
        c.end(corrid + k)
    ms = [x for t in times for x in t]
    n = s1["count"] - s0["count"]
    d = {k: (s1[k] - s0[k]) for k in s0}
    return {"clientMs": [round(x, 2) for x in ms], "clientMedianMs": round(statistics.median(ms), 2),
            "serverInferMs": round(d["infer"] / max(n, 1) / 1e6, 2),
            "serverQueueMs": round(d["queue"] / max(n, 1) / 1e6, 2),
            "serverInputMs": round(d["input"] / max(n, 1) / 1e6, 3),
            "serverOutputMs": round(d["output"] / max(n, 1) / 1e6, 3),
            "requests": n, "executions": d["executions"], "prefixSeconds": 0.0, "profiled": bool(window)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grpc", default="localhost:8021")
    ap.add_argument("--http", default="localhost:8020")
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", action="append", required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--max-id", type=int, default=30000)
    ap.add_argument("--refused", default="", help="comma-separated ids never sent")
    ap.add_argument("--first-corrid", type=int, default=70000)
    ap.add_argument("--window-start", default=None)
    ap.add_argument("--window-stop", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    refused = {int(x) for x in args.refused.split(",") if x}
    client = SequenceClient(args.grpc, args.model, "grpc")
    corrid = args.first_corrid
    results = []

    for wl in args.workload:
        parts = wl.split(":")
        kind, prefix, count = parts[0], int(parts[1]), int(parts[2])
        # decode:PREFIX:STEPS:STREAMS runs STREAMS sequences decoding at once.
        streams = int(parts[3]) if len(parts) > 3 else 1
        if streams > 1 and kind != "decode":
            raise SystemExit(f"{wl}: only decode workloads take a number of streams")
        name = f"{kind}_p{prefix}_n{count}" + (f"_s{streams}" if streams > 1 else "")
        runs = []
        for rep in range(args.repeat):
            corrid += 1
            if streams > 1:
                runs.append(concurrent_decode(args, corrid, prefix, count, streams, refused, rep == 0, name))
                corrid += streams
                continue
            ids = prompt_ids(corrid, prefix, args.max_id, refused)
            t0 = time.perf_counter()
            last = next_token(client, corrid, ids, start=True)
            prefix_s = time.perf_counter() - t0
            if kind == "decode":
                for _ in range(args.warmup):
                    last = next_token(client, corrid, [last])
                reqs = [[None] for _ in range(count)]
            else:
                reqs = [prompt_ids(corrid * 100 + k, args.chunk, args.max_id, refused) for k in range(count)]
            window = rep == 0 and args.window_start
            s0 = stats(args.http, args.model)
            if window:
                subprocess.run(args.window_start.format(name=name), shell=True, check=True)
            ms = []
            for r in reqs:
                toks = [last] if r == [None] else r
                t = time.perf_counter()
                last = next_token(client, corrid, toks)
                ms.append((time.perf_counter() - t) * 1e3)
            if window:
                subprocess.run(args.window_stop.format(name=name), shell=True, check=True)
            s1 = stats(args.http, args.model)
            client.end(corrid)
            n = s1["count"] - s0["count"]
            d = {k: (s1[k] - s0[k]) for k in s0}
            runs.append({"clientMs": [round(x, 2) for x in ms],
                         "clientMedianMs": round(statistics.median(ms), 2),
                         "serverInferMs": round(d["infer"] / max(n, 1) / 1e6, 2),
                         "serverQueueMs": round(d["queue"] / max(n, 1) / 1e6, 2),
                         "serverInputMs": round(d["input"] / max(n, 1) / 1e6, 3),
                         "serverOutputMs": round(d["output"] / max(n, 1) / 1e6, 3),
                         "requests": n, "executions": d["executions"],
                         "prefixSeconds": round(prefix_s, 2), "profiled": bool(window)})
        med = statistics.median(r["clientMedianMs"] for r in runs)
        inf = statistics.median(r["serverInferMs"] for r in runs)
        q = statistics.median(r["serverQueueMs"] for r in runs)
        line = {"workload": name, "kind": kind, "prefix": prefix, "count": count,
                "clientMedianMs": med, "serverInferMedianMs": inf, "serverQueueMedianMs": q,
                "runs": runs}
        print(f"{name}: client {med:.2f} ms per request (median of {len(runs)} run medians: "
              f"{[r['clientMedianMs'] for r in runs]}), server compute {inf:.2f} ms, "
              f"queue {q:.2f} ms", flush=True)
        results.append(line)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
