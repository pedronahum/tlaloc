#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""Prefill and decode timings of a sequence-mode model at several contexts.

For each context C given (each must be a context bucket of the model), four
sequences get a prompt of L tokens (L = C - 80 by default, so that the
prompt and the decode steps below stay inside the bucket and above the one
before it), sent one after the other with START over gRPC: the prefill time
of each is recorded. Then, with all four sequences held, decode is timed for
N = 1, 2 and 4 of them at once (up to max_batch_size): N threads each send
--steps single-token requests, the next one as soon as the last answer is
back, so Triton's sequence batcher runs their steps together. Reported per
N: the median time of one step as a client sees it (ms per token of one
sequence) and the aggregate tokens per second (N * steps over the wall time
of the phase), and how many executions (batches) Triton ran for the
N * steps requests, from the model's statistics (--http). The prompts are seeded random ids below --max-id (they skip
the model's refused ids); the tokens decoded are the greedy argmax.

The script also samples the machine's memory in use (MemTotal minus
MemAvailable, which on a unified-memory GPU includes the device's
allocations) every 0.2 s and reports the peak during each context.

  context_bench.py --grpc localhost:8001 --model muse --contexts 512,2048,8192,32768

Prints one line per measurement and, with --json FILE, writes them as JSON.
"""

import argparse
import json
import sys
import threading
import time
import urllib.request

import numpy as np

from sequence_client import SequenceClient


class Peak:
    """Samples memory in use (GiB) on a background thread."""

    def __init__(self):
        self.peak = 0.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    @staticmethod
    def used():
        vals = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, v = line.split(":", 1)
                vals[k] = int(v.split()[0])
        return (vals["MemTotal"] - vals["MemAvailable"]) / 1048576

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.used())
            time.sleep(0.2)

    def reset(self):
        self.peak = self.used()

    def stop(self):
        self._stop.set()


def stats(http, model):
    """(inference_count, execution_count) of the model: requests, and the
    batches Triton handed the backend."""
    with urllib.request.urlopen(f"http://{http}/v2/models/{model}/stats") as r:
        s = json.load(r)["model_stats"][0]
    return int(s["inference_count"]), int(s["execution_count"])


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grpc", default="localhost:8001")
    ap.add_argument("--http", default="localhost:8000", help="for the model's statistics")
    ap.add_argument("--model", required=True)
    ap.add_argument("--contexts", required=True, help="comma-separated context buckets")
    ap.add_argument("--prompt-lengths", default=None,
                    help="comma-separated prompt lengths, one per context (default: context - 80)")
    ap.add_argument("--batches", default="1,2,4")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--max-id", type=int, default=200000, help="prompt ids are drawn below this")
    ap.add_argument("--first-corrid", type=int, default=90000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    contexts = [int(c) for c in args.contexts.split(",")]
    lengths = ([int(x) for x in args.prompt_lengths.split(",")] if args.prompt_lengths
               else [c - 80 for c in contexts])
    batches = [int(b) for b in args.batches.split(",")]
    seqs = max(batches)
    peak = Peak()
    results = []
    corrid = args.first_corrid

    for ctx, n_prompt in zip(contexts, lengths):
        assert n_prompt + 3 * args.steps < ctx, f"prompt {n_prompt} + decode steps do not fit context {ctx}"
        peak.reset()
        clients = [SequenceClient(args.grpc, args.model, "grpc") for _ in range(seqs)]
        ids = list(range(corrid, corrid + seqs))
        corrid += seqs
        last = {}
        prefill_ms = []
        for k, (c, sid) in enumerate(zip(clients, ids)):
            rng = np.random.default_rng(1000 * ctx + k)
            prompt = [int(t) for t in rng.integers(1, args.max_id, n_prompt)]
            t0 = time.perf_counter()
            logits = c.step(sid, prompt, start=True)
            prefill_ms.append((time.perf_counter() - t0) * 1e3)
            last[sid] = int(np.argmax(logits))
        pre = median(prefill_ms)
        line = {"context": ctx, "promptTokens": n_prompt, "prefillMs": [round(x, 1) for x in prefill_ms],
                "prefillMedianMs": round(pre, 1), "prefillTokensPerSecond": round(n_prompt / pre * 1e3, 1)}
        print(f"context {ctx}: prefill of {n_prompt} tokens, median {pre:.0f} ms of {len(prefill_ms)} "
              f"({n_prompt / pre * 1e3:.0f} tokens/s); each {[round(x) for x in prefill_ms]}", flush=True)
        line["decode"] = []
        for n in batches:
            step_ms = {sid: [] for sid in ids[:n]}
            barrier = threading.Barrier(n + 1)

            def run(i, sid):
                barrier.wait()
                for _ in range(args.steps):
                    t0 = time.perf_counter()
                    logits = clients[i].step(sid, [last[sid]])
                    step_ms[sid].append((time.perf_counter() - t0) * 1e3)
                    last[sid] = int(np.argmax(logits))

            threads = [threading.Thread(target=run, args=(i, sid)) for i, sid in enumerate(ids[:n])]
            for t in threads:
                t.start()
            inf0, exe0 = stats(args.http, args.model)
            barrier.wait()
            t0 = time.perf_counter()
            for t in threads:
                t.join()
            wall = time.perf_counter() - t0
            inf1, exe1 = stats(args.http, args.model)
            per = median([x for v in step_ms.values() for x in v])
            agg = n * args.steps / wall
            execs = exe1 - exe0
            line["decode"].append({"sequences": n, "stepMedianMs": round(per, 1),
                                   "aggregateTokensPerSecond": round(agg, 2),
                                   "steps": inf1 - inf0, "executions": execs})
            print(f"context {ctx}: {n} sequence(s) decoding together: {per:.0f} ms per step, "
                  f"{agg:.2f} tokens/s in all; {inf1 - inf0} steps in {execs} executions", flush=True)
        for c, sid in zip(clients, ids):
            c.end(sid)
        line["peakMemoryGiB"] = round(peak.peak, 1)
        print(f"context {ctx}: peak memory in use {peak.peak:.1f} GiB", flush=True)
        results.append(line)

    peak.stop()
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
