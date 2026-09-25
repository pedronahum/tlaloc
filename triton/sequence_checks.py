#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""End-to-end checks of the tlaloc backend's sequence mode on TinyLlama-1.1B.

Run by verify.sh against a server holding the `tinyllama` sequence-mode model
(exported with a 5 s sequence idle timeout). Checks:

  prefill    the prompt as one prefill request gives the same last-token
             logits as the prompt sent one token per request (argmax equal,
             every logit within 5e-3 of the largest: TF32), and greedy
             decoding from it reproduces HuggingFace's ids, over HTTP and gRPC
  concurrent four sequences with different prompts, decoded concurrently
             from four threads, each get exactly the ids they get alone, and
             the server batched their decode steps
  end-frees  more sequences than the KV pool holds at once, one after the
             other, all succeed, because END returns their pages
  exhaustion with the pool full, one more START is refused by name, the
             server stays live, the refused sequence has no state, and after
             ending the others a new sequence runs
  idle       sequences abandoned without END lose their pages once they have
             been idle longer than max_sequence_idle_microseconds

Prints timings (prefill, ms per decode step alone and concurrently). Exit
status 0 only if every check passes. With --perturb the expected ids are
wrong and only the prefill checks run, so the run must fail.
"""

import argparse
import json
import sys
import threading
import time
import urllib.request

import numpy as np

from sequence_client import SequenceClient

HF_IDS = [3681, 29889, 13, 13, 29906, 29889]  # " Paris.\n\n2."
FRANCE = [1, 450, 7483, 310, 3444, 338]  # BOS + "The capital of France is"
PROMPTS = [
    FRANCE,
    [1, 450, 10150, 15754, 297, 278, 21635, 1788, 338],  # The largest planet in the solar system is
    [1, 1619, 25448, 2927, 338],  # My favorite color is
    [1, 450, 937, 6673, 310, 278, 3303, 3900, 471],  # The first president of the United States was
]

failures = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        failures.append(what)


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def config(url, model):
    with urllib.request.urlopen(f"http://{url}/v2/models/{model}/config") as r:
        cfg = json.load(r)
    p = {k: v["string_value"] for k, v in cfg.get("parameters", {}).items()}
    idle_us = int(cfg["sequence_batching"]["max_sequence_idle_microseconds"])
    return int(p["kv_block_size"]), int(p["kv_num_blocks"]), int(p["max_context"]), idle_us


def stats(url, model):
    with urllib.request.urlopen(f"http://{url}/v2/models/{model}/stats") as r:
        s = json.load(r)["model_stats"][0]
    return int(s["inference_count"]), int(s["execution_count"])


def refused(fn, needle):
    """(True, message) if fn() raised an error whose message contains needle."""
    try:
        fn()
    except Exception as e:  # tritonclient raises InferenceServerException
        return needle in str(e), str(e)
    return False, "no error"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--http", default="localhost:8000")
    ap.add_argument("--grpc", default="localhost:8001")
    ap.add_argument("--model", default="tinyllama")
    ap.add_argument("--perturb", action="store_true", help="expect wrong ids (negative control)")
    args = ap.parse_args()

    expect = list(HF_IDS)
    if args.perturb:
        expect[0] += 1
    block, blocks, max_context, idle_us = config(args.http, args.model)
    pages = blocks - 1
    print(f"model {args.model}: {pages} KV pages of {block} tokens, context {max_context}, "
          f"idle timeout {idle_us / 1e6:.1f} s")
    http = SequenceClient(args.http, args.model, "http")
    grpc = SequenceClient(args.grpc, args.model, "grpc")
    next_id = [1000]

    def fresh():
        next_id[0] += 1
        return next_id[0]

    # -- prefill -------------------------------------------------------------
    print("== prefill")
    http.generate(fresh(), FRANCE, 2)  # warm-up: first calls of each executable
    sid = fresh()
    pre = http.step(sid, FRANCE, start=True, end=True)
    sid = fresh()
    one = http.step(sid, FRANCE[:1], start=True)
    for t in FRANCE[1:-1]:
        http.step(sid, [t])
    loop = http.step(sid, FRANCE[-1:], end=True)
    scale = float(np.max(np.abs(loop)))
    diff = float(np.max(np.abs(pre - loop)))
    check(int(np.argmax(pre)) == int(np.argmax(loop)),
          f"prefill argmax {int(np.argmax(pre))} == one-token-per-request argmax {int(np.argmax(loop))}")
    # Two different executables (a 64-row prefill and a 1-row decode step),
    # each with f32 matmuls that XLA runs as TF32 on this GPU, through 22
    # layers: measured differences are 9e-3 to 1.3e-2 on logits of about 14
    # (the JVM interpreter, in f32, gives the two bit for bit).
    check(diff <= 5e-3 * scale,
          f"prefill logits within 5e-3 of the largest logit of the decode loop's "
          f"(max |diff| {diff:.2e}, largest {scale:.2f})")
    check(one is not None and one.shape == loop.shape, "a one-token START returns logits of the vocabulary's width")

    prefill_ms, step_ms = [], []
    for proto, c in (("http", http), ("grpc", grpc)):
        ids, p_ms, steps, _ = c.generate(fresh(), FRANCE, 6)
        prefill_ms.append(p_ms)
        step_ms += steps
        check(ids == expect, f"{proto}: greedy ids {ids} == HuggingFace's {expect}")
    for _ in range(3):
        _, p_ms, steps, _ = http.generate(fresh(), FRANCE, 6)
        prefill_ms.append(p_ms)
        step_ms += steps
    alone_step = median(step_ms)
    print(f"     prefill of {len(FRANCE)} tokens: median {median(prefill_ms):.1f} ms; "
          f"decode: median {alone_step:.1f} ms per token (one sequence)")
    if args.perturb:
        # The negative control needs only the checks above to fail.
        print(f"{len(failures)} check(s) FAILED")
        return 1 if failures else 0

    # -- concurrent ----------------------------------------------------------
    print("== concurrent")
    n_new = 12
    single = [http.generate(fresh(), p, n_new)[0] for p in PROMPTS]
    check(len({tuple(s) for s in single}) == len(PROMPTS),
          "the four prompts have four different continuations (so the comparison can fail)")
    results, errors = [None] * len(PROMPTS), []
    barrier = threading.Barrier(len(PROMPTS))
    ids0 = [fresh() for _ in PROMPTS]

    def worker(i):
        try:
            c = SequenceClient(args.http, args.model, "http")
            barrier.wait()
            results[i] = c.generate(ids0[i], PROMPTS[i], n_new)
        except Exception as e:  # noqa: BLE001 - reported below
            errors.append(f"sequence {i}: {e}")

    inf0, exe0 = stats(args.http, args.model)
    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(PROMPTS))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    inf1, exe1 = stats(args.http, args.model)
    check(not errors, "no concurrent request failed" + (f": {errors}" if errors else ""))
    for i in range(len(PROMPTS)):
        got = results[i][0] if results[i] else None
        check(got == single[i], f"sequence {i}: concurrent ids == its ids alone ({single[i][:6]}...)")
    requests = inf1 - inf0
    executions = exe1 - exe0
    check(executions < requests,
          f"decode steps were batched: {requests} requests ran in {executions} executions")
    conc_steps = [s for r in results if r for s in r[2]]
    tokens = len(PROMPTS) * n_new
    print(f"     {len(PROMPTS)} sequences x {n_new} tokens in {wall * 1e3:.0f} ms: "
          f"{tokens / wall:.1f} tokens/s, median {median(conc_steps):.1f} ms per decode step "
          f"(one sequence alone: {alone_step:.1f} ms, {1e3 / alone_step:.1f} tokens/s)")

    # -- END frees pages -------------------------------------------------------
    print("== end-frees")
    per_seq = 3  # pages per sequence below: a 3-page prompt, one decode step
    long_prompt = [1] + [450] * (per_seq * block - 2)
    fits = pages // per_seq
    ok = True
    for i in range(2 * fits + 1):
        ids, _, _, _ = http.generate(fresh(), long_prompt, 2)
        ok &= len(ids) == 2
    check(ok, f"{2 * fits + 1} sequences of {per_seq} pages, one after the other, all ran "
              f"(the pool holds {fits} at once)")

    # -- exhaustion ----------------------------------------------------------
    print("== exhaustion")
    held = []
    for i in range(fits):
        sid = fresh()
        http.step(sid, long_prompt, start=True)
        held.append(sid)
    extra = fresh()
    hit, msg = refused(lambda: http.step(extra, long_prompt, start=True), "KV page pool exhausted")
    check(hit, f"one more START is refused by name: {msg[:160]}")
    with urllib.request.urlopen(f"http://{args.http}/v2/health/live") as r:
        check(r.status == 200, "the server is live after the refusal")
    hit, msg = refused(lambda: http.step(extra, [450]), "has no KV state")
    check(hit, f"the refused sequence has no state: {msg[:120]}")
    http.end(extra)  # releases the sequence in Triton; the backend holds nothing for it
    for sid in held:
        http.end(sid)
    ids, _, _, _ = http.generate(fresh(), FRANCE, 6)
    check(ids == expect, "after ending the held sequences a new sequence decodes correctly")

    # -- idle ----------------------------------------------------------------
    print("== idle")
    abandoned = []
    for i in range(fits):
        sid = fresh()
        http.step(sid, long_prompt, start=True)
        abandoned.append(sid)
    hit, _ = refused(lambda: http.step(fresh(), long_prompt, start=True), "KV page pool exhausted")
    check(hit, "the pool is full of abandoned sequences")
    wait = idle_us / 1e6 + 2.0
    print(f"     waiting {wait:.1f} s for the idle timeout")
    time.sleep(wait)
    ids, _, _, _ = http.generate(fresh(), FRANCE, 6)
    check(ids == expect, "after the idle timeout a new sequence gets pages and decodes correctly")
    hit, msg = refused(lambda: http.step(abandoned[0], [450]), "")
    check(hit and ("START" in msg or "no KV state" in msg),
          f"a timed-out sequence's next step is refused: {msg[:160]}")
    again = []
    for i in range(fits):
        sid = fresh()
        http.step(sid, long_prompt, start=True)
        again.append(sid)
    check(True, f"{fits} sequences of {per_seq} pages hold the pool again")
    for sid in again:
        http.end(sid)

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED")
        return 1
    print("all sequence checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
