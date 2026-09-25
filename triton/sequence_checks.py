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
  pressure   with the pool full of live sequences, one of them needs a page
             mid-generation: the request is refused by name (the refusal lists
             the sequences holding pages and says the sequence keeps its KV),
             no live sequence is reclaimed, and once another sequence ends the
             same request succeeds and the sequence continues with exactly the
             ids it gets alone
  idle       sequences abandoned without END lose their pages once they have
             been idle for twice max_sequence_idle_microseconds (plus a few
             executions of queueing allowance) and a new sequence needs them;
             with --log (the server log) the pages are reclaimed least recently
             active first and only as many as a request needs

Prints timings (prefill, ms per decode step alone and concurrently). Exit
status 0 only if every check passes. With --perturb the expected ids are
wrong and only the prefill and pressure checks run, so the run must fail.

With --queued (against a copy of the model with a 200 ms idle timeout) it
checks only that the backend never frees a sequence Triton still holds: 12 to
32 sequences decode concurrently, so steps wait in Triton's queue for longer
than the timeout. Triton ends some sequences itself (their next step is
refused for lacking START, which is Triton's rule), but no step Triton
accepts may be refused by the backend for having no KV state. (A START the
full page pool refuses by name is allowed.)
"""

import argparse
import json
import os
import re
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


def reclaimed(log_path, since):
    """The sequence ids the server log says were reclaimed after byte `since`, in order."""
    if not log_path:
        return None
    time.sleep(1.0)  # the log is followed from the container: let it catch up
    with open(log_path, errors="replace") as fh:
        fh.seek(since)
        text = fh.read()
    return [int(m) for m in re.findall(r"reclaimed sequence (\d+) \(the least recently active", text)]


def log_size(log_path):
    return os.path.getsize(log_path) if log_path else 0


def text_prompt(n):
    """BOS and the four prompts' words, repeated, to n tokens: a prompt whose
    continuation is not one repeated id."""
    words = [t for p in PROMPTS for t in p[1:]]
    return [1] + (words * (n // len(words) + 1))[: n - 1]


def pressure(http, fresh, long_prompt, grower_prompt, fits, expect_solo, log_path):
    """A pool full of live sequences, and one of them needs a page."""
    print("== pressure")
    n_new = len(expect_solo)
    check(len(set(expect_solo)) > 1, f"the solo continuation {expect_solo} is not one repeated id")
    held = []
    first = None
    for i in range(fits):
        sid = fresh()
        logits = http.step(sid, grower_prompt if i == 0 else long_prompt, start=True)
        held.append(sid)
        if first is None:
            first = logits
    grower = held[0]
    mark = log_size(log_path)
    # The prompt ends one position before its last page is full: the first
    # generated token fills it, the second needs a new page.
    ids = [int(np.argmax(first))]
    logits = http.step(grower, ids[-1:])
    ids.append(int(np.argmax(logits)))
    hit, msg = refused(lambda: http.step(grower, ids[-1:]), "KV page pool exhausted")
    check(hit, f"sequence {grower} needing a page mid-generation with the pool full is refused by "
               f"name: {msg[:200]}")
    check("keeps its 3 pages and the KV of its" in msg,
          "the refusal says the sequence keeps its pages and KV and the request can be sent again")
    check(f"{fits - 1} other sequence(s) hold pages" in msg and "is not preempted" in msg,
          f"the refusal lists the {fits - 1} other sequences holding pages")
    got = reclaimed(log_path, mark)
    if got is not None:
        check(got == [], f"no live sequence was reclaimed for it (log: {got})")
    http.end(held[1])
    logits = http.step(grower, ids[-1:])
    check(logits is not None, "after another sequence ended, the same request succeeds")
    while len(ids) < n_new:
        ids.append(int(np.argmax(logits)))
        if len(ids) < n_new:
            logits = http.step(grower, ids[-1:])
    check(ids == expect_solo,
          f"the refused sequence continues with its ids alone: {ids} == {expect_solo}")
    for sid in held[:1] + held[2:]:
        http.end(sid)


def queued(url, model):
    """The backend must not free a sequence whose next request is queued."""
    import collections

    triton_ended, backend_lost, finished = 0, 0, 0
    for n, base in ((12, 70000), (24, 71000), (32, 72000)):
        errors = []
        barrier = threading.Barrier(n)

        def worker(i):
            c = SequenceClient(url, model, "http")
            barrier.wait()
            try:
                c.generate(base + i, FRANCE, 20)
            except Exception as e:  # noqa: BLE001 - classified below
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        kinds = collections.Counter(
            "triton" if "must specify the START flag" in e
            else "backend" if "has no KV state" in e
            else "pool" if "KV page pool exhausted" in e else e[:120]
            for e in errors)
        triton_ended += kinds.pop("triton", 0)
        kinds.pop("pool", 0)  # a refusal by name: 32 sequences can need more pages than the pool
        backend_lost += kinds.pop("backend", 0)
        finished += n - len(errors)
        print(f"     {n} concurrent sequences: {n - len(errors)} finished, "
              f"{len(errors)} ended early {dict(kinds) if kinds else ''}")
        check(not kinds, f"{n} sequences: no error other than Triton's idle refusal and "
                         f"the full page pool's")
    check(triton_ended > 0,
          f"steps queued longer than the timeout: Triton ended {triton_ended} sequences "
          f"itself (so the check below can fail)")
    check(backend_lost == 0,
          f"no step Triton accepted was refused for having no KV state "
          f"({backend_lost} were; {finished} sequences finished)")
    print()
    print(f"{len(failures)} check(s) FAILED" if failures else "all queued checks passed")
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--http", default="localhost:8000")
    ap.add_argument("--grpc", default="localhost:8001")
    ap.add_argument("--model", default="tinyllama")
    ap.add_argument("--perturb", action="store_true", help="expect wrong ids (negative control)")
    ap.add_argument("--queued", action="store_true",
                    help="only the queued-sequence check (a model with a short idle timeout)")
    ap.add_argument("--log", default="", help="the server log, to check which sequences were reclaimed")
    args = ap.parse_args()
    if args.queued:
        return queued(args.http, args.model)

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
    per_seq = 3  # pages per sequence below: a 3-page prompt, one decode step
    long_prompt = [1] + [450] * (per_seq * block - 2)
    fits = pages // per_seq
    grower_prompt = text_prompt(len(long_prompt))
    solo = http.generate(fresh(), grower_prompt, 6)[0]
    if args.perturb:
        solo[-1] += 1
    pressure(http, fresh, long_prompt, grower_prompt, fits, solo, args.log)
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
    # The backend frees a sequence idle for twice the timeout plus a queueing
    # allowance of a few executions (Triton ends it after one timeout, counted
    # from the arrival of its last request).
    wait = 2 * idle_us / 1e6 + 5.0
    print(f"     waiting {wait:.1f} s for twice the idle timeout")
    time.sleep(wait)
    mark = log_size(args.log)
    ids, _, _, _ = http.generate(fresh(), FRANCE, 6)
    check(ids == expect, "after the idle timeout a new sequence gets pages and decodes correctly")
    got = reclaimed(args.log, mark)
    if got is not None:
        check(got == abandoned[:1],
              f"the new sequence needed one page and got it from the least recently active "
              f"abandoned sequence only (reclaimed {got}, expected {abandoned[:1]})")
    hit, msg = refused(lambda: http.step(abandoned[0], [450]), "")
    check(hit and ("START" in msg or "no KV state" in msg),
          f"a timed-out sequence's next step is refused: {msg[:160]}")
    again = []
    mark = log_size(args.log)
    for i in range(fits):
        sid = fresh()
        http.step(sid, long_prompt, start=True)
        again.append(sid)
    check(True, f"{fits} sequences of {per_seq} pages hold the pool again")
    got = reclaimed(args.log, mark)
    if got is not None:
        check(got == abandoned[1:],
              f"the other {fits - 1} abandoned sequences were reclaimed as new ones needed pages, "
              f"least recently active first (in order: {got == abandoned[1:]})")
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
