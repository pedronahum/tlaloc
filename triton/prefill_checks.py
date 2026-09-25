#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""Checks of batched prefill through Triton's sequence batcher.

Run by verify.sh against a sequence-mode model whose artifact has prefill
entries of batch > 1. For N = 1, 2 and 4 prompts (up to the model's
max_batch_size), of different lengths:

  solo       each prompt is sent alone with START (one prefill call of batch
             1), then decoded greedily for --steps tokens; this is the
             reference
  together   the N prompts are sent at once from N threads, so that the
             sequence batcher hands them to the backend in one batch, which
             prefills them in one call; each sequence is then decoded on
             concurrently. Checks, per prompt: the last-token logits are
             within 5e-3 of the largest solo logit (the batched call is a
             different compiled entry, and the GPU runs f32 matmuls in TF32),
             the argmax is the solo argmax (required with --require-ids, and
             otherwise whenever the solo top two logits are further apart
             than the bound), and with --require-ids the decoded ids equal
             the solo ids. For N > 1 the server must have run the N prompts
             in fewer executions than N.
  throughput the median time, over --repeats rounds, from sending N prompts
             together to the last prefill answer, against the N prompts sent
             one after the other; prompts per second and prompt tokens per
             second

Prompts are given as token ids (--prompts), taken from a greedy fixture
(--fixture: its two prompts, and each followed by the first tokens of its
continuation), or seeded random ids. With --perturb each
batched row is compared with the NEXT prompt's solo row, so the run must fail.
Exit status 0 only if every check passes.
"""

import argparse
import json
import sys
import threading
import time
import urllib.request

import numpy as np

from sequence_client import SequenceClient

BOUND = 5e-3

failures = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        failures.append(what)


def model_config(http, model):
    with urllib.request.urlopen(f"http://{http}/v2/models/{model}/config") as r:
        cfg = json.load(r)
    p = {k: v["string_value"] for k, v in cfg.get("parameters", {}).items()}
    vocab = int(cfg["output"][0]["dims"][0])
    return int(cfg["max_batch_size"]), int(p["max_context"]), vocab


def stats(http, model):
    with urllib.request.urlopen(f"http://{http}/v2/models/{model}/stats") as r:
        s = json.load(r)["model_stats"][0]
    return int(s["inference_count"]), int(s["execution_count"])


def prompts_for(args, vocab):
    if args.prompts:
        return [[int(t) for t in p.split(",")] for p in args.prompts.split(";")]
    if args.fixture:
        # The fixture's two prompts, and each followed by the start of its
        # greedy continuation: four natural prompts of different lengths.
        fx = json.load(open(args.fixture))["prompts"]
        a, b = fx[0], fx[1]
        return [a["promptTokens"], b["promptTokens"],
                a["promptTokens"] + a["generatedTokens"][:6], b["promptTokens"] + b["generatedTokens"][:3]]
    rng = np.random.default_rng(args.seed)
    # The window models refuse their last two ids (placeholder stand-ins).
    return [[int(t) for t in rng.integers(0, vocab - 2, n)] for n in (13, 5, 21, 9)]


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


class Run:
    """Sequences with fresh correlation ids."""

    def __init__(self, base):
        self.next = base

    def id(self):
        self.next += 1
        return self.next


def solo(client, ids, prompt, steps):
    """Prefill alone, then greedy decoding. Returns (prefill logits, ids, prefill seconds)."""
    cid = ids.id()
    t0 = time.perf_counter()
    logits = client.step(cid, prompt, start=True)
    dt = time.perf_counter() - t0
    first = logits
    out = []
    if steps == 0:
        client.end(cid)
    for k in range(steps):
        out.append(int(np.argmax(logits)))
        logits = client.step(cid, [out[-1]], end=k == steps - 1)
    return first, out, dt


def together(url, model, protocol, ids, prompts, steps, http):
    """The prompts sent at once from one thread each, then decoded on
    concurrently. Returns ([prefill logits], [[ids]], seconds to the last
    prefill answer, executions the prefills took)."""
    n = len(prompts)
    clients = [SequenceClient(url, model, protocol) for _ in range(n)]
    cids = [ids.id() for _ in range(n)]
    go = threading.Barrier(n + 1)
    prefilled = threading.Barrier(n + 1)
    resume = threading.Barrier(n + 1)
    first = [None] * n
    out = [[] for _ in range(n)]
    done_at = [0.0] * n
    errors = []

    def worker(i):
        try:
            # Connect before the barrier, so that the N requests leave together
            # and reach the sequence batcher within its queue delay.
            clients[i].client.is_server_live()
            go.wait()
            logits = clients[i].step(cids[i], prompts[i], start=True)
            done_at[i] = time.perf_counter()
            first[i] = logits
            prefilled.wait()
            resume.wait()
            if steps == 0:
                clients[i].end(cids[i])
            for k in range(steps):
                out[i].append(int(np.argmax(logits)))
                logits = clients[i].step(cids[i], [out[i][-1]], end=k == steps - 1)
        except Exception as e:  # noqa: BLE001 - reported below
            errors.append(f"prompt {i}: {e}")
            for b in (prefilled, resume):
                b.abort()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    before = stats(http, model)
    try:
        go.wait()
        t0 = time.perf_counter()
        prefilled.wait()
        after = stats(http, model)
        resume.wait()
    except threading.BrokenBarrierError:
        pass
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError("; ".join(errors))
    return first, out, max(done_at) - t0, after[1] - before[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--http", default="localhost:8000")
    ap.add_argument("--grpc", default="localhost:8001")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", help="the prompts' token ids: comma-separated ids, prompts separated by ';'")
    ap.add_argument("--fixture", help="a greedy fixture JSON whose two prompts (and their continuations) to use")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--steps", type=int, default=8, help="greedy tokens decoded after each prefill")
    ap.add_argument("--repeats", type=int, default=5, help="rounds of the throughput measurement")
    ap.add_argument("--require-ids", action="store_true",
                    help="require the argmax and the decoded ids to equal the solo run's")
    ap.add_argument("--id-base", type=int, default=90000)
    ap.add_argument("--perturb", action="store_true",
                    help="compare each batched row with the next prompt's solo row (must fail)")
    args = ap.parse_args()

    max_batch, max_context, vocab = model_config(args.http, args.model)
    prompts = prompts_for(args, vocab)
    for p in prompts:
        if len(p) + args.steps > max_context:
            print(f"FAIL a prompt of {len(p)} tokens and {args.steps} steps exceed context {max_context}")
            return 1
    sizes = [n for n in (1, 2, 4) if n <= max_batch]
    print(f"model {args.model}: max_batch_size {max_batch}, context {max_context}; prompts of "
          f"{[len(p) for p in prompts]} tokens; N = {sizes}")
    ids = Run(args.id_base)

    ref = {}
    for i, p in enumerate(prompts):
        c = SequenceClient(args.http, args.model, "http")
        ref[i] = solo(c, ids, p, args.steps)

    for protocol, url in (("http", args.http), ("grpc", args.grpc)):
        print(f"== {protocol}: prompts prefilled together against alone")
        for n in sizes:
            first, out, _, execs = together(url, args.model, protocol, ids, prompts[:n], args.steps, args.http)
            for i in range(n):
                j = (i + 1) % n if args.perturb and n > 1 else i
                want, want_ids, _ = ref[j]
                got = first[i]
                scale = float(np.max(np.abs(want)))
                diff = float(np.max(np.abs(got - want))) / scale
                top2 = np.sort(want)[-2:]
                margin = float(top2[1] - top2[0]) / scale
                check(diff <= BOUND, f"N={n} prompt {i} ({len(prompts[i])} tokens): logits within "
                                     f"{diff:.1e} of the largest solo logit (bound {BOUND:g})")
                same = int(np.argmax(got)) == int(np.argmax(want))
                if args.require_ids or margin > BOUND:
                    check(same, f"N={n} prompt {i}: argmax {int(np.argmax(got))} == solo "
                                f"{int(np.argmax(want))} (solo top-two gap {margin:.1e})")
                else:
                    print(f"  note N={n} prompt {i}: argmax {int(np.argmax(got))}, solo "
                          f"{int(np.argmax(want))}; the solo top two are {margin:.1e} apart, within "
                          f"the bound, so equality is not required")
                if args.require_ids:
                    check(out[i] == want_ids, f"N={n} prompt {i}: {args.steps} decoded ids {out[i]} "
                                              f"== solo {want_ids}")
            if n > 1:
                check(execs < n, f"N={n}: the {n} prompts ran in {execs} execution(s)")

    print("== throughput (prefill only; END after the timed prefill)")
    client = SequenceClient(args.http, args.model, "http")
    for n in sizes:
        tokens = sum(len(p) for p in prompts[:n])
        together_s, alone_s, execs = [], [], []
        for _ in range(args.repeats):
            _, _, secs, e = together(args.http, args.model, "http", ids, prompts[:n], 0, args.http)
            together_s.append(secs)
            execs.append(e)
            t0 = time.perf_counter()
            for p in prompts[:n]:
                solo(client, ids, p, 0)
            alone_s.append(time.perf_counter() - t0)
        t, a = median(together_s), median(alone_s)
        print(f"  N={n}: {tokens} prompt tokens; together {t * 1e3:.1f} ms ({n / t:.1f} prompts/s, "
              f"{tokens / t:.0f} tokens/s; executions per round {sorted(execs)}); one after the "
              f"other {a * 1e3:.1f} ms ({n / a:.1f} prompts/s)")

    print()
    print(f"{len(failures)} check(s) FAILED" if failures else "all prefill checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
