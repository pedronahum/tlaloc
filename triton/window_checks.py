#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""Checks of the windowed KV pool through Triton.

Run by verify.sh against the example repository, which holds the same
three-layer decoder twice: `window_sequence`, whose two sliding-window layers
(window 8, pages of 4) keep their KV in a windowed pool of 3-page rings, and
`window_sequence_full`, where every layer keeps full-history pages. Checks:

  pages      one sequence grown to 60 positions, by a 13-token prompt, a
             9-token request once the window is full (split by the backend
             into calls the ring can hold), and single tokens: after every
             request KV_PAGES reports ceil(length / 4) full pages for both
             models, and min(3, ceil(length / 4)) windowed pages for
             window_sequence (0 for the full-history model)
  logits     every request's logits from window_sequence equal, within
             1e-4 of the largest, those of window_sequence_full sent the same
             calls (the two are different executables, and agree to f32
             rounding; a ring error moves them by about 1) (a request
             the backend splits for the ring is sent to the full-history model
             already split, so both run the same entries and differ only in
             where the keys and values are), over HTTP and gRPC
  unsplit    the full-history model sent the requests whole (one prefill call
             where window_sequence makes two) agrees within 5e-3 of the
             largest logit, with the same argmax: TF32 matmuls in a different
             entry, as in sequence_checks.py
  batched    three sequences stepped concurrently from three threads: each
             gets the logits it gets alone, within 5e-3 of the largest (a
             batched step runs a different compiled entry; with random weights
             two logits can be closer than that, so the argmax is reported
             and not required)
  exhausted  with the windowed pool's rings taken, one more START is refused
             by name, and after END a new sequence runs
  refused    ids 62 and 63, which the manifest lists as the model's image and
             video placeholders, are refused by name (in a START and in a
             later request, over HTTP and gRPC); the refused request leaves the
             sequence as it was (its next logits equal those of a sequence that
             never sent it); id 61 (the control) is accepted

  chunked    window_sequence_chunked, whose prefill entries take at most 6
             tokens per sequence and whose ring is 4 pages, grown the same
             way: the 13- and 9-token requests run as several calls, it holds
             at most 4 windowed pages, and its logits equal the full-history
             model's sent the same calls (within 1e-4) and sent the requests
             whole (within 5e-3, the same argmax)

Exit status 0 only if every check passes. With --perturb each request of
window_sequence is compared with the full-history model's NEXT request, so the
run must fail.
"""

import argparse
import json
import sys
import threading
import urllib.request

import numpy as np

from sequence_client import SequenceClient

failures = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        failures.append(what)


def params(http, model):
    with urllib.request.urlopen(f"http://{http}/v2/models/{model}/config") as r:
        cfg = json.load(r)
    return {k: v["string_value"] for k, v in cfg.get("parameters", {}).items()}


# Ids 62 and 63 stand in for image and video placeholders and are refused.
REFUSED = {62: "image_token_id", 63: "video_token_id"}


def tokens(seed, n, vocab=62):
    return [int(t) for t in np.random.default_rng(seed).integers(0, vocab, n)]


def split(start, n, ring, bs, window, chunk=None):
    """The calls the backend makes of a request of n tokens at `start`:
    each at most ring * bs - min(position, window - 1) tokens, and at most
    `chunk` (the prefill entries' tokens per sequence) when given."""
    calls = []
    while n > 0:
        k = min(n, max(1, ring * bs - min(start, window - 1)), chunk or n)
        calls.append(k)
        start, n = start + k, n - k
    return calls


def schedule(client, corrid, seed, ring=None, bs=4, window=8, chunk=None):
    """One sequence of 60 ids drawn with `seed`; returns [(length, logits,
    pages)] per request. With `ring`, each request is sent as the calls the
    backend would split it into for that ring, and the last call's result is
    the request's."""
    ids = tokens(seed, 60)
    requests = [ids[0:13]] + [[t] for t in ids[13:24]] + [ids[24:33]] + [[t] for t in ids[33:60]]
    out, length = [], 0
    for i, req in enumerate(requests):
        parts = [len(req)] if ring is None else split(length, len(req), ring, bs, window, chunk)
        at = 0
        for j, k in enumerate(parts):
            last = i == len(requests) - 1 and j == len(parts) - 1
            logits, pages = client.step_pages(corrid, req[at:at + k], start=length == 0, end=last)
            at += k
            length += k
        out.append((length, logits, pages))
    return out


def worst_rel(pairs):
    """The largest |a - b| over max(1, max |b|), and how many pairs are bit for bit equal."""
    worst, same, argmax = 0.0, 0, True
    for (_, a, _), (_, b, _) in pairs:
        worst = max(worst, float(np.max(np.abs(a - b))) / max(1.0, float(np.max(np.abs(b)))))
        same += int(np.array_equal(a, b))
        argmax &= int(np.argmax(a)) == int(np.argmax(b))
    return worst, same, argmax


def run(proto, url, http, perturb):
    print(f"{proto}  window_sequence vs window_sequence_full  {url}")
    p = params(http, "window_sequence")
    bs, ring = int(p["kv_block_size"]), int(p["kv_window_ring_pages"])
    check(p["kv_window"] == "8" and ring == 3 and bs == 4 and p["kv_window_layers"] == "0,1",
          f"window_sequence states a window of {p['kv_window']} over layers {p['kv_window_layers']}, "
          f"rings of {ring} pages of {bs}")
    check("kv_window" not in params(http, "window_sequence_full"), "window_sequence_full has no windowed pool")
    base = 100 if proto == "http" else 200
    win = schedule(SequenceClient(url, "window_sequence", proto), base + 1, seed=1)
    full = schedule(SequenceClient(url, "window_sequence_full", proto), base + 2, seed=1, ring=ring, bs=bs)
    whole = schedule(SequenceClient(url, "window_sequence_full", proto), base + 3, seed=1)

    want_full = [(n + bs - 1) // bs for n, _, _ in win]
    want_ring = [min(ring, (n + bs - 1) // bs) for n, _, _ in win]
    check([int(pg[0]) for _, _, pg in win] == want_full and [int(pg[1]) for _, _, pg in win] == want_ring,
          f"{proto} window_sequence holds ceil(length / {bs}) full pages and at most {ring} windowed pages")
    check([int(pg[0]) for _, _, pg in full] == want_full and all(int(pg[1]) == 0 for _, _, pg in full),
          f"{proto} window_sequence_full holds ceil(length / {bs}) full pages and no windowed page")
    shown = {n: (int(pg[0]), int(pg[1])) for n, _, pg in win if n in (8, 13, 16, 24, 33, 40, 60)}
    print("       pages held (full, windowed) at length " +
          ", ".join(f"{n}: {v}" for n, v in sorted(shown.items())))

    shift = (lambda xs: xs[1:] + xs[:1]) if perturb else (lambda xs: xs)
    worst, same, _ = worst_rel(list(zip(win, shift(full))))
    check(worst <= 1e-4, f"{proto} logits of all {len(win)} requests equal the full-history model's sent "
                         f"the same calls: worst {worst:.2e} of the largest logit, {same} of {len(win)} bit for bit")
    worst, same, argmax = worst_rel(list(zip(win, shift(whole))))
    check(worst <= 5e-3 and argmax, f"{proto} and the full-history model sent the requests whole: worst "
                                    f"{worst:.2e} of the largest logit, argmax equal: {argmax}")


def chunked(proto, url, http, perturb):
    """window_sequence_chunked: prefill entries of at most 6 tokens per
    sequence, a ring sized for a 6-token call past the window. The backend
    runs a longer request as several calls; the logits must agree with the
    full-history model sent the same calls."""
    print(f"{proto}  window_sequence_chunked vs window_sequence_full  {url}")
    p = params(http, "window_sequence_chunked")
    bs, ring = int(p["kv_block_size"]), int(p["kv_window_ring_pages"])
    check(ring == 4, f"window_sequence_chunked has rings of {ring} pages: ceil((8 - 1 + 6) / 4) = 4")
    base = 400 if proto == "http" else 450
    got = schedule(SequenceClient(url, "window_sequence_chunked", proto), base + 1, seed=1)
    full = schedule(SequenceClient(url, "window_sequence_full", proto), base + 2, seed=1, ring=ring, bs=bs, chunk=6)
    whole = schedule(SequenceClient(url, "window_sequence_full", proto), base + 3, seed=1)
    want_ring = [min(ring, (n + bs - 1) // bs) for n, _, _ in got]
    check([int(pg[1]) for _, _, pg in got] == want_ring,
          f"{proto} window_sequence_chunked holds at most {ring} windowed pages")
    shift = (lambda xs: xs[1:] + xs[:1]) if perturb else (lambda xs: xs)
    worst, same, argmax = worst_rel(list(zip(got, shift(full))))
    check(worst <= 1e-4 and argmax,
          f"{proto} logits of all {len(got)} requests (the 13- and 9-token ones run in calls of at most 6) "
          f"equal the full-history model's sent the same calls: worst {worst:.2e} of the largest logit, "
          f"{same} of {len(got)} bit for bit")
    worst, same, argmax = worst_rel(list(zip(got, shift(whole))))
    check(worst <= 5e-3 and argmax, f"{proto} and the full-history model sent the requests whole: worst "
                                    f"{worst:.2e} of the largest logit, argmax equal: {argmax}")


def batched(url):
    print(f"http  window_sequence, three sequences at once  {url}")
    results = {}

    def one(k):
        results[k] = schedule(SequenceClient(url, "window_sequence", "http"), 300 + k, seed=10 + k)

    threads = [threading.Thread(target=one, args=(k,)) for k in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    solo = {k: schedule(SequenceClient(url, "window_sequence", "http"), 310 + k, seed=10 + k) for k in range(3)}
    worst, agree, total = 0.0, 0, 0
    for k in range(3):
        w, _, _ = worst_rel(list(zip(results[k], solo[k])))
        worst = max(worst, w)
        agree += sum(int(np.argmax(a)) == int(np.argmax(b)) for (_, a, _), (_, b, _) in zip(results[k], solo[k]))
        total += len(solo[k])
    check(worst <= 5e-3, f"three concurrent sequences get the logits they get alone: worst {worst:.2e} of the "
                         f"largest logit ({agree} of {total} argmax equal)")


def exhausted(url, http):
    print(f"http  window_sequence, windowed pool exhausted  {url}")
    p = params(http, "window_sequence")
    rings = (int(p["kv_window_num_blocks"]) - 1) // int(p["kv_window_ring_pages"])
    c = SequenceClient(url, "window_sequence", "http")
    ids = tokens(7, 12)
    for k in range(rings):
        c.step(500 + k, ids, start=True)
    try:
        c.step(500 + rings, ids, start=True)
        check(False, f"a START with all {rings} rings taken is refused")
    except Exception as e:  # tritonclient raises its own exception type
        check("windowed KV page pool exhausted" in str(e), f"a START with all {rings} rings taken is refused: "
                                                          f"{str(e)[:120]}")
    c.end(500)
    c.step(500 + rings, ids, start=True, end=True)
    check(True, "after END the refused sequence runs")
    for k in range(1, rings):
        c.end(500 + k)


def refused_ids(url, protocol):
    print(f"{protocol:4}  window_sequence, refused token ids  {url}")
    c = SequenceClient(url, "window_sequence", protocol)
    prompt = tokens(11, 9)
    for sid, bad in ((700, 62), (701, 63)):
        try:
            c.step(sid, prompt[:3] + [bad] + prompt[3:], start=True, end=True)
            check(False, f"a START holding {bad} is refused")
        except Exception as e:  # noqa: BLE001 - the message is checked
            check(f"is {bad}, the model's {REFUSED[bad]} placeholder" in str(e),
                  f"a START holding {bad} is refused by name: {str(e)[:150]}")
    # Mid-sequence: the refused request changes nothing.
    ref = c.step(702, prompt, start=True)
    ref_next = c.step(702, [5], end=True)
    c.step(703, prompt, start=True)
    try:
        c.step(703, [63])
        check(False, "a later request holding 63 is refused")
    except Exception as e:  # noqa: BLE001 - the message is checked
        check("video_token_id placeholder" in str(e), f"a later request holding 63 is refused by name")
    got = c.step(703, [5], end=True)
    check(ref is not None and np.array_equal(got, ref_next),
          "after the refused request the sequence's next logits equal those of a sequence that never sent it")
    # Control: the id below them is an ordinary token.
    out = c.step(704, prompt + [61], start=True, end=True)
    check(out is not None, "id 61, which the manifest does not list, is accepted")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--http", default="localhost:8000")
    ap.add_argument("--grpc", default="localhost:8001")
    ap.add_argument("--perturb", action="store_true")
    args = ap.parse_args()
    run("http", args.http, args.http, args.perturb)
    run("grpc", args.grpc, args.http, args.perturb)
    chunked("http", args.http, args.http, args.perturb)
    chunked("grpc", args.grpc, args.http, args.perturb)
    if not args.perturb:
        batched(args.http)
        exhausted(args.http, args.http)
        refused_ids(args.http, "http")
        refused_ids(args.grpc, "grpc")
    if failures:
        print(f"{len(failures)} window check(s) FAILED")
        return 1
    print("window checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
