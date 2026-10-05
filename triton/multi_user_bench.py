#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""N concurrent users against a sequence-mode Triton model: each sends a document of --ctx tokens,
then --turns follow-up questions, decoding --max-new greedy tokens per turn.

--docs is a JSON file {"docs": [[ids...] per user], "questions": [[ids...], ...]} in the
model's vocabulary. Per turn it prints the time to the first token and the tokens per
second per user after it (the median and the worst user), and the aggregate.

Modes: a request per token (NEXT_TOKEN); --spec, a request per speculative step
(NEXT_TOKENS); --gen, one request per turn that the backend generates (request
parameter max_tokens; no first-token time); --stream, as --gen on a decoupled model,
one response per step.

  multi_user_bench.py --url localhost:8001 --model q35 --docs docs.json --users 4 --ctx 30000 --stream
"""
import argparse, json, os, statistics, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sequence_client import SequenceClient

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="localhost:8031"); ap.add_argument("--model", default="qwen35")
ap.add_argument("--docs", required=True); ap.add_argument("--users", type=int, default=4)
ap.add_argument("--ctx", type=int, default=8000); ap.add_argument("--turns", type=int, default=2)
ap.add_argument("--max-new", type=int, default=128); ap.add_argument("--stagger", type=float, default=0.0)
ap.add_argument("--label", default="")
ap.add_argument("--spec", action="store_true", help="speculative model: NEXT_TOKENS, several tokens a step")
ap.add_argument("--dump", default="", help="write each user's generated ids per turn to this JSON file")
ap.add_argument("--gen", action="store_true", help="one request per turn: the backend generates (MAX_TOKENS)")
ap.add_argument("--stream", action="store_true", help="as --gen, streamed per step (a decoupled model)")
a = ap.parse_args()
d = json.load(open(a.docs))
res = []

def user(i):
    c = SequenceClient(a.url, a.model, "grpc")
    cid = 5000 + i + int(time.time()) % 1000 * 10
    time.sleep(a.stagger * i)
    try:
        run_user(i, c, cid)
        c.close()
    except Exception as e:
        print(f"user {i}: {str(e)[:400]}")
        try:
            c.end(cid)
        except Exception:
            pass

gen = {}

def run_user(i, c, cid):
    for turn in range(a.turns):
        toks = d["docs"][i][:a.ctx] if turn == 0 else d["questions"][turn % 4]
        last = turn == a.turns - 1
        t0 = time.perf_counter()
        if a.stream:
            out, t1, steps = [], None, 0
            for t, chunk in c.stream_tokens(cid, toks, a.max_new, start=(turn == 0), end=last):
                if t1 is None:
                    t1 = t
                    ttft = t - t0
                out += chunk
                steps += 1
            rate = (len(out) - 1) / max(1e-9, time.perf_counter() - t1)
            gen.setdefault(i, []).append(out[:a.max_new])
            res.append((i, turn, len(toks), ttft, rate, (len(out) - 1) / max(1, steps - 1)))
            continue
        if a.gen:
            out = c.generate_tokens(cid, toks, a.max_new, start=(turn == 0), end=last)
            wall = time.perf_counter() - t0
            gen.setdefault(i, []).append(out[:a.max_new])
            # No first-token time: the turn's whole wall time, prompt included.
            res.append((i, turn, len(toks), wall, len(out) / wall, 0.0))
            continue
        if a.spec:
            out = c.step_tokens(cid, toks, start=(turn == 0))
            ttft = time.perf_counter() - t0
            t1 = time.perf_counter()
            steps = 0
            while len(out) < a.max_new:
                more = c.step_tokens(cid, [out[-1]])
                out += more
                steps += 1
            if last:
                c.end(cid)
            rate = (len(out) - 1) / (time.perf_counter() - t1)
            gen.setdefault(i, []).append(out[:a.max_new])
            res.append((i, turn, len(toks), ttft, rate, (len(out) - 1) / max(1, steps)))
            continue
        nxt = c.step_token(cid, toks, start=(turn == 0))
        ttft = time.perf_counter() - t0
        out = [nxt]
        t1 = time.perf_counter()
        for k in range(a.max_new - 1):
            nxt = c.step_token(cid, [nxt], end=(last and k == a.max_new - 2))
            out.append(nxt)
        rate = (a.max_new - 1) / (time.perf_counter() - t1)
        gen.setdefault(i, []).append(out)
        res.append((i, turn, len(toks), ttft, rate, 1.0))

ts = [threading.Thread(target=user, args=(i,)) for i in range(a.users)]
t0 = time.perf_counter()
for t in ts: t.start()
for t in ts: t.join()
print(f"== {a.label} users={a.users} ctx={a.ctx} max_new={a.max_new} wall={time.perf_counter()-t0:.1f}s")
for turn in range(a.turns):
    rs = [r for r in res if r[1] == turn]
    if not rs:
        print(f"turn {turn}: no user finished it"); continue
    tt = [r[3] for r in rs]; rt = [r[4] for r in rs]
    per = statistics.mean(r[5] for r in rs)
    print(f"turn {turn}: prompt {rs[0][2]} tok, TTFT median {statistics.median(tt):.2f}s worst {max(tt):.2f}s | "
          f"per-user tok/s median {statistics.median(rt):.1f} worst {min(rt):.1f} | aggregate ~{sum(rt):.0f}"
          + (f" | {per:.2f} tokens per step" if a.spec or a.stream else ""))
if a.dump:
    json.dump(gen, open(a.dump, "w"))
