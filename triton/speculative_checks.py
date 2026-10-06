#!/usr/bin/env python3
"""Generation in the backend for a speculative (MTP) model served decoupled.

Streams the Qwen3.5-0.8B greedy fixture's prompts (request parameter
max_tokens), each alone, both at once, and the second starting while the
first generates: the first 16 ids of each must be transformers'. Then two
turns of 100 tokens per prompt, both at once; --dump writes their ids, and
--compare checks them against such a dump (a server run with chained verify
steps against one without: the ids where both have them must be equal).

  speculative_checks.py --grpc localhost:8001 --model q35 --fixture qwen3_5_0_8b_greedy.json

--perturb expects a wrong first id, so the checks must fail.
"""
import argparse
import json
import sys
import threading
import time

from sequence_client import SequenceClient

ap = argparse.ArgumentParser()
ap.add_argument("--grpc", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--fixture", required=True)
ap.add_argument("--dump")
ap.add_argument("--compare")
ap.add_argument("--perturb", action="store_true")
a = ap.parse_args()

fx = json.load(open(a.fixture))
prompts = fx["prompts"]
failures = 0


def check(ok, what):
    global failures
    print(("  ok   " if ok else "FAIL ") + what)
    failures += not ok


def generate(cid, turns, max_tokens, out):
    c = SequenceClient(a.grpc, a.model, "grpc")
    got = []
    try:
        for i, toks in enumerate(turns):
            ids = []
            for _, chunk in c.stream_tokens(cid, toks, max_tokens, start=(i == 0), end=(i == len(turns) - 1)):
                ids += list(chunk)
            got.append(ids)
    except Exception as e:  # tritonclient's exception types
        print(f"FAIL sequence {cid}: {e}")
    finally:
        c.close()
    out[cid] = got


def run(how, base, turns_of, max_tokens):
    out = {}
    threads = [threading.Thread(target=generate, args=(base + i, turns_of(p), max_tokens, out)) for i, p in enumerate(prompts)]
    for t in threads:
        t.start()
        if how == "alone":
            t.join()
        elif how == "staggered":
            time.sleep(0.15)
    for t in threads:
        t.join()
    return out


for how, base in (("alone", 100), ("together", 110), ("staggered", 120)):
    out = run(how, base, lambda p: [p["promptTokens"]], 16)
    for i, p in enumerate(prompts):
        want = list(p["generatedTokens"])
        if a.perturb:
            want[0] += 1
        got = (out.get(base + i) or [[]])[0][:16]
        check(got == want, f"{how}, {p['kind']} prompt: 16 ids equal transformers'")

out = run("together", 200, lambda p: [p["promptTokens"], [11, 13, 198]], 100)
ids = {str(k): v for k, v in out.items()}
check(all(len(t) >= 100 for v in ids.values() for t in v) and len(ids) == len(prompts),
      "two turns of at least 100 tokens per prompt")
if a.dump:
    json.dump(ids, open(a.dump, "w"))
if a.compare:
    ref = json.load(open(a.compare))
    for k in sorted(ids):
        same = len(ids[k]) == len(ref.get(k, [])) and all(
            x[: min(len(x), len(y))] == y[: min(len(x), len(y))] for x, y in zip(ids[k], ref[k]))
        check(same, f"sequence {k}: the ids of {a.compare} where both have them")
print(f"{failures} failing check(s)")
sys.exit(1 if failures else 0)
