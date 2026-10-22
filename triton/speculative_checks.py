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


# Generations that end apart: the first stops on an end token (one of its own ids, which the
# host cannot foresee), so a chained step is in flight when it ends and the other takes its rows
# of it. Each must give the ids it gives alone. Then a long prompt (several prefill calls)
# arriving while the other generates runs in parts between the other's steps; both give the
# ids they give alone. The server log must show both (verify.sh checks it).
def generate_until(cid, toks, max_tokens, end_tokens, out):
    c = SequenceClient(a.grpc, a.model, "grpc")
    ids = []
    try:
        # The sequence ends after the generation, not with it: a generation is chained only
        # while its sequence goes on.
        for _, chunk in c.stream_tokens(cid, toks, max_tokens, end_tokens=end_tokens, start=True):
            ids += list(chunk)
        c.end_streamed(cid)
    except Exception as e:
        print(f"FAIL sequence {cid}: {e}")
    finally:
        c.close()
    out[cid] = ids


if len(prompts) >= 2 and not a.perturb:
    out = {}
    generate_until(150, prompts[0]["promptTokens"], 40, None, out)
    stop = out[150][12:13]  # an id it emits after a few steps
    alone = {}
    for i, n, end in ((0, 40, stop), (1, 60, None)):
        o = {}
        generate_until(151 + i, prompts[i]["promptTokens"], n, end, o)
        alone[i] = o[151 + i]
    out = {}
    threads = [threading.Thread(target=generate_until, args=(160 + i, prompts[i]["promptTokens"], n, end, out))
               for i, n, end in ((0, 40, stop), (1, 60, None))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i in (0, 1):
        got = out.get(160 + i, [])
        check(got == alone[i] and len(got) > 0, f"ending apart, {prompts[i]['kind']} prompt: the ids it gives alone ({len(got)})")
    # The pair runs before the references alone: with the prefix cache, a prompt run before
    # would be cached, and the long prompt would not need a call per budget. It starts three
    # tokens into the repeat, so no earlier prompt's cached prefix matches it either.
    long_prompt = (prompts[1]["promptTokens"] * 12)[3:163]
    out = {}
    t0 = threading.Thread(target=generate_until, args=(180, prompts[0]["promptTokens"], 200, None, out))
    t1 = threading.Thread(target=generate_until, args=(181, long_prompt, 12, None, out))
    t0.start()
    time.sleep(0.05)
    t1.start()
    t0.join()
    t1.join()
    o = {}
    generate_until(170, long_prompt, 12, None, o)
    long_alone = o[170]
    o = {}
    generate_until(171, prompts[0]["promptTokens"], 200, None, o)
    first_alone = o[171]
    check(out.get(181) == long_alone and len(long_alone) > 0, "a long prompt during a generation: its ids alone")
    check(out.get(180) == first_alone and len(first_alone) > 0, "the generation it arrived during: its ids alone")

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
# Two prompts that share their first 96 tokens, one after the other: on a server with the
# prefix cache (verify.sh's second server) the second starts from the first's cached prefix; its
# ids, and the first's, are compared with the server without one.
if not a.perturb:
    shared = (prompts[1]["promptTokens"] * 12)[:96]
    o = {}
    generate_until(190, shared + prompts[0]["promptTokens"][:4], 12, None, o)
    generate_until(191, shared + prompts[1]["promptTokens"][:6], 12, None, o)
    ids["prefix-a"] = [o.get(190, [])]
    ids["prefix-b"] = [o.get(191, [])]
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
