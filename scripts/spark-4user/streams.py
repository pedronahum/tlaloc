"""Concurrent-stream decode measurements against an OpenAI-compatible server, defined as
the community DGX Spark recipes define them, so the numbers compare with theirs.

categories: the 40 prompts of prompts-40.json (8 categories) in batches of --concurrency
  started together, each reply ending where the model stops (thinking off, temperature 0).
  Per reply: TTFT, and decode tok/s = (tokens - 1) / (wall - TTFT). Per batch: aggregate
  tok/s = the batch's tokens / its wall time. Reported: medians over replies (overall and
  per category) and over batches.

fixed: --streams replies of exactly --max-tokens tokens (ignore_eos), one prompt kind
  (prose or code), started together, --rounds rounds. Per stream: (tokens - 1) / (time
  from its first token to its last). Aggregate: all tokens / (earliest first token to
  latest last token). Reported: the median round. --distinct gives each stream a prompt of its
  own of the kind; with one prompt the greedy streams emit the same tokens in step.

Both send one warm-up request first and read the server's speculative counters around the
run: vLLM's /metrics (spec_decode accepted tokens and drafts) or Tlaloc's /stats (tokens
per step).

  streams.py categories --url http://127.0.0.1:8100 --model q36 --concurrency 4
  streams.py fixed --url http://127.0.0.1:8100 --model q36 --streams 4 --kind code --max-tokens 512
"""
import argparse, asyncio, json, os, re, statistics, time
import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
FIXED = {
    "prose": "Write a 400-word essay on why lighthouses were built where they were, for a general reader.",
    "code": "Write a short Python module with a function that parses a CSV file of transactions "
            "(date, account, amount) and returns the balance per account, with type hints and a docstring.",
}
# --distinct: stream i asks DISTINCT[kind][i % 4] (FIXED[kind] first), so the streams' tokens differ. With
# one prompt the greedy streams of a batch emit the same tokens in step, and an MoE step routes
# them to the same experts.
DISTINCT = {
    "prose": [FIXED["prose"],
              "Write a 400-word essay on why canals were dug where they were, for a general reader.",
              "Write a 400-word essay on how medieval towns chose the place of their market, for a general reader.",
              "Write a 400-word essay on why early railways followed the routes they did, for a general reader."],
    "code": [FIXED["code"],
             "Write a short Python module with a function that reads a log file of requests "
             "(timestamp, path, status) and returns the error rate per path, with type hints and a docstring.",
             "Write a short Python module with a class that keeps a fixed-size least-recently-used cache "
             "with get and put methods, with type hints and docstrings.",
             "Write a short Python module with a function that merges overlapping time intervals given as "
             "(start, end) pairs and returns them sorted, with type hints and a docstring."],
}


async def reply(client, a, prompt, max_tokens, ignore_eos):
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    if ignore_eos:
        body["ignore_eos"] = True
    t0 = time.perf_counter()
    first = last = None
    text, usage, finish = [], None, None
    async with client.stream("POST", a.url + "/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("error"):
                raise RuntimeError(ev["error"])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                finish = ch.get("finish_reason") or finish
                d = ch.get("delta", {}).get("content")
                if d:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    text.append(d)
    end = time.perf_counter()
    n = usage["completion_tokens"]
    return dict(t0=t0, first=first, last=last, end=end, ttft=(first or end) - t0, wall=end - t0, tokens=n,
                prompt_tokens=usage["prompt_tokens"], finish=finish, text="".join(text))


async def counters(client, a):
    """The server's speculative counters: ('vllm', accepted, drafts) or ('tlaloc', tokens, steps)."""
    try:
        r = await client.get(a.url + "/stats")
        if r.status_code == 200:
            v = r.json()
            return ("tlaloc", v["completion_tokens"], v["steps"])
    except Exception:
        pass
    try:
        r = await client.get(a.url + "/metrics")
        def total(name):
            return sum(float(m) for m in re.findall(rf"^vllm:{name}(?:{{[^}}]*}})? ([0-9.e+]+)$", r.text, re.M))
        return ("vllm", total("spec_decode_num_accepted_tokens_total"), total("spec_decode_num_drafts_total"))
    except Exception:
        return None


def tokens_per_step(c0, c1):
    """Mean tokens a target step emits between two counter readings."""
    if not c0 or not c1 or c1[2] == c0[2]:
        return None
    if c0[0] == "vllm":
        return 1 + (c1[1] - c0[1]) / (c1[2] - c0[2])
    return (c1[1] - c0[1]) / (c1[2] - c0[2])


async def categories(client, a):
    items = json.load(open(a.prompts))["prompts"]
    c = a.concurrency
    results, batches = [], []
    for b in range(0, len(items), c):
        group = items[b:b + c]
        t0 = time.perf_counter()
        rs = await asyncio.gather(*(reply(client, a, it["prompt"], a.max_tokens, False) for it in group))
        wall = time.perf_counter() - t0
        for it, r in zip(group, rs):
            dec = (r["tokens"] - 1) / (r["wall"] - r["ttft"]) if r["wall"] > r["ttft"] and r["tokens"] > 1 else None
            results.append(dict(id=it["id"], category=it["category"], decode_tok_s=dec, **r))
        batches.append(dict(ids=[it["id"] for it in group], wall_s=wall,
                            agg_tok_s=sum(r["tokens"] for r in rs) / wall))
    med = lambda xs: statistics.median([x for x in xs if x is not None]) if any(x is not None for x in xs) else None
    cats = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)
    summary = {k: dict(n=len(v), ttft_med_s=med([r["ttft"] for r in v]), decode_med_tok_s=med([r["decode_tok_s"] for r in v]),
                       tokens_med=med([r["tokens"] for r in v])) for k, v in cats.items()}
    overall = dict(ttft_med_s=med([r["ttft"] for r in results]), decode_med_tok_s=med([r["decode_tok_s"] for r in results]),
                   agg_tok_s_med=med([b["agg_tok_s"] for b in batches]))
    print(f"categories, concurrency {c}: decode median {overall['decode_med_tok_s']:.1f} tok/s per reply, "
          f"TTFT median {overall['ttft_med_s']:.2f} s, aggregate median {overall['agg_tok_s_med']:.1f} tok/s")
    for k, v in summary.items():
        print(f"  {k:10s} decode {v['decode_med_tok_s']:6.1f}  TTFT {v['ttft_med_s']:.2f} s  tokens {v['tokens_med']:.0f}")
    return dict(mode="categories", concurrency=c, overall=overall, summary=summary, batches=batches,
                items=[{k: v for k, v in r.items() if k not in ("t0", "first", "last", "end")} for r in results])


async def fixed(client, a):
    prompts = [DISTINCT[a.kind][i % len(DISTINCT[a.kind])] if a.distinct else FIXED[a.kind] for i in range(a.streams)]
    rounds = []
    for _ in range(a.rounds):
        rs = await asyncio.gather(*(reply(client, a, p, a.max_tokens, True) for p in prompts))
        per = [(r["tokens"] - 1) / (r["last"] - r["first"]) for r in rs]
        span = max(r["last"] for r in rs) - min(r["first"] for r in rs)
        rounds.append(dict(per_stream=per, per_stream_med=statistics.median(per),
                           aggregate=sum(r["tokens"] for r in rs) / span,
                           ttft_med=statistics.median(r["ttft"] for r in rs), tokens=[r["tokens"] for r in rs]))
    mid = sorted(rounds, key=lambda r: r["aggregate"])[len(rounds) // 2]
    print(f"fixed {a.kind}{' (distinct prompts)' if a.distinct else ''}, {a.streams} streams x {a.max_tokens} tokens, median of {a.rounds} rounds: "
          f"{mid['per_stream_med']:.1f} tok/s per stream, aggregate {mid['aggregate']:.1f}, TTFT {mid['ttft_med']:.2f} s "
          f"(rounds: {', '.join(f'{r['aggregate']:.1f}' for r in rounds)})")
    return dict(mode="fixed", kind=a.kind, distinct=a.distinct, streams=a.streams, max_tokens=a.max_tokens, median_round=mid, rounds=rounds)


async def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["categories", "fixed"])
    p.add_argument("--url", default="http://127.0.0.1:8100", help="the server, without /v1")
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--prompts", default=os.path.join(HERE, "prompts-40.json"))
    p.add_argument("--streams", type=int, default=4)
    p.add_argument("--kind", choices=sorted(FIXED), default="prose")
    p.add_argument("--distinct", action="store_true", help="fixed: a different prompt of the kind per stream (4 kept)")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=None, help="categories: 2048; fixed: 512")
    p.add_argument("--label", default="")
    p.add_argument("--out", default=None, help="append the result as one JSON line")
    a = p.parse_args()
    a.url = a.url.rstrip("/").removesuffix("/v1")
    a.max_tokens = a.max_tokens or (2048 if a.mode == "categories" else 512)
    async with httpx.AsyncClient(timeout=httpx.Timeout(3600.0)) as client:
        await reply(client, a, "Say hello.", 16, False)  # warm-up
        c0 = await counters(client, a)
        res = await (categories(client, a) if a.mode == "categories" else fixed(client, a))
        c1 = await counters(client, a)
    res["tokens_per_step"] = tokens_per_step(c0, c1)
    if res["tokens_per_step"]:
        print(f"tokens per target step: {res['tokens_per_step']:.2f}")
    res.update(label=a.label, model=a.model, ts=time.strftime("%Y-%m-%d %H:%M"))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


asyncio.run(main())
