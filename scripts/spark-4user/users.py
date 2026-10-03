"""N concurrent chat users against an OpenAI-compatible server.

Each user owns a distinct document of `--ctx` tokens (a different slice of
`--corpus` per user) and holds a multi-turn conversation about it:
turn 1 sends the document plus a question (cold prefill), later turns append
a short question to the growing history (prefix-cache hits if the server has
them). Output length is fixed with ignore_eos so runs are comparable.

Reports per turn: TTFT and per-user streaming rate (tokens after the first
divided by the time after the first), median and worst over users.
"""
import argparse, asyncio, json, statistics, time
import httpx
from transformers import AutoTokenizer

QUESTIONS = [
    "Summarize the main topics of the document above in detail.",
    "Which people or places are mentioned most often? Explain their roles.",
    "Write a critical review of the writing quality of the document.",
    "List ten facts from the document and say where each one appears.",
]


def corpus(path):
    return open(path).read()


async def one_turn(client, url, model, messages, max_tokens, t0_barrier):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    first = last = None
    text, usage = [], None
    async with client.stream("POST", url, json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {}).get("content")
                if d:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    text.append(d)
    n = usage["completion_tokens"]
    return dict(ttft=first - t0, rate=(n - 1) / (last - first) if last > first else float("nan"),
                prompt=usage["prompt_tokens"], out=n, text="".join(text),
                cached=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"))


async def user(i, args, tok, text, client, results):
    # distinct slice of the corpus per user, trimmed to args.ctx tokens
    approx = args.ctx * 5
    chunk = text[i * approx * 2: i * approx * 2 + approx]
    ids = tok(chunk, add_special_tokens=False)["input_ids"][: args.ctx]
    doc = tok.decode(ids)
    messages = [{"role": "user", "content": doc + "\n\n" + QUESTIONS[0]}]
    for t in range(args.turns):
        await asyncio.sleep(args.stagger * i if t == 0 else args.think)
        r = await one_turn(client, args.url, args.model, messages, args.max_tokens, None)
        results.append(dict(user=i, turn=t, **{k: v for k, v in r.items() if k != "text"}))
        messages += [{"role": "assistant", "content": r["text"]},
                     {"role": "user", "content": QUESTIONS[(t + 1) % len(QUESTIONS)]}]


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8100/v1/chat/completions")
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--users", type=int, default=4)
    p.add_argument("--ctx", type=int, default=100_000)
    p.add_argument("--turns", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--stagger", type=float, default=0.0, help="seconds between user arrivals")
    p.add_argument("--think", type=float, default=0.0, help="seconds between a user's turns")
    p.add_argument("--label", default="")
    p.add_argument("--out", default=None)
    p.add_argument("--corpus", required=True, help="text file the documents are cut from (corpus.sh writes one)")
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    text = corpus(args.corpus)
    results = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(3600.0)) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*(user(i, args, tok, text, client, results) for i in range(args.users)))
        wall = time.perf_counter() - t0
    print(f"== {args.label} users={args.users} ctx={args.ctx} out={args.max_tokens} wall={wall:.1f}s")
    for t in range(args.turns):
        rs = [r for r in results if r["turn"] == t]
        tt = [r["ttft"] for r in rs]; rt = [r["rate"] for r in rs]
        print(f"turn {t}: prompt~{statistics.median(r['prompt'] for r in rs):.0f} tok, cached {[r['cached'] for r in rs]}, "
              f"TTFT median {statistics.median(tt):.2f}s worst {max(tt):.2f}s | "
              f"per-user tok/s median {statistics.median(rt):.1f} worst {min(rt):.1f} | aggregate ~{sum(rt):.0f}")
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(dict(label=args.label, users=args.users, ctx=args.ctx, wall=wall, results=results)) + "\n")


asyncio.run(main())
