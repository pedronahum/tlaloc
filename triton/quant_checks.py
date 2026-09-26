#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""How far a model with quantized weights stays from the unquantized one.

A quantized model is not expected to reproduce its reference token for token,
so this script measures instead of asserting identity:

  * Greedy fixtures (--fixture, the files fixture_checks.py reads). For each
    prompt it reports how many of the fixture's generated ids the model
    reproduces when it decodes on its own (the identical prefix, and the
    positions that agree), and its top-1 agreement when it is fed the
    fixture's ids (teacher forcing: at each step the model sees the
    reference's tokens, so one early difference does not decide the rest).
    The fixture's margin (top-1 minus top-2 logit of the reference) at each
    disagreeing step is printed next to it.
  * Perplexity (--text-tokens N): the first N tokens of the wikitext-103
    test split, fed one token per decode step; the negative log-likelihood of
    each next token, and the model's argmax at each position.

--save writes the per-position results to a JSON file and --compare reads
one written by another model (the unquantized reference), and prints the
top-1 agreement between the two over the text and the difference in
perplexity.

  quant_checks.py --grpc localhost:8021 --model muse \
      --fixture ir/src/jvmTest/resources/io/tlaloc/ir/inference/muse_glimmer_30b_bf16_greedy.json \
      --tokenizer ~/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B/snapshots/<rev>/tokenizer.json \
      --text-tokens 320 --bos 200000 --save int8.json --compare bf16.json

Needs tritonclient, numpy, tokenizers and (for the text) pyarrow.
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

from sequence_client import SequenceClient

WIKITEXT = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--wikitext/snapshots/*/wikitext-103-raw-v1/test-00000-of-00001.parquet")


def log_softmax(v):
    v = v.astype(np.float64)
    m = v.max()
    return v - m - math.log(np.exp(v - m).sum())


def fixture_checks(c, path, tok, corrid):
    fx = json.load(open(path))
    rows = []
    for k, p in enumerate(fx["prompts"]):
        prompt, want = p["promptTokens"], p["generatedTokens"]
        margins = p.get("margins") or [None] * len(want)
        # Free-running greedy decoding.
        ids, _, _, _ = c.generate(corrid + 2 * k, prompt, len(want))
        same = 0
        while same < len(want) and ids[same] == want[same]:
            same += 1
        agree = sum(a == b for a, b in zip(ids, want))
        # Teacher forcing: the reference's ids in, the argmax at each step out.
        forced = []
        logits = c.step(corrid + 2 * k + 1, prompt, start=True)
        for i in range(len(want)):
            forced.append(int(np.argmax(logits)))
            if i + 1 < len(want):
                logits = c.step(corrid + 2 * k + 1, [want[i]], end=(i + 2 == len(want)))
        tf = sum(a == b for a, b in zip(forced, want))
        miss = [(i, round(margins[i], 3) if margins[i] is not None else None)
                for i in range(len(want)) if forced[i] != want[i]]
        text = tok.decode(ids) if tok else None
        print(f"{os.path.basename(path)} [{p['kind']}, {len(prompt)}-token prompt]: identical prefix "
              f"{same}/{len(want)}, free-running agreement {agree}/{len(want)}, teacher-forced top-1 "
              f"{tf}/{len(want)}" + (f"; misses (step, reference margin) {miss}" if miss else ""))
        if text is not None:
            print(f"    generated {text!r}")
            if tok and p.get("generatedText") is not None:
                print(f"    reference {p['generatedText']!r}")
        rows.append({"fixture": os.path.basename(path), "kind": p["kind"], "prefix": same,
                     "agree": agree, "forced": tf, "n": len(want), "ids": ids, "text": text})
    return rows


def text_ids(tok, n, bos):
    import pyarrow.parquet as pq
    files = glob.glob(WIKITEXT)
    if not files:
        raise SystemExit(f"no wikitext-103 test split at {WIKITEXT}")
    lines = pq.read_table(files[0]).column("text").to_pylist()
    text = "".join(lines[1:40])
    ids = tok.encode(text, add_special_tokens=False).ids
    if len(ids) < n:
        raise SystemExit(f"the text has {len(ids)} tokens, fewer than {n}")
    return ([bos] if bos is not None else []) + ids[:n]


def perplexity(c, ids, corrid):
    logits = c.step(corrid, [ids[0]], start=True)
    nll, argmax = [], []
    for i in range(1, len(ids)):
        nll.append(-float(log_softmax(logits)[ids[i]]))
        argmax.append(int(np.argmax(logits)))
        if i + 1 < len(ids):
            logits = c.step(corrid, [ids[i]])
    c.end(corrid)
    # With a BOS first, the first prediction is from BOS alone; it is kept.
    return nll, argmax


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grpc", default="localhost:8021")
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", action="append", default=[])
    ap.add_argument("--tokenizer", help="tokenizer.json of the checkpoint")
    ap.add_argument("--text-tokens", type=int, default=0)
    ap.add_argument("--bos", type=int, default=None, help="a token id sent before the text")
    ap.add_argument("--save")
    ap.add_argument("--compare")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.expanduser(args.tokenizer))
    c = SequenceClient(args.grpc, args.model, "grpc")
    out = {"label": args.label, "fixtures": []}
    for i, f in enumerate(args.fixture):
        out["fixtures"] += fixture_checks(c, f, tok, 7000 + 100 * i)
    if args.text_tokens:
        if tok is None:
            raise SystemExit("--text-tokens needs --tokenizer")
        ids = text_ids(tok, args.text_tokens, args.bos)
        nll, argmax = perplexity(c, ids, 7900)
        ppl = math.exp(sum(nll) / len(nll))
        print(f"wikitext-103 test, {len(nll)} predictions: perplexity {ppl:.4f} "
              f"(mean NLL {sum(nll) / len(nll):.5f})")
        out.update(text_ids=ids, nll=nll, argmax=argmax, perplexity=ppl)
    if args.save:
        json.dump(out, open(args.save, "w"))
    if args.compare:
        ref = json.load(open(args.compare))
        if "argmax" in ref and "argmax" in out:
            if ref["text_ids"] != out["text_ids"]:
                raise SystemExit("--compare: the two runs scored different texts")
            a, b = ref["argmax"], out["argmax"]
            agree = sum(x == y for x, y in zip(a, b))
            print(f"text top-1 agreement with {ref.get('label') or args.compare}: {agree}/{len(a)} "
                  f"({100.0 * agree / len(a):.1f}%); perplexity {ref['perplexity']:.4f} -> "
                  f"{out['perplexity']:.4f} ({100.0 * (out['perplexity'] / ref['perplexity'] - 1):+.2f}%)")
        for r, o in zip(ref.get("fixtures", []), out["fixtures"]):
            same = sum(x == y for x, y in zip(r["ids"], o["ids"]))
            print(f"{o['fixture']} [{o['kind']}]: {same}/{len(o['ids'])} generated ids equal to "
                  f"{ref.get('label') or args.compare}'s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
