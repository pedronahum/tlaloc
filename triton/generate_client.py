#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""Greedy decoding against a Tlaloc decode model served by Triton.

The model is a serving artifact written into a Triton repository by
TritonModelRepository (./gradlew :maestro:exportTritonModel). The backend
keeps the weights and the KV pools on the device; this client sends one
decode step per request (token id, position, block table, sequence length,
slot) and takes the argmax of the logits it gets back.

The prompt runs as single-token decode steps, the same way
examples/gpu-inference/serve.py runs it. The sequence gets pages 1..N of the
pool, in order; page 0 is the scratch page padded rows use. A token at
position p is written to slot table[p // blockSize] * blockSize + p % blockSize.

  generate_client.py --model tinyllama --tokenizer <checkpoint>/tokenizer.json \\
      --text "The capital of France is" --max-new 6 [--expect 3681,29889,...]

Prompt ids come from the checkpoint's own tokenizer.json (exact vocabulary
matches only, as in serve.py); --prompt takes ids instead. With --expect, the
exit status is 0 only if the generated ids are exactly those.
"""

import argparse
import json
import sys
import time

PADDING_BLOCK = 0


def load_vocab(path):
    with open(path) as fh:
        vocab = json.load(fh)["model"]["vocab"]
    return vocab, {v: k for k, v in vocab.items()}


def encode(vocab, text):
    """BOS + one id per whitespace-separated word. Exact matches only."""
    ids = [vocab.get("<s>", 1)]
    for word in text.split():
        token = "▁" + word
        if token not in vocab:
            raise SystemExit(f"the word {word!r} is not one entry of this vocabulary; pass --prompt ids")
        ids.append(vocab[token])
    return ids


def decode(rev, ids):
    out = []
    for i in ids:
        tok = rev.get(i, f"<{i}>")
        if tok.startswith("<0x") and tok.endswith(">"):
            out.append(chr(int(tok[3:-1], 16)))
        elif tok in ("<s>", "</s>"):
            continue
        else:
            out.append(tok.replace("▁", " "))
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="localhost:8000")
    ap.add_argument("--model", default="tinyllama")
    ap.add_argument("--tokenizer", default="", help="the checkpoint's tokenizer.json")
    ap.add_argument("--text", default="The capital of France is")
    ap.add_argument("--prompt", default="", help="comma-separated prompt ids (instead of --text)")
    ap.add_argument("--max-new", type=int, default=6)
    ap.add_argument("--expect", default="", help="comma-separated ids the generation must equal")
    args = ap.parse_args()

    import numpy as np
    import tritonclient.http as httpclient

    rev = None
    if args.prompt:
        prompt = [int(t) for t in args.prompt.split(",") if t.strip()]
    else:
        if not args.tokenizer:
            raise SystemExit("--text needs --tokenizer (the checkpoint's tokenizer.json)")
        vocab, rev = load_vocab(args.tokenizer)
        prompt = encode(vocab, args.text)
    if args.tokenizer and rev is None:
        _, rev = load_vocab(args.tokenizer)

    client = httpclient.InferenceServerClient(url=args.url)
    meta = client.get_model_metadata(args.model)
    tables = next(i for i in meta["inputs"] if i["name"] == "blockTables")
    width = int(tables["shape"][1])
    if width < 0:
        raise SystemExit("this client needs a model with one block-table width (one context bucket)")
    params = client.get_model_config(args.model).get("parameters", {})
    if "kv_block_size" not in params:
        raise SystemExit(f"model {args.model} has no kv_block_size parameter; it was not written "
                         "by TritonModelRepository from a decode artifact")
    block_size = int(params["kv_block_size"]["string_value"])
    capacity = width * block_size
    if len(prompt) + args.max_new > capacity:
        raise SystemExit(f"prompt {len(prompt)} + {args.max_new} new exceeds the compiled context {capacity}")

    table = [p + 1 for p in range(width)]

    def step(token, pos):
        used = table[: pos // block_size + 1]
        row = used + [PADDING_BLOCK] * (width - len(used))
        slot = table[pos // block_size] * block_size + pos % block_size
        tensors = {
            "tokenIds": np.array([[token]], dtype=np.int32),
            "positions": np.array([[pos]], dtype=np.int32),
            "blockTables": np.array([row], dtype=np.int32),
            "seqLens": np.array([pos + 1], dtype=np.int32),
            "slotMapping": np.array([slot], dtype=np.int32),
        }
        inputs = []
        for name, arr in tensors.items():
            i = httpclient.InferInput(name, list(arr.shape), "INT32")
            i.set_data_from_numpy(arr, binary_data=True)
            inputs.append(i)
        out = httpclient.InferRequestedOutput("logits", binary_data=True)
        res = client.infer(args.model, inputs, outputs=[out])
        return res.as_numpy("logits").reshape(-1)

    tokens = list(prompt)
    generated = []
    times = []
    for i in range(len(prompt) + args.max_new - 1):
        t0 = time.time()
        logits = step(tokens[i], i)
        times.append((time.time() - t0) * 1e3)
        nxt = int(np.argmax(logits))
        if i >= len(prompt) - 1:
            generated.append(nxt)
            tokens.append(nxt)
        mark = "prompt" if i < len(prompt) else "gen"
        print(f"  step {i:2d}  {mark:6s}  token {tokens[i]:6d}  -> argmax {nxt:6d}  {times[-1]:8.1f} ms")

    print(f"prompt     {prompt}")
    print(f"generated  {generated}")
    if rev is not None:
        print(f"text       {decode(rev, prompt)!r} -> {decode(rev, generated)!r}")
    later = sorted(times[1:]) or times
    print(f"first step {times[0]:.1f} ms, median step {later[len(later) // 2]:.1f} ms")
    if args.expect:
        want = [int(t) for t in args.expect.split(",") if t.strip()]
        if generated != want:
            print(f"FAIL generated {generated}, expected {want}")
            return 1
        print(f"ok   generated ids equal the expected {len(want)} ids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
