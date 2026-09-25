#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""Greedy decoding against a Tlaloc language model served by Triton.

The model is a serving artifact written into a Triton repository by
TritonModelRepository (./gradlew :maestro:exportTritonModel) in sequence
mode. The backend keeps the weights, the KV pools and each sequence's pages
on the device. This client sends the prompt as the sequence's first request
(one prefill call), then one request per generated token, and ends the
sequence with the last one (sequence_client.py).

  generate_client.py --model tinyllama --tokenizer <checkpoint>/tokenizer.json \\
      --text "The capital of France is" --max-new 6 [--expect 3681,29889,...]

Prompt ids come from the checkpoint's own tokenizer.json. With the
`tokenizers` package installed the text is tokenized the way transformers
does it, so any text works for any of the model families, including the
byte-level BPE vocabularies of Qwen3 and Muse Glimmer (as in
examples/triton-llm/chat.py). Without it, or with --words, only a
SentencePiece vocabulary is read and each whitespace-separated word must be one
entry of it (BOS + one id per word, as in examples/gpu-inference/serve.py).
--prompt takes ids instead. With --expect, the exit status is 0 only if the
generated ids are exactly those.

  generate_client.py --model qwen3 --tokenizer <snapshot>/tokenizer.json \\
      --text " The capital of France is" --max-new 16
"""

import argparse
import json
import sys


class TextTokenizer:
    """The checkpoint's tokenizer.json through the `tokenizers` package: the
    same ids as transformers' tokenizer for the text (special tokens such as
    a BOS are added as the file's post-processor says)."""

    def __init__(self, path):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(path)

    def encode(self, text):
        return self.tok.encode(text).ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)


class WordTokenizer:
    """A SentencePiece vocabulary read from tokenizer.json, exact words only."""

    def __init__(self, path):
        self.vocab, self.rev = load_vocab(path)

    def encode(self, text):
        return encode(self.vocab, text)

    def decode(self, ids):
        return decode(self.rev, ids)


def tokenizer_for(path, words=False):
    """TextTokenizer when the tokenizers package is there (and --words is not
    given), else WordTokenizer; returns (tokenizer, a name for the log)."""
    if not words:
        try:
            return TextTokenizer(path), "tokenizers"
        except ImportError:
            pass
    return WordTokenizer(path), "exact words"


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
    ap.add_argument("--protocol", default="http", choices=["http", "grpc"])
    ap.add_argument("--model", default="tinyllama")
    ap.add_argument("--tokenizer", default="", help="the checkpoint's tokenizer.json")
    ap.add_argument("--text", default="The capital of France is")
    ap.add_argument("--words", action="store_true",
                    help="read the vocabulary as exact words even if the tokenizers package is installed")
    ap.add_argument("--prompt", default="", help="comma-separated prompt ids (instead of --text)")
    ap.add_argument("--max-new", type=int, default=6)
    ap.add_argument("--sequence-id", type=int, default=1)
    ap.add_argument("--expect", default="", help="comma-separated ids the generation must equal")
    ap.add_argument("--expect-prompt", default="", help="comma-separated ids the prompt must tokenize to")
    ap.add_argument("--expect-prefix", default="",
                    help="comma-separated ids the generation must start with")
    args = ap.parse_args()

    from sequence_client import SequenceClient

    tok = None
    if args.tokenizer:
        tok, how = tokenizer_for(args.tokenizer, args.words)
        print(f"tokenizer  {args.tokenizer} ({how})")
    if args.prompt:
        prompt = [int(t) for t in args.prompt.split(",") if t.strip()]
    else:
        if tok is None:
            raise SystemExit("--text needs --tokenizer (the checkpoint's tokenizer.json)")
        prompt = tok.encode(args.text)

    client = SequenceClient(args.url, args.model, args.protocol)
    generated, prefill_ms, steps, _ = client.generate(args.sequence_id, prompt, args.max_new)

    print(f"prompt     {prompt}")
    print(f"generated  {generated}")
    if tok is not None:
        print(f"text       {tok.decode(prompt)!r} -> {tok.decode(generated)!r}")
    med = sorted(steps)[len(steps) // 2] if steps else float("nan")
    print(f"prefill of {len(prompt)} tokens {prefill_ms:.1f} ms, median decode step {med:.1f} ms")
    if args.expect_prompt:
        want = [int(t) for t in args.expect_prompt.split(",") if t.strip()]
        if prompt != want:
            print(f"FAIL the prompt tokenized to {prompt}, expected {want}")
            return 1
        print(f"ok   the prompt tokenized to the expected {len(want)} ids")
    if args.expect:
        want = [int(t) for t in args.expect.split(",") if t.strip()]
        if generated != want:
            print(f"FAIL generated {generated}, expected {want}")
            return 1
        print(f"ok   generated ids equal the expected {len(want)} ids")
    if args.expect_prefix:
        want = [int(t) for t in args.expect_prefix.split(",") if t.strip()]
        if generated[: len(want)] != want:
            print(f"FAIL generated {generated[:len(want)]}..., expected to start with {want}")
            return 1
        print(f"ok   the {len(generated)} generated ids start with the expected {len(want)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
