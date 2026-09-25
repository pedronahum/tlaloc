#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""Greedy decoding through a sequence-mode model against a HuggingFace fixture.

The fixture is JSON written by harness/python/hf_greedy_fixture.py:
transformers greedy-decoding each prompt in float32 on the CPU, with the
prompt ids, the generated ids, the top-k logits of the first generated
position and the chosen token's logit at every position. The weights are not
in it; the server holds them.

For every prompt, over HTTP and gRPC, the prompt is sent as one prefill
request and the rest one token per request (sequence_client.py). Checks:

  ids     the generated ids equal the fixture's, all of them
  logits  the top-k logits of the first position and the chosen logit at
          every position are within --tol of the largest logit magnitude
          (XLA's GPU f32 matmuls run in TF32, so this is not the CPU
          tolerance)

Then it times each prompt --repeat times over gRPC and prints the median
prefill time and the median decode step. With --perturb the expected ids are
wrong (the last id of each prompt plus one), so the run must fail.

  fixture_checks.py --fixture qwen3_0_6b_greedy.json --model qwen3
"""

import argparse
import json
import sys

import numpy as np

from sequence_client import SequenceClient

failures = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        failures.append(what)


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--http", default="localhost:8000")
    ap.add_argument("--grpc", default="localhost:8001")
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--tol", type=float, default=2e-3,
                    help="logit tolerance relative to the largest logit magnitude")
    ap.add_argument("--repeat", type=int, default=5, help="timing runs of the first prompt")
    ap.add_argument("--perturb", action="store_true", help="expect wrong ids (negative control)")
    args = ap.parse_args()

    with open(args.fixture) as fh:
        fx = json.load(fh)
    max_new = int(fx["maxNew"])
    corrid = 7000

    for proto, url in (("http", args.http), ("grpc", args.grpc)):
        client = SequenceClient(url, args.model, proto)
        for p in fx["prompts"]:
            corrid += 1
            want = list(p["generatedTokens"])
            if args.perturb:
                want[-1] += 1
            ids, _, _, logits = client.generate(corrid, p["promptTokens"], max_new)
            first_diff = next((i for i, (a, b) in enumerate(zip(ids, want)) if a != b), None)
            check(
                ids == want,
                f"{proto} {p['kind']} prompt ({len(p['promptTokens'])} tokens): {max_new} ids "
                + ("equal HuggingFace's" if ids == want
                   else f"differ from HuggingFace's at position {first_diff}: {ids} vs {want}"),
            )
            first = np.asarray(logits[0], dtype=np.float64)
            denom = max(1.0, float(np.max(np.abs(first))))
            idx = np.asarray(p["step1TopKIndices"])
            top = float(np.max(np.abs(first[idx] - np.asarray(p["step1TopKValues"])))) / denom
            chosen = 0.0
            for i, t in enumerate(ids[: len(p["chosenLogits"])]):
                row = np.asarray(logits[i], dtype=np.float64)
                d = max(1.0, float(np.max(np.abs(row))))
                chosen = max(chosen, abs(float(row[t]) - p["chosenLogits"][i]) / d)
            check(
                top <= args.tol and chosen <= args.tol,
                f"{proto} {p['kind']} prompt: step-1 top-{len(idx)} logits within {top:.2e}, "
                f"chosen logits within {chosen:.2e} of HuggingFace's (tolerance {args.tol:g})",
            )

    if not args.perturb and args.repeat > 0:
        client = SequenceClient(args.grpc, args.model, "grpc")
        for p in fx["prompts"]:
            prefills, steps = [], []
            for _ in range(args.repeat):
                corrid += 1
                _, pre, st, _ = client.generate(corrid, p["promptTokens"], max_new)
                prefills.append(pre)
                steps.extend(st)
            step = median(steps)
            print(
                f"timing (gRPC, {args.repeat} runs, {p['kind']} prompt of "
                f"{len(p['promptTokens'])} tokens): prefill median {median(prefills):.1f} ms, "
                f"decode step median {step:.1f} ms ({1000.0 / step:.1f} tokens/s for one sequence)"
            )

    if failures:
        print(f"{len(failures)} check(s) FAILED")
        return 1
    print("all fixture checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
