#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""Client for a Tlaloc language model served by Triton in sequence mode.

A sequence-mode model (written by TritonModelRepository from a serving
artifact) takes one input, TOKENS (INT32 [1, n]), and returns LOGITS (FP32
[1, vocab]): the logits of the last token sent. Each request carries the
sequence's correlation ID and Triton's START/END flags; the backend keeps the
sequence's KV pages between requests.

The first request of a sequence sends the whole prompt with START (the
backend runs it as one prefill call). Each later request sends the one token
the client chose. The request that produces the last wanted token carries
END, and the backend frees the pages. A request with no tokens and END
ends a sequence without running anything.

  sequence_client.py --model tinyllama --prompt 1,450,7483,310,3444,338 --max-new 6
"""

import argparse
import sys
import time

import numpy as np


class SequenceClient:
    """One connection to a sequence-mode model, over HTTP or gRPC."""

    def __init__(self, url="localhost:8000", model="tinyllama", protocol="http"):
        if protocol == "http":
            import tritonclient.http as tc
        elif protocol == "grpc":
            import tritonclient.grpc as tc
        else:
            raise ValueError(f"protocol must be http or grpc, got {protocol!r}")
        self.tc = tc
        self.protocol = protocol
        self.model = model
        self.client = tc.InferenceServerClient(url=url)

    def step(self, corrid, tokens, start=False, end=False):
        """Append `tokens` to sequence `corrid`; returns the last token's logits
        (a float32 vector), or None for a request with no tokens."""
        arr = np.asarray(tokens, dtype=np.int32).reshape(1, -1)
        inp = self.tc.InferInput("TOKENS", list(arr.shape), "INT32")
        if self.protocol == "http":
            inp.set_data_from_numpy(arr, binary_data=True)
            out = self.tc.InferRequestedOutput("LOGITS", binary_data=True)
        else:
            inp.set_data_from_numpy(arr)
            out = self.tc.InferRequestedOutput("LOGITS")
        res = self.client.infer(
            self.model, [inp], outputs=[out],
            sequence_id=int(corrid), sequence_start=bool(start), sequence_end=bool(end),
        )
        logits = res.as_numpy("LOGITS")
        return None if logits is None else logits.reshape(-1)

    def step_pages(self, corrid, tokens, start=False, end=False):
        """As step, and also returns KV_PAGES: the pages the sequence holds
        after the request, in the full-history and the windowed KV pools."""
        arr = np.asarray(tokens, dtype=np.int32).reshape(1, -1)
        inp = self.tc.InferInput("TOKENS", list(arr.shape), "INT32")
        if self.protocol == "http":
            inp.set_data_from_numpy(arr, binary_data=True)
        else:
            inp.set_data_from_numpy(arr)
        outs = [self.tc.InferRequestedOutput(n) for n in ("LOGITS", "KV_PAGES")]
        res = self.client.infer(
            self.model, [inp], outputs=outs,
            sequence_id=int(corrid), sequence_start=bool(start), sequence_end=bool(end),
        )
        return res.as_numpy("LOGITS").reshape(-1), res.as_numpy("KV_PAGES").reshape(-1)

    def end(self, corrid):
        """End a sequence without running a step."""
        self.step(corrid, [], end=True)

    def generate(self, corrid, prompt, max_new, keep=False):
        """Greedy decoding. Returns (ids, prefill_ms, [decode step ms], logits
        of every generated position). With `keep`, the sequence is not ended."""
        assert max_new >= 1 and len(prompt) >= 1
        t0 = time.perf_counter()
        logits = self.step(corrid, prompt, start=True, end=(max_new == 1 and not keep))
        prefill_ms = (time.perf_counter() - t0) * 1e3
        ids = [int(np.argmax(logits))]
        all_logits = [logits]
        steps = []
        for i in range(1, max_new):
            last = i == max_new - 1 and not keep
            t0 = time.perf_counter()
            logits = self.step(corrid, [ids[-1]], end=last)
            steps.append((time.perf_counter() - t0) * 1e3)
            ids.append(int(np.argmax(logits)))
            all_logits.append(logits)
        return ids, prefill_ms, steps, all_logits


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="localhost:8000")
    ap.add_argument("--protocol", default="http", choices=["http", "grpc"])
    ap.add_argument("--model", default="tinyllama")
    ap.add_argument("--prompt", required=True, help="comma-separated prompt token ids")
    ap.add_argument("--max-new", type=int, default=6)
    ap.add_argument("--sequence-id", type=int, default=1)
    args = ap.parse_args()
    prompt = [int(t) for t in args.prompt.split(",") if t.strip()]
    c = SequenceClient(args.url, args.model, args.protocol)
    ids, prefill_ms, steps, _ = c.generate(args.sequence_id, prompt, args.max_new)
    print(f"prompt     {prompt}")
    print(f"generated  {ids}")
    med = sorted(steps)[len(steps) // 2] if steps else float("nan")
    print(f"prefill {prefill_ms:.1f} ms, median decode step {med:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
