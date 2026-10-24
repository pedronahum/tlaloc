#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""A follow-up turn after a request longer than Triton's idle timeout.

Triton counts a sequence idle while the backend holds its request. One sequence
generates --max-tokens tokens in its first turn, then sends a second turn. With
--expect-refused (a repository whose max_sequence_idle_microseconds is shorter than
the first turn) Triton must refuse the second turn for want of START; without it the
second turn must generate.

  long_request_checks.py --grpc localhost:8001 --model q35 --max-tokens 100 [--expect-refused]
"""
import argparse
import sys
import time

from sequence_client import SequenceClient

ap = argparse.ArgumentParser()
ap.add_argument("--grpc", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--max-tokens", type=int, default=100)
ap.add_argument("--expect-refused", action="store_true")
a = ap.parse_args()

c = SequenceClient(a.grpc, a.model, "grpc")
cid = 300
got, err, took = [], None, 0.0
for turn, toks in enumerate(([9707, 11, 1246, 525, 498, 30] * 4, [11, 13, 198])):
    t0 = time.time()
    ids = []
    try:
        for _, chunk in c.stream_tokens(cid, toks, a.max_tokens, start=(turn == 0), end=(turn == 1)):
            ids += list(chunk)
    except Exception as e:  # tritonclient's exception types
        err = str(e)
    if turn == 0:
        took = time.time() - t0
    got.append(ids)
    if err:
        break
c.close()
print(f"  first turn: {len(got[0])} tokens in {took:.2f} s")
if a.expect_refused:
    ok = err is not None and "START" in err and len(got[0]) > 0
    print(("  ok   " if ok else "FAIL ") + "the second turn is refused for want of START" + (f" ({err[:160]})" if err else ""))
else:
    ok = err is None and len(got) == 2 and len(got[1]) > 0
    print(("  ok   " if ok else "FAIL ") + "the second turn generates" + (f" ({err[:160]})" if err else ""))
sys.exit(0 if ok else 1)
