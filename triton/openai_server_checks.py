#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""Checks for openai_server.py that need NO backend and NO model.

    ~/.local/venvs/tlaloc-client/bin/python3 triton/openai_server_checks.py

Any interpreter with numpy and tritonclient works (openai_server imports sequence_client, which
imports both); the host python3 usually has neither. No server, no GPU, no checkpoint.

Covers the two serving-audit findings fixed on the Python side: SV-9, KV reuse scoped to the
caller (the OpenAI `user` field), and SV-1's residue, unpredictable correlation ids; plus the
`stop` strings the endpoint used to ignore (SV-12's tail). Sequences.acquire/release touch no
backend, so the ownership rules are checked directly; StopStrings is checked against the same
cases the Kotlin StopStringsTest pins.

The Detokenizer's quadratic re-decode is deliberately NOT "fixed" here: see its docstring.
`decode` is not suffix-independent (146 of 175 split points of four texts disagree with the
full decode on TinyLlama's tokenizer), so an incremental decode corrupts text.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openai_server as m  # noqa: E402

fails = 0


def check(cond, what):
    global fails
    if not cond:
        fails += 1
        print("FAIL:", what)
    else:
        print("ok:", what)


def stop_strings():
    s = m.StopStrings([])
    check(s.feed("anything 🙂") == "anything 🙂", "no stops passes text through")

    s = m.StopStrings(["ab"])
    check(s.feed("xxabyy") == "xx" and s.matched, "a stop string truncates and matches")

    s = m.StopStrings(["abc"])
    check(s.feed("xxab") == "xx" and not s.matched, "a partial stop is held back")
    check(s.feed("c") == "" and s.matched, "the held stop completes")

    # Python strings are code points, so a slice cannot split a character (the Kotlin SV-11
    # was a UTF-16 cut). These pin that the hold-back still returns the whole text.
    s = m.StopStrings(["ab"])
    check(s.feed("Hi 🙂") + s.finish() == "Hi 🙂", "an emoji straddling the hold-back cut stays whole")

    s = m.StopStrings(["stop"])
    text = "a🙂b🙃c😀d"
    out = "".join(s.feed(text[i:i + 3]) for i in range(0, len(text), 3)) + s.finish()
    check(out == text, "several emoji across feeds stay whole")


def ownership():
    with tempfile.TemporaryDirectory() as d:
        seqs = m.Sequences("localhost:1", "fake", 4, os.path.join(d, "state.json"))
        ids = [1, 2, 3, 4]
        a, send, start = seqs.acquire(ids, "alice")
        check(start and send == ids, "a first request starts a sequence")
        seqs.release(a, ids + [5, 6], 7)
        longer = ids + [5, 6, 8]
        b, _, start_b = seqs.acquire(longer, "bob")
        check(start_b, "another caller does not reuse alice's sequence")
        seqs.release(b, longer + [9], 10)
        c, send_c, start_c = seqs.acquire(longer + [9, 11], "alice")
        check(not start_c and c is a and send_c == [8, 9, 11],
              "alice reuses her own sequence and sends only the tail")
        check(abs(a.corrid - b.corrid) > 1, "ids are not sequential")
        check(a.corrid > 0 and b.corrid > 0, "ids are positive")
        # Absent user keeps today's behaviour: callers that send none share reuse.
        e, _, start_e = seqs.acquire([100, 101], "")
        seqs.release(e, [100, 101, 102], 103)
        f, _, start_f = seqs.acquire([100, 101, 102, 104], "")
        check(not start_f and f is e, "with no user field, reuse is unchanged")


if __name__ == "__main__":
    stop_strings()
    ownership()
    print("FAILURES:", fails)
    sys.exit(1 if fails else 0)
