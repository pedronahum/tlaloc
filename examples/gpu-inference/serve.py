#!/usr/bin/env python3
"""HALF TWO OF TWO — the serving side, with no framework in the process.

This script is what a deployment looks like. Read its imports: `argparse`,
`json`, `pathlib`, `sys`, `time`, and `tlaloc_serve`. There is no jax here, no
torch, no numpy, no transformers, and — the whole point — **no JVM**. Kotlin
built the directory this reads and then exited; nothing below calls back into
it, and nothing below knows Kotlin exists.

WHAT THE SERVING PROCESS ACTUALLY DEPENDS ON
============================================

Two files, and this script names both out loud when it starts:

  1. a **PJRT plugin `.so`** — `xla_cuda_plugin.so` on an NVIDIA box,
     `/lib/libtpu.so` on a Cloud TPU VM. `tlaloc_serve` `dlopen`s it through
     `ctypes` and calls `GetPjrtApi`. It is looked up as a FILE, never
     imported: `$TLALOC_PJRT_PLUGIN_PATH` first, then `/lib/libtpu.so`, then a
     directory walk of any `jax_plugins/*/` that happens to be installed.
  2. a **driver** for whatever that plugin drives.

`pip install -e harness/python` pulls in nothing at all: the distribution's
dependency list is empty, and a test pins it empty.

WHAT IT DOES
============

Loads the artifact, re-hashes every body against its own filename, stages the
weight table if there is one, and then greedy-decodes: one token per call, KV
pools threaded through the loop, argmax on the host. The prompt runs as N
single-token decode steps rather than one prefill call, because the ragged
chunked-prefill form of PAGED_ATTENTION is a still-open performance deferral
in this repo — not a correctness one.

THE PAGE ARITHMETIC, WHICH IS THE PART THAT CAN BE WRONG SILENTLY
=================================================================

Page 0 is the reserved scratch page and is NEVER given to a live sequence: a
padded row's block table is all zeros, so a real sequence parked on page 0
would read another row's writes. This one sequence gets pages 1..mbs, in
order, so the slot a token at position `p` is written to is

    slot(p) = blockTable[p // blockSize] * blockSize + (p % blockSize)

and `seqLens` at step `p` is `p + 1` — the token being written is part of the
context it attends over.

EXIT CODES
==========

`0` it ran, or the environment honestly cannot (no plugin, no device — a
laptop should be able to run this file), `1` something was wrong with the
artifact or the run. A missing GPU is not a failure of this example; a wrong
answer would be.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# A repo checkout has the loader here; a deployment has it from
# `pip install -e harness/python` and does not need this line.
_HARNESS = Path(__file__).resolve().parents[2] / "harness" / "python"
if _HARNESS.is_dir():
    sys.path.insert(0, str(_HARNESS))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", default=str(Path(__file__).resolve().parent / "build" / "artifact"))
    ap.add_argument("--platform", default="cuda", help="cuda | tpu | cpu")
    ap.add_argument("--prompt", default="", help="comma-separated token IDS (not text)")
    ap.add_argument("--text", default="", help="a prompt in WORDS, encoded with the model's own tokenizer.json")
    ap.add_argument("--tokenizer", default="",
                    help="path to the model's tokenizer.json (default: the checkpoint cache)")
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--verify-weights", action="store_true",
                    help="re-hash every staged weight file (slow, honest)")
    args = ap.parse_args()

    try:
        import tlaloc_serve
    except ImportError as e:
        print(f"SKIP: cannot import tlaloc_serve ({e}).")
        print("      pip install -e <tlaloc>/harness/python   # pulls in nothing")
        return 0

    # ---- dependency 1 of 2: the plugin .so ---------------------------------
    try:
        plugin = tlaloc_serve.find_pjrt_plugin(platform=args.platform)
    except Exception as e:
        print(f"SKIP: no PJRT {args.platform} plugin found, so there is nothing to run on.")
        for line in str(e).splitlines():
            print(f"      {line}")
        print("      Half one still ran: the artifact in --artifact is the deployment,")
        print("      and it is complete whether or not this box can execute it.")
        return 0

    art_dir = Path(args.artifact)
    if not (art_dir / "tlaloc-serving.json").is_file():
        print(f"no artifact at {art_dir} — run half one first:", file=sys.stderr)
        print("    ./gradlew -p examples/gpu-inference run", file=sys.stderr)
        return 1

    print("the entire runtime of this process:")
    print(f"  PJRT plugin   {plugin}")
    # `--platform` is a REQUEST, not an observation: the plugin decides what it
    # actually opens, and a plugin built for one backend will happily ignore a
    # name meant for another. So this line says what was asked for, and the
    # `engine` line below — which asks the live client for its own platform
    # name — says what answered. If the two disagree, the mismatch is printed.
    print(f"  platform      {args.platform} (requested; the client's own answer is below)")
    print(f"  Python        {sys.executable} ({sys.version.split()[0]})")
    print(f"  frameworks    {frameworks()}")
    print()

    try:
        art = tlaloc_serve.ServingArtifact.load(str(art_dir), platform=args.platform)
    except Exception as e:
        print(f"SKIP: no {args.platform} device visible to that plugin ({e}).")
        return 0

    with art:
        # The body filenames ARE hashes of the bodies. Checking them is one line
        # and it is the difference between "a directory" and "a deployment".
        art.verify_bodies()
        if args.verify_weights and art.weight_table:
            art.verify_weights()

        staged = len(art.weight_table)
        print(f"artifact      {art_dir}")
        print(f"  model       {art.model_name}")
        print(f"  hash        {art.model_hash}")
        print(f"  layers      {art.model['numLayers']}   vocab {art.model['vocabSize']}")
        print(f"  bodies      {len(art.entries)} compiled "
              f"{'entry' if len(art.entries) == 1 else 'entries'}, all re-hashed OK")
        print(f"  weights     {staged} staged operand files"
              if staged else "  weights     in-body constants (nothing staged)")
        # The client's own platform name, read back off the live PJRT client —
        # this is the device claim, and the only one in this program.
        actual = art.engine.platform_name()
        print(f"  engine      {type(art.engine).__name__} on {actual}")
        if actual.lower() != args.platform.lower():
            print(f"  MISMATCH    you asked for '{args.platform}' and the plugin opened "
                  f"'{actual}'. The plugin .so, not the flag, chose the device.")
        print()

        block_size = art.model["blockSize"]
        mbs = max(e.max_blocks_per_seq for e in art.entries)
        capacity = block_size * mbs

        if args.text:
            tok_path = find_tokenizer(art, args.tokenizer)
            if tok_path is None:
                raise SystemExit("--text needs the model's tokenizer.json; pass --tokenizer PATH")
            prompt = encode(load_vocab(tok_path)[0], args.text)
        else:
            prompt = [int(t) for t in args.prompt.split(",") if t.strip()] or default_prompt(art)
        max_new = min(args.max_new, capacity - len(prompt))
        if max_new < 1:
            print(f"prompt of {len(prompt)} fills this artifact's compiled context "
                  f"{capacity} ({mbs} pages x {block_size}). The ladder REFUSES rather "
                  f"than truncating: a clamped context is a wrong answer dressed as a "
                  f"slow one, and only a scheduler can split the request.", file=sys.stderr)
            return 1

        print(f"greedy decode  prompt {prompt} + {max_new} new "
              f"(compiled context {capacity} = {mbs} pages x {block_size})")

        # Pages 1..mbs. Page 0 is the scratch page and stays unallocated.
        table = [p + 1 for p in range(mbs)]
        pools = art.empty_pools()

        tokens = list(prompt)
        generated: list[int] = []
        first_s = None
        step_ms: list[float] = []

        for i in range(len(prompt) + max_new):
            if i >= len(tokens):
                break
            t0 = time.time()
            logits, pools = art.run_decode(
                token_ids=[tokens[i]],
                positions=[i],
                block_tables=[table[: (i // block_size) + 1]],
                seq_lens=[i + 1],
                slot_mapping=[table[i // block_size] * block_size + (i % block_size)],
                kv_pools=pools,
            )
            dt = time.time() - t0
            if first_s is None:
                # The first step paid for the weight upload AND the XLA compile.
                # Reporting it as a per-token time would be a lie about both.
                first_s = dt
            else:
                step_ms.append(dt * 1e3)

            # The token axis is KEPT even at length 1, so this indexing is the
            # same the day a prefill entry exists.
            row = logits[0][-1] if isinstance(logits[0][0], list) else logits[0]
            nxt = max(range(len(row)), key=row.__getitem__)
            if i >= len(prompt) - 1 and len(generated) < max_new:
                generated.append(nxt)
                tokens.append(nxt)
            mark = "prompt" if i < len(prompt) else "gen"
            print(f"  step {i:2d}  {mark:6s}  token {tokens[i]:6d}  "
                  f"-> argmax {nxt:6d}   {dt * 1e3:8.1f} ms")

        print()
        print(f"prompt     {prompt}")
        print(f"generated  {generated}")
        print(f"all        {tokens}")
        print()
        print(f"  XLA compiles     {art.compile_count} "
              f"(one per distinct ladder point actually touched)")
        print(f"  first step       {first_s * 1e3:.0f} ms  (weight upload + compile)")
        if step_ms:
            print(f"  median step      {sorted(step_ms)[len(step_ms) // 2]:.0f} ms")
        print()
        tok_path = find_tokenizer(art, args.tokenizer)
        if tok_path is not None:
            _, rev = load_vocab(tok_path)
            print("  in words, using the vocabulary that ships with the checkpoint:")
            print()
            print(f"    prompt      {decode(rev, prompt)!r}")
            print(f"    completion  {decode(rev, generated)!r}")
            print()
            print("There is still no tokenizer LIBRARY in this process — that is a JSON")
            print("file and the standard library's `json`. Turning text into ids is a")
            print(f"frontend's job, and the manifest names '{art.model_name}' precisely")
            print("so a frontend can find the vocabulary that belongs to these weights.")
        else:
            print("Those are token IDS, in and out. There is no tokenizer in this process")
            print("and none in the artifact: turning text into ids is a frontend's job, and")
            print(f"the manifest names '{art.model_name}' precisely so a frontend can find")
            print("the tokenizer that belongs to these weights.")
    return 0


def frameworks() -> str:
    """Report whether the frameworks are even INSTALLED in this interpreter.

    Not "not imported" — `find_spec` asks the import system whether it could
    find them at all, and the honest answer on a serving box is that it cannot.
    Run this script with `/usr/bin/python3` to see that answer.
    """
    import importlib.util

    present = [m for m in ("jax", "jaxlib", "torch", "numpy", "transformers")
               if importlib.util.find_spec(m) is not None]
    if not present:
        return "none installed in this interpreter (jax, torch, numpy: not found)"
    return f"{', '.join(present)} installed but UNUSED by this script"


# ---------------------------------------------------------------------------
# The tokenizer: not imported, READ.
#
# There is still no tokenizer library in this process. `tokenizer.json` is a
# JSON file that ships with the checkpoint, and `json` is in the standard
# library — so the ids below come from the model's OWN vocabulary rather than
# from a number written by hand, which was always the rule here. Encoding is
# exact-match only: if a word is not one vocabulary entry, this refuses instead
# of guessing, because a prompt that does not match the oracle's tokenization
# would make the whole comparison meaningless.
# ---------------------------------------------------------------------------

DEFAULT_TEXT = "The capital of France is"


def find_tokenizer(art, explicit: str):
    """The model's own tokenizer.json, or None."""
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    name = getattr(art, "model_name", None) or ""
    if not name:
        return None
    p = Path.home() / ".cache" / "tlaloc-checkpoints" / name / "tokenizer.json"
    return p if p.is_file() else None


def load_vocab(path):
    """{token: id} and {id: token} from a tokenizer.json, with json alone."""
    with open(path) as fh:
        vocab = json.load(fh)["model"]["vocab"]
    return vocab, {v: k for k, v in vocab.items()}


def encode(vocab, text: str) -> list:
    """BOS + one id per whitespace-separated word. Exact matches only."""
    ids = [vocab.get("<s>", 1)]
    for word in text.split():
        token = "\u2581" + word
        if token not in vocab:
            raise SystemExit(
                f"the word {word!r} is not a single entry in this model's vocabulary.\n"
                "This encoder is exact-match only — it will not guess a split that\n"
                "might differ from the tokenizer HuggingFace would use. Pass --prompt\n"
                "with ids instead."
            )
        ids.append(vocab[token])
    return ids


def decode(rev, ids) -> str:
    """Ids back to text: SentencePiece marks a word start with U+2581."""
    out = []
    for i in ids:
        tok = rev.get(i)
        if tok is None:
            out.append(f"<{i}>")
        elif tok.startswith("<0x") and tok.endswith(">"):
            out.append(chr(int(tok[3:-1], 16)))
        elif tok in ("<s>", "</s>"):
            continue
        else:
            out.append(tok.replace("\u2581", " "))
    return "".join(out)


def default_prompt(art) -> list:
    """A prompt that is in range for whatever was exported.

    The reference decode graph has an 11-word vocabulary and 4 tokens of
    context, so it gets a toy prompt. A real model gets DEFAULT_TEXT, encoded
    with its own tokenizer.json — and if that file is not next to the weights,
    this still refuses to invent ids.
    """
    if not art.weight_table:
        return [1, 2]
    path = find_tokenizer(art, "")
    if path is None:
        raise SystemExit(
            "this artifact is a real model — pass --prompt with ids from its own\n"
            "tokenizer, or --tokenizer with the path to its tokenizer.json.\n"
            "Nothing here writes a token id by hand."
        )
    vocab, _ = load_vocab(path)
    return encode(vocab, DEFAULT_TEXT)


if __name__ == "__main__":
    sys.exit(main())
