"""§0.4.480 (H3c-3) — greedy-decode a REAL Llama through a Tlaloc serving artifact.

    python3 harness/python/run_llama_generate.py \
        --artifact /tmp/tl-llama-full --request /tmp/req.json --output /tmp/out.json

**Stdlib + `tlaloc_serve` only.** No jax, no torch, no numpy, no transformers,
no tokenizer: the request carries token IDS and the result carries token IDS.
That is not a convenience, it is the line this whole arc is about — turning
text into ids is a frontend's job (vLLM and `transformers` both do it from the
checkpoint directory `modelName` names), and the runtime that answers is a
plugin `.so` and a driver (§0.4.476).

WHAT IT DOES
============

Loads the artifact, stages its weight table once, and runs the prompt as one
prefill call when the artifact has a prefill entry that holds it
(`ServingArtifact.run_prefill`), else as one single-token decode step per
prompt token; then one decode step per generated token, with the KV pools
threaded through, picking the argmax at each step. `--no-prefill` walks the
prompt with decode steps even when a prefill entry exists (the control).

THE PAGE ARITHMETIC, WHICH IS THE PART THAT CAN BE WRONG SILENTLY
=================================================================

Page 0 is the reserved scratch page and is NEVER allocated to a live sequence
(the padding convention in `tlaloc_serve`: a padded row's block table is all
zeros, so a live sequence on page 0 would read another row's writes). Pages
1..maxBlocksPerSeq are this one sequence's, in order, so

    slot(p) = blockTable[p // blockSize] * blockSize + (p % blockSize)

and `seqLens` at step p is `p + 1` — the token being written is part of the
context it attends over, which is what makes the first step's softmax a single
term (§0.4.466's round-trip check rests on exactly that).

Exit codes mirror `run_tlaloc_serve_check.py`: 0 ran, 1 the run failed (the
interesting failure), 2 the environment cannot run it at all.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--request", required=True, help="JSON: {promptTokens:[…], maxNewTokens:n}")
    ap.add_argument("--output", required=True)
    ap.add_argument("--platform", default="cuda")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--verify-weights", action="store_true")
    ap.add_argument("--no-prefill", action="store_true",
                    help="walk the prompt with decode steps even if a prefill entry exists")
    args = ap.parse_args()

    try:
        import tlaloc_serve
    except Exception as e:  # pragma: no cover - environment probe
        print(f"cannot import tlaloc_serve: {e}", file=sys.stderr)
        return 2

    req = json.loads(Path(args.request).read_text())
    prompt = [int(t) for t in req["promptTokens"]]
    max_new = int(req.get("maxNewTokens", 8))
    if not prompt:
        print("promptTokens is empty — there is nothing to condition on", file=sys.stderr)
        return 1

    try:
        tlaloc_serve.find_pjrt_plugin(platform=args.platform)
    except Exception as e:
        print(f"no PJRT plugin: {e}", file=sys.stderr)
        return 2

    try:
        art = tlaloc_serve.ServingArtifact.load(
            args.artifact, platform=args.platform, engine=args.engine,
        )
    except Exception as e:
        print(f"cannot load artifact: {e}", file=sys.stderr)
        return 1

    with art:
        art.verify_bodies()
        if args.verify_weights:
            art.verify_weights()
        entry = next(e for e in art.entries if e.kind == "decode")
        block_size = art.block_size
        mbs = entry.max_blocks_per_seq
        capacity = block_size * mbs
        total = len(prompt) + max_new
        if total > capacity:
            print(
                f"prompt {len(prompt)} + {max_new} new = {total} tokens exceeds this "
                f"artifact's compiled context {capacity} ({mbs} pages x {block_size}). "
                f"The ladder REFUSES rather than truncating — a clamped context is a "
                f"wrong answer dressed as a slow one",
                file=sys.stderr,
            )
            return 1

        # Pages 1..mbs. Page 0 is the reserved scratch page; see the module doc.
        table = [p + 1 for p in range(mbs)]
        pools = art.empty_pools()

        def argmax_of(logits):
            row = logits[0][-1] if isinstance(logits[0][0], list) else logits[0]
            return max(range(len(row)), key=row.__getitem__)

        t0 = time.time()
        stage_t = None
        tokens = list(prompt)
        generated: list[int] = []
        step_ms: list[float] = []
        prefill = None if args.no_prefill or len(prompt) < 2 else art.prefill_entry(1, len(prompt))
        if prefill is not None:
            # The whole prompt in one call; its last token's logits give the
            # first generated id.
            s0 = time.time()
            pages = (len(prompt) + block_size - 1) // block_size
            logits, pools = art.run_prefill([prompt], [0], [table[:pages]], pools)
            stage_t = time.time() - s0
            nxt = argmax_of(logits)
            generated.append(nxt)
            tokens.append(nxt)
            first = len(prompt)
        else:
            first = 0
        for i in range(first, total):
            if i >= len(tokens) or len(generated) >= max_new:
                break
            tok = tokens[i]
            slot = table[i // block_size] * block_size + (i % block_size)
            s0 = time.time()
            logits, pools = art.run_decode(
                token_ids=[tok], positions=[i], block_tables=[table[: (i // block_size) + 1]],
                seq_lens=[i + 1], slot_mapping=[slot], kv_pools=pools,
            )
            if stage_t is None:
                # The first step paid for the weight upload AND the compile;
                # reporting it as a per-token time would be a lie about both.
                stage_t = time.time() - s0
            else:
                step_ms.append((time.time() - s0) * 1e3)
            nxt = argmax_of(logits)
            if i >= len(prompt) - 1 and len(generated) < max_new:
                generated.append(nxt)
                tokens.append(nxt)

        result = {
            "artifact": str(Path(args.artifact).resolve()),
            "modelName": art.model_name,
            "modelHash": art.model_hash,
            "platform": art.engine.platform_name(),
            "engine": type(art.engine).__name__,
            "numLayers": art.model["numLayers"],
            "weightSlots": len(art.weight_table),
            "weightBytes": sum(int(w["byteLength"]) for w in art.weight_table),
            "promptTokens": prompt,
            "generatedTokens": generated,
            "allTokens": tokens,
            "blockSize": block_size,
            "maxBlocksPerSeq": mbs,
            "compileCount": art.compile_count,
            "firstStepSeconds": stage_t,
            "prefillEntry": prefill.entry_id if prefill is not None else None,
            "promptCalls": 1 if prefill is not None else len(prompt),
            "medianStepMs": sorted(step_ms)[len(step_ms) // 2] if step_ms else None,
            "totalSeconds": time.time() - t0,
        }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in (
        "modelName", "numLayers", "weightSlots", "generatedTokens", "prefillEntry",
        "promptCalls", "firstStepSeconds", "medianStepMs")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
