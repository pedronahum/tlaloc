"""§0.4.470 — Phase H3b: the plugin-driven end-to-end check.

Drives `vllm_tlaloc.TlalocModelRunner` — the class vLLM's worker delegates
to — over a REAL serving artifact on real PJRT, with no vLLM installed and
no JVM in the process. What this certifies is the whole plugin below the
vLLM API surface: page allocation, slot arithmetic, block tables, bucket
selection, the pool swap across steps, the artifact binding, and host-side
sampling. What it does not touch is `platform.py` / `worker.py`, which is
the honest boundary recorded in the audit.

    python run_vllm_tlaloc_check.py --artifact DIR --request req.json \
        --output result.json [--platform cuda]

Exit codes follow `run_tlaloc_serve_check.py`: 0 ran, 1 the run failed
(the interesting failure), 2 the environment cannot run it at all.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, type=Path)
    ap.add_argument("--request", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--platform", default="cuda")
    args = ap.parse_args()

    def emit(payload: dict, code: int) -> int:
        args.output.write_text(json.dumps(payload) + "\n")
        return code

    if not args.artifact.exists() or not args.request.exists():
        return emit({"ok": False, "stage": "input", "error": "missing artifact or request"}, 2)

    try:
        import jax  # noqa: F401
        import numpy as np
    except Exception as e:  # pragma: no cover - environment gate
        return emit({"ok": False, "stage": "import", "error": f"{type(e).__name__}: {e}"}, 2)

    try:
        import jax

        if args.platform in ("cuda", "gpu") and not any(
            d.platform in ("cuda", "gpu") for d in jax.devices()
        ):
            return emit({"ok": False, "stage": "device", "error": "no CUDA device"}, 2)
    except Exception as e:  # pragma: no cover - environment gate
        return emit({"ok": False, "stage": "device", "error": f"{type(e).__name__}: {e}"}, 2)

    req = json.loads(args.request.read_text())

    try:
        import tlaloc_serve
        from vllm_tlaloc.runner import TlalocModelRunner

        tlaloc_serve.check_padding_constants(req["padding"])
        runner = TlalocModelRunner(args.artifact, platform=args.platform)
    except Exception as e:
        return emit(
            {"ok": False, "stage": "load", "error": f"{type(e).__name__}: {e}",
             "traceback": traceback.format_exc()},
            1,
        )

    try:
        prompts = req["prompts"]  # one token per sequence: the decode-only admission
        seq_ids = list(range(len(prompts)))
        for sid, p in zip(seq_ids, prompts):
            runner.add_sequence(sid, [p])

        steps = []
        feed = [int(p) for p in prompts]
        for _ in range(int(req["steps"])):
            call, logits, sampled = runner.step(list(zip(seq_ids, feed)))
            steps.append({
                "bucket": list(call.bucket),
                "context": call.context,
                "positions": list(call.positions),
                "seqLens": list(call.seq_lens),
                "slotMapping": list(call.slot_mapping),
                "blockTables": [list(t) for t in call.block_tables],
                "logits": [float(x) for x in np.asarray(logits).reshape(-1)],
                "logitsShape": list(np.asarray(logits).shape),
                "sampled": [int(t) for t in sampled],
            })
            feed = [int(t) for t in sampled]

        # Free one sequence and admit a new one: the pool's pages must come
        # back, and the newcomer must get the lowest of them. A leak here is
        # invisible to a single-request test and fatal to a server.
        # Every step so far ran at one bucket, so the executable cache must
        # hold exactly one program. Read BEFORE the lifecycle step below,
        # which is a batch of one and therefore a DIFFERENT bucket.
        compiles_after_steps = runner.artifact.compile_count

        freed_seq = seq_ids[-1]
        free_before = runner.pool.free_pages
        runner.free_sequence(freed_seq)
        free_after_release = runner.pool.free_pages
        runner.add_sequence(99, [int(prompts[0])])
        runner.step([(99, int(prompts[0]))])
        reused = list(runner.pool.sequence(99).blocks)

        payload = {
            "ok": True,
            "stage": "ok",
            "modelName": runner.artifact.model_name,
            "entryCount": len(runner.artifact.entries),
            "blockSize": runner.block_size,
            "numGpuBlocks": runner.num_gpu_blocks,
            "maxBatch": runner.max_batch,
            "maxContext": runner.max_context,
            "poolCapacity": runner.pool.capacity,
            "steps": steps,
            "compileCountAfterSteps": compiles_after_steps,
            "compileCount": runner.artifact.compile_count,
            "freePagesBeforeRelease": free_before,
            "freePagesAfterRelease": free_after_release,
            "reusedBlocks": reused,
            "historyOfSeq0": runner.sequence_tokens(seq_ids[0]),
        }
        return emit(payload, 0)
    except Exception as e:
        return emit(
            {"ok": False, "stage": "run", "error": f"{type(e).__name__}: {e}",
             "traceback": traceback.format_exc()},
            1,
        )


if __name__ == "__main__":
    sys.exit(main())
