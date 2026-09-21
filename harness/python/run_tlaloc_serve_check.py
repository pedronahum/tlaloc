"""§0.4.469 — Phase H3a: the export-then-run certification's Python half.

Loads a Tlaloc serving artifact DIRECTORY written by
`io.tlaloc.maestro.serving.ServingArtifactWriter`, runs one decode step
through jaxlib/PJRT, and writes the outputs as JSON for the Kotlin side to
compare against its own reference interpreter.

There is no JVM in this process and no gradle in the loop — the artifact
directory is the entire input. That is the claim under test.

    python run_tlaloc_serve_check.py --artifact DIR --request req.json \
        --output result.json

Exit codes: 0 ran, 1 the run failed (the interesting failure), 2 the
environment cannot run it at all (no jax, no CUDA device, bad inputs) —
the Kotlin side self-skips on 2 and fails on 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tlaloc_serve  # noqa: E402


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
    except Exception as e:  # pragma: no cover - environment gate
        return emit({"ok": False, "stage": "import", "error": f"{type(e).__name__}: {e}"}, 2)

    req = json.loads(args.request.read_text())

    try:
        art = tlaloc_serve.ServingArtifact.load(args.artifact, platform=args.platform)
        art.verify_bodies()
        tlaloc_serve.check_padding_constants(req["padding"])
    except Exception as e:
        return emit(
            {"ok": False, "stage": "load", "error": f"{type(e).__name__}: {e}",
             "traceback": traceback.format_exc()},
            1,
        )

    try:
        import jax

        if not any(d.platform in ("cuda", "gpu") for d in jax.devices()):
            return emit({"ok": False, "stage": "device", "error": "no CUDA device"}, 2)
    except Exception as e:  # pragma: no cover - environment gate
        return emit({"ok": False, "stage": "device", "error": f"{type(e).__name__}: {e}"}, 2)

    try:
        pools = [req["keyPool"], req["valuePool"]]
        import numpy as np

        dims = tuple(art.model["kvPoolDims"])
        pools = [np.asarray(p, dtype=np.float32).reshape(dims) for p in pools]

        logits, out_pools = art.run_decode(
            token_ids=req["tokenIds"],
            positions=req["positions"],
            block_tables=req["blockTables"],
            seq_lens=req["seqLens"],
            slot_mapping=req["slotMapping"],
            kv_pools=pools,
            context=req.get("context"),
        )
        first_compiles = art.compile_count
        # Run it a second time at the same bucket: the executable cache is
        # keyed by the manifest's cacheKey, so this must not compile again.
        logits2, _ = art.run_decode(
            token_ids=req["tokenIds"],
            positions=req["positions"],
            block_tables=req["blockTables"],
            seq_lens=req["seqLens"],
            slot_mapping=req["slotMapping"],
            kv_pools=pools,
            context=req.get("context"),
        )
        payload = {
            "ok": True,
            "stage": "ok",
            "modelName": art.model_name,
            "bucket": list(art.select_bucket(len(req["tokenIds"]), req.get("context") or max(req["seqLens"]))),
            "entryCount": len(art.entries),
            "compileCountAfterFirst": first_compiles,
            "compileCountAfterSecond": art.compile_count,
            "logits": [float(x) for x in np.asarray(logits).reshape(-1)],
            "logitsShape": list(np.asarray(logits).shape),
            "repeatLogitsMatch": bool(np.array_equal(np.asarray(logits), np.asarray(logits2))),
            "keyPool": [float(x) for x in np.asarray(out_pools[0]).reshape(-1)],
            "valuePool": [float(x) for x in np.asarray(out_pools[1]).reshape(-1)],
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
