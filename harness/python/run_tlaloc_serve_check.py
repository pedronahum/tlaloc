"""§0.4.469 (H3a), rewired in §0.4.476 (H6b) — the export-then-run
certification's Python half.

Loads a Tlaloc serving artifact DIRECTORY written by
`io.tlaloc.maestro.serving.ServingArtifactWriter`, runs one decode step, and
writes the outputs as JSON for the Kotlin side to compare against its own
reference interpreter.

There is no JVM in this process and no gradle in the loop — the artifact
directory is the entire input. **And in `--engine ctypes`, which is the
default and the deployment path, there is no framework either**: the whole
run happens under a `sys.meta_path` guard that raises on jax, jaxlib, torch
or numpy, in the venv where jax *is* installed. That is H6b's headline
claim, and this file is where it is measured.

    python run_tlaloc_serve_check.py --artifact DIR --request req.json \
        --output result.json [--engine ctypes|jax] [--platform cuda|cpu]

`--engine jax` is the ORACLE lane and nothing else: jaxlib ships no CPU PJRT
plugin `.so`, so the tight 1e-5 XLA-CPU semantics floor has no ctypes route
on this machine. It runs WITHOUT the guard, obviously, and says so in its
report, so no reader can mistake which lane proved which claim.

Exit codes: 0 ran, 1 the run failed (the interesting failure), 2 the
environment cannot run it at all (no plugin, no device, no jax, bad inputs) —
the Kotlin side self-skips on 2 and fails on 1. The distinction lives here so
a broken artifact cannot present itself as an absent GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import import_guard  # noqa: E402

_GUARD = None


def _install_guard():
    global _GUARD
    _GUARD = import_guard.install(
        why="§0.4.476 (H6b) runs the WHOLE serving path — manifest, bucket "
            "selection, compile, staging, execute — through ctypes-bound PJRT, so "
            "that a deployment needs a plugin .so and a driver and nothing else. "
            "An import here means that claim is false.",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, type=Path)
    ap.add_argument("--request", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--platform", default="cuda")
    ap.add_argument("--engine", default="ctypes", choices=("ctypes", "jax"))
    ap.add_argument("--plugin", default=None)
    args = ap.parse_args()

    def emit(payload: dict, code: int) -> int:
        if _GUARD is not None:
            payload["guard"] = import_guard.report(_GUARD)
        else:
            payload["guard"] = {"installed": False, "engine": args.engine}
        args.output.write_text(json.dumps(payload) + "\n")
        return code

    if not args.artifact.exists() or not args.request.exists():
        return emit({"ok": False, "stage": "input", "error": "missing artifact or request"}, 2)

    # The guard goes up BEFORE tlaloc_serve is imported, so the import itself
    # is under test: the module's whole dependency surface is asserted, not
    # just what its methods happen to touch at run time.
    if args.engine == "ctypes":
        _install_guard()
    else:
        try:
            import jax  # noqa: F401
        except Exception as e:  # pragma: no cover - environment gate
            return emit({"ok": False, "stage": "import", "error": f"{type(e).__name__}: {e}"}, 2)

    import tlaloc_serve

    req = json.loads(args.request.read_text())

    try:
        plugin = None
        if args.engine == "ctypes":
            # Resolving the plugin is an ENVIRONMENT question: no plugin file
            # is a skip (2), not a failure (1).
            plugin = tlaloc_serve.find_pjrt_plugin(args.plugin, args.platform)
    except Exception as e:  # pragma: no cover - environment gate
        return emit({"ok": False, "stage": "plugin", "error": f"{type(e).__name__}: {e}"}, 2)

    try:
        art = tlaloc_serve.ServingArtifact.load(
            args.artifact, platform=args.platform, engine=args.engine, plugin_path=plugin,
        )
        art.verify_bodies()
        tlaloc_serve.check_padding_constants(req["padding"])
    except Exception as e:
        return emit(
            {"ok": False, "stage": "load", "error": f"{type(e).__name__}: {e}",
             "traceback": traceback.format_exc()},
            1,
        )

    try:
        platform_name = art.engine.platform_name()
    except Exception as e:  # pragma: no cover - environment gate
        return emit(
            {"ok": False, "stage": "device",
             "error": f"{type(e).__name__}: {e}", "plugin": plugin}, 2,
        )

    try:
        pools = [req["keyPool"], req["valuePool"]]
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
        flat = [float(x) for x in tlaloc_serve.flatten(logits)]
        flat2 = [float(x) for x in tlaloc_serve.flatten(logits2)]
        bucket = art.select_bucket(
            len(req["tokenIds"]), req.get("context") or max(req["seqLens"]),
        )
        # The shape is REPORTED from the entry's declared logits slot with the
        # batch axis replaced by the real row count — derived from the manifest
        # rather than reconstructed here, so a nesting convention cannot drift
        # between the two processes.
        logits_dims = art.entry_for("decode", bucket[0], bucket[1]).outputs[0].dims
        payload = {
            "ok": True,
            "stage": "ok",
            "engine": args.engine,
            "plugin": plugin,
            "platformName": platform_name,
            "modelName": art.model_name,
            "bucket": list(bucket),
            "entryCount": len(art.entries),
            "compileCountAfterFirst": first_compiles,
            "compileCountAfterSecond": art.compile_count,
            "logits": flat,
            "logitsShape": [len(logits)] + [int(d) for d in logits_dims[1:]],
            "repeatLogitsMatch": bool(flat == flat2),
            "keyPool": [float(x) for x in out_pools[0]],
            "valuePool": [float(x) for x in out_pools[1]],
        }
        return emit(payload, 0)
    except Exception as e:
        return emit(
            {"ok": False, "stage": "run", "error": f"{type(e).__name__}: {e}",
             "traceback": traceback.format_exc()},
            1,
        )
    finally:
        try:
            art.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
