"""Runs requests through a bounded-program artifact with `tlaloc_bounded` on a PJRT plugin, for
`BoundedArtifactRunTest` (and the mixed-length compile-count measurement).

    python run_bounded_check.py --artifact DIR --request REQ.json --output OUT.json

REQ.json: {"cases": [{"inputs": [{"values": [...], "dims": [...]}, ...]}, ...]}
OUT.json: {"ok": true, "platform": ..., "results": [{"outputs": [{"values": [...], "dims": [...]}, ...]}, ...],
           "compileCount": N, "compileSeconds": s, "runSeconds": s, "guard": {...}}

Exit 0: ran. Exit 2: the environment cannot run it (no plugin, no device); the caller skips
by name. Exit 1: a failure. The run is under the same import guard as the serving check:
no jax, jaxlib, torch or numpy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import import_guard  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, type=Path)
    ap.add_argument("--request", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--platform", default="cuda")
    ap.add_argument("--plugin", default=None)
    args = ap.parse_args()

    guard = import_guard.install(
        why="a bounded-program artifact runs through ctypes-bound PJRT with the standard library only",
    )

    def emit(payload: dict, code: int) -> int:
        payload["guard"] = import_guard.report(guard)
        args.output.write_text(json.dumps(payload) + "\n")
        return code

    import tlaloc_bounded
    import tlaloc_serve

    try:
        plugin = tlaloc_serve.find_pjrt_plugin(args.plugin, args.platform)
    except Exception as e:  # environment gate
        return emit({"ok": False, "stage": "plugin", "error": f"{type(e).__name__}: {e}"}, 2)

    try:
        art = tlaloc_bounded.BoundedArtifact.load(args.artifact, platform=args.platform, plugin_path=plugin)
    except Exception as e:
        return emit({"ok": False, "stage": "load", "error": f"{type(e).__name__}: {e}",
                     "traceback": traceback.format_exc()}, 1)

    with art:
        try:
            platform_name = art.engine.platform_name()
        except Exception as e:  # environment gate
            return emit({"ok": False, "stage": "device", "error": f"{type(e).__name__}: {e}"}, 2)

        compile_seconds = 0.0
        real_compile = art.engine.compile

        def timed_compile(text):
            nonlocal compile_seconds
            t0 = time.perf_counter()
            try:
                return real_compile(text)
            finally:
                compile_seconds += time.perf_counter() - t0

        art.engine.compile = timed_compile
        req = json.loads(args.request.read_text())
        results = []
        t0 = time.perf_counter()
        try:
            for case in req["cases"]:
                outs = art.run_all([(i["values"], i["dims"]) for i in case["inputs"]])
                results.append({"outputs": [{"values": v, "dims": d} for v, d in outs]})
        except Exception as e:
            return emit({"ok": False, "stage": "run", "error": f"{type(e).__name__}: {e}",
                         "traceback": traceback.format_exc()}, 1)
        run_seconds = time.perf_counter() - t0
        return emit({
            "ok": True,
            "platform": platform_name,
            "results": results,
            "compileCount": art.compile_count,
            "compileSeconds": compile_seconds,
            "runSeconds": run_seconds,
        }, 0)


if __name__ == "__main__":
    sys.exit(main())
