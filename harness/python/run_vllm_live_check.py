"""§0.4.477 — Phase H7: the LIVE vLLM lane, run with vLLM actually installed.

This is the script that settles `docs/INFERENCE_SERVING_AUDIT.md` §5's
WRITTEN-BUT-UNCERTIFIED item 1. Until H7 there were exactly two files in
this repo that imported vLLM — `vllm_tlaloc/platform.py` and
`vllm_tlaloc/worker.py` — and neither had ever been executed, because
installing vLLM into `~/.local/venvs/iree` would have replaced the torch
every cross-language oracle in the repo is measured against (186 packages,
torch 2.13+cu130, 33 CUDA-13 wheels — the §0.4.470 dry-run record).

H6 is what makes this runnable without that trade. The serving path reaches
PJRT through `tlaloc_pjrt`'s ctypes binding, so the venv this script runs in
needs **vLLM and nothing else of ours**: no jax, no jaxlib, no framework at
all below the plugin `.so` that `$TLALOC_PJRT_PLUGIN_PATH` names. That `.so`
is allowed to live inside the oracle venv's site-packages — `dlopen` of a
file in another directory is not a dependency on that directory's Python.

    python run_vllm_live_check.py --artifact DIR --request req.json \
        --output result.json [--platform cuda]

Exit codes follow `run_vllm_tlaloc_check.py`: 0 ran, 1 the run failed (the
interesting failure), 2 the environment cannot run it at all.

## The three stages, and what each one is worth

1. **DISCOVERY** — `from vllm.platforms import current_platform` must land
   on `vllm_tlaloc.platform.TlalocPlatform`. This exercises the whole
   out-of-tree path: the entry point in `pyproject.toml`, `register()`
   returning a dotted path, vLLM importing it, and the `_enum =
   PlatformEnum.OOT` contract holding against the vLLM that is installed
   rather than against the one the docstrings were written from.

2. **THE CONFIG HOOK** — `check_and_update_config` against REAL vLLM config
   objects (`CacheConfig`, `ParallelConfig`), not stand-ins. The hook's
   whole design is that it REFUSES rather than adjusts, so the check is not
   "does it run" but "does each of the four refusals fire, by name, on a
   config vLLM itself built". A refusal that only fires against a
   `SimpleNamespace` is a refusal that has never met a pydantic validator.

3. **THE WORKER LANE** — `TlalocWorker` driven through the v1 worker API
   with a duck-typed `SchedulerOutput`, returning a real
   `vllm.v1.outputs.ModelRunnerOutput`. The scheduler-output stand-in is
   deliberate and is NOT a weakening: `decode_requests_from_scheduler_output`
   is duck-typed on purpose (its docstring says why — the dataclass is
   internal and has been renamed between vLLM versions), so a stand-in
   carrying the v1 field names tests exactly the contract the adapter
   claims. What is NOT a stand-in is everything the adapter hands its result
   to: `WorkerBase`, `ModelRunnerOutput`, and the config the platform
   validated.

## What this script does NOT do, stated rather than implied

It does not call `LLM.generate()` or start `vllm serve`. Both require a
HuggingFace config and tokenizer for a real model, and the artifact this
repo can export today is `ReferenceDecodeGraph` — a deliberately tiny LCG
model with a 16-token vocabulary and no tokenizer. That gap is H3c (the
weight name-mapping and a real Llama), already the audit's largest open
item; it is a MODEL-COVERAGE gap and not a plugin gap, and pretending
otherwise by wrapping the reference model in a fake `config.json` would
certify the fake. So the lane certified here is: vLLM's platform discovery,
vLLM's config validation, and vLLM's worker interface — every line in the
two files that import vLLM except the ones an engine-level `generate()`
would drive.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _flat(x) -> list:
    """Flatten nested lists / any ndarray-alike to a flat list of floats.

    The ctypes engine's logits are nested Python lists; a numpy build would
    hand back an ndarray. Written to accept either without importing numpy,
    because this venv's numpy belongs to vLLM and the serving path must not
    start depending on it by accident (§0.4.476)."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    out: list = []
    stack = [x]
    while stack:
        v = stack.pop()
        if isinstance(v, (list, tuple)):
            stack.extend(reversed(v))
        else:
            out.append(float(v))
    return out


def _refusal(fn) -> str:
    """Run something expected to raise; return the message, or "" if it did
    not raise. The empty string is what makes a missing refusal FAIL on the
    JVM side rather than silently pass as "some string was recorded"."""
    try:
        fn()
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return ""


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
        import vllm
    except Exception as e:  # pragma: no cover - environment gate
        return emit({"ok": False, "stage": "import", "error": f"{type(e).__name__}: {e}"}, 2)

    import os

    os.environ.setdefault("TLALOC_SERVING_ARTIFACT", str(args.artifact))
    req = json.loads(args.request.read_text())
    payload: dict = {"vllmVersion": vllm.__version__, "platformArg": args.platform}

    # --- stage 1: discovery ---------------------------------------------
    try:
        from vllm.platforms import current_platform

        payload["platformClass"] = (
            type(current_platform).__module__ + "." + type(current_platform).__qualname__
        )
        payload["deviceName"] = current_platform.device_name
        payload["getDeviceName"] = current_platform.get_device_name()
        payload["supportsV1"] = bool(current_platform.supports_v1(None))
        payload["isAsyncOutputSupported"] = bool(
            current_platform.is_async_output_supported(True)
        )
        # The attention-backend refusal: compiled-in attention has no
        # runtime-selectable backend, and vLLM asking must get an answer.
        payload["attnBackendRefusal"] = _refusal(current_platform.get_attn_backend_cls)
    except Exception as e:
        return emit(
            {**payload, "ok": False, "stage": "discovery", "error": f"{type(e).__name__}: {e}",
             "traceback": traceback.format_exc()}, 1,
        )

    # --- stage 2: the config hook against real vLLM config objects -------
    try:
        from vllm.config import CacheConfig, ParallelConfig
        from vllm_tlaloc.platform import TlalocPlatform
        from vllm_tlaloc.runner import TlalocModelRunner

        probe = TlalocModelRunner(args.artifact, platform=args.platform)
        payload["blockSize"] = probe.block_size
        payload["numGpuBlocks"] = probe.num_gpu_blocks
        payload["maxBatch"] = probe.max_batch
        payload["maxContext"] = probe.max_context

        def cfg(block_size=None, max_model_len=None, max_num_seqs=None, world_size=1):
            cache = CacheConfig()
            cache.block_size = block_size if block_size is not None else probe.block_size
            parallel = ParallelConfig()
            parallel.world_size = world_size
            return SimpleNamespace(
                cache_config=cache,
                parallel_config=parallel,
                model_config=SimpleNamespace(max_model_len=max_model_len),
                scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
            )

        happy = cfg(max_model_len=probe.max_context, max_num_seqs=probe.max_batch)
        TlalocPlatform.check_and_update_config(happy)
        payload["configHook"] = {
            "workerCls": happy.parallel_config.worker_cls,
            "blockSize": happy.cache_config.block_size,
            "numGpuBlocksOverride": happy.cache_config.num_gpu_blocks_override,
            "cacheConfigClass": type(happy.cache_config).__module__
            + "."
            + type(happy.cache_config).__qualname__,
        }
        payload["refusals"] = {
            "blockSize": _refusal(
                lambda: TlalocPlatform.check_and_update_config(
                    cfg(block_size=probe.block_size * 2)
                )
            ),
            "maxModelLen": _refusal(
                lambda: TlalocPlatform.check_and_update_config(
                    cfg(max_model_len=probe.max_context + 1)
                )
            ),
            "maxNumSeqs": _refusal(
                lambda: TlalocPlatform.check_and_update_config(
                    cfg(max_num_seqs=probe.max_batch + 1)
                )
            ),
            "worldSize": _refusal(
                lambda: TlalocPlatform.check_and_update_config(cfg(world_size=2))
            ),
        }
    except Exception as e:
        return emit(
            {**payload, "ok": False, "stage": "config", "error": f"{type(e).__name__}: {e}",
             "traceback": traceback.format_exc()}, 1,
        )

    # --- stage 3: the worker lane ----------------------------------------
    try:
        import tlaloc_serve

        tlaloc_serve.check_padding_constants(req["padding"])

        from vllm_tlaloc.worker import TlalocWorker

        worker = TlalocWorker(vllm_config=happy)
        worker.init_device()
        worker.load_model()
        payload["availableMemoryBytes"] = worker.determine_available_memory()
        spec = worker.get_kv_cache_spec()
        payload["kvCacheSpecLayers"] = sorted(spec.keys())
        payload["kvCacheSpecLayer0"] = spec["layer.0"]
        # vLLM's KV-cache config is CHECKED against the compiled pool, not
        # applied: agreeing is silent, disagreeing must be named.
        worker.initialize_from_config([SimpleNamespace(num_blocks=probe.num_gpu_blocks)])
        payload["kvConfigRefusal"] = _refusal(
            lambda: worker.initialize_from_config(
                [SimpleNamespace(num_blocks=probe.num_gpu_blocks + 1)]
            )
        )

        prompts = [int(p) for p in req["prompts"]]
        seq_ids = [str(i) for i in range(len(prompts))]

        steps = []
        for i in range(int(req["steps"])):
            # The v1 SchedulerOutput shape: new requests on step 0, cached
            # ones after. The scheduler says WHICH requests run and how many
            # tokens each gets; the runner owns what token each feeds in.
            if i == 0:
                so = SimpleNamespace(
                    scheduled_new_reqs=[
                        SimpleNamespace(req_id=r, prompt_token_ids=[p])
                        for r, p in zip(seq_ids, prompts)
                    ],
                    scheduled_cached_reqs=None,
                    num_scheduled_tokens={r: 1 for r in seq_ids},
                    finished_req_ids=[],
                )
            else:
                so = SimpleNamespace(
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=SimpleNamespace(req_ids=list(seq_ids)),
                    num_scheduled_tokens={r: 1 for r in seq_ids},
                    finished_req_ids=[],
                )
            out = worker.execute_model(so)
            r = worker.model_runner
            call = r.last_call
            steps.append({
                "outputClass": type(out).__module__ + "." + type(out).__qualname__,
                "reqIds": list(out.req_ids),
                "reqIdToIndex": {k: int(v) for k, v in out.req_id_to_index.items()},
                "sampled": [int(t[0]) for t in out.sampled_token_ids],
                "logits": _flat(r.last_logits),
                "bucket": list(call.bucket) if call is not None else None,
            })

        payload["steps"] = steps
        payload["compileCount"] = worker.model_runner.artifact.compile_count

        # The chunked-prefill refusal, on the live adapter: a request
        # scheduled for more than one token has no compiled entry, and the
        # scheduler must be told rather than handed a wrong batch.
        payload["chunkedPrefillRefusal"] = _refusal(
            lambda: worker.execute_model(
                SimpleNamespace(
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=SimpleNamespace(req_ids=[seq_ids[0]]),
                    num_scheduled_tokens={seq_ids[0]: 3},
                    finished_req_ids=[],
                )
            )
        )

        payload["ok"] = True
        payload["stage"] = "ok"
        return emit(payload, 0)
    except Exception as e:
        return emit(
            {**payload, "ok": False, "stage": "worker", "error": f"{type(e).__name__}: {e}",
             "traceback": traceback.format_exc()}, 1,
        )


if __name__ == "__main__":
    raise SystemExit(main())
