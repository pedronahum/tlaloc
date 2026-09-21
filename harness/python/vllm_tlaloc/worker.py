"""§0.4.470 — `TlalocWorker`: vLLM's worker interface over
`TlalocModelRunner`.

The second and last module that imports vLLM, and therefore the second and
last module the certification cannot run (see `platform.py`'s docstring and
`docs/INFERENCE_SERVING_AUDIT.md` §5 H3b). Everything it delegates to is
certified; the delegation itself is written against the documented v1
worker API and is honestly untested.

## Decisions

* **`determine_available_memory` reports the artifact's pool, not a
  measurement.** vLLM profiles a forward pass to find out how much KV cache
  fits. Here the KV pool is a COMPILED shape: `numBlocks × blockSize ×
  numKvHeads × headDim`, fixed at export. Reporting a measured number would
  let vLLM's block manager hand out blocks the program has no slots for —
  a profile that is *more* generous than the truth is a silent
  out-of-bounds write. So this returns the bytes the pool actually is.

* **`execute_model` does the whole step, sampling included**, and returns
  vLLM's `ModelRunnerOutput`. REJECTED: returning raw logits and letting
  vLLM's sampler run — its sampler is torch code over torch tensors on a
  torch device, and there is no torch device in this path. Host-side
  greedy sampling in v1 is the audit's stated scope; adapting to vLLM's
  sampler means materialising logits as torch CPU tensors, which is a
  measurable cost and a decision for the slice that has a real vocabulary
  to measure it on.

* **The worker is the thing that remembers what each request last
  decoded.** vLLM's scheduler output says WHICH requests run, not what
  token they feed in; the runner already keeps each sequence's history for
  exactly this reason, so the worker reads it there instead of keeping a
  second copy that can disagree.
"""

from __future__ import annotations

from .batching import decode_requests_from_scheduler_output

try:  # pragma: no cover - exercised only where vLLM is installed
    from vllm.v1.worker.worker_base import WorkerBase
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "vllm_tlaloc.worker requires vLLM to be installed; it subclasses "
        "vllm.v1.worker.worker_base.WorkerBase. The execution it delegates to "
        "is in vllm_tlaloc.runner, which imports without vLLM."
    ) from e


class TlalocWorker(WorkerBase):
    """One worker, one device, one serving artifact."""

    def __init__(self, vllm_config, local_rank: int = 0, rank: int = 0,
                 distributed_init_method: str | None = None, is_driver_worker: bool = True,
                 **kwargs):
        self.vllm_config = vllm_config
        self.local_rank = local_rank
        self.rank = rank
        self.is_driver_worker = is_driver_worker
        self.model_runner = None

    # --- lifecycle -------------------------------------------------------

    def init_device(self) -> None:
        """No device to initialise: PJRT's client is created on first
        compile, inside `tlaloc_serve`, which is also the only place that
        knows the §0.4.333 rule about never creating a client without
        create_options."""

    def load_model(self) -> None:
        """'Loading the model' is loading the ARTIFACT: manifest, bodies,
        content-address verification. Compilation is lazy and per bucket,
        cached by the manifest's `cacheKey` — see the warm-up deferral in
        H3a: which buckets a deployment compiles at startup is still a
        tuning question with no answer, so v1 compiles the ones it is
        asked for."""
        from .platform import TlalocPlatform
        from .runner import TlalocModelRunner

        self.model_runner = TlalocModelRunner(
            TlalocPlatform.artifact_path(), platform=TlalocPlatform.device_type,
        )

    def determine_available_memory(self) -> int:
        """Bytes of KV cache available — the artifact's pool, exactly."""
        r = self._runner()
        m = r.model
        elem = {"f32": 4, "bf16": 2, "f64": 8, "i32": 4, "i64": 8}.get(m["kvDtype"])
        if elem is None:
            raise ValueError(
                f"tlaloc: KV dtype '{m['kvDtype']}' has no element size here; a worker "
                f"that guesses this number reports a pool that does not exist"
            )
        pool_elems = 1
        for d in m["kvPoolDims"]:
            pool_elems *= d
        return pool_elems * elem * 2 * m["numLayers"]

    def get_kv_cache_spec(self) -> dict:
        """vLLM asks each layer's KV shape so its block manager can size
        itself. Answered from the manifest, per layer, with the axis order
        the artifact STATES (`kvPoolDims`) rather than the one every
        backend happens to use."""
        r = self._runner()
        m = r.model
        return {
            f"layer.{i}": {
                "block_size": m["blockSize"],
                "num_kv_heads": m["numKvHeads"],
                "head_size": m["headDim"],
                "dtype": m["kvDtype"],
                "pool_dims": list(m["kvPoolDims"]),
            }
            for i in range(m["numLayers"])
        }

    def initialize_from_config(self, kv_cache_configs) -> None:
        """The pools already exist — `TlalocModelRunner` allocated them to
        the artifact's declared layout at load. vLLM's KV-cache config is
        accepted and CHECKED against that layout rather than applied: two
        sources of truth for a compiled shape is one too many."""
        r = self._runner()
        for cfg in (kv_cache_configs or []):
            n = getattr(cfg, "num_blocks", None)
            if n is not None and n != r.num_gpu_blocks:
                raise ValueError(
                    f"vLLM sized the KV cache at {n} blocks; the artifact's compiled "
                    f"pool has {r.num_gpu_blocks}. The pool is a compiled shape and "
                    f"cannot be resized by a config"
                )

    # --- the step --------------------------------------------------------

    def execute_model(self, scheduler_output):
        r = self._runner()
        new_reqs, decode_reqs, finished = decode_requests_from_scheduler_output(
            scheduler_output, last_token_of=lambda rid: r.sequence_tokens(rid)[-1],
        )
        for rid, prompt in new_reqs:
            r.add_sequence(rid, prompt)
        call, logits, sampled = r.step(decode_reqs)
        for rid in finished:
            r.free_sequence(rid)

        from vllm.v1.outputs import ModelRunnerOutput

        return ModelRunnerOutput(
            req_ids=list(call.seq_ids),
            req_id_to_index={rid: i for i, rid in enumerate(call.seq_ids)},
            sampled_token_ids=[[int(t)] for t in sampled],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )

    def _runner(self):
        if self.model_runner is None:
            raise RuntimeError(
                "TlalocWorker: load_model() has not run; there is no artifact loaded"
            )
        return self.model_runner
