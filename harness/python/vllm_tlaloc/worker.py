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

from . import kv_layout
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
        # §0.4.492 — what `execute_model` computed and `sample_tokens` will
        # adapt. Exactly one step deep: vLLM's engine calls the two in
        # immediate succession, and a queue here would be a second opinion
        # about how many steps are in flight.
        self._pending = None

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
        # §0.4.491: the element-size table and the pool arithmetic moved to
        # `kv_layout`, which the attention backend class reads the same
        # answer out of. Two tables is how a reported pool stops being the
        # pool that exists.
        return kv_layout.pool_bytes(self._runner().model)

    def get_kv_cache_spec(self) -> dict:
        """vLLM asks each layer's KV shape so its block manager can size
        itself. Answered from the manifest, per layer, with the axis order
        the artifact STATES (`kvPoolDims`) rather than the one every
        backend happens to use.

        §0.4.492 (H3c-4b) — THE RETURN TYPE IS vLLM's, NOT A DICT. Until this
        slice the method answered with a dict of plain dicts, which reads
        correctly and is the wrong thing: `EngineCore._initialize_kv_caches`
        passes every returned value straight to `resolve_kv_cache_layout`,
        which asks it for `.num_heads` / `.num_states` / `.page_size_bytes`.
        A dict has none of those, and the engine died with
        `AttributeError: 'dict' object has no attribute 'num_heads'` — the
        first thing a real `LLM(...)` construction found. The fields are the
        same fields; what changed is that vLLM's own dataclass now computes
        the derived ones, which is also what makes `KV_PAGE_BYTES_AGREE`
        checkable at runtime instead of only in a test.
        """
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        from .attention import torch_kv_dtype

        m = self._runner().model
        spec = FullAttentionSpec(
            block_size=m["blockSize"],
            num_kv_heads=m["numKvHeads"],
            head_size=m["headDim"],
            dtype=torch_kv_dtype(m),
        )
        # The page byte count is the one number vLLM's block manager spends,
        # and it now has two independent derivations live in one process.
        # Disagreeing here means the plugin would hand out blocks of a size
        # the compiled pool does not have; that is a refusal, not a warning.
        ours = kv_layout.page_size_bytes(m)
        if int(spec.page_size_bytes) != ours:
            raise ValueError(
                f"vLLM's FullAttentionSpec.page_size_bytes is {spec.page_size_bytes} and "
                f"the artifact's manifest-derived page is {ours} bytes; the block manager "
                f"would hand out pages the compiled KV pool does not have"
            )
        return {f"layer.{i}": spec for i in range(m["numLayers"])}

    def get_supported_kv_cache_layouts(self) -> list:
        """§0.4.492 — the layouts this worker can be given, derived from the
        artifact's `kvPoolAxisOrder` (H3c-4a's `kv_cache_layout_name`).

        `resolve_kv_cache_layout` intersects these across workers and asserts
        the list is non-empty; the base class's default is not usable here
        because it is the torch backends' set. One entry: a compiled pool has
        exactly one axis order."""
        return [kv_layout.kv_cache_layout_name(self._runner().model)]

    def set_kv_cache_layout(self, kv_cache_layout: str) -> None:
        """vLLM tells the worker which layout it resolved. It can only be the
        one this worker offered — anything else is vLLM having resolved past
        the artifact, and is refused rather than recorded."""
        ours = kv_layout.kv_cache_layout_name(self._runner().model)
        if kv_cache_layout != ours:
            raise ValueError(
                f"vLLM resolved the KV cache layout to {kv_cache_layout}; this artifact's "
                f"pool is {ours} (from kvPoolAxisOrder). The layout is a compiled fact"
            )

    def initialize_from_config(self, kv_cache_config) -> None:
        """The pools already exist — `TlalocModelRunner` allocated them to
        the artifact's declared layout at load. vLLM's KV-cache config is
        accepted and CHECKED against that layout rather than applied: two
        sources of truth for a compiled shape is one too many.

        §0.4.492 (H3c-4b) — THE ARGUMENT IS ONE CONFIG, NOT A LIST. The
        plural spelling came from `WorkerBase.initialize_from_config`'s
        `kv_cache_configs: list[Any]`, which is the WRAPPER's signature: it
        indexes that list by rank and hands each worker its own
        `KVCacheConfig` (`gpu_worker.py` spells the parameter singular for
        exactly this reason). A real engine start found it —
        `TypeError: 'KVCacheConfig' object is not iterable` — and this is the
        second thing in this file that a signature read off a base class got
        wrong. Both spellings are accepted now, because the live lane drives
        it one way and vLLM drives it the other, and a worker that only works
        when called by the test is not a worker.
        """
        r = self._runner()
        configs = kv_cache_config if isinstance(kv_cache_config, (list, tuple)) else [kv_cache_config]
        for cfg in (c for c in configs if c is not None):
            layout = getattr(cfg, "kv_cache_layout", None)
            if layout is not None:
                self.set_kv_cache_layout(layout)
            n = getattr(cfg, "num_blocks", None)
            if n is not None and n != r.num_gpu_blocks:
                raise ValueError(
                    f"vLLM sized the KV cache at {n} blocks; the artifact's compiled "
                    f"pool has {r.num_gpu_blocks}. The pool is a compiled shape and "
                    f"cannot be resized by a config"
                )

    def get_supported_tasks(self) -> tuple:
        """§0.4.492 — `("generate",)`, and nothing else.

        The fourth thing a real engine start demanded. A torch worker
        forwards this to its model runner, which inspects the loaded
        `nn.Module` for pooling/classification heads. There is no module to
        inspect here: the artifact carries ONE compiled entry kind,
        `decode`, whose output is a logits tensor. Embedding, classification
        and reward tasks would each need their own exported entry, so
        claiming them would be claiming a program that is not in the
        directory.
        """
        return ("generate",)

    def compile_or_warm_up_model(self):
        """§0.4.492 (H3c-4b) — the third thing a real engine start demanded.

        `WorkerBase.compile_or_warm_up_model` is `raise NotImplementedError`,
        and the executor calls it unconditionally after the KV caches are
        sized. It is where a torch backend does its CUDA-graph capture and
        `torch.compile` warm-up.

        Here it returns ZEROS, and that is a statement rather than a stub:
        compilation in this plugin is PJRT's, it is lazy and per bucket, and
        it is cached by the manifest's `cacheKey` (H3a). Which buckets a
        deployment should compile at startup is H3a's still-open warm-up
        deferral — there is no answer yet, so v1 compiles the buckets it is
        asked for and reports no warm-up time rather than an invented one.

        REJECTED: running a decode step here to force the first compile. It
        would make startup report a number, but the step would write KV into
        the pool for a sequence no scheduler admitted, and the runner would
        then be one token ahead of vLLM's bookkeeping for the rest of the
        process.
        """
        from vllm.v1.worker.worker_base import CompilationTimes

        return CompilationTimes(language_model=0.0, encoder=0.0)

    # --- the step --------------------------------------------------------

    def execute_model(self, scheduler_output):
        """Run the step and return **None**.

        §0.4.492 (H3c-4b) — until this slice this method ran the step AND
        returned the `ModelRunnerOutput`, which reads like the whole
        contract and is half of it. vLLM 0.29.0 splits the two:
        `execute_model` returning `None` means "call `sample_tokens` now",
        and `EngineCore.step_with_batch_queue` calls it unconditionally for
        a generation model and takes the RESULT OF `sample_tokens`,
        discarding whatever `execute_model` returned. A worker that
        returned the output here and did not implement `sample_tokens`
        therefore ran the model correctly and then died in
        `WorkerBase.sample_tokens`'s `raise NotImplementedError` — which is
        exactly what the first real `generate()` did.

        The split exists so that structured-output grammars can mask logits
        between the forward pass and the sampler. This worker has the
        logits in `runner.last_logits` for the same reason it kept them
        since §0.4.477, so honouring the split costs nothing but saying
        which half does what.
        """
        # §0.4.492 — an EMPTY batch is a real thing vLLM schedules. When the
        # batch queue drains and the scheduler has requests but nothing to
        # run for them this step, `total_num_scheduled_tokens` is 0, and the
        # batch-queue path then uses THIS call's return value directly
        # (`if self.is_pooling_model or not model_executed`) — `sample_tokens`
        # is never called for it. So the answer is vLLM's own empty output,
        # not `None`, and not a decode step: `build_decode_call` refused the
        # empty batch by name and it was right to, which is how this arrived
        # here as a clean `ValueError` rather than as a zero-row graph call.
        if getattr(scheduler_output, "total_num_scheduled_tokens", None) == 0:
            from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

            return EMPTY_MODEL_RUNNER_OUTPUT

        r = self._runner()
        new_reqs, decode_reqs, finished = decode_requests_from_scheduler_output(
            scheduler_output, last_token_of=lambda rid: r.sequence_tokens(rid)[-1],
        )
        for rid, prompt in new_reqs:
            r.add_sequence(rid, prompt)
        call, _logits, sampled = r.step(decode_reqs)
        for rid in finished:
            r.free_sequence(rid)
        self._pending = (call, sampled)
        return None

    def sample_tokens(self, grammar_output=None):
        """The second half of the step: turn the logits already computed into
        vLLM's `ModelRunnerOutput`.

        Sampling itself happened in `runner.step` (host-side greedy, the
        audit's §2.6 scope) — this method is the ADAPTER, and it exists as
        its own method because vLLM's engine calls it as its own method.

        A non-null `grammar_output` is refused BY NAME: a grammar bitmask is
        a mask over the logits vector that must be applied BEFORE the
        argmax, and this worker's argmax has already run. Implementing it
        means moving `greedy_sample` out of `runner.step` and into here,
        which is a real slice and not a line — so it is a named deferral
        rather than a silently ignored argument. Structured outputs would
        otherwise come back unconstrained and look like they worked.
        """
        if grammar_output is not None:
            raise NotImplementedError(
                "tlaloc: structured outputs (a grammar bitmask) are not supported; "
                "the mask must be applied to the logits before the argmax and this "
                "worker samples inside execute_model. Run without a guided-decoding "
                "backend, or move greedy_sample into sample_tokens first"
            )
        pending = getattr(self, "_pending", None)
        if pending is None:
            raise RuntimeError(
                "TlalocWorker.sample_tokens: no step is pending; vLLM calls this only "
                "after an execute_model that returned None"
            )
        self._pending = None
        call, sampled = pending

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
