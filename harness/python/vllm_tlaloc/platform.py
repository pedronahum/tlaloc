"""§0.4.470 — `TlalocPlatform`, the class vLLM's out-of-tree platform
discovery loads.

**This is the one module (with `worker.py`) that imports vLLM**, and it is
therefore the one module the certification cannot run — see the honest
record in `docs/INFERENCE_SERVING_AUDIT.md` §5 (H3b): vLLM is NOT installed
in this repo's venv and installing it would replace the torch every
certification oracle runs on. What is written here is written against
vLLM's documented out-of-tree platform-plugin API; what is *certified* is
everything under it (`paging`, `batching`, `runner`), end to end against a
real artifact on real PJRT.

The contract, as documented:

* an entry point in group `vllm.platform_plugins` returns the dotted path
  of a `Platform` subclass (or `None` to decline);
* the subclass sets `_enum = PlatformEnum.OOT` and identifies the device;
* `check_and_update_config(vllm_config)` is vLLM's single hook for a
  backend to impose its own truths on the config — it is where this
  platform names its worker class and refuses configurations it cannot
  honour;
* the worker class does the rest.

## Decisions

* **`check_and_update_config` REFUSES rather than adjusts, wherever the
  mismatch is a user's intent.** A `block_size` that disagrees with the
  artifact's compiled `blockSize` is not a preference: it is a different
  KV layout, and silently overriding it would make a user's `--block-size`
  a no-op. The same for a `max_model_len` past the ladder's top — H3a's
  loader already refuses an over-cap context BY NAME rather than clamping,
  for the same reason (a clamped context truncates history), and the
  platform should refuse it at startup rather than per request.
  It DOES set `worker_cls`, because that is what the hook is for.

* **The artifact is loaded once, in the platform, and consulted for the
  answers vLLM asks the platform for** (`block_size`, number of KV blocks,
  max context). REJECTED: profiling device memory to decide the block
  count, which is what the GPU backends do — a Tlaloc deployment's KV pool
  is a COMPILED shape. The pool size is a fact of the artifact, and
  discovering it by measurement would be discovering it wrong.

* **CUDA today, and the device is not hard-coded anywhere below this
  file.** `tlaloc_serve` takes a PJRT platform string; the day G2b lands,
  serving on TPU is this file naming a different one, and the artifact
  does not change. That is the whole StableHLO bet, stated as a one-line
  diff.
"""

from __future__ import annotations

import os

from . import ARTIFACT_ENV_VAR

try:  # pragma: no cover - exercised only where vLLM is installed
    from vllm.platforms.interface import Platform, PlatformEnum
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "vllm_tlaloc.platform requires vLLM to be installed; it subclasses "
        "vllm.platforms.interface.Platform. Everything the plugin does that is "
        "not vLLM's API — paging, batching, the model runner, artifact loading "
        "— lives in vllm_tlaloc.paging / .batching / .runner and imports "
        "without vLLM on purpose."
    ) from e


class TlalocPlatform(Platform):
    """vLLM out-of-tree platform backed by a Tlaloc serving artifact."""

    _enum = PlatformEnum.OOT
    device_name: str = "tlaloc"
    # The PJRT device the artifact is executed on. Not "cuda" the torch
    # device: nothing in this plugin allocates a torch tensor on a device.
    device_type: str = "cuda"
    dispatch_key: str = "CPU"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
    simple_compile_backend: str = "eager"
    supported_quantization: list = []

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return f"tlaloc-pjrt-{cls.device_type}:{device_id}"

    @classmethod
    def is_async_output_supported(cls, enforce_eager) -> bool:
        # One step in, one step out. Async output staging is a scheduler
        # optimisation that assumes the runner can be ahead of the sampler;
        # this runner samples on the host inside the step.
        return False

    @classmethod
    def inference_mode(cls):
        # vLLM wraps execution in torch.inference_mode() on torch backends.
        # There is no torch in this execution path at all, so the context
        # manager is a no-op — stated, rather than inherited by accident.
        import contextlib

        return contextlib.nullcontext()

    @classmethod
    def supports_v1(cls, model_config) -> bool:
        return True

    @classmethod
    def artifact_path(cls) -> str:
        path = os.environ.get(ARTIFACT_ENV_VAR)
        if not path:
            raise ValueError(
                f"{cls.device_name}: ${ARTIFACT_ENV_VAR} is unset. A Tlaloc deployment's "
                f"unit is the serving-artifact DIRECTORY written by "
                f"io.tlaloc.maestro.serving.ServingArtifactWriter, not a checkpoint path; "
                f"vLLM's --model names the tokenizer/config, this names the compiled "
                f"programs."
            )
        return path

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        """vLLM's one hook for a backend to impose its truths on the config.

        Names the worker class, and refuses — by name — the two settings a
        compiled artifact cannot honour.
        """
        from .runner import TlalocModelRunner  # local: keeps discovery cheap

        runner = TlalocModelRunner(cls.artifact_path(), platform=cls.device_type)

        cache = getattr(vllm_config, "cache_config", None)
        if cache is not None:
            if getattr(cache, "block_size", None) not in (None, runner.block_size):
                raise ValueError(
                    f"--block-size {cache.block_size} disagrees with the serving "
                    f"artifact's compiled blockSize {runner.block_size}; the KV page "
                    f"layout is baked into the exported programs, so this is a "
                    f"different artifact and not a different flag"
                )
            cache.block_size = runner.block_size
            cache.num_gpu_blocks_override = runner.num_gpu_blocks

        model = getattr(vllm_config, "model_config", None)
        if model is not None:
            max_len = getattr(model, "max_model_len", None)
            if max_len is not None and max_len > runner.max_context:
                raise ValueError(
                    f"--max-model-len {max_len} exceeds the artifact's top context "
                    f"bucket {runner.max_context}; clamping it here would silently "
                    f"truncate a user's history, and extending it is a re-export"
                )

        sched = getattr(vllm_config, "scheduler_config", None)
        if sched is not None:
            cap = getattr(sched, "max_num_seqs", None)
            if cap is not None and cap > runner.max_batch:
                raise ValueError(
                    f"--max-num-seqs {cap} exceeds the artifact's top batch bucket "
                    f"{runner.max_batch}; the scheduler must not form a batch no "
                    f"compiled entry can execute"
                )

        parallel = getattr(vllm_config, "parallel_config", None)
        if parallel is not None:
            if getattr(parallel, "world_size", 1) != 1:
                raise ValueError(
                    "tlaloc: multi-device serving is a named deferral (the artifact "
                    "carries one program per bucket for one device); world_size must "
                    "be 1"
                )
            parallel.worker_cls = "vllm_tlaloc.worker.TlalocWorker"

    @classmethod
    def get_attn_backend_cls(cls, *args, **kwargs) -> str:
        """There is no pluggable attention backend: attention is inside the
        compiled program (`PAGED_ATTENTION`, §0.4.465), chosen at export
        time. Refused by name so that a config asking for FlashAttention
        gets an answer instead of a backend that is quietly ignored."""
        raise NotImplementedError(
            "tlaloc: attention is compiled into the serving artifact's programs "
            "(OpKind.PAGED_ATTENTION); there is no runtime-selectable attention "
            "backend to name"
        )
