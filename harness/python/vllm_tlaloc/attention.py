"""§0.4.491 — Phase H3c-4a: the attention backend CLASS, which vLLM asks for
unconditionally and whose `forward` is never reached.

This is the third and last module that imports vLLM (`platform.py` and
`worker.py` are the other two). Everything under it that can be checked
without vLLM installed is in `kv_layout.py`.

## Why this file exists at all

Attention is INSIDE the compiled artifact. `OpKind.PAGED_ATTENTION`
(§0.4.465) is an op in the exported StableHLO program, chosen at export
time, and `TlalocWorker.execute_model` runs that program. There is no
runtime-selectable attention kernel here, so from §0.4.470 until this slice
`TlalocPlatform.get_attn_backend_cls` refused by name.

The refusal was right about Tlaloc and wrong about vLLM. vLLM 0.29.0's v1
engine core calls that classmethod **unconditionally** while building the
model runner (`vllm/v1/attention/selector.py:202`), so it is not an
optional hook a backend may decline, and §0.4.480 died there with a real
artifact in hand. What vLLM wants is not a kernel: it is an object to ask
shape and capability questions of. That object is what this file is.

## What vLLM 0.29.0 actually asks for — checked, not remembered

The audit's H3c-4 contract was written as "a backend class whose
`get_kv_cache_shape` agrees with the manifest". Read against the installed
vLLM, **`AttentionBackend` in 0.29.0 has no `get_kv_cache_shape`**. The KV
page shape moved onto `AttentionSpec` + `KVCacheLayout`
(`compute_layer_kv_cache_shape_bytes`, RFC #42082), and what the BACKEND
publishes instead is a set of capability predicates: supported head sizes,
supported kernel block sizes, supported dtypes, preferred layouts. The
contract's intent is unchanged and its spelling is not, so:

* every predicate below is derived from the manifest, at call time —
  nothing about the pool is a literal in this file;
* `get_kv_cache_shape` is implemented anyway, in the classic
  `(2, num_blocks, block_size, num_kv_heads, head_size)` spelling the
  contract meant, and DOCUMENTED as a compatibility surface vLLM 0.29.0
  does not call. It is not dead code: it is the both-ways pin, and the
  certification drives it in both directions against a real manifest.

`get_attn_backend_cls` returns a **dotted string**, not a class — vLLM runs
it through `resolve_obj_by_qualname`. That is why this class is a
module-level name that reads the manifest lazily, and not a class built per
artifact: a dynamically created class has no qualname to resolve.

## The forward that is never reached

`TlalocAttentionImpl.forward` raises `TlalocAttentionNotReached`. That is
not a stub pretending to be a kernel; it is the one honest thing a
never-reached method can do, and its message names WHY control arriving
there means vLLM took a path this plugin does not implement (a torch
`Attention` layer got built and called, which means a torch model got
built, which means execution did not go through `TlalocWorker`). REJECTED:
returning zeros, or silently passing `query` through — both make a
structural gap look like a numerical one, and this arc has refused that
trade since §0.4.465.
"""

from __future__ import annotations

from . import ARTIFACT_ENV_VAR
from . import kv_layout

try:  # pragma: no cover - exercised only where vLLM is installed
    # vLLM FIRST, then torch. The order is load-bearing, and the unit lane
    # found it: this module's torch need is downstream of vLLM's, and the
    # oracle venv has torch but no vLLM — so importing torch first would
    # leave it resident in a process that then correctly refused to load
    # this module, which is exactly the framework leak
    # `test_importing_the_loader_pulls_in_no_framework` exists to catch.
    from vllm.v1.attention.backend import AttentionBackend, AttentionImpl, AttentionMetadataBuilder
    from vllm.v1.kv_cache_layout import KVCacheLayout

    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "vllm_tlaloc.attention requires vLLM to be installed; it subclasses "
        "vllm.v1.attention.backend.AttentionBackend. The layout arithmetic it "
        "answers with is in vllm_tlaloc.kv_layout, which imports without vLLM "
        "on purpose."
    ) from e


#: Manifest dtype names to torch dtypes. Only the ones a Tlaloc graph
#: boundary actually carries; an unknown one is refused rather than mapped
#: to a plausible neighbour.
_TORCH_DTYPE = {
    "f32": torch.float32,
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f64": torch.float64,
}

#: Manifest dtype names to vLLM `CacheDType` strings, for the KV pool.
_CACHE_DTYPE = {"f32": "float32", "bf16": "bfloat16", "f16": "float16"}


def torch_kv_dtype(model: dict):
    """§0.4.492 — the torch dtype of the artifact's KV pool, refused by name
    if the manifest states one this plugin has no mapping for.

    It lives HERE rather than in `kv_layout` because it is the one KV fact
    that cannot be answered without torch, and `kv_layout` is the module
    that stays importable with neither torch nor vLLM present. The worker
    needs it to build vLLM's `FullAttentionSpec` (see
    `TlalocWorker.get_kv_cache_spec`), and going through this function
    rather than re-spelling the table is what stops the spec's dtype and
    `supports_kv_cache_dtype`'s from drifting apart.
    """
    name = model["kvDtype"]
    dtype = _TORCH_DTYPE.get(name)
    if dtype is None:
        raise ValueError(
            f"tlaloc: the artifact's KV pool dtype '{name}' has no torch dtype here; "
            f"vLLM's KVCacheSpec needs a real torch.dtype and guessing one would size "
            f"the block manager's pages off a dtype the compiled program does not use"
        )
    return dtype


class TlalocAttentionNotReached(NotImplementedError):
    """Control reached an attention method that a Tlaloc deployment has no
    path to.

    Raised by name so that the failure says which assumption broke. Every
    method that raises it is a method vLLM only calls when it has built a
    torch attention layer — and a Tlaloc deployment never builds one,
    because `TlalocWorker.execute_model` runs a compiled program whose
    attention is an op inside it.
    """


def _model() -> dict:
    """The manifest's model block, read at call time.

    Not cached. vLLM asks these questions a handful of times during engine
    startup, the file is small JSON, and a cache keyed on an env var that a
    test process changes between cases is a cache that answers the previous
    test's question.
    """
    from .platform import TlalocPlatform

    return kv_layout.read_model(TlalocPlatform.artifact_path())


class TlalocAttentionImpl(AttentionImpl):
    """The impl vLLM would call if it ever ran attention itself. It does not."""

    def __init__(self, *args, **kwargs):
        raise TlalocAttentionNotReached(
            "tlaloc: vLLM constructed an attention IMPL. Attention in a Tlaloc "
            "deployment is OpKind.PAGED_ATTENTION inside the exported StableHLO "
            "program (§0.4.465), executed by TlalocWorker.execute_model; there is no "
            "torch attention layer to build. Reaching this constructor means vLLM "
            "built a torch model, which is a path vllm_tlaloc does not implement"
        )

    def forward(self, *args, **kwargs):  # pragma: no cover - unreachable by construction
        raise TlalocAttentionNotReached(
            "tlaloc: attention forward() was called. Attention lives inside the "
            "compiled artifact and is never a runtime call here; reaching this method "
            "means vLLM took a path the plugin does not implement"
        )


class TlalocAttentionMetadataBuilder(AttentionMetadataBuilder):
    """The metadata builder, for the same reason and with the same answer.

    vLLM builds per-batch attention metadata for the kernel that will
    consume it. The block tables and slot mapping a Tlaloc step needs are
    built by `vllm_tlaloc.batching` from the scheduler output and handed to
    the compiled program as operands; a second, torch-shaped copy of the
    same bookkeeping would be a second source of truth for the slot
    arithmetic H1b compiled.
    """

    def __init__(self, *args, **kwargs):
        raise TlalocAttentionNotReached(
            "tlaloc: vLLM asked for an attention METADATA BUILDER. Block tables and "
            "slot mapping for a Tlaloc step are built in vllm_tlaloc.batching and "
            "passed as operands of the compiled program; building them again in "
            "torch would be a second source of truth for H1b's slot arithmetic"
        )

    def build(self, *args, **kwargs):  # pragma: no cover - unreachable by construction
        raise TlalocAttentionNotReached("tlaloc: attention metadata build() was called")


class TlalocAttentionBackend(AttentionBackend):
    """What `get_attn_backend_cls` resolves to. Every fact it publishes is
    read out of the serving artifact's manifest."""

    # The compiled program writes its own KV: KV_CACHE_WRITE (§0.4.466) is
    # an op in the same graph as PAGED_ATTENTION. Stated rather than
    # inherited, because the inherited default happens to be right and a
    # default that is right by luck is a default that changes silently.
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "TLALOC"

    @staticmethod
    def get_impl_cls():
        return TlalocAttentionImpl

    @staticmethod
    def get_builder_cls():
        return TlalocAttentionMetadataBuilder

    # --- the manifest-derived facts --------------------------------------

    @classmethod
    def get_supported_head_sizes(cls) -> list:
        """Exactly the artifact's `headDim`. Not a list of head sizes this
        code could handle — a list of the one head size that is compiled."""
        return [int(_model()["headDim"])]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list:
        """Exactly the artifact's `blockSize`, and NOT `MultipleOf(...)`.

        vLLM's default is `MultipleOf(1)` — any block size — and every
        torch backend narrows it to what its kernel can index. Here the
        page layout is baked into the exported program, so the honest
        answer is a single fixed size. `supports_block_size` then refuses
        a mismatch on vLLM's own terms, which is the same refusal
        `check_and_update_config` makes about `--block-size`, arriving by
        a second route.
        """
        return [int(_model()["blockSize"])]

    @classmethod
    def supports_block_size(cls, block_size) -> bool:
        """EQUALITY, not divisibility — and this override is a finding.

        vLLM's inherited `supports_block_size` accepts any block size that
        is a MULTIPLE of a supported one, because 0.29.0 can subdivide a
        manager block into several kernel blocks (`group_kernel_blocks`,
        `compute_layer_kv_cache_shape_bytes(..., kernel_block_size)`). So
        publishing "kernel block size 2" and stopping there declares
        `--block-size 16` supported: eight kernel blocks per manager block.

        A Tlaloc artifact cannot do that. `blockSize` is baked into the
        exported program's slot arithmetic (H1b's `KV_CACHE_WRITE` scatter
        and H1a's gather both index it), and there is no splitting pass
        between vLLM's manager blocks and the compiled pool. Inheriting the
        default would have let a scheduler page memory the program has no
        slots for, which is the same class of error
        `determine_available_memory` refuses to make by profiling.
        """
        if block_size is None:
            return True
        return block_size == int(_model()["blockSize"])

    @classmethod
    def supports_dtype(cls, dtype) -> bool:
        """The activation dtype the graph was exported at, and no other."""
        return dtype is _TORCH_DTYPE.get(_model()["dtype"])

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool:
        """`auto` (vLLM's "match the model"), or the manifest's KV dtype
        spelled vLLM's way. Under `kvQuant` the pool holds integer CODES
        (§0.4.472) and vLLM's fp8 KV path is not that, so a quantized
        artifact accepts only `auto` — the codes' dtype is the graph
        boundary's business and not a cache-dtype flag."""
        if kv_cache_dtype is None or kv_cache_dtype == "auto":
            return True
        model = _model()
        if model.get("kvQuant") is not None:
            return False
        return kv_cache_dtype == _CACHE_DTYPE.get(model["kvDtype"])

    @classmethod
    def supported_kv_cache_layouts(cls):
        """The layout `kvPoolAxisOrder` spells, resolved against vLLM's enum."""
        name = kv_layout.kv_cache_layout_name(_model())
        try:
            return (KVCacheLayout[name],)
        except KeyError as e:  # pragma: no cover - a manifest kv_layout accepted
            raise kv_layout.KvLayoutError(
                f"the artifact's kvPoolAxisOrder spells KV layout '{name}', which "
                f"vLLM {getattr(KVCacheLayout, '__module__', '')} does not define"
            ) from e

    # Everything below is a capability this deployment does not have. Each
    # one is the inherited default, restated so that the class says what it
    # is rather than what it forgot to override.
    @classmethod
    def is_mla(cls) -> bool:
        return False

    @classmethod
    def is_sparse(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    # --- the contract's own spelling, kept and labelled ------------------

    @classmethod
    def get_kv_cache_shape(
        cls,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        *,
        cache_dtype_str: str = "auto",
    ) -> tuple:
        """`(2, num_blocks, block_size, num_kv_heads, head_size)`.

        **vLLM 0.29.0 does not call this.** It is the H3c-4 contract's own
        spelling — the classic vLLM backend surface, which 0.29.0 replaced
        with `AttentionSpec` + `compute_layer_kv_cache_shape_bytes` — and
        it is kept for two reasons. First, it is the shape a Tlaloc pool
        LITERALLY has (separate K and V tensors over `kvPoolDims`), so it
        is the tuple a reader comparing the artifact to vLLM's accounting
        wants to see. Second, it is checkable in both directions against a
        real manifest, which is what this slice can certify without a
        server.

        Every argument is checked against the manifest and a disagreement
        is refused BY NAME. A compiled pool cannot be reshaped by a caller,
        so a method that quietly returned the caller's numbers would be
        agreeing with whoever asked last.
        """
        model = _model()
        expected = kv_layout.kv_cache_shape(model)
        names = kv_layout.kv_cache_axis_names(model)
        asked = (kv_layout.KV_KINDS, num_blocks, block_size, num_kv_heads, head_size)
        for name, want, got in zip(names, expected, asked):
            if want != got:
                raise ValueError(
                    f"tlaloc: KV cache axis '{name}' asked for {got} and the serving "
                    f"artifact's compiled pool is {want} (kvPoolDims "
                    f"{list(model['kvPoolDims'])}). The pool is a compiled shape; a "
                    f"backend that returned the caller's number here would be agreeing "
                    f"with whoever asked last"
                )
        return expected

    @classmethod
    def page_size_bytes(cls) -> int:
        """Bytes of one block, one layer, K and V — the number vLLM's block
        manager spends, and the one both readings of the layout agree on."""
        return kv_layout.check_page_bytes_agree(_model())

    @classmethod
    def describe(cls) -> dict:
        """Everything this class publishes, in one dict.

        Exists so the live lane can hand the whole surface across a process
        boundary in one call and the Kotlin side can pin it field by field,
        instead of the certification growing one accessor per fact.
        """
        model = _model()
        return {
            "name": cls.get_name(),
            "artifactEnvVar": ARTIFACT_ENV_VAR,
            "dtype": model["dtype"],
            "kvDtype": model["kvDtype"],
            "numLayers": int(model["numLayers"]),
            "kvCacheShape": list(cls.get_kv_cache_shape(*kv_layout.kv_cache_shape(model)[1:])),
            "kvCacheAxisNames": list(kv_layout.kv_cache_axis_names(model)),
            "vllmLogicalPageShapeBytes": list(kv_layout.vllm_logical_page_shape_bytes(model)),
            "pageSizeBytes": cls.page_size_bytes(),
            "poolBytes": kv_layout.pool_bytes(model),
            "kvCacheLayout": kv_layout.kv_cache_layout_name(model),
            "supportedHeadSizes": list(cls.get_supported_head_sizes()),
            "supportedKernelBlockSizes": [int(b) for b in cls.get_supported_kernel_block_sizes()],
            "forwardIncludesKvCacheUpdate": bool(cls.forward_includes_kv_cache_update),
            "isMla": cls.is_mla(),
            "isSparse": cls.is_sparse(),
        }
