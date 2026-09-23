"""§0.4.491 — Phase H3c-4a: the KV page layout, read out of the manifest BY
NAME, in a module that does not import vLLM.

This is the arithmetic `attention.py` needs and the arithmetic the
certification must be able to run. `attention.py` imports vLLM (it
subclasses `vllm.v1.attention.backend.AttentionBackend`), so everything
under it that can be checked without 186 packages present lives here — the
same split, for the same reason, as `paging` / `batching` / `runner` under
`platform` / `worker`.

## The one fact this module exists to keep honest

The artifact STATES its pool layout:

    "kvPoolAxisOrder": ["numBlocks", "blockSize", "numKvHeads", "headDim"],
    "kvPoolDims":      [numBlocks,   blockSize,   numKvHeads,   headDim]

`ServingModelShape` writes those two fields side by side precisely so that a
consumer never has to infer which axis is which from four integers — an
inference that is right until the day two of them are equal. So every
function here indexes `kvPoolDims` **through** `kvPoolAxisOrder`, and
refuses an axis order it does not recognise rather than assuming position.

## The divergence from vLLM, named rather than smoothed over

vLLM 0.29.0's logical per-layer page is `[B, H, N, C]` with
`C = (head_size + head_size_v) * itemsize` — that is, K and V INTERLEAVED
into one content cell (`AttentionSpec.state_content_size_bytes`). Tlaloc's
pools are **separate K and V tensors per layer**, which is the classic
vLLM spelling `(2, num_blocks, block_size, num_kv_heads, head_size)` — the
`2` this module puts in front of `kvPoolDims`.

The two do not agree axis for axis and they are not made to. What they DO
agree on is the number vLLM's block manager actually spends:
`page_size_bytes` — one block, one layer, K and V — is identical under both
readings, because interleaving K and V into a cell and storing them as two
tensors move the same bytes. `KV_PAGE_BYTES_AGREE` is that claim, and
`vllm_tlaloc_test.py` pins it here while the live lane pins it against
vLLM's own `compute_layer_kv_cache_shape_bytes`.
"""

from __future__ import annotations

import json
from pathlib import Path

# The manifest file inside a serving-artifact directory. Restated here
# rather than imported from `tlaloc_serve` because this module is loaded
# inside vLLM's process, where the serving loader is not on the path.
MANIFEST_NAME = "tlaloc-serving.json"

# The axis order H1a compiled and `ServingModelShape` writes. A manifest
# that states a different one is a different artifact, and this module says
# so instead of reading the dims positionally anyway.
KV_POOL_AXIS_ORDER = ("numBlocks", "blockSize", "numKvHeads", "headDim")

# Logical axis letters, in vLLM's `[L, B, H, N, C]` vocabulary (RFC #42082),
# for each axis the artifact names. `blockSize` is N (states within a page)
# and `headDim` is C (the content axis) — the mapping is stated because the
# two vocabularies use "block" for different things.
_AXIS_LETTER = {
    "numBlocks": "B",
    "blockSize": "N",
    "numKvHeads": "H",
    "headDim": "C",
}

# Element sizes for the dtypes a KV pool can ride at. Kept here, in the
# vLLM-free half, because `worker.determine_available_memory` and the
# backend class must not each keep their own table: two tables is how a
# reported pool stops being the pool that exists.
ELEMENT_SIZE_BYTES = {"f32": 4, "bf16": 2, "f16": 2, "f64": 8, "i32": 4, "i64": 8, "i8": 1}

# K and V. The leading axis of `kv_cache_shape`, and the factor in
# `page_size_bytes`. One name, so the two cannot drift apart.
KV_KINDS = 2


class KvLayoutError(ValueError):
    """A manifest whose pool layout this module will not guess at.

    Raised BY NAME rather than falling back to a positional read: the whole
    point of `kvPoolAxisOrder` being in the artifact is that nobody has to
    guess, and a consumer that guesses anyway when the field surprises it
    has thrown away the field.
    """


def read_model(artifact_dir) -> dict:
    """The manifest's `model` block, from a serving-artifact DIRECTORY.

    Deliberately not `tlaloc_serve.ServingArtifact`: the backend class is
    asked for its shape facts during vLLM's engine-core startup, long
    before any program is compiled, and loading the whole artifact (bodies,
    content-address verification) to answer "how big is a page" would make
    a question about JSON cost a hash of every program.
    """
    path = Path(artifact_dir) / MANIFEST_NAME
    if not path.is_file():
        raise KvLayoutError(
            f"no {MANIFEST_NAME} in {artifact_dir}: a Tlaloc deployment's unit is the "
            f"serving-artifact DIRECTORY written by ServingArtifactWriter, and this "
            f"path does not hold one"
        )
    with path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise KvLayoutError(f"{path}: the manifest has no 'model' block")
    return model


def _axes(model: dict) -> dict:
    """`{axis name: extent}`, checked. The one place dims meet names."""
    order = model.get("kvPoolAxisOrder")
    dims = model.get("kvPoolDims")
    if not isinstance(order, list) or not isinstance(dims, list):
        raise KvLayoutError(
            "the manifest states no kvPoolAxisOrder/kvPoolDims; this loader will not "
            "reconstruct a pool layout from the scalar fields, because the axis order "
            "is exactly the thing they do not carry"
        )
    if len(order) != len(dims):
        raise KvLayoutError(
            f"kvPoolAxisOrder has {len(order)} axes and kvPoolDims has {len(dims)} "
            f"extents; they are written as a pair and must be read as one"
        )
    if tuple(order) != KV_POOL_AXIS_ORDER:
        raise KvLayoutError(
            f"kvPoolAxisOrder is {list(order)}; this plugin knows "
            f"{list(KV_POOL_AXIS_ORDER)} (the PAGED_ATTENTION operand order). "
            f"A pool in a different axis order is a different compiled artifact, and "
            f"reading its dims positionally would hand vLLM a page shape that is "
            f"right only when two of the four extents happen to be equal"
        )
    for name, extent in zip(order, dims):
        if not isinstance(extent, int) or extent <= 0:
            raise KvLayoutError(f"kvPoolDims axis '{name}' is {extent!r}, not a positive extent")
    return dict(zip(order, dims))


def element_size(dtype: str) -> int:
    """Bytes per element of a KV-pool dtype, refusing an unknown one."""
    size = ELEMENT_SIZE_BYTES.get(dtype)
    if size is None:
        raise KvLayoutError(
            f"KV dtype '{dtype}' has no element size here; a consumer that guesses this "
            f"number reports a pool that does not exist"
        )
    return size


def kv_cache_shape(model: dict) -> tuple:
    """`(2, numBlocks, blockSize, numKvHeads, headDim)` — one layer's pool.

    The leading `2` is K and V, which Tlaloc stores as two tensors rather
    than as one interleaved content cell (see the module docstring). The
    four that follow are `kvPoolDims` in the order the artifact states.
    """
    axes = _axes(model)
    return (KV_KINDS,) + tuple(axes[name] for name in KV_POOL_AXIS_ORDER)


def kv_cache_axis_names(model: dict) -> tuple:
    """The names of `kv_cache_shape`'s axes, so a caller comparing two
    tuples can say WHICH axis disagreed instead of which index."""
    _axes(model)
    return ("kv",) + KV_POOL_AXIS_ORDER


def kv_cache_layout_name(model: dict) -> str:
    """vLLM's `KVCacheLayout` name for this pool, derived from the axis
    order rather than chosen.

    `L` is outermost by construction — Tlaloc's pools are separate tensors
    per layer, so a layer's bytes are contiguous whatever the rest does —
    and the remaining four letters are `kvPoolAxisOrder` translated through
    `_AXIS_LETTER`. For H1a's order that spells **LBNHC**, which is a real
    member of vLLM's enum and the layer-compact layout its own kernel-block
    splitting recommends.
    """
    axes = _axes(model)
    return "L" + "".join(_AXIS_LETTER[name] for name in axes)


def page_size_bytes(model: dict) -> int:
    """Bytes of ONE block, ONE layer, K and V together.

    This is the number vLLM's block manager spends, and it is the number
    both readings of the layout agree on — see `KV_PAGE_BYTES_AGREE`.
    """
    axes = _axes(model)
    elem = element_size(model["kvDtype"])
    return (
        KV_KINDS
        * axes["blockSize"]
        * axes["numKvHeads"]
        * axes["headDim"]
        * elem
    )


def pool_bytes(model: dict) -> int:
    """Bytes of the WHOLE compiled KV pool: every block, every layer, K and V.

    What `TlalocWorker.determine_available_memory` reports. It is the pool
    that exists, not a memory profile — a profile more generous than the
    truth is a silent out-of-bounds write.
    """
    axes = _axes(model)
    return page_size_bytes(model) * axes["numBlocks"] * int(model["numLayers"])


def vllm_logical_page_shape_bytes(model: dict) -> tuple:
    """vLLM 0.29.0's `(B, H, N, C_bytes)` reading of the same page.

    Computed here so the certification can compare it against vLLM's own
    `compute_layer_kv_cache_shape_bytes` and see the two agree — and see
    WHERE they differ from `kv_cache_shape`, which is the axis order and
    not the byte count.
    """
    axes = _axes(model)
    elem = element_size(model["kvDtype"])
    return (
        axes["numBlocks"],
        axes["numKvHeads"],
        axes["blockSize"],
        KV_KINDS * axes["headDim"] * elem,
    )


def check_page_bytes_agree(model: dict) -> int:
    """The claim `KV_PAGE_BYTES_AGREE` names, checked and returned.

    `prod(vllm_logical_page_shape_bytes[1:]) == page_size_bytes` — the two
    layouts move the same bytes per block even though they order them
    differently. If this ever stops holding, a Tlaloc artifact's pool and
    vLLM's accounting of it have diverged in SIZE and not merely in shape,
    which is the difference between a cosmetic mismatch and blocks handed
    out that the program has no slots for.
    """
    b, h, n, c = vllm_logical_page_shape_bytes(model)
    logical = h * n * c
    ours = page_size_bytes(model)
    if logical != ours:
        raise KvLayoutError(
            f"page bytes disagree: vLLM's [B,H,N,C] reading says {logical} and the "
            f"artifact's separate K/V pools say {ours}. These are two readings of ONE "
            f"pool and they must at least spend the same bytes"
        )
    return ours


#: The name the audit and the tests use for the invariant above.
KV_PAGE_BYTES_AGREE = "vLLM's [B,H,N,C] page and Tlaloc's separate K/V pools spend equal bytes"
