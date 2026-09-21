"""§0.4.469 — Phase H3a: the Python side of a Tlaloc serving artifact.

This module is the other half of `docs/INFERENCE_SERVING_AUDIT.md`'s
central claim: **no JVM in the serving path**. Kotlin AOT-compiles a
family of decode graphs into an artifact DIRECTORY (see
`io.tlaloc.maestro.serving.ServingArtifactWriter`); this file reads that
directory, hands the StableHLO to jaxlib/PJRT, and runs decode steps. It
imports nothing of Tlaloc's, starts no JVM, and never calls back.

The shape follows vLLM's `tpu-inference` precedent (audit §1): the
serving loop's scheduler, continuous batching and paged-KV bookkeeping
stay in Python; what this replaces is EXECUTION.

    art = ServingArtifact.load("build/serving-artifact")
    logits, pools = art.run_decode(
        token_ids=[3, 1, 4], positions=[0, 0, 0],
        block_tables=[[2], [5], [3]], seq_lens=[1, 1, 1],
        slot_mapping=[4, 10, 6], kv_pools=pools,
    )

Design notes, and what they rejected:

* **Bucket selection is read from the manifest, not re-derived.** The
  artifact carries its ladder (H1c named this as the manifest field it
  was deferring), so the rounding rule lives in one place. REJECTED:
  hard-coding powers of two here — two implementations of one policy is
  how a request gets padded to a shape nobody compiled.
* **Padding constants are duplicated from `DecodePadding`, deliberately
  and loudly.** They are a WIRE convention between two processes, so
  they are restated here with the reason attached (`seq_lens` padding is
  1 and never 0: a zero-length row makes the paged-attention softmax
  0/0). `check_padding_constants()` exists so a test can pin the two
  copies together rather than trusting a comment.
* **The body is compiled, never edited.** The §0.4.325 spike regex-renamed
  the entry function to `@main` before compiling; the exporter names it
  `main` instead and the manifest states it, so a loader does not rewrite
  a program it was asked to run.
* **Executables are cached by the manifest's `cacheKey`** — the same
  string `DecodeGraphSpec.executableCacheKey` builds, so the JVM-side and
  serving-side caches agree by construction.

Named deferrals: buffer DONATION (the manifest carries `donationPairs`;
wiring them into `CompileOptions` is the next slice's measurable win),
staged weights (v1 bodies carry them as constants), sampling (host-side,
outside this module, per the audit), and multi-device execution.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# GB10 is unified-memory: JAX's default 75% preallocation would claim ~90 GB
# of system RAM. Must be set before `import jax` (§0.4.333, the reboot).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

SCHEMA_VERSION = "tlaloc-serving-v1"

# --- the wire-level padding convention (mirror of DecodePadding) --------
PADDING_TOKEN_ID = 0
PADDING_POSITION = 0
PADDING_BLOCK = 0
PADDING_SEQ_LEN = 1  # NOT 0: a zero-length row is a 0/0 softmax => NaN
PADDING_SLOT = -1  # the KV_CACHE_WRITE convention: the write is dropped


def check_padding_constants(expected: dict) -> None:
    """Refuse by name when the Kotlin side's constants and ours disagree.

    Called by the certification with the values read out of
    `DecodePadding`, so the duplication above is a pinned mirror and not
    a hopeful one.
    """
    ours = {
        "PADDING_TOKEN_ID": PADDING_TOKEN_ID,
        "PADDING_POSITION": PADDING_POSITION,
        "PADDING_BLOCK": PADDING_BLOCK,
        "PADDING_SEQ_LEN": PADDING_SEQ_LEN,
        "PADDING_SLOT": PADDING_SLOT,
    }
    for k, v in expected.items():
        if k not in ours:
            raise ValueError(f"unknown padding constant '{k}'")
        if ours[k] != v:
            raise ValueError(
                f"padding constant '{k}': the artifact's producer says {v}, "
                f"tlaloc_serve.py says {ours[k]} — the two processes would "
                f"pad a batch differently"
            )


# Keys are `io.tlaloc.core.DType.name` VERBATIM (lower case — `f32`, not
# `F32`). The manifest writes that name and this reads it; folding case
# here would let an artifact and a loader disagree about a dtype spelling
# and still agree about the dtype, which is the sort of latitude that hides
# a real mismatch later.
_DTYPE_TO_NUMPY = {
    "f32": "float32",
    "f64": "float64",
    "i32": "int32",
    "i64": "int64",
    "bf16": "bfloat16",
    "bool": "bool",
}


def _numpy_dtype(name: str):
    import numpy as np

    if name == "bf16":
        import ml_dtypes

        return ml_dtypes.bfloat16
    try:
        return np.dtype(_DTYPE_TO_NUMPY[name])
    except KeyError:
        raise ValueError(
            f"Tlaloc dtype '{name}' has no numpy mapping in tlaloc_serve.py; "
            f"known: {sorted(_DTYPE_TO_NUMPY)}"
        ) from None


@dataclass(frozen=True)
class Slot:
    name: str
    role: str
    dtype: str
    dims: tuple

    @staticmethod
    def parse(o: dict) -> "Slot":
        t = o["type"]
        return Slot(o["name"], o["role"], t["dtype"], tuple(t["dims"]))


@dataclass(frozen=True)
class Entry:
    kind: str
    batch: int
    context: int
    tokens_per_seq: int
    max_blocks_per_seq: int
    cache_key: str
    entry_point: str
    body_path: str
    body_hash: str
    program_path: str
    inputs: tuple
    outputs: tuple
    donation_pairs: tuple

    @staticmethod
    def parse(o: dict) -> "Entry":
        return Entry(
            kind=o["kind"],
            batch=o["batch"],
            context=o["context"],
            tokens_per_seq=o["tokensPerSeq"],
            max_blocks_per_seq=o["maxBlocksPerSeq"],
            cache_key=o["cacheKey"],
            entry_point=o["entryPoint"],
            body_path=o["bodyPath"],
            body_hash=o["bodyHash"],
            program_path=o["programPath"],
            inputs=tuple(Slot.parse(s) for s in o["inputs"]),
            outputs=tuple(Slot.parse(s) for s in o["outputs"]),
            donation_pairs=tuple((p[0], p[1]) for p in o["donationPairs"]),
        )

    def role_indices(self, role: str, among) -> list:
        return [i for i, s in enumerate(among) if s.role == role]


class ServingArtifact:
    """A loaded Tlaloc serving artifact: manifest, bodies, executables."""

    def __init__(self, root: Path, manifest: dict, platform: str = "cuda"):
        if manifest.get("schemaVersion") != SCHEMA_VERSION:
            raise ValueError(
                f"{root}: schemaVersion {manifest.get('schemaVersion')!r} is not "
                f"{SCHEMA_VERSION!r}; refusing an artifact of unknown shape"
            )
        self.root = root
        self.manifest = manifest
        self.platform = platform
        self.model_name = manifest["modelName"]
        self.model_hash = manifest["modelHash"]
        self.model = manifest["model"]
        ladder = manifest["bucketLadder"]
        self.block_size = ladder["blockSize"]
        self.batch_ladder = list(ladder["batch"])
        self.context_ladder = list(ladder["context"])
        self.weights = manifest["weights"]
        self.entries = [Entry.parse(e) for e in manifest["entries"]]
        self._exe_cache: dict = {}
        self._backend = None
        self._device_list = None

    # --- loading -------------------------------------------------------

    @classmethod
    def load(cls, path, platform: str = "cuda") -> "ServingArtifact":
        root = Path(path)
        mf = root / "tlaloc-serving.json"
        if not mf.exists():
            raise FileNotFoundError(f"{root} is not a Tlaloc serving artifact: no {mf.name}")
        return cls(root, json.loads(mf.read_text()), platform=platform)

    def verify_bodies(self) -> None:
        """Content-address check: every body hashes to the name it is filed
        under. The artifact crossed a process boundary; this is the cheapest
        possible statement that it arrived intact."""
        import hashlib

        for e in self.entries:
            data = (self.root / e.body_path).read_bytes()
            got = hashlib.sha256(data).hexdigest()
            if got != e.body_hash:
                raise ValueError(
                    f"{e.body_path}: body hashes to {got} but the manifest "
                    f"(and the filename) say {e.body_hash}"
                )

    # --- bucket selection (H1c's policy, read off the ladder) -----------

    def select_bucket(self, batch: int, context: int) -> tuple:
        """Round (batch, context) UP to a compiled ladder point.

        Refuses over-cap by name rather than clamping: clamping a context
        request silently truncates a sequence's history, which is a wrong
        answer dressed as a slow one. Splitting is the scheduler's job and
        it can only do it if it is told."""
        if batch < 1 or context < 1:
            raise ValueError(f"select_bucket: batch={batch}, context={context} — both must be >= 1")
        b = next((x for x in self.batch_ladder if x >= batch), None)
        if b is None:
            raise ValueError(
                f"select_bucket: batch {batch} exceeds the artifact's maxBatch "
                f"{self.batch_ladder[-1]}; the scheduler must split the request"
            )
        c = next((x for x in self.context_ladder if x >= context), None)
        if c is None:
            raise ValueError(
                f"select_bucket: context {context} exceeds the artifact's maxContext "
                f"{self.context_ladder[-1]}; the scheduler must split the request"
            )
        return b, c

    def entry_for(self, kind: str, batch: int, context: int) -> Entry:
        for e in self.entries:
            if e.kind == kind and e.batch == batch and e.context == context:
                return e
        raise ValueError(
            f"no {kind} entry for (batch={batch}, context={context}); compiled: "
            + ", ".join(f"({e.batch},{e.context})" for e in self.entries if e.kind == kind)
        )

    # --- compilation ---------------------------------------------------

    def _ensure_backend(self):
        if self._backend is not None:
            return
        import jax
        import jax._src.xla_bridge as xb
        import jaxlib._jax as _jax

        backend = xb.backends()[self.platform]
        devs = backend.local_devices()
        if not devs:
            raise RuntimeError(f"PJRT backend '{self.platform}' has no local devices")
        self._backend = backend
        self._devices = devs
        self._device_list = _jax.DeviceList(tuple(devs[:1]))
        self._jax = jax
        self._jaxlib = _jax

    def compiled(self, entry: Entry):
        """Compile (once per `cacheKey`) and return a LoadedExecutable."""
        hit = self._exe_cache.get(entry.cache_key)
        if hit is not None:
            return hit
        self._ensure_backend()
        import jaxlib.mlir.ir as ir
        from jaxlib.mlir._mlir_libs._jax_mlir_ext import register_dialects
        import jaxlib.mlir.dialects.stablehlo as stablehlo

        text = (self.root / entry.body_path).read_text()
        reg = ir.DialectRegistry()
        register_dialects(reg)
        ctx = ir.Context()
        ctx.append_dialect_registry(reg)
        ctx.load_all_available_dialects()
        stablehlo.register_dialect(ctx)
        with ctx, ir.Location.unknown(ctx):
            module = ir.Module.parse(text)
        co = self._jaxlib.CompileOptions()
        exe = self._backend.compile_and_load(module, self._device_list, co)
        self._exe_cache[entry.cache_key] = exe
        return exe

    @property
    def compile_count(self) -> int:
        """How many distinct executables this process holds. The cache's
        observable: two calls at one bucket must not move it."""
        return len(self._exe_cache)

    # --- execution -----------------------------------------------------

    def run_decode(
        self,
        token_ids: Sequence[int],
        positions: Sequence[int],
        block_tables: Sequence[Sequence[int]],
        seq_lens: Sequence[int],
        slot_mapping: Sequence[int],
        kv_pools: Sequence[Any],
        context: int | None = None,
    ):
        """One decode step for `len(token_ids)` sequences.

        Returns `(logits, kv_pools)` with `logits` sliced back to the real
        rows — the padding rows the bucket added never leave this function.
        The pools come back WHOLE, because a pool is device state shared by
        every sequence and slicing it would be meaningless.
        """
        import numpy as np

        n = len(token_ids)
        if not (len(positions) == len(seq_lens) == len(block_tables) == n):
            raise ValueError(
                f"run_decode: ragged request — {n} tokens, {len(positions)} positions, "
                f"{len(seq_lens)} seq_lens, {len(block_tables)} block tables"
            )
        if len(slot_mapping) != n:
            raise ValueError(
                f"run_decode: slot_mapping has {len(slot_mapping)} entries for {n} "
                f"sequences (decode writes exactly one slot per sequence)"
            )
        want_context = context if context is not None else max(seq_lens)
        b, c = self.select_bucket(n, want_context)
        entry = self.entry_for("decode", b, c)

        mbs = entry.max_blocks_per_seq

        def pad_rows(rows, width, fill):
            out = np.full((b, width), fill, dtype=np.int32)
            for i, r in enumerate(rows):
                if len(r) > width:
                    raise ValueError(
                        f"run_decode: sequence {i} names {len(r)} blocks but the "
                        f"(batch={b}, context={c}) bucket's table is {width} wide"
                    )
                out[i, : len(r)] = r
            return out

        args_np = {
            "TOKEN_IDS": np.array(
                [list(token_ids) + [PADDING_TOKEN_ID] * (b - n)], dtype=np.int32
            ).reshape(b, entry.tokens_per_seq),
            "POSITIONS": np.array(
                [list(positions) + [PADDING_POSITION] * (b - n)], dtype=np.int32
            ).reshape(b, entry.tokens_per_seq),
            "BLOCK_TABLES": pad_rows(block_tables, mbs, PADDING_BLOCK),
            "SEQ_LENS": np.array(
                list(seq_lens) + [PADDING_SEQ_LEN] * (b - n), dtype=np.int32
            ),
            "SLOT_MAPPING": np.array(
                list(slot_mapping) + [PADDING_SLOT] * (b - n), dtype=np.int32
            ),
        }

        pools = list(kv_pools)
        expected_pools = sum(1 for s in entry.inputs if s.role == "KV_POOL_IN")
        if len(pools) != expected_pools:
            raise ValueError(
                f"run_decode: {len(pools)} KV pools supplied, the entry's signature "
                f"has {expected_pools} (two per layer, (key, value) order, layers ascending)"
            )

        self._ensure_backend()
        jnp = self._jax.numpy
        dev = self._devices[0]
        ordered = []
        pool_i = 0
        for slot in entry.inputs:
            if slot.role == "KV_POOL_IN":
                a = np.asarray(pools[pool_i], dtype=_numpy_dtype(slot.dtype))
                pool_i += 1
            else:
                a = args_np[slot.role]
            if tuple(a.shape) != slot.dims:
                raise ValueError(
                    f"run_decode: operand '{slot.name}' is {tuple(a.shape)} but the "
                    f"compiled entry declares {slot.dims}"
                )
            ordered.append(
                self._jax.device_put(jnp.asarray(a, dtype=_numpy_dtype(slot.dtype)), dev)
            )

        results = self.compiled(entry).execute(ordered)
        out = [np.asarray(r) for r in results]
        logits = out[0][:n]
        return logits, out[1:]

    def empty_pools(self, fill: float = 0.0):
        """Freshly-zeroed KV pools of the artifact's declared layout."""
        import numpy as np

        m = self.model
        dims = tuple(m["kvPoolDims"])
        dt = _numpy_dtype(m["kvDtype"])
        return [np.full(dims, fill, dtype=dt) for _ in range(2 * m["numLayers"])]
