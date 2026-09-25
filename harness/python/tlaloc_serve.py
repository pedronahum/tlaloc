"""§0.4.469 (H3a), rewired in §0.4.476 (H6b) — the Python side of a Tlaloc
serving artifact, executing through **PJRT bound by ctypes**.

WHAT CHANGED IN H6b, AND WHY IT IS THE POINT
============================================

H3a wrote this module against **jaxlib**: `jax._src.xla_bridge` for the
backend, `jaxlib.mlir` for the module, `jaxlib._jax.CompileOptions` for the
compile, numpy for every host array. That worked, and it certified — but it
meant the deployment claim was "no JVM", with a 500 MB CUDA-12 wheel family
standing quietly behind it.

H6a (`tlaloc_pjrt.py`) bound the PJRT C API with `ctypes` alone. This file
now runs on that binding, and the claim becomes the one the audit wanted:

    no JVM, no JAX, no torch, no numpy — just the PJRT plugin `.so`.

At module scope this file imports `json`, `os`, `hashlib`, `pathlib`,
`dataclasses`, `typing` and `tlaloc_pjrt`. Nothing else. That is checkable,
and `run_tlaloc_serve_check.py` checks it: the whole serving path runs under
a `sys.meta_path` guard that raises on `jax`, `jaxlib`, `torch` or `numpy`,
in the venv where jax *is* installed, and the guard is made to fire before
the run reports so a guard that never fired cannot be mistaken for a clean
run (§0.4.474's lesson, §0.4.475's pattern).

    art = ServingArtifact.load("build/serving-artifact")   # engine="ctypes"
    with art:
        logits, pools = art.run_decode(
            token_ids=[3, 1, 4], positions=[0, 0, 0],
            block_tables=[[2], [5], [3]], seq_lens=[1, 1, 1],
            slot_mapping=[4, 10, 6], kv_pools=art.empty_pools(),
        )

THE HOST WIRE FORMAT IS FLAT LISTS, DELIBERATELY
================================================

Without numpy there is no ndarray, so every host-side tensor here is a
**flat row-major sequence of Python numbers** and the shape comes from the
manifest — which is where it was authoritative anyway. `run_decode` returns
`logits` split into the real rows (`list[list[float]]`, the bucket's padding
rows never leave this function) and the KV pools WHOLE and flat, because a
pool is device state shared by every sequence and slicing it would be
meaningless. `unflatten()` is here for a caller that wants nesting;
`empty_pools()` hands back the same flat form it accepts.

REJECTED: carrying a tiny ndarray-alike so the surface looked like the old
one. A shim that is 5% of numpy is a thing every caller has to learn *and*
cannot trust; a flat list is a thing every Python caller already knows.

TWO ENGINES, AND THE HONEST REASON THERE ARE TWO
================================================

`engine="ctypes"` is **the serving engine**, and the default for every
platform that has a plugin file — see `default_engine_for`, which is the one
place the rule lives. It is what the deployment runs and what the claim above
is about.

`engine="jax"` is kept, and kept ONLY as an **oracle**: jaxlib ships no CPU
PJRT plugin `.so` — its CPU client is a C++ class inside the jaxlib
extension, not a loadable plugin — so the 1e-5 XLA-CPU *semantics* lane
(`ServingArtifactExportRunTest`, the tight floor that says the program MEANS
what the interpreter means) has no ctypes route on this machine. Deleting
the jax engine would have deleted a certified row to make a sentence
tidier. Its imports live INSIDE its methods, so selecting `engine="ctypes"`
imports none of it and the import guard is a true statement about which
engine ran, not a hopeful one.

The house rule is one engine per job. The jobs are different: one is the
deployment, one is the measurement apparatus. The day a CPU PJRT plugin
`.so` exists here (or the TPU VM's `libtpu.so` does, G2b), the ctypes engine
covers that lane unchanged and the jax engine goes.

WHAT IS STILL DEFERRED
======================

Buffer DONATION (the manifest carries `donationPairs`; neither engine wires
them into compile options yet — every step still round-trips whole pools
through the host, which is the next measurable win and the reason this is
a correctness artifact rather than a throughput one), sampling (host-side,
outside this module), multi-device execution, and bf16/int8 pools end to end
(the dtype mapping below handles them; nothing has exported one yet).

STAGED WEIGHTS ARRIVED IN 0.4.480 (H3c-3). An artifact may now carry a
`weights.table` — one raw little-endian file per WEIGHT operand, already in
math layout, written by `ServingArtifactWriter`. They are uploaded ONCE per
artifact through `tlaloc_pjrt.buffer_from_file`, which never creates a Python
number for them (1.1e9 Python floats is not a slow path, it is an impossible
one), and are held as live device buffers for the artifact's lifetime. What
is NOT done: the KV pools still round-trip to the host every step, so a
22-layer model pays that in Python list construction per token. That is the
donation item above, and it is now the dominant cost of a real decode.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import tlaloc_pjrt as P

SCHEMA_VERSION = "tlaloc-serving-v2"
# v1 has decode entries only; v2 adds prefill entries (a right-aligned chunk
# of tokens that returns the last token's logits). Both are read.
READABLE_SCHEMA_VERSIONS = ("tlaloc-serving-v1", SCHEMA_VERSION)

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


# ---------------------------------------------------------------------------
# Finding a PJRT plugin, which is the whole dependency story.
# ---------------------------------------------------------------------------

def find_pjrt_plugin(explicit: str | None = None, platform: str = "cuda") -> str:
    """Locate a PJRT plugin `.so` for `platform`. **Imports nothing** — it looks
    for a FILE.

    Delegates to `tlaloc_pjrt.find_plugin`, which searches the same places as
    the JVM's `PjrtBinaries`, in the same order:

      1. `explicit`, then `TLALOC_PJRT_PLUGIN_PATH` — the variable a deployment
         that ships its own plugin sets. For `tpu` the variable counts only
         when it names a tpu-shaped file.
      2. For cuda: `jax_plugins/*cuda*/*.so` in the site-packages of
         `$VIRTUAL_ENV`, `~/.local/venvs/*`, `~/.venv`, `~/venv`, `~/.local`,
         `/usr/local`, `/usr`, then this interpreter's own site-packages. A jax
         install is used as a place a file sits; nothing is imported.
      3. For tpu: the libtpu wheel's `libtpu/libtpu.so`, then `/lib/libtpu.so`
         and `/usr/lib/libtpu.so`.

    On failure the FileNotFoundError carries every place looked.
    """
    return P.find_plugin(platform, explicit)


# ---------------------------------------------------------------------------
# Dtypes. Keys are `io.tlaloc.core.DType.name` VERBATIM (lower case — `f32`,
# not `F32`). The manifest writes that name and this reads it; folding case
# here would let an artifact and a loader disagree about a dtype spelling and
# still agree about the dtype, which is the sort of latitude that hides a real
# mismatch later.
#
# This table is where H6b LIFTS the jax path's practical single-dtype squeeze:
# every slot is staged as its own PJRT buffer of its own declared type, so an
# I32 block table is a real device i32 buffer rather than something a
# framework had to be talked into. (The manifest always described per-slot
# dtypes — `ServingManifest`'s slots are exactly that — so this is honouring a
# schema that was already right, not inventing one.)
# ---------------------------------------------------------------------------

_STAGE = {
    "f32": ("buffer_from_host_f32", "to_f32", float),
    "i32": ("buffer_from_host_i32", "to_i32", int),
    "bf16": ("buffer_from_host_bf16", "to_bf16_patterns", int),
}


def _stage_for(dtype: str):
    try:
        return _STAGE[dtype]
    except KeyError:
        raise ValueError(
            f"Tlaloc dtype '{dtype}' has no PJRT staging in tlaloc_serve.py; "
            f"known: {sorted(_STAGE)}. f64/i64/bool are refused BY NAME rather "
            f"than coerced — a silently widened slot is a wrong answer that "
            f"runs."
        ) from None


def numel(dims: Sequence[int]) -> int:
    n = 1
    for d in dims:
        n *= int(d)
    return n


def unflatten(flat: Sequence[Any], dims: Sequence[int]) -> list:
    """Row-major nesting, for a caller that wants it. The loader itself never
    needs it — shapes live in the manifest."""
    if len(dims) <= 1:
        return list(flat)
    stride = numel(dims[1:])
    return [unflatten(flat[i * stride:(i + 1) * stride], dims[1:]) for i in range(dims[0])]


def flatten(nested) -> list:
    """Row-major flattening, the inverse of [unflatten]."""
    out = []
    stack = [nested]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(reversed(item))
        else:
            out.append(item)
    return out


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

    @property
    def count(self) -> int:
        return numel(self.dims)


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


# ---------------------------------------------------------------------------
# Engines.
# ---------------------------------------------------------------------------


class CtypesEngine:
    """THE serving engine: `tlaloc_pjrt` and the standard library.

    Owns the plugin handle, the client and the executable cache; a `close()`
    destroys them in the order PJRT requires (executables, then client),
    because a leaked CUDA client is a pinned BFC pool that outlives the
    process that wanted it.
    """

    name = "ctypes"

    def __init__(self, platform: str, plugin_path: str | None):
        self.platform = platform
        self.plugin_path = find_pjrt_plugin(plugin_path, platform)
        self._api = None
        self._client = None
        self._device = None

    def _ensure(self):
        if self._client is not None:
            return
        self._api = P.PjrtApi.load(self.plugin_path, self.platform)
        # create_options are NOT optional on CUDA (§0.4.333): without them the
        # plugin preallocates 75% of unified memory and takes the GB10 down.
        # `create_client` refuses `None` for a GPU platform by name.
        self._client = self._api.create_client(self.platform)
        self._device = self._client.addressable_devices()[0]

    def platform_name(self) -> str:
        self._ensure()
        return self._client.platform_name()

    def compile(self, text: str):
        self._ensure()
        return self._client.compile(text)

    def stage_weights(self, root, table: Sequence[dict]) -> dict:
        """§0.4.480 — upload the staged weight table once, return name -> buffer.

        Straight to `tlaloc_pjrt.buffer_from_file`: the file IS the operand, so
        no Python number is ever created for a weight. If any upload fails the
        ones already done are closed — a half-staged model is device memory
        nobody holds a handle to.
        """
        self._ensure()
        bufs: dict = {}
        try:
            for w in table:
                bufs[w["name"]] = self._client.buffer_from_file(
                    self._device, Path(root) / w["path"], w["dtype"], tuple(w["dims"]),
                )
        except BaseException:
            for b in bufs.values():
                b.close()
            raise
        return bufs

    def run(self, exe, staged: Sequence[tuple], outputs: Sequence[Slot]) -> list:
        """`staged` is [(Slot, values)] in the entry's input order, where
        `values` is a flat host sequence OR an already-uploaded device buffer
        (the staged weights — see `ServingArtifact.weight_buffers`).

        Every input buffer THIS CALL MADE and every output buffer is destroyed
        before it returns: one decode step must not leave device memory behind,
        or a serving loop is an allocator leak with a model attached. A buffer
        passed in is NOT closed here — its owner is the artifact, not the step.
        """
        self._ensure()
        bufs = []
        mine = []
        try:
            for slot, values in staged:
                if isinstance(values, P.PjrtBuffer):
                    bufs.append(values)
                    continue
                maker, _, _ = _stage_for(slot.dtype)
                b = getattr(self._client, maker)(self._device, values, slot.dims)
                bufs.append(b)
                mine.append(b)
            results = exe.execute(bufs, self._device)
            try:
                out = []
                for i, slot in enumerate(outputs):
                    _, reader, _ = _stage_for(slot.dtype)
                    out.append(getattr(results[i], reader)(slot.count))
                return out
            finally:
                for r in results:
                    r.close()
        finally:
            for b in mine:
                b.close()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._api = None
        self._device = None


class JaxEngine:
    """ORACLE ONLY — not the deployment path. See the module docstring.

    Exists because jaxlib ships no CPU PJRT plugin `.so`, so the tight
    XLA-CPU semantics lane has no ctypes route here. Every import is inside a
    method on purpose: choosing the ctypes engine must import none of this,
    and that is what makes the import guard a fact rather than a wish.
    """

    name = "jax"

    def __init__(self, platform: str, plugin_path: str | None):
        self.platform = platform
        self.plugin_path = plugin_path
        self._backend = None

    def _ensure(self):
        if self._backend is not None:
            return
        # GB10 is unified-memory: JAX's default 75% preallocation would claim
        # ~90 GB of system RAM. Must be set before `import jax` (§0.4.333).
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
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

    def platform_name(self) -> str:
        self._ensure()
        return self.platform

    def compile(self, text: str):
        self._ensure()
        import jaxlib.mlir.ir as ir
        from jaxlib.mlir._mlir_libs._jax_mlir_ext import register_dialects
        import jaxlib.mlir.dialects.stablehlo as stablehlo

        reg = ir.DialectRegistry()
        register_dialects(reg)
        ctx = ir.Context()
        ctx.append_dialect_registry(reg)
        ctx.load_all_available_dialects()
        stablehlo.register_dialect(ctx)
        with ctx, ir.Location.unknown(ctx):
            module = ir.Module.parse(text)
        return self._backend.compile_and_load(module, self._device_list, self._jaxlib.CompileOptions())

    def stage_weights(self, root, table: Sequence[dict]) -> dict:
        """REFUSED BY NAME (§0.4.480).

        The jax engine exists for ONE reason — jaxlib ships no CPU PJRT plugin
        `.so`, so the tight 1e-5 XLA-CPU semantics lane has nothing for the
        ctypes engine to dlopen (`default_engine_for`). Staging a real
        checkpoint's weight table through it would mean `np.fromfile` and a
        second, differently-shaped upload path, in the one engine the
        deployment never runs — an oracle that diverges from the thing it is
        an oracle for.

        The consequence is stated rather than worked around: **a staged-weight
        artifact has no CPU lane in this loader.** Its oracle is the JVM
        interpreter and HuggingFace transformers, which is what §0.4.479
        certified the graph against in the first place.
        """
        raise ValueError(
            f"the jax ORACLE engine does not stage weights ({len(table)} slots asked "
            f"for). A staged-weight artifact runs on the ctypes engine, which is the "
            f"deployment path; the jax engine carries the CPU semantics lane for "
            f"artifacts whose weights are body constants."
        )

    def run(self, exe, staged: Sequence[tuple], outputs: Sequence[Slot]) -> list:
        self._ensure()
        import numpy as np

        np_of = {"f32": np.float32, "i32": np.int32}
        ordered = []
        for slot, values in staged:
            if slot.dtype not in np_of:
                raise ValueError(
                    f"the jax oracle engine stages f32/i32 only (slot '{slot.name}' is "
                    f"{slot.dtype}); the ctypes engine is the one that carries bf16"
                )
            a = np.asarray(list(values), dtype=np_of[slot.dtype]).reshape(slot.dims)
            ordered.append(self._jax.device_put(a, self._devices[0]))
        results = exe.execute(ordered)
        return [
            [t.item() for t in np.asarray(r).reshape(-1)]
            for r in results
        ]

    def close(self) -> None:
        self._backend = None


_ENGINES = {"ctypes": CtypesEngine, "jax": JaxEngine}


def default_engine_for(platform: str) -> str:
    """Which engine a platform gets when the caller does not say.

    One rule, in one place, and it is a statement about jaxlib rather than a
    preference: **there is no CPU PJRT plugin `.so`**. jaxlib's CPU client is
    a C++ class inside its own extension module, so the ctypes binding — which
    dlopens a file and calls `GetPjrtApi` — has nothing to open for `"cpu"`.
    Every accelerator platform (cuda, rocm, tpu) ships a real plugin file and
    therefore gets the ctypes engine, which is the deployment path.

    The day a CPU plugin `.so` exists here this function is the only thing
    that changes.
    """
    return "jax" if platform == "cpu" else "ctypes"


class ServingArtifact:
    """A loaded Tlaloc serving artifact: manifest, bodies, executables."""

    def __init__(self, root: Path, manifest: dict, platform: str = "cuda",
                 engine: str | None = None, plugin_path: str | None = None):
        engine = engine or default_engine_for(platform)
        if manifest.get("schemaVersion") not in READABLE_SCHEMA_VERSIONS:
            raise ValueError(
                f"{root}: schemaVersion {manifest.get('schemaVersion')!r} is not one of "
                f"{READABLE_SCHEMA_VERSIONS!r}; refusing an artifact of unknown shape"
            )
        if engine not in _ENGINES:
            raise ValueError(f"unknown engine {engine!r}; known: {sorted(_ENGINES)}")
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
        # §0.4.480 — the staged weight table. Absent reads as empty: an
        # artifact written before this slice bakes its weights into the bodies
        # and binds no weight operands, and it stays a legal artifact.
        self.weight_table = list(self.weights.get("table") or [])
        self.entries = [Entry.parse(e) for e in manifest["entries"]]
        self._exe_cache: dict = {}
        self._weight_bufs: dict | None = None
        self.engine = _ENGINES[engine](platform, plugin_path)

    # --- loading -------------------------------------------------------

    @classmethod
    def load(cls, path, platform: str = "cuda", engine: str | None = None,
             plugin_path: str | None = None) -> "ServingArtifact":
        root = Path(path)
        mf = root / "tlaloc-serving.json"
        if not mf.exists():
            raise FileNotFoundError(f"{root} is not a Tlaloc serving artifact: no {mf.name}")
        return cls(root, json.loads(mf.read_text()), platform=platform,
                   engine=engine, plugin_path=plugin_path)

    def __enter__(self) -> "ServingArtifact":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """Drop the executables, then the client. A serving process that
        reloads an artifact must not accumulate PJRT clients."""
        for exe in self._exe_cache.values():
            close = getattr(exe, "close", None)
            if close is not None:
                close()
        self._exe_cache.clear()
        if self._weight_bufs is not None:
            for b in self._weight_bufs.values():
                b.close()
            self._weight_bufs = None
        self.engine.close()

    # --- staged weights (§0.4.480) --------------------------------------

    def weight_buffers(self) -> dict:
        """The staged weight operands, as LIVE DEVICE BUFFERS, uploaded once.

        This is the one place in the loader where a buffer outlives a call, and
        it is the reason the weight table exists at all. A weight is the same
        bytes on every decode step of every bucket; re-uploading 4.4 GB per
        token is not a slow serving loop, it is a different program. So they
        are staged on first use and closed with the artifact, and
        `run_decode`'s per-step staging binds them by NAME against the entry's
        WEIGHT slots.

        Lazy rather than eager in `__init__`: `select_bucket` / `entry_for` /
        `verify_bodies` are useful without a device, and `ServingArtifact.load`
        must not need one.
        """
        if self._weight_bufs is None:
            if not self.weight_table:
                self._weight_bufs = {}
            else:
                self._weight_bufs = self.engine.stage_weights(self.root, self.weight_table)
        return self._weight_bufs

    def verify_weights(self) -> None:
        """Re-hash every staged weight file against the manifest.

        SEPARATE FROM `verify_bodies` and never automatic. A body is kilobytes
        and hashing it on load is free; a weight table is gigabytes, and a
        deployment restarting a worker should not pay a full SHA-256 of the
        model to learn what its filesystem already told it. The check exists,
        it is exact, and choosing to run it is the operator's.
        """
        for w in self.weight_table:
            p = self.root / w["path"]
            size = p.stat().st_size
            if size != w["byteLength"]:
                raise ValueError(
                    f"{w['path']}: {size} bytes on disk, manifest says {w['byteLength']} "
                    f"for slot '{w['name']}' {w['dims']}"
                )
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
            if h.hexdigest() != w["sha256"]:
                raise ValueError(
                    f"{w['path']}: hashes to {h.hexdigest()} but the manifest says "
                    f"{w['sha256']} — this artifact's weights are not the ones it was "
                    f"exported with"
                )

    def verify_bodies(self) -> None:
        """Content-address check: every body hashes to the name it is filed
        under. The artifact crossed a process boundary; this is the cheapest
        possible statement that it arrived intact."""
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

    def compiled(self, entry: Entry):
        """Compile (once per `cacheKey`) and return a loaded executable.

        The body is compiled, never edited: the exporter names the entry
        function `main` and the manifest states it, so a loader does not
        rewrite a program it was asked to run (the §0.4.325 spike had to
        regex-rename; that was a smell, and the fix went in the exporter).
        """
        hit = self._exe_cache.get(entry.cache_key)
        if hit is not None:
            return hit
        exe = self.engine.compile((self.root / entry.body_path).read_text())
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
        kv_pools: Sequence[Sequence[float]],
        context: int | None = None,
    ):
        """One decode step for `len(token_ids)` sequences.

        Returns `(logits, kv_pools)`. `logits` is `list[list[float]]`, one row
        per REAL sequence — the padding rows the bucket added never leave this
        function. The pools come back WHOLE and flat, because a pool is device
        state shared by every sequence and slicing it would be meaningless.
        """
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

        def pad_tables(rows):
            out = []
            for i in range(b):
                r = list(rows[i]) if i < len(rows) else []
                if len(r) > mbs:
                    raise ValueError(
                        f"run_decode: sequence {i} names {len(r)} blocks but the "
                        f"(batch={b}, context={c}) bucket's table is {mbs} wide"
                    )
                out.extend(r + [PADDING_BLOCK] * (mbs - len(r)))
            return out

        args = {
            "TOKEN_IDS": list(token_ids) + [PADDING_TOKEN_ID] * (b - n),
            "POSITIONS": list(positions) + [PADDING_POSITION] * (b - n),
            "BLOCK_TABLES": pad_tables(block_tables),
            "SEQ_LENS": list(seq_lens) + [PADDING_SEQ_LEN] * (b - n),
            "SLOT_MAPPING": list(slot_mapping) + [PADDING_SLOT] * (b - n),
        }

        pools = [list(p) for p in kv_pools]
        expected_pools = sum(1 for s in entry.inputs if s.role == "KV_POOL_IN")
        if len(pools) != expected_pools:
            raise ValueError(
                f"run_decode: {len(pools)} KV pools supplied, the entry's signature "
                f"has {expected_pools} (two per layer, (key, value) order, layers ascending)"
            )

        # §0.4.480 — the staged weights, uploaded once and reused by every
        # step of every bucket. They are bound by NAME, not by position:
        # position is already the contract between the manifest and XLA, and
        # re-deriving it here would be a second copy of an ordering that can
        # only disagree.
        wbufs = self.weight_buffers()
        staged = []
        pool_i = 0
        for slot in entry.inputs:
            if slot.role == "WEIGHT":
                try:
                    staged.append((slot, wbufs[slot.name]))
                except KeyError:
                    raise ValueError(
                        f"run_decode: entry '{entry.entry_id}' binds a WEIGHT operand "
                        f"'{slot.name}' that the manifest's weight table does not name "
                        f"(it has {len(self.weight_table)} entries). The artifact promises "
                        f"an operand it cannot supply"
                    ) from None
                continue
            if slot.role == "KV_POOL_IN":
                values = pools[pool_i]
                pool_i += 1
            else:
                values = args[slot.role]
            if len(values) != slot.count:
                raise ValueError(
                    f"run_decode: operand '{slot.name}' has {len(values)} elements but the "
                    f"compiled entry declares {list(slot.dims)} = {slot.count}"
                )
            staged.append((slot, values))

        out = self.engine.run(self.compiled(entry), staged, entry.outputs)
        logits_slot = entry.outputs[0]
        # A row is nested to the slot's DECLARED trailing dims — `[tokensPerSeq,
        # vocab]` for decode. The token axis is kept even at `tokensPerSeq == 1`,
        # because the same contract describes a prefill entry where it is not 1
        # and `last_token_logits` has to be able to find the last position
        # (`vllm_tlaloc.batching`, which rejected flattening for exactly that
        # reason). Everything below the row is the manifest's shape, not a
        # convention invented here.
        stride = numel(logits_slot.dims[1:]) if len(logits_slot.dims) > 1 else len(out[0])
        logits = [
            unflatten(out[0][i * stride:(i + 1) * stride], logits_slot.dims[1:])
            for i in range(n)
        ]
        return logits, out[1:]

    def empty_pools(self, fill: float = 0.0) -> list:
        """Freshly-zeroed KV pools of the artifact's declared layout, in the
        flat host form `run_decode` takes and returns."""
        m = self.model
        n = numel(tuple(m["kvPoolDims"]))
        _stage_for(m["kvDtype"])  # refuse an unstageable pool dtype here, by name
        value = int(fill) if m["kvDtype"] in ("i32", "bf16") else float(fill)
        return [[value] * n for _ in range(2 * m["numLayers"])]
