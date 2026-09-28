"""The Python runtime for a Tlaloc bounded-program artifact (`tlaloc-bounded.json`,
schema `tlaloc-bounded-v1`; docs/design/bounded-dims.md).

A bounded program has inputs whose axes are bounded (`Bounded<MaxSeq>` in the Kotlin type).
The exporter wrote one static StableHLO body per combination of buckets. For a request this
module:

1. reads each bound's size from the DATA inputs (every axis of one bound has one size),
2. picks the smallest bucket of each bound that holds it,
3. pads each DATA input with the manifest's padding value along its bounded axes,
4. fills each VALID_MASK input (`[bucket]`, 1.0 for a real position) and each VALID_LENGTH
   input (the real size, as an f32 scalar),
5. runs that bucket's body (compiled once per body, on first use), and
6. slices the output's bounded axes back to the real sizes.

Host tensors are flat row-major lists with their dims given alongside, as in
`tlaloc_serve.py`. Imports are the standard library, `tlaloc_serve` and `tlaloc_pjrt`; no
numpy, jax or torch.

    art = BoundedArtifact.load("build/bounded-artifact")
    with art:
        y, dims = art.run([(x_flat, [5, 16])])

The reader refuses a manifest it does not fully know: another schema version, an unknown key,
role or axis form, or entries that do not cover every combination of buckets.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import tlaloc_serve as S

SCHEMA_VERSION = "tlaloc-bounded-v1"
MANIFEST_NAME = "tlaloc-bounded.json"
ROLES = ("DATA", "VALID_MASK", "VALID_LENGTH")

_TOP_KEYS = {"schemaVersion", "name", "bounds", "padding", "inputs", "outputs", "entries", "paddingCheck"}
_BOUND_KEYS = {"name", "max", "buckets"}
_TENSOR_KEYS = {"name", "role", "dtype", "axes", "bound"}
_ENTRY_KEYS = {"id", "sizes", "entryPoint", "bodyPath", "bodyHash", "programPath"}
_CHECK_KEYS = {"sizes", "maxDifference", "tolerance"}


class ManifestError(ValueError):
    """A bounded manifest this reader refuses, with the reason."""


def _only(obj: dict, allowed: set, what: str) -> None:
    extra = set(obj) - allowed
    if extra:
        raise ManifestError(f"{what} has unknown key(s) {sorted(extra)}; this reader knows {sorted(allowed)}")


def _int(v, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ManifestError(f"{what} is not an integer: {v!r}")
    return v


def validate(manifest: dict) -> None:
    """Raise ManifestError unless `manifest` is a complete tlaloc-bounded-v1 manifest."""
    if not isinstance(manifest, dict):
        raise ManifestError("the manifest is not a JSON object")
    version = manifest.get("schemaVersion")
    if version != SCHEMA_VERSION:
        raise ManifestError(f"schemaVersion {version!r} is not one this reader knows ({SCHEMA_VERSION!r})")
    _only(manifest, _TOP_KEYS, "the manifest")
    missing = _TOP_KEYS - set(manifest)
    if missing:
        raise ManifestError(f"the manifest lacks {sorted(missing)}")
    bounds = manifest["bounds"]
    names = []
    for b in bounds:
        _only(b, _BOUND_KEYS, f"bound {b.get('name')!r}")
        mx = _int(b["max"], f"bound {b['name']} max")
        buckets = [_int(v, f"bound {b['name']} bucket") for v in b["buckets"]]
        if not buckets or buckets[0] < 1 or any(x >= y for x, y in zip(buckets, buckets[1:])) or buckets[-1] != mx:
            raise ManifestError(f"bound {b['name']}: buckets {buckets} must ascend strictly from >= 1 to max {mx}")
        names.append(b["name"])
    if len(set(names)) != len(names) or not names:
        raise ManifestError(f"bound names {names} are empty or repeat")
    _only(manifest["padding"], {"value"}, "padding")
    for t in manifest["inputs"] + manifest["outputs"]:
        _only(t, _TENSOR_KEYS, f"tensor {t.get('name')!r}")
        if t["role"] not in ROLES:
            raise ManifestError(f"tensor {t['name']}: unknown role {t['role']!r}; known {list(ROLES)}")
        if t["dtype"] not in ("f32", "i32"):
            raise ManifestError(f"tensor {t['name']}: dtype {t['dtype']!r} is not f32 or i32")
        for a in t["axes"]:
            _only(a, {"size", "bound"}, f"an axis of {t['name']}")
            if ("size" in a) == ("bound" in a):
                raise ManifestError(f"an axis of {t['name']} must have exactly one of size and bound: {a}")
            if "bound" in a and a["bound"] not in names:
                raise ManifestError(f"tensor {t['name']} uses undeclared bound {a['bound']!r}")
        if t["role"] != "DATA" and t.get("bound") not in names:
            raise ManifestError(f"{t['role']} input {t['name']} must name a declared bound")
    for t in manifest["outputs"]:
        if t["role"] != "DATA":
            raise ManifestError(f"output {t['name']} has role {t['role']}; outputs are DATA")
    for e in manifest["entries"]:
        _only(e, _ENTRY_KEYS, f"entry {e.get('id')!r}")
    _only(manifest["paddingCheck"], _CHECK_KEYS, "paddingCheck")
    want = [{}]
    for b in bounds:
        want = [dict(m, **{b["name"]: n}) for m in want for n in b["buckets"]]
    got = [e["sizes"] for e in manifest["entries"]]
    key = lambda m: tuple(sorted(m.items()))
    if sorted(map(key, got)) != sorted(map(key, want)):
        raise ManifestError(f"entries cover {len(got)} bucket combinations; the buckets give {len(want)}")


def numel(dims: Sequence[int]) -> int:
    n = 1
    for d in dims:
        n *= int(d)
    return n


def pad_to(values: Sequence, dims: Sequence[int], target: Sequence[int], fill=0):
    """Row-major `values` of `dims`, padded with `fill` at the end of each axis to `target`."""
    if len(dims) != len(target) or any(d > t for d, t in zip(dims, target)):
        raise ValueError(f"pad_to: {list(dims)} does not fit in {list(target)}")
    if list(dims) == list(target):
        return list(values)
    out = [fill] * numel(target)
    _copy_block(values, dims, out, target, dims)
    return out


def slice_to(values: Sequence, dims: Sequence[int], target: Sequence[int]):
    """The leading `target` block of row-major `values` of `dims`."""
    if len(dims) != len(target) or any(t > d for d, t in zip(dims, target)):
        raise ValueError(f"slice_to: {list(target)} is not inside {list(dims)}")
    if list(dims) == list(target):
        return list(values)
    out = [0] * numel(target)
    _copy_block(values, dims, out, target, target)
    return out


def _strides(dims):
    s, acc = [0] * len(dims), 1
    for i in range(len(dims) - 1, -1, -1):
        s[i] = acc
        acc *= dims[i]
    return s


def _copy_block(src, src_dims, dst, dst_dims, block):
    rank = len(block)
    if rank == 0:
        dst[0] = src[0]
        return
    if any(b == 0 for b in block):
        return
    ss, ds = _strides(src_dims), _strides(dst_dims)
    inner = block[-1]
    idx = [0] * (rank - 1)
    while True:
        so = sum(i * s for i, s in zip(idx, ss))
        do = sum(i * s for i, s in zip(idx, ds))
        dst[do:do + inner] = src[so:so + inner]
        a = rank - 2
        while a >= 0:
            idx[a] += 1
            if idx[a] < block[a]:
                break
            idx[a] = 0
            a -= 1
        if a < 0:
            return


class BoundedArtifact:
    """A loaded bounded-program artifact: manifest, bodies, executables (compiled lazily)."""

    def __init__(self, root: Path, manifest: dict, platform: str = "cuda", engine=None,
                 plugin_path: str | None = None):
        validate(manifest)
        self.root = Path(root)
        self.manifest = manifest
        self.name = manifest["name"]
        self.bounds = {b["name"]: b for b in manifest["bounds"]}
        self.inputs = manifest["inputs"]
        self.outputs = manifest["outputs"]
        self.padding = manifest["padding"]["value"]
        self._entries = {tuple(sorted(e["sizes"].items())): e for e in manifest["entries"]}
        self._exe_cache: dict = {}
        if engine is None:
            engine = S.CtypesEngine(platform, plugin_path)
        self.engine = engine

    @classmethod
    def load(cls, path, platform: str = "cuda", engine=None, plugin_path: str | None = None) -> "BoundedArtifact":
        root = Path(path)
        mf = root / MANIFEST_NAME
        if not mf.exists():
            if (root / "tlaloc-serving.json").exists():
                raise ValueError(f"{root} holds tlaloc-serving.json, a language-model artifact; load it with tlaloc_serve")
            raise FileNotFoundError(f"{root} is not a Tlaloc bounded-program artifact: no {MANIFEST_NAME}")
        return cls(root, json.loads(mf.read_text()), platform=platform, engine=engine, plugin_path=plugin_path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        for exe in self._exe_cache.values():
            close = getattr(exe, "close", None)
            if close is not None:
                close()
        self._exe_cache.clear()
        close = getattr(self.engine, "close", None)
        if close is not None:
            close()

    @property
    def compile_count(self) -> int:
        """Distinct bodies compiled by this process."""
        return len(self._exe_cache)

    def data_inputs(self) -> list:
        return [t for t in self.inputs if t["role"] == "DATA"]

    def sizes_of(self, data: Sequence[tuple]) -> dict:
        """Each bound's size in the DATA inputs `[(flat, dims), ...]`, checked against the manifest."""
        specs = self.data_inputs()
        if len(data) != len(specs):
            raise ValueError(f"{self.name}: {len(data)} inputs given, {len(specs)} expected")
        sizes: dict = {}
        for k, ((values, dims), spec) in enumerate(zip(data, specs)):
            axes = spec["axes"]
            if len(dims) != len(axes):
                raise ValueError(f"{self.name}: input {spec['name']} has rank {len(dims)}, expected {len(axes)}")
            if len(values) != numel(dims):
                raise ValueError(f"{self.name}: input {spec['name']} has {len(values)} values for dims {list(dims)}")
            for i, (d, a) in enumerate(zip(dims, axes)):
                if "size" in a:
                    if d != a["size"]:
                        raise ValueError(f"{self.name}: input {spec['name']} axis {i} is {d}, fixed at {a['size']}")
                    continue
                b = self.bounds[a["bound"]]
                if not 1 <= d <= b["max"]:
                    raise ValueError(f"{self.name}: input {spec['name']} axis {i} is {d}, outside 1..{b['max']} of {b['name']}")
                prev = sizes.setdefault(b["name"], d)
                if prev != d:
                    raise ValueError(f"{self.name}: bound {b['name']} is {prev} on one axis and {d} on {spec['name']} axis {i}")
        return sizes

    def buckets_for(self, sizes: dict) -> dict:
        """The smallest bucket of each bound that holds its size."""
        if set(sizes) != set(self.bounds):
            raise ValueError(f"{self.name}: sizes for {sorted(sizes)} given, the bounds are {sorted(self.bounds)}")
        out = {}
        for name, n in sizes.items():
            b = self.bounds[name]
            if not 1 <= n <= b["max"]:
                raise ValueError(f"{self.name}: {name} = {n} is outside 1..{b['max']}")
            out[name] = next(x for x in b["buckets"] if x >= n)
        return out

    def entry_for(self, buckets: dict) -> dict:
        return self._entries[tuple(sorted(buckets.items()))]

    def compiled(self, entry: dict):
        key = entry["bodyHash"]
        exe = self._exe_cache.get(key)
        if exe is None:
            exe = self.engine.compile((self.root / entry["bodyPath"]).read_text())
            self._exe_cache[key] = exe
        return exe

    def run(self, data: Sequence[tuple]):
        """Run on DATA inputs `[(flat, dims), ...]`; returns `(flat, dims)` of the one output at the
        real sizes. An artifact with several outputs (a training step: the value, then gradients)
        is read with `run_all`."""
        if len(self.outputs) != 1:
            raise ValueError(f"{self.name} has {len(self.outputs)} outputs; use run_all")
        return self.run_all(data)[0]

    def run_all(self, data: Sequence[tuple]) -> list:
        """Run on DATA inputs `[(flat, dims), ...]`; returns `[(flat, dims), ...]`, one per output,
        each at the real sizes."""
        sizes = self.sizes_of(data)
        buckets = self.buckets_for(sizes)
        entry = self.entry_for(buckets)
        staged = []
        it = iter(data)
        for t in self.inputs:
            dims = tuple(a["size"] if "size" in a else buckets[a["bound"]] for a in t["axes"])
            slot = S.Slot(t["name"], t["role"], t["dtype"], dims)
            if t["role"] == "DATA":
                values, real = next(it)
                fill = int(self.padding) if t["dtype"] == "i32" else float(self.padding)
                staged.append((slot, pad_to(values, real, dims, fill)))
            elif t["role"] == "VALID_MASK":
                n = sizes[t["bound"]]
                staged.append((slot, [1.0] * n + [0.0] * (buckets[t["bound"]] - n)))
            else:
                staged.append((slot, [float(sizes[t["bound"]])]))
        slots = []
        for out in self.outputs:
            out_dims = tuple(a["size"] if "size" in a else buckets[a["bound"]] for a in out["axes"])
            slots.append(S.Slot(out["name"], out["role"], out["dtype"], out_dims))
        results = self.engine.run(self.compiled(entry), staged, slots)
        outs = []
        for out, slot, result in zip(self.outputs, slots, results):
            real = [a["size"] if "size" in a else sizes[a["bound"]] for a in out["axes"]]
            outs.append((slice_to(result, slot.dims, real), real))
        return outs
