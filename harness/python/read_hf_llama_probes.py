#!/usr/bin/env python3
"""§0.4.478 (H3c-1) — the ORACLE for HF Llama weight ingestion.

Reads a REAL HuggingFace checkpoint with the reference stack (`safetensors` +
`torch`) and prints, as JSON on stdout, the facts the Kotlin loader must
reproduce: every mapped tensor's dtype and shape, and the RAW BIT PATTERN of a
handful of chosen elements.

This is NOT serving-path code. It imports torch on purpose: the whole point is
that the bytes are interpreted by the implementation that WROTE them, so a
shared misreading between Tlaloc's reader and Tlaloc's test is impossible. It
runs in `~/.local/venvs/vllm` (torch 2.13.0+cu130, safetensors), never in the
frozen oracle venv and never in a serving process.

Usage:

    python read_hf_llama_probes.py <checkpoint-dir> <probe-spec> [<probe-spec> ...]

where a probe spec is `tensor.name@flatIndex` (row-major flat index into the
tensor as stored, which is exactly how Tlaloc's LoadedTensor is laid out).

Why bit patterns and not floats: a float printed at 6 digits and re-parsed is a
comparison with a hidden tolerance in it. Reading bytes is not arithmetic, so
the floor is EXACT and the oracle must be exact to make that claim checkable.
"""

import json
import sys

import torch
from safetensors import safe_open


def _raw_bits(t: torch.Tensor, index: int) -> int:
    """The element's storage bit pattern as an unsigned integer."""
    flat = t.reshape(-1)
    v = flat[index]
    if t.dtype == torch.bfloat16:
        return int(v.view(torch.int16).item()) & 0xFFFF
    if t.dtype == torch.float16:
        return int(v.view(torch.int16).item()) & 0xFFFF
    if t.dtype == torch.float32:
        return int(v.view(torch.int32).item()) & 0xFFFFFFFF
    if t.dtype == torch.float64:
        return int(v.view(torch.int64).item()) & 0xFFFFFFFFFFFFFFFF
    raise SystemExit(f"read_hf_llama_probes: unhandled dtype {t.dtype} for a probe")


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    ckpt_dir = argv[1]
    specs = argv[2:]

    wanted = {}
    for spec in specs:
        name, _, idx = spec.rpartition("@")
        if not name:
            raise SystemExit(f"read_hf_llama_probes: bad probe spec '{spec}'")
        wanted.setdefault(name, []).append(int(idx))

    out = {"checkpoint": ckpt_dir, "torch": torch.__version__, "tensors": {}}

    # Single-file only for now; the sharded form would consult
    # model.safetensors.index.json. The Kotlin side handles both (§0.4.468) and
    # the checkpoint this oracle was written against is single-file, so adding
    # an untested shard walk HERE would be an oracle nobody has checked.
    path = f"{ckpt_dir}/model.safetensors"
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        out["tensor_count"] = len(keys)
        out["names"] = sorted(keys)
        for name, indices in wanted.items():
            if name not in keys:
                raise SystemExit(f"read_hf_llama_probes: no tensor '{name}' in {path}")
            t = f.get_tensor(name)
            out["tensors"][name] = {
                "dtype": str(t.dtype).removeprefix("torch."),
                "shape": list(t.shape),
                "numel": int(t.numel()),
                "probes": {str(i): _raw_bits(t, i) for i in indices},
            }
        # Shapes for EVERY tensor, cheaply: get_slice reads the header only.
        out["shapes"] = {k: list(f.get_slice(k).get_shape()) for k in keys}
        out["dtypes"] = {k: f.get_slice(k).get_dtype() for k in keys}

    json.dump(out, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv)
