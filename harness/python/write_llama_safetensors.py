"""§0.4.468 (Phase H2) — synthesize a Llama-decoder checkpoint as a REAL
safetensors file, written by the reference implementation.

Why synthesize rather than fetch. Phase H2's job is to certify (a) Tlaloc's
Kotlin safetensors reader against the format as the real producer writes it,
and (b) the decode graph against PyTorch on the weights that reader returns.
A downloaded TinyLlama would add a network dependency and a multi-gigabyte
disk footprint to the certification suite without strengthening either claim:
the bytes of a random f32 tensor and the bytes of a trained one are the same
kind of bytes, and `LlamaDecoderPrimal`'s graph is not TinyLlama's graph
anyway (single layer, single head, no GQA — see LlamaDecoderPrimal.kt's scope
notes). What a real checkpoint WOULD add is HF's tensor-naming convention and
the sharded-index form; the reader handles both and pins them separately
(SafetensorsFileTest), and loading a real Llama end to end is named in the
audit as an H3 item, where the model-layer mapping belongs.

What this script writes into --out-dir:

  llama_tiny.safetensors     every LlamaDecoderPrimal param, F32
  llama_bf16.safetensors     the same numbers stored at BF16
  f32/<name>.npy             the F32 values, for run_pytorch_llama.py
  bf16/<name>.npy            the same values ROUNDED THROUGH torch.bfloat16
                             and widened back to f32 — i.e. exactly the
                             numbers a bf16 checkpoint contains
  probes.json                per-tensor raw element bit patterns at chosen
                             flat indices, so the Kotlin reader can be pinned
                             bit-for-bit against what Python wrote

The bf16 pair is the load-bearing one: it makes the PyTorch reference run on
the bf16-rounded values, so Tlaloc reading the bf16 file and widening
(HostBf16Storage -> f32, exact) must land on the same numbers. It also
cross-checks Tlaloc's round-to-nearest-even (§0.4.455 floatToBf16Bits)
against torch's own bfloat16 conversion, since both sides must produce the
identical 16-bit patterns for the comparison to hold at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_arrays(tokens: int, d: int, dff: int, vocab: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    def n(shape, scale):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    return {
        # Activations / inputs.
        "x_in": n((tokens, d), 0.5),
        "theta": n((tokens, d), 0.25),
        # Labels: a proper one-hot per token, so the loss is a real
        # log-likelihood rather than a weighted sum of noise.
        "labels": np.eye(vocab, dtype=np.float32)[rng.integers(0, vocab, size=tokens)],
        # Weights, scaled 1/sqrt(fan_in) so activations stay O(1) and the
        # softmaxes do not saturate — a saturated softmax would hide a real
        # disagreement behind an exp() underflow on both sides.
        "q_w": n((d, d), d ** -0.5),
        "k_w": n((d, d), d ** -0.5),
        "v_w": n((d, d), d ** -0.5),
        "out_w": n((d, d), d ** -0.5),
        "gate_w": n((d, dff), d ** -0.5),
        "up_w": n((d, dff), d ** -0.5),
        "down_w": n((dff, d), dff ** -0.5),
        "lm_head_w": n((d, vocab), d ** -0.5),
        # RmsNorm eps, broadcast-shaped [tokens, 1] to match the primal.
        "eps_attn": np.full((tokens, 1), 1e-6, dtype=np.float32),
        "eps_mlp": np.full((tokens, 1), 1e-6, dtype=np.float32),
    }


def probe_indices(count: int) -> list[int]:
    """A handful of flat indices: both ends plus a spread through the middle.

    Both ends matter most — an off-by-one in offset slicing shows up at a
    boundary first — and the middle catches a stride or endianness error that
    happens to leave the ends intact.
    """
    if count == 0:
        return []
    picks = {0, count - 1}
    for k in (1, 2, 3, 5, 7):
        picks.add((count * k) // 8 % count)
    return sorted(picks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--tokens", required=True, type=int)
    ap.add_argument("--dmodel", required=True, type=int)
    ap.add_argument("--dff", required=True, type=int)
    ap.add_argument("--vocab", required=True, type=int)
    ap.add_argument("--seed", default=20260921, type=int)
    args = ap.parse_args()

    import torch
    from safetensors.numpy import save_file

    out = args.out_dir
    (out / "f32").mkdir(parents=True, exist_ok=True)
    (out / "bf16").mkdir(parents=True, exist_ok=True)

    arrs = build_arrays(args.tokens, args.dmodel, args.dff, args.vocab, args.seed)

    # --- F32 checkpoint + the .npy the torch reference consumes -----------
    save_file(arrs, str(out / "llama_tiny.safetensors"), metadata={"format": "pt", "tlaloc": "h2"})
    for name, a in arrs.items():
        np.save(out / "f32" / f"{name}.npy", a)

    # --- BF16 checkpoint: torch does the narrowing, so the rounding under
    #     test is the reference's own RNE, not ours. ----------------------
    bf16_store = {}
    bf16_widened = {}
    for name, a in arrs.items():
        t = torch.from_numpy(a).to(torch.bfloat16)
        # Raw 16-bit patterns: view the bf16 tensor's storage as int16, then
        # carry it as uint16 (numpy has no bfloat16 dtype, and safetensors
        # takes the dtype from the array — so the file is written from the
        # torch side, where BF16 is a real dtype).
        bf16_store[name] = t
        bf16_widened[name] = t.float().numpy()
        np.save(out / "bf16" / f"{name}.npy", bf16_widened[name])

    from safetensors.torch import save_file as save_file_torch
    save_file_torch(
        {k: v.contiguous() for k, v in bf16_store.items()},
        str(out / "llama_bf16.safetensors"),
        metadata={"format": "pt", "tlaloc": "h2-bf16"},
    )

    # --- probes: raw element bit patterns Python actually wrote ----------
    probes = {}
    for name, a in arrs.items():
        flat = a.reshape(-1)
        idx = probe_indices(flat.size)
        probes[name] = {
            "dtype": "F32",
            "shape": list(a.shape),
            "count": int(flat.size),
            "indices": idx,
            # int32 view of the f32 bits, signed — Kotlin's Float.toRawBits().
            "bits": [int(np.asarray(flat[i], dtype=np.float32).view(np.int32)) for i in idx],
        }
    bf16_probes = {}
    for name, t in bf16_store.items():
        flat16 = t.reshape(-1).view(torch.int16).numpy()
        idx = probe_indices(flat16.size)
        bf16_probes[name] = {
            "dtype": "BF16",
            "shape": list(t.shape),
            "count": int(flat16.size),
            "indices": idx,
            # int16 view, signed — Kotlin's Short from HostBf16Storage.
            "bits": [int(flat16[i]) for i in idx],
        }
    (out / "probes.json").write_text(json.dumps({"f32": probes, "bf16": bf16_probes}))

    print(
        f"[write_llama_safetensors] wrote {len(arrs)} tensors "
        f"(tokens={args.tokens} d={args.dmodel} dff={args.dff} vocab={args.vocab} seed={args.seed})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
