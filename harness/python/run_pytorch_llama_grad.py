"""§0.4.292 — PyTorch backward reference for LlamaDecoderPrimal.

Loads the same `.npy` inputs `run_pytorch_llama.py` uses, runs the same
forward, then calls `torch.autograd.grad(loss, [...all 13 params...])`
to compute one ∂loss/∂param tensor per param. Writes the gradients as a
JSON dict keyed by param name (positionally aligned with the Tlaloc
DxirFunction.params order).

Output JSON shape:
    {"x_in": [v0, v1, ...], "labels": [...], ..., "eps_mlp": [...]}

Each value array is a flat row-major listing of the gradient's elements.
The Kotlin test parses this trivially.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _expected_param_names() -> list[str]:
    return [
        "x_in", "labels", "theta",
        "q_w", "k_w", "v_w", "out_w",
        "gate_w", "up_w", "down_w", "lm_head_w",
        "eps_attn", "eps_mlp",
    ]


def _load_inputs(inputs_dir: Path):
    arrs = {}
    for name in _expected_param_names():
        path = inputs_dir / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        arrs[name] = np.load(path)
    return arrs


def forward_and_grad(arrs: dict) -> dict:
    import torch
    import torch.nn.functional as F

    tensors: dict[str, "torch.Tensor"] = {}
    for name in _expected_param_names():
        tensors[name] = torch.from_numpy(arrs[name].astype(np.float32)).requires_grad_(True)

    x_in = tensors["x_in"]
    labels = tensors["labels"]
    theta = tensors["theta"]
    q_w, k_w, v_w, out_w = tensors["q_w"], tensors["k_w"], tensors["v_w"], tensors["out_w"]
    gate_w, up_w, down_w = tensors["gate_w"], tensors["up_w"], tensors["down_w"]
    lm_head_w = tensors["lm_head_w"]
    eps_attn, eps_mlp = tensors["eps_attn"], tensors["eps_mlp"]

    # ---- Forward (mirrors run_pytorch_llama.py 1:1) ---------------------------
    sq1 = x_in * x_in
    mean1 = sq1.mean(dim=1, keepdim=True)
    rsq1 = torch.rsqrt(mean1 + eps_attn)
    x_norm_attn = x_in * rsq1

    q = x_norm_attn @ q_w
    k = x_norm_attn @ k_w
    v = x_norm_attn @ v_w

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    q_rot = (q * cos_t) - (theta * sin_t)

    k_t = k.transpose(0, 1)
    s = q_rot @ k_t
    p = torch.softmax(s, dim=-1)
    attn_out = p @ v

    attn_proj = attn_out @ out_w
    x_attn = x_in + attn_proj

    sq2 = x_attn * x_attn
    mean2 = sq2.mean(dim=1, keepdim=True)
    rsq2 = torch.rsqrt(mean2 + eps_mlp)
    x_norm_mlp = x_attn * rsq2

    gate_proj = x_norm_mlp @ gate_w
    up_proj = x_norm_mlp @ up_w
    swiglu = F.silu(gate_proj) * up_proj

    mlp_out = swiglu @ down_w
    x_out = x_attn + mlp_out

    logits = x_out @ lm_head_w
    probs = torch.softmax(logits, dim=-1)
    logp = torch.log(probs)
    loss = (labels * logp).sum()

    # ---- Reverse via torch.autograd.grad --------------------------------------
    grad_list = torch.autograd.grad(
        loss,
        [tensors[name] for name in _expected_param_names()],
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    out: dict[str, list[float]] = {}
    for name, g in zip(_expected_param_names(), grad_list):
        if g is None:
            # `allow_unused=True` allows None for params that don't contribute to the
            # loss graph; emit a zero-filled array of the correct size so the Kotlin
            # comparison sees an explicit zero (matching Tlaloc's gradient-of-unused = 0).
            n = int(np.prod(arrs[name].shape) or 1)
            out[name] = [0.0] * n
        else:
            flat = g.detach().contiguous().flatten().numpy().astype(np.float32)
            out[name] = [float(v) for v in flat]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    arrs = _load_inputs(args.inputs_dir)
    grads = forward_and_grad(arrs)

    args.output.write_text(json.dumps(grads))
    sizes = {k: len(v) for k, v in grads.items()}
    print(f"[run_pytorch_llama_grad] grads written; sizes={sizes}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
