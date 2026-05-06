"""§0.4.289 — PyTorch reference for LlamaDecoderPrimal.

Mirrors the forward pass in
`benchmarks/src/jvmTest/kotlin/io/tlaloc/benchmarks/LlamaDecoderPrimal.kt`
exactly. Loads inputs as .npy files from --inputs-dir (one per param,
keyed by name), runs the forward, writes the loss to --output as JSON.

The point is to compare against the Tlaloc-IREE-CPU loss on the same
inputs. This script is invoked by
`LlamaDecoderIreeVsPytorchTest.kt` as a subprocess; it self-skips
silently if torch is missing (just exits non-zero).

# Math (matches LlamaDecoderPrimal.kt lines 188-257 1:1)

    1. Pre-attention RmsNorm (eps form)
       sq1 = x_in * x_in
       mean1 = sq1.mean(dim=1, keepdim=True)
       rsq1 = rsqrt(mean1 + eps_attn)
       x_norm_attn = x_in * rsq1                # broadcasts [tokens, 1] over [tokens, d]

    2. Q, K, V projections (no bias)
       q = x_norm_attn @ q_w
       k = x_norm_attn @ k_w
       v = x_norm_attn @ v_w

    3. RoPE on Q only (single rotation; theta as imag-half proxy)
       q_rot = (q * cos(theta)) - (theta * sin(theta))

    4. Attention with pre-transposed K (Tlaloc convention: S = A · B,
       contract last(A) × first(B); Q·K^T needs K pre-transposed):
       k_t = k.transpose(0, 1)
       s = q_rot @ k_t                           # [tokens, tokens]
       p = softmax(s, dim=-1)
       attn_out = p @ v

    5. Output projection + residual (against UNNORMALIZED x_in)
       attn_proj = attn_out @ out_w
       x_attn = x_in + attn_proj

    6. Pre-MLP RmsNorm (same eps form)
       x_norm_mlp = x_attn * rsqrt(mean(x_attn²) + eps_mlp)

    7. SwiGLU MLP
       gate_proj = x_norm_mlp @ gate_w
       up_proj   = x_norm_mlp @ up_w
       swiglu    = silu(gate_proj) * up_proj

    8. Down projection + residual (against post-attention x_attn, not normed)
       mlp_out = swiglu @ down_w
       x_out = x_attn + mlp_out

    9. LM head + cross-entropy-shape loss
       logits = x_out @ lm_head_w
       loss   = sum(labels * log(softmax(logits)))    # scalar

Note that Tlaloc's "loss" is unsigned `Σ labels · log p` (i.e., negative
cross-entropy / log-likelihood — no minus sign). The Python reference
matches that sign convention so the numbers compare directly.
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


def forward(arrs: dict) -> float:
    import torch
    import torch.nn.functional as F

    def t(name: str) -> "torch.Tensor":
        return torch.from_numpy(arrs[name].astype(np.float32))

    x_in = t("x_in")
    labels = t("labels")
    theta = t("theta")
    q_w, k_w, v_w, out_w = t("q_w"), t("k_w"), t("v_w"), t("out_w")
    gate_w, up_w, down_w = t("gate_w"), t("up_w"), t("down_w")
    lm_head_w = t("lm_head_w")
    eps_attn, eps_mlp = t("eps_attn"), t("eps_mlp")

    # Pre-attention RmsNorm (eps form)
    sq1 = x_in * x_in
    mean1 = sq1.mean(dim=1, keepdim=True)
    rsq1 = torch.rsqrt(mean1 + eps_attn)
    x_norm_attn = x_in * rsq1

    # Q, K, V projections
    q = x_norm_attn @ q_w
    k = x_norm_attn @ k_w
    v = x_norm_attn @ v_w

    # RoPE on Q only (theta is the imag-half proxy in this primal — see kt comments)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    q_rot = (q * cos_t) - (theta * sin_t)

    # Attention with pre-transposed K (matches §0.4.286's TRANSPOSE(k, [1,0]))
    k_t = k.transpose(0, 1)
    s = q_rot @ k_t
    p = torch.softmax(s, dim=-1)
    attn_out = p @ v

    # Output projection + residual against UNNORMALIZED x_in
    attn_proj = attn_out @ out_w
    x_attn = x_in + attn_proj

    # Pre-MLP RmsNorm (eps form, on x_attn)
    sq2 = x_attn * x_attn
    mean2 = sq2.mean(dim=1, keepdim=True)
    rsq2 = torch.rsqrt(mean2 + eps_mlp)
    x_norm_mlp = x_attn * rsq2

    # SwiGLU MLP
    gate_proj = x_norm_mlp @ gate_w
    up_proj = x_norm_mlp @ up_w
    swiglu = F.silu(gate_proj) * up_proj

    # Down + residual against UNNORMALIZED x_attn
    mlp_out = swiglu @ down_w
    x_out = x_attn + mlp_out

    # LM head + (negative) cross-entropy loss
    logits = x_out @ lm_head_w
    probs = torch.softmax(logits, dim=-1)
    logp = torch.log(probs)
    loss = (labels * logp).sum()

    return float(loss.item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    arrs = _load_inputs(args.inputs_dir)
    loss = forward(arrs)

    args.output.write_text(json.dumps({"loss": loss}))
    print(f"[run_pytorch_llama] loss={loss:.9g}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
