"""PyTorch reference for :nn's CausalLM: loss, gradients and an AdamW loss curve.

Reads the weights, token ids and targets the Kotlin test writes as .npy files
(names are the Kotlin parameter keys; Dense weights are [in, out]) and a
config.json, builds the same Llama-architecture model in plain torch
(RMSNorm, per-head q/k RMSNorm when the weights have it, rotate_half RoPE,
grouped-query causal attention, SwiGLU, optionally tied embeddings), and
writes JSON:

    {"loss": l, "grads": {key: [flat values]}, "losses": [...], "final_loss": l}

`grads` is the gradient of the step-0 loss. `losses[k]` is the loss before
AdamW update k, and `final_loss` the loss after the last update.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    torch.use_deterministic_algorithms(True)
    cfg = json.loads((args.inputs_dir / "config.json").read_text())
    keys = cfg["keys"]
    params = {
        k: torch.from_numpy(np.load(args.inputs_dir / f"{k}.npy").astype(np.float32)).requires_grad_(True)
        for k in keys
    }
    ids = torch.from_numpy(np.load(args.inputs_dir / "ids.npy").astype(np.int64))
    targets = torch.from_numpy(np.load(args.inputs_dir / "targets.npy").astype(np.int64))

    n_heads, n_kv, head_dim = cfg["numHeads"], cfg["numKvHeads"], cfg["headDim"]
    eps, theta, tied = cfg["normEps"], cfg["ropeTheta"], cfg["tiedEmbeddings"]
    batch, seq = ids.shape

    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(seq, dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()

    def rotate_half(x):
        x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def rms(x, w):
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * w

    causal = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)

    def forward():
        p = params
        h = p["embed.table"][ids]
        for i in range(cfg["numLayers"]):
            pre = f"blocks.{i}."
            x = rms(h, p[pre + "attnNorm.weight"])
            q = (x @ p[pre + "attn.q.w"]).view(batch, seq, n_heads, head_dim).transpose(1, 2)
            k = (x @ p[pre + "attn.k.w"]).view(batch, seq, n_kv, head_dim).transpose(1, 2)
            v = (x @ p[pre + "attn.v.w"]).view(batch, seq, n_kv, head_dim).transpose(1, 2)
            if pre + "attn.qNorm.weight" in p:
                q = rms(q, p[pre + "attn.qNorm.weight"])
                k = rms(k, p[pre + "attn.kNorm.weight"])
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            k = k.repeat_interleave(n_heads // n_kv, dim=1)
            v = v.repeat_interleave(n_heads // n_kv, dim=1)
            scores = (q @ k.transpose(-1, -2)) / (head_dim ** 0.5)
            scores = scores.masked_fill(causal, float("-inf"))
            attn = torch.softmax(scores, dim=-1) @ v
            attn = attn.transpose(1, 2).reshape(batch, seq, n_heads * head_dim)
            h = h + attn @ p[pre + "attn.o.w"]
            x = rms(h, p[pre + "mlpNorm.weight"])
            gate = x @ p[pre + "mlp.gate.w"]
            h = h + (F.silu(gate) * (x @ p[pre + "mlp.up.w"])) @ p[pre + "mlp.down.w"]
        h = rms(h, p["norm.weight"])
        logits = h @ (p["embed.table"].T if tied else p["head.w"])
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)

    loss = forward()
    loss.backward()
    grads = {k: params[k].grad.detach().numpy().reshape(-1).tolist() for k in keys}
    step0 = float(loss.item())

    for t in params.values():
        t.grad = None
    opt = torch.optim.AdamW(
        [params[k] for k in keys], lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg["weightDecay"]
    )
    losses = []
    for _ in range(cfg["steps"]):
        opt.zero_grad()
        loss = forward()
        losses.append(float(loss.item()))
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = float(forward().item())

    args.output.write_text(json.dumps({"loss": step0, "grads": grads, "losses": losses, "final_loss": final}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
