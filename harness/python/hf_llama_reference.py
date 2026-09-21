#!/usr/bin/env python3
"""§0.4.479 (Phase H3c-2) — THE ORACLE for the real-Llama decode graph.

This script is **not serving-path code** and it says so loudly: it imports
``torch`` and ``transformers`` ON PURPOSE, and it runs in the vLLM venv
(``~/.local/venvs/vllm``: torch 2.13.0+cu130, transformers 5.17.0, no jax).
The §0.4.474 rail is about the frozen ORACLE venv and about what the RUNTIME
imports; a reference implementation is allowed — indeed required — to be the
thing we are claiming parity with.

What it does
------------

Load a HuggingFace Llama checkpoint, optionally **reduced to its first N
layers**, run one forward pass over a token sequence in float32 on CPU, and
print the logits (and the argmax per position) as JSON on stdout.

The reduction is arithmetic, not approximate: ``LlamaForCausalLM`` with
``num_hidden_layers = N`` over the checkpoint's real layers ``0 .. N-1``, the
real embedding table, the real final norm and the real head, is *exactly* the
model Tlaloc's graph builds when handed ``config.copy(numLayers = N)``. It is
not TinyLlama; it is a well-defined 2-layer Llama whose every weight came out
of TinyLlama's file, and that is the thing both sides compute.

float32 on CPU, deliberately
----------------------------

The checkpoint's bytes are bf16. ``bf16 -> f32`` is exact (a 16-bit shift), so
running the reference in f32 reads the checkpoint's real numbers and makes the
comparison one about ARITHMETIC ORDER rather than about width — which is the
comparison Tlaloc's f32 graph can actually be held to. CPU because a CUDA
matmul would add TF32 to the same question; the GPU lane's own floor is the
serving lane's 1e-3 and is named where it is measured.

Usage::

    python hf_llama_reference.py --checkpoint DIR --tokens 1 2 3 --layers 2

Output JSON::

    {"config": {...}, "tokens": [...], "logits": [[...], ...],
     "argmax": [...], "top5": [[...], ...]}
"""

import argparse
import json
import sys


def _rope_theta(cfg) -> float:
    theta = getattr(cfg, "rope_theta", None)
    if theta is None:
        params = getattr(cfg, "rope_parameters", None) or {}
        theta = params.get("rope_theta")
    if theta is None:
        raise SystemExit("hf_llama_reference: this config states no rope theta")
    return float(theta)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokens", required=True, nargs="+", type=int)
    ap.add_argument("--layers", type=int, default=0, help="0 = all layers")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    import torch  # noqa: PLC0415 - oracle-only import, see the module docstring
    from transformers import AutoConfig, AutoModelForCausalLM  # noqa: PLC0415

    cfg = AutoConfig.from_pretrained(args.checkpoint)
    full_layers = cfg.num_hidden_layers
    if args.layers:
        cfg.num_hidden_layers = args.layers

    # from_pretrained with a modified config keeps only the layers the config
    # names; transformers reports the rest as unused, which is the whole point
    # of a reduced slice and not a warning worth suppressing.
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, config=cfg, dtype=torch.float32
    )
    model.eval()

    ids = torch.tensor([args.tokens], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    logits = out.logits[0].to(torch.float32)

    topk = torch.topk(logits, k=min(args.topk, logits.shape[-1]), dim=-1)
    payload = {
        "config": {
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "full_num_hidden_layers": full_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "vocab_size": cfg.vocab_size,
            "rms_norm_eps": float(cfg.rms_norm_eps),
            # transformers 5.x moved `rope_theta` under `rope_parameters`;
            # both spellings are read so the recorded oracle config states
            # the theta that was actually used rather than omitting it.
            "rope_theta": _rope_theta(cfg),
        },
        "torch_version": torch.__version__,
        "tokens": args.tokens,
        "logits": logits.tolist(),
        "argmax": logits.argmax(dim=-1).tolist(),
        "topk_indices": topk.indices.tolist(),
        "topk_values": topk.values.tolist(),
    }
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
