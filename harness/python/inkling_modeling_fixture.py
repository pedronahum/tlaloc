"""Oracle for Inkling 2 (looped): transformers' own modeling code on a tiny random checkpoint.

Imports torch and transformers on purpose: this is the oracle side of a comparison, not
serving code. It runs in a venv with transformers 5.24.0 (the first release with
``inkling2_loop``), for example ~/.local/venvs/inkling, and never installs anything.

It builds a tiny ``Inkling2LoopForCausalLM`` (float32, seeded), saves it with
``save_pretrained`` (which writes the checkpoint's original tensor names, as the released
Inkling-2-Mini files have them: ``model.llm.*``, interleaved ``w13``, the split message
queries), loads the saved files back with ``from_pretrained`` (the released loading path)
and records the logits of each prompt at every position, with eager attention:

  ~/.local/venvs/inkling/bin/python -I harness/python/inkling_modeling_fixture.py \\
      --out nn/src/jvmTest/resources/io/tlaloc/nn/tiny_inkling

The tiny model has six layers in blocks of two, the middle block looped three times, a
full-attention layer inside and outside the loop, a sliding window of 8, a relative extent
of 6, log scaling from position 3, and clip thresholds low enough to clip.
"""

import argparse
import json
import math
import os
import sys

import torch
import transformers
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM

PINNED_TRANSFORMERS = "5.24.0"

PROMPTS = [
    [3, 5, 9, 12, 33, 7],
    [3, 4, 50, 11, 12, 13, 14, 20, 21, 22, 40, 41, 42, 43, 8, 9, 10, 44, 45],
]


def tiny_config():
    return Inkling2LoopTextConfig(
        vocab_size=136,
        unpadded_vocab_size=128,
        hidden_size=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=8,
        sliding_window_size=8,
        d_rel=4,
        rel_extent=6,
        log_scaling_n_floor=4,
        log_scaling_alpha=0.1,
        # full attention at layers 3 (inside the loop) and 5 (after it)
        layer_types=["hybrid_sliding", "hybrid_sliding", "hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid"],
        rms_norm_eps=1e-6,
        conv_kernel_size=4,
        moe_intermediate_size=12,
        n_routed_experts=8,
        num_experts_per_tok=3,
        route_scale=2.5,
        qk_l2_clip_threshold=1.5,
        rel_l2_clip_threshold=0.8,
        n_layers_per_attn_res_block=2,
        mlp_gate_softcap=1.5,
        mlp_up_softcap=2.0,
        final_logit_softcapping=6.0,
        loop_config={"core_start": 2, "core_end": 4, "num_passes": 3, "gate_rank": 5},
        eos_token_id=1,
        tie_word_embeddings=False,
    )


def randomize(model, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if name.endswith("norm.weight") or name.endswith("layernorm.weight") or name.endswith("embed_norm.weight"):
                p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
            elif name.endswith(".scale"):
                p.copy_(1.0 + 0.3 * torch.randn(p.shape, generator=g))
            else:
                fan_in = p.shape[-1] if p.dim() > 1 else 1
                p.copy_(torch.randn(p.shape, generator=g) * (1.2 / math.sqrt(fan_in)))
        for name, b in sorted(model.named_buffers()):
            if name.endswith("e_score_correction_bias"):
                b.copy_(0.3 * torch.randn(b.shape, generator=g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261123)
    a = ap.parse_args()
    if transformers.__version__ != PINNED_TRANSFORMERS:
        sys.exit(f"transformers {transformers.__version__}, not the pinned {PINNED_TRANSFORMERS}: read the new modeling code first")
    torch.use_deterministic_algorithms(True)
    cfg = tiny_config()
    cfg._attn_implementation = "eager"
    model = Inkling2LoopForCausalLM(cfg).to(torch.float32).eval()
    randomize(model, a.seed)
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)

    loaded = Inkling2LoopForCausalLM.from_pretrained(a.out, dtype=torch.float32, attn_implementation="eager").eval()
    # The saved files must load back to the same parameters, or the names were not understood.
    want = dict(model.state_dict())
    got = dict(loaded.state_dict())
    assert want.keys() == got.keys(), sorted(set(want) ^ set(got))
    for k in want:
        assert torch.equal(want[k], got[k]), k

    results = []
    with torch.no_grad():
        for ids in PROMPTS:
            out = loaded(input_ids=torch.tensor([ids]), use_cache=False)
            logits = out.logits[0].float()
            # Greedy continuation through the cache (the decode path), recorded for later checks.
            gen = loaded.generate(torch.tensor([ids]), max_new_tokens=6, do_sample=False, use_cache=True)
            results.append({
                "ids": ids,
                "logits": [[float(v) for v in row] for row in logits],
                "greedy": [int(t) for t in gen[0, len(ids):]],
            })
    with open(os.path.join(a.out, "logits.json"), "w") as f:
        json.dump({"transformers": transformers.__version__, "torch": torch.__version__, "seed": a.seed, "prompts": results}, f)
    names = sorted(json.load(open(os.path.join(a.out, "model.safetensors.index.json")))["weight_map"]) if os.path.exists(
        os.path.join(a.out, "model.safetensors.index.json")) else None
    print("wrote", a.out, "tensors:", len(names) if names else "single file")


if __name__ == "__main__":
    main()
