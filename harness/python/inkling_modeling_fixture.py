"""Oracle for Inkling 2 (looped): transformers' own modeling code on a tiny random checkpoint.

INKLING_QUANT_ONLY=1 (argument: the output directory) writes only the expert-dequantization
fixture; INKLING_GRADS_ONLY=1 only grads.json, from the checkpoint already there.

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

The tiny model has eight layers in blocks of two, layers 2-5 looped three times, a
full-attention layer inside and after the loop, a sliding window of 8, a relative extent
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
        num_hidden_layers=8,
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
        # full attention at layers 3 (inside the loop) and 7 (after it)
        layer_types=["hybrid_sliding", "hybrid_sliding", "hybrid_sliding", "hybrid",
                     "hybrid_sliding", "hybrid_sliding", "hybrid_sliding", "hybrid"],
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
        # two blocks inside the loop, so a later pass's messages mix two states
        loop_config={"core_start": 2, "core_end": 6, "num_passes": 3, "gate_rank": 5},
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


if __name__ == "__main__" and not os.environ.get("INKLING_QUANT_ONLY") and not os.environ.get("INKLING_GRADS_ONLY"):
    main()


def quantized_experts_fixture(out_dir, seed):
    """Random NVFP4 and MXFP8 expert stacks in the released checkpoint's layout, and what
    transformers' Inkling2LoopDequantizeExperts makes of them (float32)."""
    from safetensors.torch import save_file
    from transformers.core_model_loading import Inkling2LoopDequantizeExperts

    g = torch.Generator().manual_seed(seed)
    e, rows, cols = 3, 4, 64
    nv = {
        "nv.w": torch.randint(0, 256, (e, rows, cols // 2), generator=g, dtype=torch.int32).to(torch.uint8),
        "nv.w_scale": (torch.rand((e, rows, cols // 16), generator=g) * 4 + 0.05).to(torch.float8_e4m3fn),
        "nv.w_amax": torch.rand((e,), generator=g) * 3 + 0.5,
        "nv.w_outer_scale_denominator": torch.tensor([2688.0]),
    }
    mx = {
        "mx.w": (torch.randn((e, rows, cols), generator=g) * 3).to(torch.float8_e4m3fn),
        "mx.w_scale": torch.randint(118, 134, (e, rows, cols // 32), generator=g, dtype=torch.int32).to(torch.uint8),
    }
    op = Inkling2LoopDequantizeExperts()
    pats = ["w", "w_scale", "w_amax", "w_outer_scale_denominator"]
    nv_out = op.convert({p: nv["nv." + p] for p in pats}, pats, ["out"])["out"].float()
    mx_out = op.convert({"w": mx["mx.w"], "w_scale": mx["mx.w_scale"], "w_amax": None, "w_outer_scale_denominator": None},
                        pats, ["out"])["out"].float()
    save_file({**nv, **mx}, os.path.join(out_dir, "quantized_experts.safetensors"))
    with open(os.path.join(out_dir, "quantized_experts.json"), "w") as f:
        json.dump({"experts": e, "rows": rows, "cols": cols,
                   "nvfp4": [float(v) for v in nv_out.flatten()], "mxfp8": [float(v) for v in mx_out.flatten()]}, f)


if __name__ == "__main__" and os.environ.get("INKLING_QUANT_ONLY"):
    quantized_experts_fixture(sys.argv[1], 20261124)


GRAD_PARAMS = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.relative_position_bias.proj",
    "model.layers.1.post_attention_convolution.conv1d.weight",
    "model.layers.2.mlp.experts.down_proj",
    "model.layers.2.q_message_gate.down_proj.weight",
    "model.layers.3.attention_input.query",
    "model.layers.3.self_attn.q_proj.linear.weight",
    "model.layers.4.mlp.gate.weight",
    "model.layers.7.self_attn.k_proj.sconv.static_weight",
    "model.message_input.query",
    "lm_head.weight",
]


def grads_fixture(out_dir):
    """The mean next-token cross entropy of the second prompt on the committed tiny checkpoint, and
    its gradient with respect to GRAD_PARAMS (torch autograd, float32, eager attention)."""
    torch.use_deterministic_algorithms(True)
    model = Inkling2LoopForCausalLM.from_pretrained(out_dir, dtype=torch.float32, attn_implementation="eager")
    model.train(False)
    ids = torch.tensor([PROMPTS[1]])
    logits = model(input_ids=ids, use_cache=False).logits[0].float()
    loss = torch.nn.functional.cross_entropy(logits[:-1], ids[0, 1:])
    loss.backward()
    params = dict(model.named_parameters())
    out = {"ids": PROMPTS[1], "loss": float(loss), "grads": {
        n: {"shape": list(params[n].shape), "values": [float(v) for v in params[n].grad.flatten()]} for n in GRAD_PARAMS}}
    with open(os.path.join(out_dir, "grads.json"), "w") as f:
        json.dump(out, f)
    print("loss", float(loss))


if __name__ == "__main__" and os.environ.get("INKLING_GRADS_ONLY"):
    grads_fixture(sys.argv[1])
