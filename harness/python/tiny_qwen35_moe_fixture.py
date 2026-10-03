#!/usr/bin/env python3
"""A tiny random Qwen3.5-MoE checkpoint and its greedy oracle, for the qwen3_5_moe
family's parity test. Builds Qwen3_5MoeForConditionalGeneration from a tiny config
(seeded), saves it with save_pretrained (float32 safetensors), and greedy-decodes two
prompts in float32 on the CPU with eager attention. Run in ~/.local/venvs/vllm:

    python tiny_qwen35_moe_fixture.py ir/src/jvmTest/resources/io/tlaloc/ir/inference/tiny_qwen3_5_moe
"""
import json
import sys
from pathlib import Path

import torch
from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration


def main():
    out = Path(sys.argv[1])
    torch.manual_seed(20261003)
    text = dict(
        vocab_size=64, hidden_size=16, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_conv_kernel_dim=4, linear_key_head_dim=4, linear_value_head_dim=4, linear_num_key_heads=2,
        linear_num_value_heads=4, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=8,
        shared_expert_intermediate_size=8, max_position_embeddings=128, rms_norm_eps=1e-6, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                         "mrope_interleaved": True, "mrope_section": [1, 1, 0]},
    )
    vision = dict(depth=1, hidden_size=16, intermediate_size=16, num_heads=2, out_hidden_size=16,
                  patch_size=2, spatial_merge_size=1, temporal_patch_size=1, num_position_embeddings=16)
    cfg = Qwen3_5MoeConfig(text_config=text, vision_config=vision, image_token_id=62, video_token_id=63,
                           tie_word_embeddings=False)
    model = Qwen3_5MoeForConditionalGeneration(cfg).float().eval()
    with torch.no_grad():
        # Non-trivial norms, decays and gates: the random init leaves some of them at constants.
        for name, p in model.named_parameters():
            if p.dim() == 1:
                p.copy_(torch.randn_like(p) * 0.3)
    model.save_pretrained(out, safe_serialization=True)
    prompts = [[1, 5, 9, 2, 33, 7], [3, 4, 50, 11, 12, 13, 14, 20, 21, 22, 40]]
    fixture = {"oracle": "transformers Qwen3_5MoeForConditionalGeneration, float32, CPU, eager, greedy",
               "maxNew": 12, "prompts": []}
    with torch.no_grad():
        for ids in prompts:
            x = torch.tensor([ids])
            gen = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=12, do_sample=False,
                                 eos_token_id=None, pad_token_id=0, output_logits=True, return_dict_in_generate=True)
            new = gen.sequences[0, len(ids):].tolist()
            fixture["prompts"].append({
                "promptTokens": ids, "generatedTokens": new,
                "logits": [lg[0].float().tolist() for lg in gen.logits],
            })
    (out / "greedy.json").write_text(json.dumps(fixture))
    print([p["generatedTokens"] for p in fixture["prompts"]])


if __name__ == "__main__":
    main()
