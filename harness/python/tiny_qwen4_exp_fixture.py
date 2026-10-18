#!/usr/bin/env python3
"""A tiny random Qwen4-Exp (Qwen3.8-Flash-Next) checkpoint and its greedy oracle, for the
qwen4_exp family's parity test. Four layers (three Gated DeltaNet, the second with the
n-gram PLE layer, then one QSA-indexed attention layer), four residual streams, a mixture of
8 experts. The indexer keeps 2 blocks of 4 tokens, so prompts past 11 tokens attend to a
selection; a second prompt carries the eos id, which resets the n-gram history. Builds
Qwen4ExpForConditionalGeneration from the tiny config (seeded), saves it with save_pretrained
(float32 safetensors), and greedy-decodes the prompts in float32 on the CPU with eager
attention. Every indexer top-k is checked to have a gap at its boundary, so the selection
does not depend on how ties break. Run in ~/.local/venvs/vllm:

    python tiny_qwen4_exp_fixture.py ir/src/jvmTest/resources/io/tlaloc/ir/inference/tiny_qwen4_exp
"""
import json
import sys
from pathlib import Path

import torch
from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration
from transformers.models.qwen4_exp import modeling_qwen4_exp

EOS = 1


def checked_topk(original):
    """torch.Tensor.topk that refuses a selection whose k-th and (k+1)-th scores are close."""
    def topk(self, k, dim=-1, *args, **kwargs):
        if self.dim() == 1 and k < self.shape[0]:
            s = self.sort(descending=True).values
            gap = (s[k - 1] - s[k]).item()
            assert gap > 1e-3 * max(1.0, abs(s[k - 1].item())), f"indexer top-{k} boundary gap {gap}: {s.tolist()}"
        return original(self, k, dim, *args, **kwargs)
    return topk


def main():
    out = Path(sys.argv[1])
    torch.manual_seed(20261018)
    text = dict(
        vocab_size=64, hidden_size=16, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_conv_kernel_dim=4, linear_key_head_dim=4, linear_value_head_dim=4, linear_num_key_heads=2,
        linear_num_value_heads=4, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=8,
        shared_expert_intermediate_size=8, max_position_embeddings=128, rms_norm_eps=1e-6, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                         "mrope_interleaved": True, "mrope_section": [1, 1, 0]},
        hc_count=4, hc_lowrank=6, ple_layer_ids=[2], ple_embed_dim=16, ple_conv_kernel_size=4, ngram_size=3,
        heads_per_ngram=2, ngram_vocab_size_base=50, make_ngram_vocab_size_divisible_by=8, seed=1234,
        split_ngram_parts=2, indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=8, indexer_budget=8,
        indexer_compress_ratio=4, output_gate_type="sigmoid", eos_token_id=EOS, bos_token_id=EOS,
    )
    vision = dict(depth=1, hidden_size=16, intermediate_size=16, num_heads=2, out_hidden_size=16,
                  patch_size=2, spatial_merge_size=1, temporal_patch_size=1, num_position_embeddings=16)
    cfg = Qwen4ExpConfig(text_config=text, vision_config=vision, image_token_id=62, video_token_id=63,
                         tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    model = Qwen4ExpForConditionalGeneration(cfg).float().eval()
    with torch.no_grad():
        # Non-trivial norms, decays, gates and n-gram rows: the init leaves some at constants or zero.
        for name, p in model.named_parameters():
            if p.dim() == 1:
                p.copy_(torch.randn_like(p) * 0.3)
            elif "ple.conv1d" in name or "ngram_embedding" in name:
                p.copy_(torch.randn_like(p) * 0.5)
            elif "hyper_connection" in name or "indexer" in name or ".ple." in name:
                p.copy_(torch.randn_like(p) * 0.3)
    model.save_pretrained(out, safe_serialization=True)
    prompts = [
        [3, 5, 9, 2, 33, 7],
        [3, 4, 50, 11, 12, 13, 14, 20, 21, 22, 40, EOS, 41, 42, 43, 8, 9, 10, 44, 45, 46, 47],
    ]
    fixture = {"oracle": "transformers Qwen4ExpForConditionalGeneration, float32, CPU, eager, greedy",
               "maxNew": 12, "prompts": []}
    original = torch.Tensor.topk
    indexer_forward = modeling_qwen4_exp.Qwen4ExpTextQSAIndexer.forward

    def guarded(self, *args, **kwargs):
        torch.Tensor.topk = checked_topk(original)
        try:
            return indexer_forward(self, *args, **kwargs)
        finally:
            torch.Tensor.topk = original
    modeling_qwen4_exp.Qwen4ExpTextQSAIndexer.forward = guarded
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
