#!/usr/bin/env python3
"""A tiny random Qwen4-Exp (Qwen3.8-Flash-Next) checkpoint and its greedy oracle, for the
qwen4_exp family's parity test. Four layers (three Gated DeltaNet, the second with the
n-gram PLE layer, then one QSA-indexed attention layer), four residual streams, a mixture of
8 experts. The indexer keeps 2 blocks of 4 tokens, so prompts past 11 tokens attend to a
selection; a second prompt carries the eos id, which resets the n-gram history. Builds
Qwen4ExpForConditionalGeneration from the tiny config (seeded), saves it with save_pretrained
(float32 safetensors), and greedy-decodes the prompts in float32 on the CPU with eager
attention. Every indexer top-k is checked to have a gap at its boundary, so the selection
does not depend on how ties break. Then a random MTP head goes into the checkpoint and its
drafts into mtp.json (see mtp_fixture). Run in ~/.local/venvs/vllm:

    python tiny_qwen4_exp_fixture.py ir/src/jvmTest/resources/io/tlaloc/ir/inference/tiny_qwen4_exp
"""
import copy
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration
from transformers.models.qwen4_exp import modeling_qwen4_exp

EOS = 1
DRAFTS = 3
MTP_SEED = 20261020


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
        mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=False,
        mtp={"hybrid": True, "layer_types": ["full_attention"], "mtp_use_hidden_state_from_layer": None,
             "num_hidden_layers": 1, "rope_theta": 10000.0},
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
    with torch.no_grad():
        mtp_fixture(model, out, fixture)


def mtp_fixture(model, out, fixture):
    """The MTP head (`mtp.*`), random, appended to model.safetensors, and its drafts in mtp.json.

    transformers does not run it, so it is built from transformers' own pieces as vLLM's
    Qwen4ExpMultiTokenPredictor wires them: at position p, from the token x at p + 1 and the
    target's four streams h at p (before the last mix),

        e        = fc_embedding(pre_fc_norm_embedding(embed(x)))
        stream i = fc_hidden(pre_fc_norm_hidden(h)[i]) + e     pre_fc_norm_hidden: one norm over all streams
        s        = decoder_layer(streams)                       an indexed attention layer, no n-gram memory
        logits   = lm_head(hyper_connection_mixer(s))           its own last mix

    A draft feeds the next one its token and s. For each position q from the prompt's end on,
    the drafts after the token at q: the head over x_1..x_q with the target's streams at
    0..q-1, then DRAFTS - 1 more steps; each draft's argmax is checked to lead the next logit by 1e-4.
    """
    torch.manual_seed(MTP_SEED)
    lm = model.model.language_model
    tc = model.config.text_config
    h, n = tc.hidden_size, tc.hc_count
    layer = copy.deepcopy(lm.layers[3])
    norm_e = modeling_qwen4_exp.Qwen4ExpTextRMSNorm(h, eps=tc.rms_norm_eps)
    norm_h = modeling_qwen4_exp.Qwen4ExpTextRMSNorm(n * h, eps=tc.rms_norm_eps)
    fc_e = torch.nn.Linear(h, h, bias=False)
    fc_h = torch.nn.Linear(h, h, bias=False)
    mixer = modeling_qwen4_exp.Qwen4ExpTextGatedResidual(tc, use_combine=False)
    for m in (layer, norm_e, norm_h, fc_e, fc_h, mixer):
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * (0.3 if p.dim() == 1 else 0.4))
    # The layer's tensors under the names save_pretrained gives layer 3's.
    target3 = lm.layers[3]
    lm.layers[3] = layer
    tmp = out / "mtp_tmp"
    model.save_pretrained(tmp, safe_serialization=True)
    lm.layers[3] = target3
    prefix = "model.language_model.layers.3."
    tensors = dict(load_file(out / "model.safetensors"))
    with safe_open(tmp / "model.safetensors", "pt") as f:
        for k in f.keys():
            if k.startswith(prefix):
                tensors["mtp.layers.0." + k[len(prefix):]] = f.get_tensor(k)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    tensors["mtp.pre_fc_norm_embedding.weight"] = norm_e.weight.detach().clone()
    tensors["mtp.pre_fc_norm_hidden.weight"] = norm_h.weight.detach().clone()
    tensors["mtp.fc_embedding.weight"] = fc_e.weight.detach().clone()
    tensors["mtp.fc_hidden.weight"] = fc_h.weight.detach().clone()
    for k, v in mixer.state_dict().items():
        tensors["mtp.hyper_connection_mixer." + k] = v.detach().clone()
    save_file({k: v.contiguous() for k, v in tensors.items()}, out / "model.safetensors", metadata={"format": "pt"})

    streams = {}
    lm.hyper_connection_mixer.register_forward_pre_hook(lambda mod, args: streams.__setitem__("s", args[0]))
    embed, head = lm.embed_tokens, model.lm_head

    def mtp_pass(tokens, hidden):
        t = tokens.shape[1]
        e = fc_e(norm_e(embed(tokens)))
        s = fc_h(norm_h(hidden).unflatten(-1, (n, h))) + e.unsqueeze(-2)
        pos = torch.arange(t).view(1, 1, -1).expand(3, 1, t)
        pe = lm.rotary_emb(s, pos)
        mask = torch.full((t, t), torch.finfo(torch.float32).min).triu(1).view(1, 1, t, t)
        y = layer(s.flatten(-2), position_embeddings=pe, attention_mask=mask)
        return y, head(mixer(y))

    def lead(logits):
        top = logits.topk(2).values
        gap = (top[0] - top[1]).item()
        assert gap > 1e-4, f"draft argmax leads by {gap}"
        return int(logits.argmax())

    # The indexer's top-k boundary check (main's) stays installed.
    mtp = {"oracle": "transformers' decoder layer, norms and gated residual with the mtp.* tensors, as vLLM's "
                     "Qwen4ExpMultiTokenPredictor wires them; float32, CPU, eager",
           "drafts": DRAFTS, "prompts": []}
    for p in fixture["prompts"]:
        seq = p["promptTokens"] + p["generatedTokens"]
        model.model.language_model(input_ids=torch.tensor([seq]))
        hid = streams["s"]
        rows = []
        for q in range(len(p["promptTokens"]), len(seq)):
            tokens = torch.tensor([seq[1:q + 1]])
            hidden = hid[:, :q]
            y, logits = mtp_pass(tokens, hidden)
            drafts = [lead(logits[0, -1])]
            for _ in range(DRAFTS - 1):
                tokens = torch.cat([tokens, torch.tensor([[drafts[-1]]])], dim=1)
                hidden = torch.cat([hidden, y[:, -1:]], dim=1)
                y, logits = mtp_pass(tokens, hidden)
                drafts.append(lead(logits[0, -1]))
            rows.append(drafts)
        mtp["prompts"].append({"promptTokens": p["promptTokens"], "drafts": rows})
        print("drafts after each position from", len(p["promptTokens"]), rows)
    (out / "mtp.json").write_text(json.dumps(mtp))


if __name__ == "__main__":
    main()
