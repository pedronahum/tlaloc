#!/usr/bin/env python3
"""The MTP head of Qwen/Qwen3.5-0.8B over two prompts, for the MTP parity test.

transformers does not run a Qwen3.5 checkpoint's `mtp.*` tensors, so this builds the
head from transformers' own pieces, as vLLM's Qwen3_5MultiTokenPredictor wires them:

    e      = pre_fc_norm_embedding(embed(token))
    m      = fc([e | pre_fc_norm_hidden(h)])          h: the target's final-norm hidden state
    out    = norm(decoder_layer(m))                   one full-attention Qwen3_5DecoderLayer
    logits = lm_head(out)

At position p the head reads the token at p + 1 and the target's hidden state at p (RoPE
position p) and predicts the token at p + 2. Over a prompt x_0..x_{n-1} with the target's
greedy next token g, the tokens are x_1..x_{n-1}, g; two more draft steps feed each draft
back with the head's own output as the hidden state. Float32, CPU, eager attention.

Run in ~/.local/venvs/vllm:
    python qwen35_mtp_fixture.py ir/src/jvmTest/resources/io/tlaloc/ir/inference/qwen3_5_0_8b_mtp.json
"""
import copy
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_5 import modeling_qwen3_5 as q35

REPO = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
TOP_K = 20
DRAFTS = 3


def main():
    out = Path(sys.argv[1])
    path = Path(snapshot_download(REPO, revision=REVISION))
    model = AutoModelForCausalLM.from_pretrained(path, revision=REVISION, torch_dtype=torch.float32,
                                                 attn_implementation="eager").eval()
    lm = model.model
    tc = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    tc = copy.deepcopy(tc)
    tc.layer_types = ["full_attention"] * tc.num_hidden_layers
    tc._attn_implementation = "eager"
    layer = q35.Qwen3_5DecoderLayer(tc, layer_idx=0).float().eval()
    h = tc.hidden_size
    fc = torch.nn.Linear(2 * h, h, bias=False)
    norms = {n: q35.Qwen3_5RMSNorm(h, eps=tc.rms_norm_eps) for n in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm")}
    mtp = {}
    for f in path.glob("*.safetensors"):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if k.startswith("mtp."):
                    mtp[k[len("mtp."):]] = s.get_tensor(k).float()
    with torch.no_grad():
        fc.weight.copy_(mtp["fc.weight"])
        for n, m in norms.items():
            m.weight.copy_(mtp[f"{n}.weight"])
        layer.load_state_dict({k[len("layers.0."):]: v for k, v in mtp.items() if k.startswith("layers.0.")})
    embed = lm.get_input_embeddings() if hasattr(lm, "get_input_embeddings") else model.get_input_embeddings()
    head = model.get_output_embeddings()
    text_model = lm.language_model if hasattr(lm, "language_model") else lm

    def mtp_pass(tokens, hidden):
        t = tokens.shape[1]
        x = fc(torch.cat([norms["pre_fc_norm_embedding"](embed(tokens)), norms["pre_fc_norm_hidden"](hidden)], dim=-1))
        pos = torch.arange(t).view(1, 1, -1).expand(3, 1, t)
        pe = text_model.rotary_emb(x, pos)
        mask = torch.full((t, t), float("-inf")).triu(1).view(1, 1, t, t)
        y = layer(x, position_embeddings=pe, attention_mask=mask, position_ids=pos[0])
        y = y[0] if isinstance(y, tuple) else y
        o = norms["norm"](y)
        return o, head(o)

    fixture = {"checkpoint": REPO, "revision": REVISION,
               "oracle": "transformers Qwen3_5DecoderLayer + the checkpoint's mtp.* tensors, as vLLM wires them; float32, CPU, eager",
               "topK": TOP_K, "drafts": DRAFTS, "prompts": []}
    prompts = [[561, 6511, 314, 9338, 369], [248045, 846, 198, 3838, 374, 279, 6511, 314, 9338, 30, 248046, 198]]
    with torch.no_grad():
        for ids in prompts:
            x = torch.tensor([ids])
            hid = text_model(input_ids=x).last_hidden_state
            g = int(head(hid[:, -1]).argmax(-1))
            tokens = torch.tensor([ids[1:] + [g]])
            hidden = hid
            outs, logits = mtp_pass(tokens, hidden)
            drafts = [int(logits[0, -1].argmax())]
            for _ in range(DRAFTS - 1):
                tokens = torch.cat([tokens, torch.tensor([[drafts[-1]]])], dim=1)
                hidden = torch.cat([hidden, outs[:, -1:]], dim=1)
                outs, logits = mtp_pass(tokens, hidden)
                drafts.append(int(logits[0, -1].argmax()))
            n = len(ids)
            first = logits[0, :n]
            top = first.topk(TOP_K, dim=-1)
            fixture["prompts"].append({
                "promptTokens": ids, "targetNext": g, "drafts": drafts,
                "argmax": first.argmax(-1).tolist(),
                "topKIndices": top.indices.tolist(), "topKValues": top.values.tolist(),
            })
            print(ids, "target next", g, "drafts", drafts)
    out.write_text(json.dumps(fixture))


if __name__ == "__main__":
    main()
