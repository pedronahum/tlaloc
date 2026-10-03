#!/usr/bin/env python3
"""Oracle fixture for MOE_EXPERTS: transformers' Qwen3_5MoeTopKRouter and
Qwen3_5MoeExperts in float32 on the CPU, over random weights. Run in
~/.local/venvs/vllm:

    python moe_ops_fixture.py > ir/src/jvmTest/resources/io/tlaloc/ir/passes/moe_ops_fixture.json
"""
import json
import sys

import torch
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

R, H, E, I, K = 5, 6, 8, 4, 3


def main():
    torch.manual_seed(5)
    cfg = Qwen3_5MoeTextConfig(hidden_size=H, num_experts=E, num_experts_per_tok=K, moe_intermediate_size=I)
    router = m.Qwen3_5MoeTopKRouter(cfg)
    experts = m.Qwen3_5MoeExperts(cfg)
    with torch.no_grad():
        router.weight.copy_(torch.randn(E, H))
        experts.gate_up_proj.copy_(torch.randn(E, 2 * I, H) * 0.5)
        experts.down_proj.copy_(torch.randn(E, H, I) * 0.5)
        x = torch.randn(R, H)
        logits, scores, indices = router(x)
        y = experts(x, indices, scores)
    json.dump({
        "dims": {"R": R, "H": H, "E": E, "I": I, "K": K},
        "x": x.flatten().tolist(), "logits": logits.flatten().tolist(),
        "gateUp": experts.gate_up_proj.detach().flatten().tolist(),
        "down": experts.down_proj.detach().flatten().tolist(),
        "indices": indices.flatten().tolist(), "y": y.flatten().tolist(),
    }, sys.stdout)


if __name__ == "__main__":
    main()
