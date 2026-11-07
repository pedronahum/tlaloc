"""Oracle for the granite family: transformers' own GraniteForCausalLM on a tiny checkpoint.

Imports torch and transformers on purpose: this is the oracle side of a comparison,
not serving code. It runs in a venv that has both (for example ~/.local/venvs/vllm,
transformers 5.17) and never installs anything. The modeling code is the one
transformers ships (``transformers/models/granite/modeling_granite.py``); nothing is
downloaded.

The checkpoint is the committed ``tiny_granite`` fixture, which ``HfGraniteParityTest``
writes from a seeded ``java.util.Random`` and checks byte for byte (three layers, the
four Granite multipliers all away from 1):

  ~/.local/venvs/vllm/bin/python -I harness/python/granite_modeling_fixture.py \\
      --checkpoint ir/src/jvmTest/resources/io/tlaloc/ir/inference/tiny_granite \\
      --out ir/src/jvmTest/resources/io/tlaloc/ir/inference/granite_modeling_logits.json

The output holds the logits at every position of each prompt, in float32, with eager
attention.
"""

import argparse
import json
import os
import sys

PROMPTS = [
    [3, 5, 9, 12, 33, 7],
    [3, 4, 50, 11, 12, 13, 14, 20, 21, 22, 40, 41, 42, 43, 8, 9, 10, 44, 45],
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, help="the tiny checkpoint (config.json, model.safetensors)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    import transformers
    from safetensors.torch import load_file
    from transformers import GraniteConfig, GraniteForCausalLM

    with open(os.path.join(a.checkpoint, "config.json")) as f:
        cfg = json.load(f)
    config = GraniteConfig(**{k: v for k, v in cfg.items() if k not in ("architectures", "model_type")})
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = GraniteForCausalLM(config).float().eval()
    state = load_file(os.path.join(a.checkpoint, "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    params = {n for n, _ in model.named_parameters()}
    if [m for m in missing if m in params] or unexpected:
        sys.exit(f"checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    out = {"source": "transformers " + transformers.__version__ + " GraniteForCausalLM",
           "torch": torch.__version__, "prompts": []}
    with torch.no_grad():
        for ids in PROMPTS:
            logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0]
            out["prompts"].append({"ids": ids, "logits": [[float(x) for x in row] for row in logits.float()]})
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {a.out}: {len(PROMPTS)} prompts, vocab {logits.shape[-1]}")


if __name__ == "__main__":
    main()
