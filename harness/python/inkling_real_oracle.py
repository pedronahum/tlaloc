"""Oracle for the released Inkling-2-Mini: transformers' modeling code on the real weights.

Imports torch and transformers on purpose: this is the oracle side of a comparison, not
serving code. It runs in a venv with transformers 5.24.0 and a CUDA torch (for example
~/.local/venvs/inkling) and never installs anything. transformers dequantizes the NVFP4 and
MXFP8 experts to bf16 on load (``Inkling2LoopDequantizeExperts``); the model runs in bf16
with eager attention, about 55 GB on the device.

  ~/.local/venvs/inkling/bin/python -I harness/python/inkling_real_oracle.py \\
      --model ~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/<rev> \\
      --out build/inkling-oracle.json

Writes the chat prompt's ids, the logits at its last position (float32), and the greedy
continuation, which PjrtInklingTest.releasedModelMatchesTheModelingCode compares with.
"""

import argparse
import json
import sys
import time

import torch
import transformers
from transformers import AutoTokenizer
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForConditionalGeneration

PINNED_TRANSFORMERS = "5.24.0"
MESSAGES = [{"role": "user", "content": "Name three boroughs of New York City, in one sentence."}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--new", type=int, default=24)
    a = ap.parse_args()
    if transformers.__version__ != PINNED_TRANSFORMERS:
        sys.exit(f"transformers {transformers.__version__}, not the pinned {PINNED_TRANSFORMERS}")
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok.apply_chat_template(MESSAGES, add_generation_prompt=True, tokenize=True)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    ids = [int(i) for i in ids]
    t0 = time.time()
    model = Inkling2LoopForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda").eval()  # loaded on the host, then moved: device_map needs accelerate
    print(f"loaded in {time.time() - t0:.0f} s; {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    with torch.no_grad():
        x = torch.tensor([ids], device="cuda")
        out = model(input_ids=x, use_cache=False)
        last = out.logits[0, -1].float().cpu()
        gen = model.generate(x, max_new_tokens=a.new, do_sample=False, use_cache=True)
    greedy = [int(t) for t in gen[0, len(ids):]]
    with open(a.out, "w") as f:
        json.dump({
            "transformers": transformers.__version__, "torch": torch.__version__, "messages": MESSAGES,
            "ids": ids, "last_logits": [float(v) for v in last], "greedy": greedy,
            "greedy_text": tok.decode(greedy),
        }, f)
    print("prompt", len(ids), "tokens; greedy:", repr(tok.decode(greedy)))


if __name__ == "__main__":
    main()
