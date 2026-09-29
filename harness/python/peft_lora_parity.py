"""PEFT side of the LoRA parity test (nn/src/jvmTest/.../HfLoraAdapterTest.kt).

    python peft_lora_parity.py read  BASE_DIR ADAPTER_DIR IDS_JSON OUT_JSON
    python peft_lora_parity.py write BASE_DIR ADAPTER_DIR IDS_JSON OUT_JSON
    python peft_lora_parity.py plain BASE_DIR -           IDS_JSON OUT_JSON

`read` loads the base checkpoint with transformers (f32, eager attention),
applies the adapter in ADAPTER_DIR with PeftModel.from_pretrained, and writes
the logits for the token ids, then the logits of merge_and_unload().

`plain` loads BASE_DIR with transformers alone (ADAPTER_DIR is ignored) and
writes its logits: a merged checkpoint needs no PEFT to be read.

`write` adds a fresh LoRA adapter with PEFT itself (q_proj and v_proj, r=2,
alpha=4, random B so the adapter is not the identity), saves it to
ADAPTER_DIR with save_pretrained, and writes its logits.

Needs torch, transformers and peft; nothing else.
"""

import json
import sys

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM


def logits(model, ids):
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], dtype=torch.long)).logits
    return out.reshape(-1).tolist()


def main():
    mode, base_dir, adapter_dir, ids_path, out_path = sys.argv[1:6]
    with open(ids_path) as f:
        ids = json.load(f)
    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32, attn_implementation="eager")
    base.eval()
    result = {}
    if mode == "read":
        model = PeftModel.from_pretrained(base, adapter_dir)
        model.eval()
        result["adapted"] = logits(model, ids)
        result["merged"] = logits(model.merge_and_unload(), ids)
    elif mode == "plain":
        result["adapted"] = logits(base, ids)
    elif mode == "write":
        config = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], lora_dropout=0.0,
                            init_lora_weights=False, task_type="CAUSAL_LM")
        model = get_peft_model(base, config)
        model.eval()
        model.save_pretrained(adapter_dir)
        result["adapted"] = logits(model, ids)
    else:
        raise SystemExit(f"unknown mode {mode}")
    with open(out_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
