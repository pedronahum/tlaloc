"""Write a greedy-decoding fixture for a HuggingFace checkpoint.

Imports torch and transformers on purpose: this is the oracle side of a
comparison, not serving code. It runs in a venv that has both (for example
~/.local/venvs/vllm) and never installs anything.

For each prompt it greedy-decodes a fixed number of tokens with
AutoModelForCausalLM in float32 on the CPU with eager attention, and records:

- the prompt ids and the generated ids,
- the top-k logits (indices and values) of the first generated position,
- the logit of the chosen token at every generated position, and the
  margin between the best and second-best logit there.

A prompt is either plain text (tokenized as given) or a chat message rendered
with the checkpoint's own chat template. The output is small JSON meant to be
committed as a test fixture; the weights are not.

    ~/.local/venvs/vllm/bin/python harness/python/hf_greedy_fixture.py \\
        --checkpoint Qwen/Qwen3-0.6B --max-new 16 --top-k 20 \\
        --text " The capital of France is" \\
        --chat "What is the capital of France? Answer in one word." \\
        --output ir/src/jvmTest/resources/io/tlaloc/ir/inference/qwen3_0_6b_greedy.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="HF repo id or local directory")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--text", action="append", default=[], help="a plain-text prompt")
    ap.add_argument("--chat", action="append", default=[],
                    help="a user message rendered with the chat template")
    ap.add_argument("--no-thinking", action="store_true",
                    help="pass enable_thinking=False to the chat template (Qwen3)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.float32, attn_implementation="eager",
    )
    model.eval()

    prompts = []
    for text in args.text:
        prompts.append({"kind": "text", "input": text,
                        "ids": tok(text, add_special_tokens=True).input_ids})
    for msg in args.chat:
        kw = {"enable_thinking": False} if args.no_thinking else {}
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, **kw,
        )
        prompts.append({"kind": "chat", "input": msg, "rendered": rendered,
                        "ids": tok(rendered, add_special_tokens=False).input_ids})

    results = []
    with torch.no_grad():
        for p in prompts:
            ids = torch.tensor([p["ids"]])
            gen = model.generate(
                ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new,
                do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
                # A fixed budget: EOS does not cut the compared prefix short.
                eos_token_id=None, pad_token_id=tok.pad_token_id or 0,
                output_logits=True, return_dict_in_generate=True,
            )
            new = gen.sequences[0, len(p["ids"]):].tolist()
            step_logits = [lg[0].float() for lg in gen.logits]
            first = step_logits[0]
            vals, idx = torch.topk(first, args.top_k)
            entry = dict(p)
            entry["promptTokens"] = entry.pop("ids")
            entry["generatedTokens"] = new
            entry["generatedText"] = tok.decode(new, skip_special_tokens=False)
            entry["step1TopKIndices"] = idx.tolist()
            entry["step1TopKValues"] = [float(v) for v in vals]
            entry["chosenLogits"] = [float(step_logits[i][t]) for i, t in enumerate(new)]
            # top-1 minus top-2 at each position: how far a numerically
            # different implementation is from choosing another token.
            entry["margins"] = [float(torch.topk(lg, 2).values[0] - torch.topk(lg, 2).values[1])
                                for lg in step_logits]
            results.append(entry)

    cfg = model.config
    out = {
        "checkpoint": args.checkpoint,
        "revision": getattr(cfg, "_commit_hash", None),
        "oracle": "transformers AutoModelForCausalLM, float32, CPU, eager attention, greedy",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "maxNew": args.max_new,
        "topK": args.top_k,
        "numHiddenLayers": cfg.num_hidden_layers,
        "vocabSize": cfg.vocab_size,
        "prompts": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")
    for r in results:
        print(r["kind"], r["promptTokens"], "->", r["generatedTokens"], repr(r["generatedText"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
