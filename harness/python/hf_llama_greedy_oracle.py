"""§0.4.480 (H3c-3) — the ORACLE for a real Llama generating through Tlaloc.

Imports `torch` and `transformers` ON PURPOSE. **This is not serving-path
code.** It runs in `~/.local/venvs/vllm` (torch 2.13.0+cu130, transformers
5.17.0), it is the other side of the comparison, and the runtime it is an
oracle for still imports nothing but the standard library and `tlaloc_pjrt`.

Two jobs, and they are deliberately in one file:

1. **Tokenize** a prompt with the checkpoint's own tokenizer and emit the ids,
   so `run_llama_generate.py` can be handed ids without this repo ever writing
   a token id down by hand. §0.4.477 refused to fabricate a `config.json`; a
   hand-typed token id is the same refusal at a smaller scale.
2. **Greedy-decode** the same prompt with `AutoModelForCausalLM`, **fp32 on
   CPU**, and emit the ids it produces plus the text.

FP32 ON CPU, for the same reason §0.4.479 gave: bf16 -> f32 is exact
(§0.4.468), and a CPU oracle keeps TF32 out of the comparison entirely, so
what is left between the two sides is accumulation ORDER. The artifact under
test runs on XLA-GPU, whose default f32 `dot_general` policy IS TF32, so the
two sides differ by ~5e-4 per matmul by construction — which is precisely why
the claim this oracle certifies is stated as **token agreement over a prefix**
and not as a tolerance on logits.

    ~/.local/venvs/vllm/bin/python harness/python/hf_llama_greedy_oracle.py \
        --checkpoint <dir> --prompt "The capital of France is" \
        --max-new 6 --output /tmp/oracle.json [--num-layers 2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new", type=int, default=6)
    ap.add_argument("--num-layers", type=int, default=None,
                    help="reduce to a prefix of layers, as the parity lane does")
    ap.add_argument("--output", required=True)
    ap.add_argument("--decode-ids", default=None,
                    help="JSON list of ids to additionally render as text")
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # pragma: no cover - environment probe
        print(f"no torch/transformers here: {e}", file=sys.stderr)
        return 2

    ck = args.checkpoint
    tok = AutoTokenizer.from_pretrained(ck)
    ids = tok(args.prompt, return_tensors="pt").input_ids

    cfg = AutoConfig.from_pretrained(ck)
    if args.num_layers is not None:
        cfg.num_hidden_layers = args.num_layers
    model = AutoModelForCausalLM.from_pretrained(ck, config=cfg, dtype=torch.float32)
    model.eval()

    out = {
        "checkpoint": str(Path(ck).resolve()),
        "prompt": args.prompt,
        "promptTokens": ids[0].tolist(),
        "numLayers": cfg.num_hidden_layers,
        "torch": torch.__version__,
    }
    with torch.no_grad():
        gen = model.generate(
            ids, max_new_tokens=args.max_new, do_sample=False, num_beams=1,
            # A fixed budget is the claim; letting EOS cut it short would make
            # the compared prefix depend on the model's own choice.
            eos_token_id=None, pad_token_id=tok.eos_token_id,
        )
    full = gen[0].tolist()
    out["generatedTokens"] = full[len(out["promptTokens"]):]
    out["allTokens"] = full
    out["text"] = tok.decode(full, skip_special_tokens=True)
    out["generatedText"] = tok.decode(out["generatedTokens"], skip_special_tokens=True)
    if args.decode_ids:
        other = json.loads(args.decode_ids)
        out["decodedIds"] = tok.decode(other, skip_special_tokens=True)
        out["decodedIdsPieces"] = [tok.decode([i]) for i in other]
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in out if k != "allTokens"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
