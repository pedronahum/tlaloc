"""§0.4.492 (H3c-4b) — drive **vLLM's own `LLM.generate()`** against a Tlaloc
serving artifact and report the token ids it produced.

    ~/.local/venvs/vllm/bin/python run_vllm_generate_check.py \
        --artifact DIR --checkpoint CKPT --prompt "The capital of France is" \
        --max-new 6 --output out.json

This is the last uncertified step of the serving arc: from §0.4.470 to
§0.4.491 the plugin was driven by hand-built vLLM objects
(`run_vllm_live_check.py`) because the engine core refused to start. It
starts now, and this script is the thing that proves it — vLLM's scheduler,
its block manager, its tokenizer, its detokenizer, its `LLM` entry point,
and underneath them a compiled Tlaloc StableHLO program on PJRT.

WHAT IT MUST AND MUST NOT DO
============================

* It must let **vLLM** tokenize. The prompt goes in as a STRING and the ids
  come back out of `RequestOutput.prompt_token_ids`, so the comparison
  against `run_llama_generate.py` is a comparison of two callers driving one
  artifact and not of two people typing the same list (§0.4.477's principle,
  and §0.4.480's refusal to write a token id by hand).
* It must NOT set `load_format`, a memory fraction, or any other knob that
  makes vLLM believe it loaded something. `TlalocWorker.load_model` loads the
  ARTIFACT; vLLM's weight loading never runs. Passing `load_format="dummy"`
  would work and would be a lie about which weights answered.
* The three settings it DOES pass are the artifact's own compiled facts —
  `max_model_len`, `max_num_seqs`, `block_size` — which the platform's
  `check_and_update_config` refuses to have disagree with the manifest. They
  are passed rather than discovered because vLLM validates them against the
  HF config before the platform hook ever runs.
* `enable_prefix_caching=False`, by name: this runner's KV pool is its own
  and holds no cross-request prefix, so a prefix-cache hit would start a
  sequence in the middle of a context it never wrote. `batching.py` refuses
  that case; this turns it off rather than relying on the refusal.

Exit codes follow the rest of the harness: 0 ran, 1 the run failed (the
interesting failure), 2 the environment cannot run it at all.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--checkpoint", required=True, help="tokenizer + HF config; NOT the weights")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new", type=int, default=6)
    ap.add_argument("--max-context", type=int, required=True)
    ap.add_argument("--block-size", type=int, required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import os

    os.environ["TLALOC_SERVING_ARTIFACT"] = args.artifact

    try:
        import vllm
        from vllm import LLM, SamplingParams
    except Exception as e:  # pragma: no cover - environment probe
        print(f"cannot import vllm: {e}", file=sys.stderr)
        return 2
    if not os.environ.get("TLALOC_PJRT_PLUGIN_PATH"):
        print("TLALOC_PJRT_PLUGIN_PATH is unset; there is no PJRT plugin to run on",
              file=sys.stderr)
        return 2

    payload: dict = {
        "vllmVersion": vllm.__version__,
        "artifact": str(Path(args.artifact).resolve()),
        "prompt": args.prompt,
        "maxNew": args.max_new,
    }

    def emit(code: int) -> int:
        Path(args.output).write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        return code

    try:
        t0 = time.time()
        llm = LLM(
            model=args.checkpoint,
            max_model_len=args.max_context,
            max_num_seqs=1,
            block_size=args.block_size,
            enforce_eager=True,
            enable_prefix_caching=False,
        )
        payload["constructSeconds"] = time.time() - t0
        payload["platformName"] = type(llm.llm_engine.vllm_config.device_config).__name__

        t1 = time.time()
        outs = llm.generate(
            [args.prompt],
            SamplingParams(temperature=0.0, max_tokens=args.max_new),
        )
        payload["generateSeconds"] = time.time() - t1
    except Exception as e:
        payload["error"] = f"{type(e).__name__}: {e}"
        return emit(1)

    out = outs[0]
    payload["promptTokens"] = [int(t) for t in out.prompt_token_ids]
    payload["generatedTokens"] = [int(t) for t in out.outputs[0].token_ids]
    payload["text"] = out.outputs[0].text
    payload["finishReason"] = out.outputs[0].finish_reason
    return emit(0)


if __name__ == "__main__":
    sys.exit(main())
