#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""Ask a Tlaloc language model served by Triton one question, and print the
answer as it is generated.

The model is a Triton model in sequence mode (run.sh writes it and starts the
server). The server holds the weights, the KV cache and every sequence's
pages; this client sends token ids and gets logits back:

  1. the question is rendered with the model's own chat template and
     tokenized with its own tokenizer.json (the `tokenizers` package; no
     torch, no transformers);
  2. the prompt goes to the server as ONE request with START: one prefill
     call;
  3. each generated token goes back as one request with the same sequence
     id: one decode step. The client picks the token (greedy: the argmax)
     and prints the text it adds;
  4. an empty request with END frees the sequence's pages on the server.

  chat.py --model qwen3 --checkpoint <HF snapshot dir> --question "..."

Dependencies: tritonclient[grpc], tokenizers, jinja2, numpy.
"""

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# The prompt: the checkpoint's chat template, rendered the way transformers
# renders it, then tokenized by the checkpoint's tokenizer.json.
# ---------------------------------------------------------------------------

def load_chat_template(ckpt):
    """The template text: chat_template.jinja if present, else the
    `chat_template` field of tokenizer_config.json."""
    jinja_file = os.path.join(ckpt, "chat_template.jinja")
    config = json.load(open(os.path.join(ckpt, "tokenizer_config.json")))
    if os.path.isfile(jinja_file):
        template = open(jinja_file).read()
    else:
        template = config.get("chat_template")
    if not template:
        raise SystemExit(f"{ckpt} has no chat template (chat_template.jinja or "
                         "tokenizer_config.json's chat_template)")

    def special(name):
        v = config.get(name)
        return v.get("content") if isinstance(v, dict) else v

    return template, {"bos_token": special("bos_token"), "eos_token": special("eos_token")}


def render_chat(template, specials, question, variables):
    """Render one user message with add_generation_prompt, in the same jinja2
    environment transformers' apply_chat_template uses: sandboxed,
    trim_blocks, lstrip_blocks, the loop-controls extension, and the
    raise_exception and strftime_now globals."""
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise ValueError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    return env.from_string(template).render(
        messages=[{"role": "user", "content": question}],
        add_generation_prompt=True,
        # Qwen3: answer directly instead of writing a <think> block first.
        # Templates that do not know the variable ignore it.
        enable_thinking=False,
        **specials,
        **variables,
    )


def stop_ids(ckpt):
    """The ids that end an answer: generation_config.json's eos_token_id."""
    path = os.path.join(ckpt, "generation_config.json")
    if not os.path.isfile(path):
        return set()
    eos = json.load(open(path)).get("eos_token_id")
    return set(eos) if isinstance(eos, list) else ({eos} if eos is not None else set())


# ---------------------------------------------------------------------------
# The server: one request per step of one sequence.
# ---------------------------------------------------------------------------

class Sequence:
    """One sequence on a sequence-mode Tlaloc model, over gRPC. The request
    input is TOKENS (INT32 [1, n]); the response is LOGITS (FP32 [1, vocab]),
    the logits of the last token sent."""

    def __init__(self, url, model, sequence_id):
        import tritonclient.grpc as grpcclient
        self.tc = grpcclient
        self.client = grpcclient.InferenceServerClient(url=url)
        self.model = model
        self.id = sequence_id

    def send(self, tokens, start=False, end=False):
        arr = np.asarray(tokens, dtype=np.int32).reshape(1, -1)
        inp = self.tc.InferInput("TOKENS", list(arr.shape), "INT32")
        inp.set_data_from_numpy(arr)
        res = self.client.infer(self.model, [inp],
                                outputs=[self.tc.InferRequestedOutput("LOGITS")],
                                sequence_id=self.id, sequence_start=start, sequence_end=end)
        logits = res.as_numpy("LOGITS")
        return None if logits is None else logits.reshape(-1)


def plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="localhost:8001", help="the server's gRPC endpoint")
    ap.add_argument("--model", required=True, help="the Triton model name")
    ap.add_argument("--checkpoint", required=True,
                    help="the checkpoint directory (tokenizer.json, tokenizer_config.json)")
    ap.add_argument("--question", default="Why is the sky blue? Answer in two sentences.")
    ap.add_argument("--max-new", type=int, default=96, help="stop after this many tokens")
    ap.add_argument("--context", type=int, default=256,
                    help="the largest context the model was compiled for")
    ap.add_argument("--template-var", action="append", default=[],
                    help="k=v passed to the chat template")
    ap.add_argument("--sequence-id", type=int, default=1)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(os.path.join(args.checkpoint, "tokenizer.json"))
    template, specials = load_chat_template(args.checkpoint)
    variables = dict(kv.split("=", 1) for kv in args.template_var)
    rendered = render_chat(template, specials, args.question, variables)
    # The template already wrote any special tokens (BOS, role markers), so
    # the tokenizer must not add its own: apply_chat_template does the same.
    prompt = tokenizer.encode(rendered, add_special_tokens=False).ids
    stops = stop_ids(args.checkpoint)
    # The server refuses a sequence longer than its largest compiled context.
    max_new = min(args.max_new, args.context - len(prompt))
    if max_new < 1:
        raise SystemExit(f"the prompt is {len(prompt)} tokens; the model's context is {args.context}")

    print(f"question  {args.question}")
    print(f"prompt    {len(prompt)} tokens after the chat template")
    print("answer    ", end="", flush=True)

    seq = Sequence(args.url, args.model, args.sequence_id)
    t0 = time.perf_counter()
    logits = seq.send(prompt, start=True)
    prefill_ms = (time.perf_counter() - t0) * 1e3

    generated, shown, steps = [], "", []
    stopped_by = "the token limit"
    while True:
        token = int(np.argmax(logits))
        if token in stops:
            stopped_by = f"end-of-turn token {token}"
            break
        generated.append(token)
        # Decode everything so far and print only what is new: a token can be
        # half of a character, or change how the previous one is spelled.
        # Special tokens are printed: a model that writes channel markers
        # (Muse Glimmer's reasoning channel) shows them.
        text = tokenizer.decode(generated, skip_special_tokens=False)
        if text.startswith(shown) and not text.endswith("\ufffd"):
            print(text[len(shown):].replace("\n", "\n          "), end="", flush=True)
            shown = text
        if len(generated) == max_new:
            break
        t0 = time.perf_counter()
        logits = seq.send([token])
        steps.append((time.perf_counter() - t0) * 1e3)
    seq.send([], end=True)
    print()
    print()

    med = sorted(steps)[len(steps) // 2] if steps else float("nan")
    print(f"generated {plural(len(generated), 'token')}, stopped by {stopped_by}")
    print(f"prefill   {len(prompt)} tokens in one request: {prefill_ms:.0f} ms")
    if steps:
        print(f"decode    median {med:.1f} ms a token over {plural(len(steps), 'request')} "
              f"({1e3 / med:.1f} tokens/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
