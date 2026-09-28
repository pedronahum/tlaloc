"""Write the golden files the :tokenizer tests compare against.

Imports transformers and tokenizers on purpose: this is the oracle side of the
comparison. It runs in a venv that has both (for example ~/.local/venvs/vllm)
and never installs anything.

For each tokenizer family it loads the checkpoint with AutoTokenizer and
records, over a fixed corpus plus seeded random strings:

- input_ids with and without special tokens, and with special-token strings
  treated as plain text (split_special_tokens),
- decode of those ids with and without skip_special_tokens,
- decode of partial id sequences (cut mid-character, single tokens),
- the chunks tokenizers' DecodeStream emits token by token,
- apply_chat_template(tokenize=False) strings and their ids, for the families
  whose chat template :tokenizer renders.

    ~/.local/venvs/vllm/bin/python harness/python/tokenizer_golden.py \\
        --output tokenizer/src/jvmTest/resources/io/tlaloc/tokenizer

A family whose checkpoint is not in the local Hugging Face cache is skipped
by name.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))

FAMILIES = {
    "qwen3": ("hub", "Qwen/Qwen3-0.6B"),
    "muse_glimmer": ("hub", "meta-models/Muse-Glimmer-30B"),
    "gpt2": ("hub", "gpt2"),
    "tinyllama": ("dir", "~/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0"),
    "gemma4": ("hub", "google/gemma-4-e4b-it"),
}

SPECIALS = {
    "qwen3": ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"],
    "muse_glimmer": ["<|begin_of_text|>", "<|eot|>", "<|start|>", "<|message|>"],
    "gpt2": ["<|endoftext|>"],
    "tinyllama": ["<s>", "</s>", "<unk>"],
    "gemma4": ["<bos>", "<eos>", "<|turn>", "<turn|>"],
}

CORPUS = [
    "",
    " ",
    "Hello world",
    "Hello, world! How are you?",
    " The capital of France is",
    "don't won't can't I'm you're we've they'll he'd it's",
    "DON'T WON'T I'M YOU'RE WE'VE THEY'LL HE'D IT'S",
    "Don'T wOn't I'm You'Re We'vE",
    "  leading   spaces\n\n\ttab",
    "trailing spaces   ",
    "tabs\tand\t\ttabs\t",
    "line one\r\nline two\r\n\r\nline three\n",
    "\n\n\n",
    "   \n  \n",
    "nbsp\u00a0here and\u00a0\u00a0there",
    "en\u2002em\u2003thin\u2009hair\u200aideographic\u3000space",
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a",
    "zero\u200bwidth\u200cjoiner\u200d",
    "1 12 123 1234 12345 123456 1234567",
    "3.14159 2,718 1e-9 0x1F",
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3067\u3059\u3002",
    "\u4e2d\u6587\u6587\u672c\uff0c\u6ca1\u6709\u7a7a\u683c\u3002\u4f60\u597d\u4e16\u754c\uff01",
    "\ud55c\uad6d\uc5b4 \ud14d\uc2a4\ud2b8",
    "caf\u00e9 vs cafe\u0301",
    "e\u0301\u0302\u0303 a\u0308 n\u0303",
    "\u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u05e2\u05d1\u05e8\u05d9\u05ea",
    "\u0939\u093f\u0928\u094d\u0926\u0940 \u0e20\u0e32\u0e29\u0e32\u0e44\u0e17\u0e22",
    "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440! \u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac",
    "emoji \U0001f44d\U0001f3fd family \U0001f468\u200d\U0001f469\u200d\U0001f467 flag \U0001f1eb\U0001f1f7",
    "\U0001f600\U0001f601\U0001f602 \u2764\ufe0f \U0001f9d1\U0001f3ff\u200d\U0001f4bb",
    "astral \U0001d400\U0001d401\U0001d402 \U00010380\U00010381 \U00020000\U0002a6d6",
    "math \u2200x\u2208\u211d: x\u00b2 \u2265 0 \u2192 \u221e",
    "symbols !@#$%^&*()_+-=[]{}|;:'\",.<>/?`~",
    "path/to/file.txt and https://example.com/a?b=c&d=e",
    "CamelCaseWord HTTPServer iPhone XMLHttpRequest",
    "def f(x):\n    return x * 2  # comment\n",
    "x = [1, 2, 3]\nfor i in x:\n\tprint(i)\n",
    "'s 's 'S 'll 'LL 're",
    "a'b 'a' '' ''' \"q\"",
    "\u017f's 'S\u017f",
    "\u00c0\u00c9\u00ce\u00d5\u00dc \u00df \u0130 \u0131",
    "\u2126 \u212a \u212b",
    "control\u0001chars\u0007here\u007f",
    "The quick brown fox jumps over the lazy dog. " * 3,
    "It was the best of times, it was the worst of times, it was the age of "
    "wisdom, it was the age of foolishness, it was the epoch of belief, it was "
    "the epoch of incredulity, it was the season of Light, it was the season of "
    "Darkness, it was the spring of hope, it was the winter of despair.",
    "Tlaloc compiles Kotlin programs to StableHLO; gradients print as Kotlin "
    "source that compiles and runs. A transposed weight is a compile error.\n\n"
    "    implementation(\"io.github.pedronahum:tlaloc-nn\")\n",
]


def special_cases(family: str) -> list[str]:
    s = SPECIALS[family]
    a, b = s[0], s[1 % len(s)]
    return [
        f"hi {a} there",
        f"{a}hi{b}",
        f"{a}{b}{a}",
        f"text{a}",
        f"  {a}  spaced  {b}  ",
        f"{a}user\nWhat is 2+2?{b}\n",
        " ".join(s),
        f"{a[:-1]} almost {a[1:]}",
        f"<s>hi</s> <|im_end|> <|endoftext|>",
    ]


# (unit, times): the text is unit * times. Stored that way, not spelled out.
LONG = [("abcdefghij", 1000), ("\u00e9", 1000), ("\U0001f600", 200), ("word ", 500)]


def random_strings(seed: int, n: int) -> list[str]:
    rng = random.Random(seed)
    pool = (
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        + list(" \t\n\r'.,!?-_/\\\"()[]{}<>|@#$%^&*~`+=:;")
        + ["\u00a0", "\u2003", "\u3000", "\u200b", "\u200d", "\u0301", "\u0308",
           "\u00e9", "\u00df", "\u017f", "\u0130", "\u212a",
           "\u65e5", "\u672c", "\u3042", "\u30c6", "\ud55c", "\u0627", "\u05e2",
           "\u0939", "\u094d", "\u0e20", "\u041f", "\u03a9",
           "\U0001f44d", "\U0001f3fd", "\U0001f468", "\U0001f600", "\U0001d400",
           "\U00020000", "\u2764", "\ufe0f", "\u2581", "\u0120", "\uffff", "\ufffd"]
    )
    out = []
    for _ in range(n):
        k = rng.randint(1, 40)
        out.append("".join(rng.choice(pool) for _ in range(k)))
    return out


def resolve(kind: str, where: str) -> Path | None:
    if kind == "dir":
        p = Path(os.path.expanduser(where))
        return p if (p / "tokenizer.json").is_file() else None
    repo = HUB / ("models--" + where.replace("/", "--"))
    ref = repo / "refs" / "main"
    if not ref.is_file():
        return None
    snap = repo / "snapshots" / ref.read_text().strip()
    return snap if (snap / "tokenizer.json").is_file() else None


def chat_cases(family: str):
    q = "What is the capital of France? Answer in one word."
    if family == "qwen3":
        convs = [
            [{"role": "user", "content": q}],
            [{"role": "system", "content": "You are terse."}, {"role": "user", "content": q}],
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"},
             {"role": "user", "content": q}],
            [{"role": "user", "content": "Hi"},
             {"role": "assistant", "content": "<think>\nThe user greets.\n</think>\n\nHello!"}],
            [{"role": "user", "content": "Hi"},
             {"role": "assistant", "content": "<think>\nThe user greets.\n</think>\n\nHello!"},
             {"role": "user", "content": q}],
            [{"role": "user", "content": "Hi"},
             {"role": "assistant", "content": "Hello!", "reasoning_content": "\nGreeting back.\n"}],
            [{"role": "system", "content": "A"}, {"role": "user", "content": "B"},
             {"role": "system", "content": "C"}, {"role": "user", "content": "D"}],
            [{"role": "user", "content": "\u65e5\u672c\u8a9e \U0001f44d\n\ttab"}],
        ]
        variants = [dict(), dict(enable_thinking=False), dict(enable_thinking=True)]
    elif family == "tinyllama":
        convs = [
            [{"role": "user", "content": q}],
            [{"role": "system", "content": "You are a friendly chatbot."},
             {"role": "user", "content": q}],
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"},
             {"role": "user", "content": q}],
            [{"role": "user", "content": "  spaced\n\ncontent  "}],
        ]
        variants = [dict()]
    elif family == "muse_glimmer":
        convs = [
            [{"role": "user", "content": q}],
            [{"role": "system", "content": "You are terse."}, {"role": "user", "content": q}],
            [{"role": "system", "content": "Be brief. Reasoning effort: low."},
             {"role": "user", "content": q}],
            [{"role": "system", "content": "Reasoning Strength: medium"},
             {"role": "user", "content": q}],
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"},
             {"role": "user", "content": q}],
            [{"role": "user", "content": "Hi"},
             {"role": "assistant", "content": "Hello!", "reasoning_content": "Greet back."},
             {"role": "user", "content": q}],
            [{"role": "user", "content": "A"}, {"role": "user", "content": "B"}],
        ]
        variants = [
            dict(current_date="2026-09-25"),
            dict(current_date="2026-09-25", reasoning_strength="low"),
            dict(current_date="2026-01-01", knowledge_cutoff="2025-06-30"),
        ]
    else:
        return []
    out = []
    for conv in convs:
        for v in variants:
            for agp in (True, False):
                out.append((conv, v, agp))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--random", type=int, default=80, help="seeded random strings per family")
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import tokenizers
    import transformers
    from tokenizers.decoders import DecodeStream
    from transformers import AutoTokenizer

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for family, (kind, where) in FAMILIES.items():
        ckpt = resolve(kind, where)
        if ckpt is None:
            print(f"skip {family}: no tokenizer.json for {where}", file=sys.stderr)
            continue
        tok = AutoTokenizer.from_pretrained(str(ckpt))
        split_tok = AutoTokenizer.from_pretrained(str(ckpt), split_special_tokens=True)
        backend = tok.backend_tokenizer

        def encode_case(text: str, full: bool) -> dict:
            ids = tok(text).input_ids
            if not full:
                # Long inputs: the ids, and whether decoding them gives the
                # text back, instead of three more copies of 10k characters.
                return {"text": text, "ids": ids,
                        "decodeSkipIsText": tok.decode(ids, skip_special_tokens=True) == text}
            return {"text": text, "ids": ids, "decode": tok.decode(ids),
                    "decodeSkip": tok.decode(ids, skip_special_tokens=True),
                    "idsNoSpecial": tok(text, add_special_tokens=False).input_ids,
                    "idsSplitSpecial": split_tok(text, add_special_tokens=False).input_ids}

        cases = [encode_case(t, True) for t in CORPUS + special_cases(family)]
        for unit, times in LONG:
            case = encode_case(unit * times, False)
            del case["text"]
            cases.append({"repeat": unit, "times": times, **case})
        randoms = [encode_case(t, True) for t in random_strings(1234, args.random)]

        # Partial id sequences: every suffix of a multi-byte case, and each
        # token of it alone, which cuts characters in the middle.
        decodes = []
        for text in ["emoji \U0001f44d\U0001f3fd family \U0001f468\u200d\U0001f469\u200d\U0001f467",
                     "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8", "caf\u00e9 \u00e9t\u00e9"]:
            ids = tok(text, add_special_tokens=False).input_ids
            for i in range(len(ids)):
                decodes.append(ids[i:])
                decodes.append([ids[i]])
        vocab_size = len(tok)
        rng = random.Random(99)
        for _ in range(40):
            decodes.append([rng.randrange(vocab_size) for _ in range(rng.randint(1, 12))])
        decode_cases = [{"ids": d, "text": tok.decode(d),
                         "textSkip": tok.decode(d, skip_special_tokens=True)} for d in decodes]

        streams = []
        stream_texts = [t for t in CORPUS if t.startswith(("emoji", "\u65e5", "  leading", "Tlaloc"))]
        for text in stream_texts + [special_cases(family)[4]]:
            ids = tok(text).input_ids
            for skip in (False, True):
                ds = DecodeStream(skip_special_tokens=skip)
                chunks = [ds.step(backend, i) for i in ids]
                streams.append({"ids": ids, "skip": skip, "chunks": chunks})

        chats = []
        for conv, variables, agp in chat_cases(family):
            rendered = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=agp,
                                               **variables)
            chats.append({"messages": conv, "vars": variables, "addGenerationPrompt": agp,
                          "rendered": rendered,
                          "ids": tok(rendered, add_special_tokens=False).input_ids})

        golden = {
            "family": family,
            "checkpoint": where if kind == "hub" else Path(where).name,
            "revision": ckpt.name if kind == "hub" else None,
            "tokenizerClass": type(tok).__name__,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "cases": cases,
            "random": randoms,
            "decodes": decode_cases,
            "streams": streams,
            "chats": chats,
        }
        path = out_dir / f"{family}.json"
        path.write_text(json.dumps(golden, ensure_ascii=True, separators=(",", ":")) + "\n")
        print(f"{family}: {len(cases)} cases, {len(randoms)} random, {len(decode_cases)} decodes, "
              f"{len(streams)} streams, {len(chats)} chats -> {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
