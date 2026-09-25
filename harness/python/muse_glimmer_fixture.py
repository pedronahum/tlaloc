"""Oracles for the Muse Glimmer decoder, from transformers' own modeling code.

Imports torch and transformers on purpose: this is the oracle side of a
comparison, not serving code. It runs in a venv that has both (for example
~/.local/venvs/vllm, transformers 5.17, which carries model_type
muse_glimmer) and never installs anything.

Two modes.

``tiny``: a five-layer Muse Glimmer (hidden 32, window 3, soft-cap 3) whose
text weights are a closed-form function of the tensor name and element index,
so the Kotlin test rebuilds the same bf16 values without a weight file. It
records the config.json transformers writes for it and the logits at every
position of one prompt, computed with bf16 weights and f32 activations (see
``--precision``), twice: as transformers computes them (``logits``), and with
the rotary cos/sin tables computed in float64 and narrowed, the way Tlaloc
builds them (``logitsF64Rope``). The two differ because a one-ulp change in a
table entry can move a projection input across a bf16 rounding boundary. The
output is small JSON, committed as a test fixture.

``real``: greedy-decode prompts with the real checkpoint and record the ids,
the step-1 top-k logits, the chosen logit and the top-1 margin at every step.

Precision (``--precision``):

- ``bf16``: the checkpoint as transformers runs it by default in bfloat16:
  activations are bf16 between ops, norms and softmax are computed in f32
  and rounded back.
- ``mixed``: bf16 weights, f32 activations. Every nn.Linear rounds its input
  to bf16 and multiplies it by the bf16 weight in f32 (exact products, f32
  sums); the embedding rows are widened to f32 before the embedding norm;
  everything after follows in f32 because the model's ops keep their input
  dtype. This is the arithmetic of Tlaloc's bf16-weight decode graph.

    python muse_glimmer_fixture.py tiny --output .../muse_glimmer_tiny.json
    python muse_glimmer_fixture.py real --checkpoint meta-models/Muse-Glimmer-30B \\
        --precision bf16 --text " The capital of France is" --chat "..." --output ...
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

TINY_TEXT_CONFIG = dict(
    vocab_size=96, hidden_size=32, intermediate_size=48, num_hidden_layers=5,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=64,
    rms_norm_eps=1e-5, post_norm_eps=1e-8, sliding_window=3, final_logit_softcapping=3.0,
    output_multiplier=0.5, qk_scale_factor=3.87,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    bos_token_id=1, eos_token_id=2, pad_token_id=None,
)
TINY_PROMPT = [5, 17, 3, 88, 41, 9, 60, 23, 77, 12, 30]


def closed_form(name: str, k: int, shape, torch):
    """The value of every element of text tensor number ``k`` (sorted by name).

    m = (i * 7919 + (k + 1) * 104729) mod 2003 over the row-major index i,
    u = (m - 1001) / 1001, scaled by the tensor's kind, widened to float64,
    narrowed to float32 and then to bfloat16 (round to nearest even both
    times, which is what the Kotlin side does).
    """
    n = 1
    for d in shape:
        n *= d
    if name.endswith("embed_tokens.weight"):
        scale, offset = 1.0, 0.0
    elif name.endswith("language_model.norm.weight"):
        scale, offset = 0.2, 1.0
    elif len(shape) == 1:
        scale, offset = 0.2, 0.0
    else:
        scale, offset = 0.35, 0.0
    vals = [offset + scale * (((i * 7919 + (k + 1) * 104729) % 2003 - 1001) / 1001.0) for i in range(n)]
    t = torch.tensor(vals, dtype=torch.float64).to(torch.float32).to(torch.bfloat16)
    return t.reshape(shape)


# The activation dtype of the mixed mode: float32, or float64 to measure how
# far float32 arithmetic is from exact with the same bf16 rounding points.
ACTIVATIONS = {}


def patch_mixed(torch):
    """bf16 weights, f32 activations (see the module docstring)."""
    import torch.nn.functional as F
    from transformers.models.muse_glimmer import modeling_muse_glimmer as mm

    ACTIVATIONS["dtype"] = torch.float32

    def linear_forward(self, x):
        w = self.weight
        dt = ACTIVATIONS["dtype"]
        if w.dtype == torch.bfloat16:
            return F.linear(x.to(torch.bfloat16).to(dt), w.to(dt),
                            None if self.bias is None else self.bias.to(dt))
        return F.linear(x, w, self.bias)

    torch.nn.Linear.forward = linear_forward

    def embed_forward(self, input_ids):
        rows = torch.nn.Embedding.forward(self, input_ids)
        return self.embed_norm(rows.to(ACTIVATIONS["dtype"]))

    mm.MuseGlimmerTextNormedEmbedding.forward = embed_forward

    # The norms multiply by their gain in float32; in float64 they must
    # multiply in float64 (identical arithmetic in float32).
    def rms_forward(self, h):
        o = self._norm(h)
        return o * self.weight.to(h.dtype) if self.with_scale else o

    def centered_forward(self, x):
        return self._norm(x) * (1.0 + self.weight.to(x.dtype))

    mm.MuseGlimmerRMSNorm.forward = rms_forward
    mm.MuseGlimmerTextCenteredRMSNorm.forward = centered_forward


def patch_f64_rope(torch, text_config):
    """Rotary tables in float64, narrowed to float32: Tlaloc's ropeTables formula."""
    import math

    from transformers.models.muse_glimmer import modeling_muse_glimmer as mm

    hd = text_config.head_dim
    theta = text_config.rope_parameters["rope_theta"]

    def forward(self, x, position_ids):
        pos = position_ids[0].tolist()
        cos = torch.zeros(1, len(pos), hd, dtype=torch.float32)
        sin = torch.zeros_like(cos)
        for j, p in enumerate(pos):
            for i in range(hd // 2):
                a = p * theta ** (-2.0 * i / hd)
                cos[0, j, i] = cos[0, j, i + hd // 2] = math.cos(a)
                sin[0, j, i] = sin[0, j, i + hd // 2] = math.sin(a)
        return cos.to(x.dtype), sin.to(x.dtype)

    mm.MuseGlimmerTextRotaryEmbedding.forward = forward


def tiny(args) -> int:
    import torch
    import transformers
    from transformers import MuseGlimmerConfig, MuseGlimmerForConditionalGeneration

    torch.manual_seed(0)
    cfg = MuseGlimmerConfig(
        text_config=dict(TINY_TEXT_CONFIG),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                           num_attention_heads=2),
        image_token_id=94, video_token_id=95, out_hidden_size=64, projector_hidden_size=32,
    )
    model = MuseGlimmerForConditionalGeneration(cfg).to(torch.bfloat16).eval()
    sd = model.state_dict()
    text = sorted(n for n in sd if n.startswith("model.language_model.") or n == "lm_head.weight")
    with torch.no_grad():
        for k, n in enumerate(text):
            p = dict(model.named_parameters())[n]
            p.copy_(closed_form(n, k, tuple(p.shape), torch))
    # The head must be its own tensor, not the embedding table.
    assert model.lm_head.weight.data_ptr() != model.model.language_model.embed_tokens.weight.data_ptr()

    if args.precision == "mixed":
        patch_mixed(torch)
    model.config.architectures = ["MuseGlimmerForConditionalGeneration"]
    with tempfile.TemporaryDirectory() as d:
        model.config.save_pretrained(d)
        config_text = (Path(d) / "config.json").read_text()

    ids = torch.tensor([TINY_PROMPT])
    model.set_attn_implementation("eager")
    assert model.model.language_model.config._attn_implementation == "eager"
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0].float()
        patch_f64_rope(torch, cfg.text_config)
        logits64 = model(input_ids=ids).logits[0].float()
    out = {
        "oracle": f"transformers {transformers.__version__} MuseGlimmerForConditionalGeneration, "
                  f"tiny closed-form weights, precision {args.precision}, CPU, eager attention, "
                  "one forward over the whole prompt",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "precision": args.precision,
        "textTensors": text,
        "configJson": config_text,
        "prompt": TINY_PROMPT,
        "logits": [[float(v) for v in row] for row in logits],
        "logitsF64Rope": [[float(v) for v in row] for row in logits64],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", args.output, "argmax per position:", logits.argmax(-1).tolist())
    return 0


def real(args) -> int:
    import time

    import torch
    import transformers
    from transformers import AutoTokenizer, MuseGlimmerForConditionalGeneration

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    if args.noise and args.precision != "mixed":
        raise SystemExit("--noise measures the mixed precision; pass --precision mixed")
    if args.precision == "mixed":
        patch_mixed(torch)
    from transformers import AutoConfig

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    cfg = AutoConfig.from_pretrained(args.checkpoint)
    if args.layers:
        tc = cfg.text_config
        tc.num_hidden_layers = args.layers
        tc.layer_types = tc.layer_types[: args.layers]
        tc.layer_rope_theta = tc.layer_rope_theta[: args.layers]
    if args.f64_rope:
        patch_f64_rope(torch, cfg.text_config)
    t0 = time.time()
    model = MuseGlimmerForConditionalGeneration.from_pretrained(
        args.checkpoint, config=cfg, dtype=torch.bfloat16, attn_implementation="eager",
    ).eval()
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    head = model.lm_head.weight
    table = model.model.language_model.embed_tokens.weight
    tied = head.data_ptr() == table.data_ptr()
    print("lm_head tied to the embedding table:", tied, flush=True)

    image_id, video_id = model.config.image_token_id, model.config.video_token_id
    prompts = []
    for t in args.text:
        prompts.append({"kind": "text", "input": t, "ids": tok(t, add_special_tokens=True).input_ids})
    # The template writes today's date unless told one; a fixture needs a fixed one.
    tvars = dict(kv.split("=", 1) for kv in args.template_var)
    for msg in args.chat:
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, **tvars,
        )
        prompts.append({"kind": "chat", "input": msg, "rendered": rendered, "templateVars": tvars,
                        "ids": tok(rendered, add_special_tokens=False).input_ids})
    for p in prompts:
        bad = [i for i in p["ids"] if i in (image_id, video_id)]
        if bad:
            raise SystemExit(f"prompt {p['input']!r} carries image/video placeholder ids {bad}")

    results = []
    with torch.no_grad():
        for p in prompts:
            ids = torch.tensor([p["ids"]])
            t0 = time.time()
            gen = model.generate(
                ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new,
                do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
                eos_token_id=None, pad_token_id=tok.pad_token_id or 0,
                output_logits=True, return_dict_in_generate=True,
            )
            secs = time.time() - t0
            new = gen.sequences[0, len(p["ids"]):].tolist()
            step_logits = [lg[0].float() for lg in gen.logits]
            vals, idx = torch.topk(step_logits[0], args.top_k)
            entry = dict(p)
            entry["promptTokens"] = entry.pop("ids")
            entry["generatedTokens"] = new
            entry["generatedText"] = tok.decode(new, skip_special_tokens=False)
            entry["step1TopKIndices"] = idx.tolist()
            entry["step1TopKValues"] = [float(v) for v in vals]
            entry["chosenLogits"] = [float(step_logits[i][t]) for i, t in enumerate(new)]
            entry["margins"] = [float(torch.topk(lg, 2).values[0] - torch.topk(lg, 2).values[1])
                                for lg in step_logits]
            entry["seconds"] = round(secs, 1)
            if args.noise:
                # The same sequence, teacher-forced, with float32 and float64
                # activations: how far float32 is from exact at each step.
                seq = torch.tensor([entry["promptTokens"] + new[:-1]])
                n = len(entry["promptTokens"]) - 1
                runs = {}
                for dt in (torch.float32, torch.float64):
                    ACTIVATIONS["dtype"] = dt
                    runs[dt] = model(input_ids=seq).logits[0, n:].double()
                ACTIVATIONS["dtype"] = torch.float32
                exact = runs[torch.float64]
                diff = (runs[torch.float32] - exact).abs().max(-1).values
                entry["referenceNoise"] = [float(v) for v in (diff / exact.abs().max(-1).values.clamp(min=1.0))]
                print(p["kind"], "float32 vs float64 activations, relative:",
                      ["%.1e" % v for v in entry["referenceNoise"]], flush=True)
            results.append(entry)
            print(p["kind"], entry["promptTokens"], "->", new, repr(entry["generatedText"]),
                  f"{secs:.0f} s", "min margin", min(entry["margins"]), flush=True)

    cfg = model.config
    out = {
        "checkpoint": args.checkpoint,
        "revision": getattr(cfg, "_commit_hash", None),
        "oracle": f"transformers MuseGlimmerForConditionalGeneration, weights bfloat16, "
                  f"precision {args.precision}, "
                  f"{'float64' if args.f64_rope else 'float32'} rotary tables, CPU, eager attention, greedy",
        "precision": args.precision,
        "f64Rope": args.f64_rope,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "lmHeadTied": tied,
        "maxNew": args.max_new,
        "topK": args.top_k,
        "numHiddenLayers": cfg.text_config.num_hidden_layers,
        "vocabSize": cfg.text_config.vocab_size,
        "prompts": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("tiny")
    t.add_argument("--precision", choices=["mixed", "bf16"], default="mixed")
    t.add_argument("--output", required=True)
    r = sub.add_parser("real")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--precision", choices=["mixed", "bf16"], default="bf16")
    r.add_argument("--max-new", type=int, default=16)
    r.add_argument("--top-k", type=int, default=20)
    r.add_argument("--threads", type=int, default=16)
    r.add_argument("--text", action="append", default=[])
    r.add_argument("--chat", action="append", default=[])
    r.add_argument("--template-var", action="append", default=[],
                   help="k=v passed to the chat template, e.g. current_date=2026-09-25")
    r.add_argument("--noise", action="store_true",
                   help="mixed only: also record referenceNoise, float32 vs float64 activations")
    r.add_argument("--f64-rope", action="store_true",
                   help="rotary tables in float64, narrowed, as Tlaloc builds them")
    r.add_argument("--layers", type=int, default=0,
                   help="keep only the first N decoder layers (0: all), for a reduced oracle")
    r.add_argument("--output", required=True)
    args = ap.parse_args()
    return tiny(args) if args.mode == "tiny" else real(args)


if __name__ == "__main__":
    sys.exit(main())
