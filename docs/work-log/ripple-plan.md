# Ripple-65B Preview: plan

Ricursive's open-weight model, released 2026-11-05 under Apache 2.0.
Page: <https://www.ricursive.com/ripple>.

| Repo | Contents | Size |
|---|---|---|
| `ricursiveAI/Ripple-65B-Preview` | BF16, instruction-tuned | 120.7 GiB |
| `ricursiveAI/Ripple-65B-Preview-NVFP4` | MLP NVFP4, attention, embeddings and head BF16 | 53.3 GiB |
| `ricursiveAI/Ripple-65B-Base-Preview` | BF16, pretrain only | not fetched |
| `ricursiveAI/Ripple-65B-Preview-DFlash2` | BF16 draft, 4.04B parameters | 7.5 GiB |

## The model

`RicursiveForCausalLM`, `model_type` `ricursive`, custom modeling code.

- 80 layers, hidden 5,120, MLP 40,960, 80 query heads over 8 KV heads, head_dim 128,
  vocabulary 100,352, untied head.
- Four sliding layers (window 1,024, RoPE), then one full layer without RoPE, repeating.
- Four RMSNorms per layer, gains `w`. Per-head q/k RMSNorms with gains, before RoPE.
- The attention output times `sigmoid(g_proj(x))`, computed in f32.
- Value residual: `v_proj` reads `x0 + sigmoid(w) (x - x0)`, with `x0` the scaled embeddings
  and `w` one scalar per layer. q, k and the gate read `x`.
- `gate.clamp(max=20)` and `up.clamp(-20, 20)` before the SwiGLU.
- Embeddings times `embed_scale` = 71.554.

Bytes a target step reads, from the sizes above: attention 26.8 GB in BF16 (q, g and o are
105 MB per layer, k and v 10 MB), MLP 25 GB in NVFP4, table and head 1 GB each.

## How it maps onto Tlaloc

Muse Glimmer already has the layer shape: `HfModelFamily.Ricursive` differs in the names of
its norms, q/k gains, no soft cap and the three items below.

| New | Where |
|---|---|
| `DecoderLayerSpec.valueResidual`, `DecoderLayerPart.VALUE_RES_LAMBDA` | `attention()`: the blend is what `v_proj` reads |
| `HfDecoderConfig.embedScale` | after the embedding, before the blend's source is taken |
| `HfDecoderConfig.ffnClampThreshold` | `mlpOut`, as COMPARE + WHERE; target layers only |
| `HfDecoderConfig.attentionQuant` | `-PattentionQuant=none`: NVFP4 MLP and BF16 attention, as the checkpoint stores them |

## DFlash2

Tensor list from the checkpoint header; behaviour from vLLM's `qwen3_dflash2.py`,
`spec_decode/dflash2/speculator.py`, its Ripple PR (#65745), and the draft's model card.
The draft repository ships no modeling code.

```
context at each position the target ran:
  c = hidden_norm(fc(concat(output of target layers 1, 20, 39, 58, 77)))
  each draft layer: K = k_norm(k_proj(c)), RoPE unless NoPE; V = v_proj(c); written to its KV pool

query rows: the next token, then k mask tokens, embedded by the target's table, unscaled
each draft layer:
  x  = norm(h);  x' = conv_a(x);  a = attention(x');  h += conv_a'(a)
  y  = norm(h);  y' = conv_m(y);  m = mlp(y');        h += conv_m'(m)

tokens:
  mask rows through the target's LM head, top 32 candidates
  score[p, c] = logit[c] + sum_r pred[p, r] view[r] succ[c, r]    rank 256
  walk from the next token: the best candidate after the previous choice
```

Layout in Tlaloc:

- **One call.** The verify entry is the target, the context writes and the draft, as the MTP
  head's. It returns tokens, accepted count, drafts and the chained inputs, so the backend's
  protocol does not change.
- **Not causal:** `PAGED_ATTENTION` has per-row lengths, so every query row gets the block's
  end as its length. No new mask.
- **Rollback:** the query rows start right after the accepted positions and cover every
  rejected one, so their K/V writes replace the rejected context. The draft has no other state.
- **No state slots.** A speculative entry takes `stateSlots` and `stateCommit` only when the
  model has a `LinearStatePool`; the backend reads a v5 manifest without one as this case.
- **Pages:** a step reaches two positions further than an MTP step (`extra` in
  `SequenceInstance`). The ring adds `1 + drafts` to what a call writes.
- **Window of the full draft layer:** 16,384. Ricursive's serving code caps it there because
  the draft was trained on 16K rows; vLLM does not cap it.
- **Quantization:** the draft stays in its checkpoint's BF16 under an NVFP4 target;
  `-PmtpLayerQuant=fp8` quantizes it.

## Validation

`HfRippleParityTest` and `HfRippleDFlash2Test` compare the graphs with a reference written in
Kotlin from reading the upstream modeling code. `harness/python/ripple_modeling_fixture.py` ran
that code (`modeling_ricursive.py`, sha256-pinned, transformers 5.17, eager attention, f32) on
`tiny_ripple`; the Kotlin reference's logits agree within 1.6e-5 over two prompts
(`ripple_modeling_logits.json`). The real weights check the rest:

1. The output is coherent and solves problems.
2. Acceptance by draft position against vLLM's published table for GSM8K 5-shot and 15 drafts
   (mean accepted run 7.34; p1 .91, p2 .82, p3 .73, p4 .65, p5 .58).
3. GSM8K accuracy against vLLM's NVFP4 result (5-shot 0.916).

The tokenizer has its own oracle: `tokenizer_golden.py` writes transformers' ids, decodes and
chat prompts to `ripple.json` and `GoldenTest` compares them, 989 checks, including tool calls,
tool lists and `clear_thinking`.

## Results

Real weights on the GB10 (121.7 GiB), `ricursiveAI/Ripple-65B-Preview-NVFP4` and the DFlash2 draft.

```bash
./gradlew :maestro:exportHfServingArtifact -PexportHeap=24g \
    -PckptDir=<NVFP4 snapshot> -PdraftCkptDir=<DFlash2 snapshot> -PoutDir=<artifact> \
    -PmodelName=ricursiveAI/Ripple-65B-Preview-NVFP4 \
    -PweightDType=bf16 -PweightQuant=nvfp4 -PattentionQuant=none -PkvDtype=fp8 -PcudaKernels=true \
    -PmtpDraftTokens=15 -PmtpVerifyDrafts=3,7 \
    -PmaxBatch=4 -PcontextLadder=8192,32768,131072 -PnumBlocks=8400 -PprefillChunk=128,512 -PprefillMaxBatch=2 \
    -Psampling=true -PprefillActQuant=projections
./gradlew :maestro:exportTritonModel -PartifactDir=<artifact> -PoutDir=<repository> -PmodelName=ripple -Pdecoupled=true
TLALOC_PJRT_MEMORY_FRACTION=0.1 TLALOC_PJRT_CACHE_DIR=<dir> MODEL_REPOSITORY=<repository> triton/run_server.sh
./gradlew :endpoint:installEndpoint
endpoint/build/endpoint/bin/tlaloc-endpoint --model ripple --tokenizer <NVFP4 snapshot> --port 8100 --sampling
```

The manifest carries the draft policy (a step cost of 0.02 per draft, at most 16 verified rows),
which the runs below before it set with `TLALOC_DRAFT_COST` and `TLALOC_DRAFT_WIDTH_ROWS`. Those
runs also used `triton/openai_server.py`:

```bash
python triton/openai_server.py --model ripple --tokenizer <NVFP4 snapshot> --max-model-len 131072
```

Export: 39 entries, 1,519 weights, 62,291 MiB, about 4 min. First load: 1,477 s of compiles;
from the executable cache about 105 s for 3 entries, 190 s for 21.

Memory: 6,282 MiB of KV pools (FP8), 31 GiB available when idle, about 20 during the full
GSM8K run, 16 at the lowest. KV in f32 is 23.6 GiB and leaves 9.
With a ring per draft window (4,096 for four layers, 16,384 for the fifth) the pools are 5,246
MiB, 1,036 MiB less, and the full GSM8K run left 25 GiB at the lowest.

### Acceptance by draft position

GSM8K 5-shot completions, 15 drafts, first 200 questions, one stream, against vLLM's table
(PR #65745, the row with `no_rope_layers` honored, all 1,319 questions):

| | p1 | p2 | p3 | p4 | p5 | p6 | p8 | p10 | p12 | p15 | tokens per step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Tlaloc, f32 KV | .91 | .81 | .73 | .66 | .59 | .51 | .37 | .25 | .16 | .08 | 7.24 |
| Tlaloc, FP8 KV | .91 | .81 | .72 | .66 | .59 | .52 | .38 | .25 | .16 | .08 | 7.24 |
| vLLM | .91 | .82 | .73 | .65 | .58 | .52 | .38 | .26 | .17 | .09 | 7.34 |

The FP8 row ran at four streams. vLLM's table calls 7.34 the mean acceptance length; its
positions sum to about 6.3, so it appears to count the verified token, as the last column
does here. vLLM's row with the draft's NoPE layer wrongly rotated is .86 .71 .58 at p1 to p3.

### Speed

A verify step of one stream is 388 ms: kernels 377 ms, of which BF16 attention and the draft
through XLA's GEMM 173 ms (about 34 GB) and the NVFP4 MLP kernel 160 ms (about 26 GB). The
62,291 MiB of weights at 273 GB/s are 239 ms: the step is bound by memory bandwidth, not by the draft count
(3 drafts 318 ms, 15 drafts 388 ms).

Above 16 rows `NVFP4_MATMUL` used to unpack the whole weight on each call: 32 rows took
2,220 ms and 64 rows 2,344 ms. It now runs one kernel call per 16 rows up to 128 rows.

| Stream count | Drafts | Step | Tokens per step | Tokens/s, total |
|---|---|---|---|---|
| 1 | 15 | 388 ms | 7.24 | 18.6 |
| 4, before | 15 | 2,344 ms | 24.6 | 10.5 |
| 4 | 3 | 379 ms | 12.7 | 33.6 |

Over the full GSM8K run: 8,261 steps of four streams at 389 ms and 12.9 tokens per step, 33.2
tokens/s. Prefill took 7,431 s of device time and decode 3,549 s, so prefill is what limits a
short-prompt workload at four streams.

| Prefill call | Context | Time |
|---|---|---|
| 128 tokens, one sequence | 8,192 | 1.8 s |
| 512 tokens, one sequence | 8,192 | 3.4 s |
| 512 tokens, two sequences | 8,192 | 5.1 s |
| 512 tokens, one sequence | 32,768 | 4.5 s |
| 512 tokens, one sequence | 131,072 | 8.3 s |

A decode step at context 131,072 with 7 drafts is 715 ms against about 350 ms at 8,192.

With `-PprefillActQuant=projections` (W4A4: each prefill row quantized to NVFP4 and multiplied on
the block-scaled FP4 instruction) a 512-token call at context 8,192 is 1.84 s (1.73 s at best),
1.85 times faster. The MLP projections of one call took about 3.0 s at W4A16 and take about
0.94 s. At Ripple's shapes, 512 rows:

| Projection | W4A4 | W4A16 |
|---|---|---|
| gate and up | 6.79 ms | 24.13 ms |
| down | 4.90 ms | 13.56 ms |

The kernel reaches 46 to 67 TFLOPS there. GSM8K's first 200 questions at four streams through
the Kotlin endpoint, with W4A4 prefill: 0.940, 188 right, as in the full run's first 200 (each
has two right that the other has wrong), 3.87 tokens per step at 381 ms.

### Long context

A code word in a long prompt, asked for at its end:

| Prompt tokens | Word's position in the prompt | Result |
|---|---|---|
| 1.6K, 4.9K, 7.3K | three positions each | 9 of 9 |
| 16,280 | 50% | found, 130 s in all |
| 102,530 | 10% | found, 1,437 s in all |

Through the Kotlin endpoint, which ends the least recently used idle sequence when the backend
refuses a new one for pages:

| Prompt tokens | Word's position in the prompt | Result |
|---|---|---|
| 16,280 | 50% | found, 152 s |
| 102,530 | 10% | found, 1,594 s |
| 102,530 | 90% | found, 1,642 s; two idle sequences ended to make room |

With draft rings and W4A4 prefill, where both draft rings wrap:

| Prompt tokens | Word's position in the prompt | Result |
|---|---|---|
| 16,280 | 50% | found, 80 s |
| 102,530 | 10% | found, 1,062 s |
| 102,530 | 90% | found, 1,057 s |

The two 102K requests took 4.96 tokens per step against 4.71 without rings.

### Accuracy

GSM8K 5-shot completions, 256 tokens, the stops vLLM's PR uses:

| | Questions | Accuracy | Invalid |
|---|---|---|---|
| Tlaloc, FP8 KV, four streams, widths 3, 7 and 15 | 1,319 | 0.923 | 0 |
| The same with draft rings and W4A4 prefill | 1,319 | 0.921 | 0 |
| vLLM, NVFP4 | 1,319 | 0.916 | not published |

The gap of 0.007 is 9 questions of 1,319, about one standard error, so this shows the two agree
and not that either is better. 2 of the first 200 answers differ between one stream with f32 KV
and four streams with FP8 KV. The run with draft rings got 16 questions right that the first got
wrong and 18 the other way, at 3.86 tokens per step and 382 ms per step against 3.87 and 381 ms
for W4A4 prefill without rings; it took 6,973 s.
