# Qwen3.8-Flash-Next for four users: plan

Branch `feat/flash-next` from `main` at `f191eae2` (2026-10-17).

Goal: serve `Qwen3.8-Flash-Next` (`qwen4_exp`) on one GB10 to four concurrent users, faster
per user than the published single-Spark recipes, measured as they measure.

## Published numbers (one Spark, four streams, tokens/s per stream)

| Recipe | Prose | Code |
|---|---|---|
| vLLM, NVFP4 ([MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)) | 27.7–28.4 | |
| vLLM, NVFP4, table on disk ([tonyd2wild](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark)) | 26.4 (40 prompts) | |
| TensorFold, NVFP4 ([Mia](https://mia-ai.net/models/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)) | 37.2 | 49.6 |
| TensorFold, INT4-AutoRound, top-5 experts | 40.2 (temperature 0.6) | |

What the faster ones do:

- **The n-gram table in memory.** Mia's checkpoint stores it as NVFP4 codes (26.8 GiB
  against NVIDIA's 47.7 GiB FP8); TensorFold keeps an FP8 table on the GPU. A table on disk
  needs its rows read on the host before each step (tonyd2wild stages them before the
  forward so CUDA graphs still run).
- **More tokens per step.** At four streams the routed experts are read at about the memory
  rate by every engine, so speed follows acceptance. vLLM: 3 drafts win at every
  concurrency, a 65K draft vocabulary +25%. TensorFold: draft depth set by confidence
  (`--mtp-drafts 15 --mtp-confidence 0.70`) and copy drafts from earlier text; 2.1 tokens
  per step on fresh prose, 5 on code, 15.5 on verbatim copies.
- **bf16 recurrent state:** +8.5% at eight streams (vLLM).
- **Exactness checked:** TensorFold compares every drafted reply with the same request
  undrafted (`"draft": false`).

## How they measure

- OpenAI chat endpoint, thinking off, temperature 0, warm server.
- 40 prompts in 8 categories (tonyd2wild), in batches of the concurrency started together,
  each ending where the model stops. Per reply: (tokens − 1) / (wall − TTFT). Per batch:
  tokens / wall.
- Fixed length (Mia, TensorFold): N streams of 256–600 tokens; per stream (tokens − 1) /
  (last token − first token); aggregate all tokens / (earliest first token to latest last).
  Prose and code reported apart (acceptance differs about 1.5×). Median of 3.
- Acceptance per position from the server's counters.
- Prefill ladder 8K–256K with a nonce per prompt; decode gaps (p50/p95/p99) while another
  user prefills 64K.

Tlaloc now runs the same: `triton/openai_server.py` (OpenAI endpoint over the Triton
backend) and `scripts/spark-4user/streams.py` (`categories` with tonyd2wild's 40 prompts in
`prompts-40.json`, `fixed` for prose and code). Raw results:
[flash-next-results.jsonl](flash-next-results.jsonl).

## Baseline: Qwen3.6-35B-A3B, Tlaloc against vLLM 0.29 (2026-10-17)

Same machine, same harness. Tlaloc: NVFP4 weights and heads, FP8 KV, 3 drafts (draft
vocabulary 65536+288), `-PcudaKernels=true`, contexts up to 32K. vLLM: the configuration of
[spark-4user-serving.md](spark-4user-serving.md) (NVFP4, FP8 KV, MTP 3, prefix caching).

| | Tlaloc | vLLM |
|---|---|---|
| 40 prompts, 1 stream: decode median / TTFT | 132.1 / 0.14 s | 128.4 / 0.09 s |
| 40 prompts, 4 streams: decode median / TTFT | 73.3 / 0.28 s | 77.9 / 0.14 s |
| prose 512, 1 stream | 91.5 | 88.5 |
| prose 512, 4 streams: per stream / aggregate | 52.6 / 203.5 | 56.1 / 219.7 |
| code 512, 1 stream | 132.7 | 124.8 |
| code 512, 4 streams: per stream / aggregate | 81.0 / 323.3 | 78.0 / 311.6 |
| tokens per step, 4 streams: prompts / prose / code | 2.87 / 2.23 / 3.32 | 3.02 / 2.31 / 3.40 |

The step times match (prose at four streams: 42.4 ms against 41.2 ms); the difference is
acceptance (our drafts use an NVFP4 head over 65,824 ids) and the first token (twice
vLLM's).

## Order of work

1. ~~OpenAI endpoint and the community harness; Tlaloc against vLLM on the 35B.~~
2. ~~bf16 Gated DeltaNet state.~~ `-PstateDtype=bf16`: the recurrent pools are bf16; the
   kernel and the XLA forms widen a row's state when read, carry it in f32 over the row's
   tokens and round it when written. Server compute of a four-stream verify step
   (`profile.sh`, `decode:256:48:4` / `decode:8000:48:4`): 49.31 → 46.66 ms and 53.93 →
   52.10 ms. The Qwen3.5-0.8B fixture keeps transformers' 16 ids (alone, batched,
   staggered). Served, four streams: prose 52.6 → 56.0 tokens/s, 40 prompts 73.3 → 74.2.
3. Acceptance (measured; no change kept). Tokens per step, one stream, bf16 state:

   | | 40 prompts | prose | code |
   |---|---|---|---|
   | 3 drafts (NVFP4 draft head, 65,824 ids) | 2.94 | 2.28 | 3.38 |
   | MTP layer bf16 (`-PmtpLayerQuant=none`) | 2.89 | 2.26 | 3.41 |
   | FP8 draft head, whole vocabulary | 2.94 | 2.25 | 3.40 |
   | 5 drafts | 3.44 | 2.33 | 4.27 |
   | 7 drafts | 3.64 | 2.49 | 4.94 |

   The draft head, its vocabulary and the MTP layer's precision do not move acceptance;
   vLLM's 2.98–3.02 on the 40 prompts comes from a target whose attention and Gated
   DeltaNet projections stay bf16 (ours are FP8 under `-PweightQuant=nvfp4`), so its
   replies differ. A draft costs about 2 ms a step at one stream (24.5, 28.2, 32.3 ms for
   3, 5, 7 drafts on prose): 5 drafts gain alone on the 40 prompts (129.7 → 148.6 tokens/s)
   and code (137.6 → 146.7) and lose on prose (92.2 → 82.6). At four streams a 5-draft
   step takes 59 ms against 40 (24 rows pass `tlaloc_fp4_gemm`'s 16 and read more distinct
   experts): prose 56.0 → 41.2, code 78.1 → 68.0. Depth set by confidence pays at four
   streams only if a dropped draft costs nothing in the experts, the 4-bit GEMM, attention
   and the state writes; its bound is the one-stream gain above. Copy drafts need the host
   to choose a step's drafts, which chained steps (issued before the host reads the step
   before) do not allow. Both left for after Flash-Next.
4. Time to the first token (part done). One request of a 60-token prompt over gRPC: 95 ms,
   the prefill call (`prefill_b1_c2048_t128`: 27 ms to launch, 68 ms more until its token is
   on the host; 128 rows reach nearly every expert, about the weights' 18 GB). The endpoint
   added up to 34 ms: ending an evicted sequence before sending the prompt (now in the
   background; Triton queues the new START until the END frees a slot) and small streamed
   writes without TCP_NODELAY. A 64-token prefill entry (`-PprefillChunk=64,128,2048`)
   for short prompts: 40 prompts at one stream 0.13 → 0.10 s (vLLM 0.09). Four prompts sent
   together are prefilled in one call, but padded to 128 rows each: 512 rows, about 250 ms,
   so four streams stay at 0.26 s (vLLM 0.14; four prompts under 64 tokens: 0.14 s). Left:
   prefill without padding, either a row mask through the MoE kernels and attention or
   packed prompts of different lengths, as vLLM does.
5. ~~`qwen4_exp`, a 4-layer parity fixture~~ (target model; the MTP head is left). The
   family `HfModelFamily.Qwen4Exp` reads Qwen3.8-Flash-Next's config and every tensor of
   its text model:
   - four residual streams (`GatedResidualConfig`): per sublayer a grouped RMSNorm of the
     streams, a rank-320 gated mix as the input, the output added to each stream with its
     own weight; a last mix of the streams in place of the final norm;
   - the n-gram memory on layer 1 (`PleConfig`): `NGRAM_HASH` (a new op: the splitmix
     multipliers, a prime per head, 64-bit products and XORs), the table's 16 rows of 160
     per token, gated key and value projections, the conv (kernel 4, dilation 3) run as a
     10-tap `CAUSAL_CONV1D`. The token history is a `CAUSAL_CONV1D` too, over `id - eos`
     with one-hot taps, so the state slots and speculative writes carry it unchanged;
   - query-selective sparse attention on the 12 attention layers (`IndexerConfig`): the
     indexer keys and each 4-position chunk's pooled key (mean, norm, RoPE at the chunk's
     start, written where the chunk ends) in two paged pools; `INDEXER_TOPK` (new) keeps
     the 512 best chunks and the incomplete one; `PAGED_ATTENTION` takes the keep mask as
     a sixth operand. A context of at most 2,051 positions keeps every chunk, so entries
     up to 2,048 skip the selection (exact);
   - the Gated DeltaNet's gated norm by `sigmoid(z)`.

   Checks: `tiny_qwen4_exp/` (transformers 5.17, f32, 4 layers, 4 streams, n-gram layer,
   indexer keeping 2 chunks of 4, a prompt of 22 tokens with an eos inside):
   `HfQwen4ExpParityTest` gives transformers' ids with every logit within 2.2e-8 (one
   sequence; two sequences prefilled in calls of 5 tokens). Attending to every chunk
   instead is 9.4e-5 away, so the tolerance is 1e-6. On the GB10 (`PjrtQwen4ExpParityTest`)
   the ids are transformers', logits within 5.4e-5; `PjrtQwen4ExpOpsTest` has the
   emitted hash equal to the reference at Flash-Next's own constants (ids up to 3.2e8),
   the keep masks equal, masked attention within 3e-8. `HfQwen4ExpCheckpointTest` reads
   nvidia/Qwen3.8-Flash-Next-NVFP4: 48 layers, the table's 128 FP8 shards stack to
   320,001,536 x 160 rows, the stored hash constants equal the computed ones.

   The MTP head (as vLLM's `qwen4_exp/nvidia/mtp.py` wires it; transformers does not run
   it): `pre_fc_norm_hidden` normalizes the target's four streams before the last mix
   together (one RMS over 4 × 2560); stream i of the layer's input is
   `fc_hidden(stream i) + fc_embedding(pre_fc_norm_embedding(e))`; one gated-residual
   indexed attention layer (its experts FP8 in 128 × 128 blocks in the checkpoint), then
   the head's own last mix for the logits. The next draft reads the layer's streams before
   that mix, so `mtpHidden` carries 4 × 2560 per state slot, and the head's layer has its
   own indexer key and chunk key pools (`mtpIndexKeys`, `mtpChunkKeys`). Past 2,048
   positions every draft runs its own chunk selection; vLLM reuses the first draft's.

   Checks: `tiny_qwen4_exp/` now holds a random head and `mtp.json`, the drafts after every
   position of both prompts (transformers' decoder layer, norms and gated residual with the
   `mtp.*` tensors, each argmax ahead of the next logit by 1e-4, every indexer selection
   with a gap at its boundary). `HfQwen4ExpMtpTest` (interpreter, 3 drafts, a 48-position
   bucket, so the head's indexer selects): the speculative prefill gives the target's next
   token and the oracle's drafts; verify steps feed the head's drafts, the target's
   continuation, or its first token and a wrong one (accepted 0, 3, 1, 0, 3), and after
   every step the drafts are the oracle's and the ids transformers'; chained steps too.
   On the GB10 (`PjrtQwen4ExpMtpTest`) the same ids and drafts after every step. `HfQwen4ExpCheckpointTest` maps every tensor of
   Flash-Next's head as well.

   The indexer and the masked attention in the CUDA kernels: step 6.
6. ~~Export and serve with the n-gram table as NVFP4 in memory, chained steps kept~~ (done
   for contexts up to 8,192; see "Step 6" below).
7. ~~Measure against the table above~~ (done; see "Step 7" below). Target: above 37
   tokens/s per stream on prose and 50 on code at four streams. Not met: 24.7 and 36.9.

## Step 6: Qwen3.8-Flash-Next served (2026-10-24)

- **n-gram table as NVFP4** (`-PpleTableQuant=nvfp4`, op `NVFP4_EMBEDDING`): codes
  `[320,001,536, 80]` and an e4m3 scale per 16 values `[320,001,536, 10]`, 26.8 GiB against
  the checkpoint's 47.7 GiB of FP8. The FP8 shards are rounded shard by shard with tensor
  scale `s / 6`; this equals `Nvfp4Quantizer` on the widened table. An independent numpy
  rounding of 4,608 sampled rows of the checkpoint gives the exported bytes exactly. The
  NVFP4 values differ from the FP8 ones by 8.4% of their magnitude on average
  (`Σ|v − v8| / Σ|v8|`). On the tiny fixture the interpreter's logits equal an f32 table's
  holding the same values bit for bit, and the GB10's are within 3.8e-5 of the
  interpreter's with the same ids.
- **Sparse attention in the CUDA kernels** (`-PcudaKernels=true`). `INDEXER_TOPK` runs as
  `tlaloc_indexer_topk`: f32 scores against each chunk's pooled key, then a radix select
  that keeps ties in chunk order. `PAGED_ATTENTION` with its keep operand runs as
  `tlaloc_paged_attention`: every form masks dropped chunks, and the f16 prefill form skips
  a 32-position block that none of a warp's rows keeps. `PjrtFusedSparseAttentionTest`: the
  reference's keep masks, and the interpreter's outputs at Flash-Next's layout.
- **Export** (`exportHfServingArtifact`, the flags of the 35B exports plus
  `-PpleTableQuant=nvfp4`):
  - NVFP4 weights and heads, 3 drafts, FP8 KV, `-PcudaKernels=true`, 8 state slots.
  - Context 8,192, 2,200 pages, decode batches 1, 2, 4, prefill calls of 128 and 512 tokens
    for up to 2 sequences: 7 entries, 1,633 weights, 100,029 MiB, written in 8.4 min.
  - The first program compile needs a PJRT fraction that holds the 23.9 GiB table, because
    XLA's autotuner allocates the operands of the fusion that reads it (it fails at 0.045).
    The 7 programs were compiled once at 0.6 into `TLALOC_PJRT_CACHE_DIR` and are loaded from
    there in 18 s.
- **Memory on the GB10** (121.7 GiB):
  - The weights are outside the PJRT pool, in device memory of their own (backend default
    since this step). In the pool at 0.83, the 97.7 GiB took MemAvailable to 3 GiB before
    the last weights were in.
  - Served with `TLALOC_PJRT_MEMORY_FRACTION=0.045`: 1,896 MiB of KV, indexer and state
    pools; ready in 150–175 s (weights upload 130 s); 11 GiB available when idle, 6 GiB at
    the lowest under four streams.
  - XLA's temporaries per prefill call: 1.5 GiB (one sequence × 512 tokens), 2.9 GiB (2 ×
    512), 5.4 GiB (1 × 2,048), 9.7 GiB (4 × 2,048). That is why calls are 512 tokens and at
    most 2 sequences.
  - Contexts of 32,768 (36 entries, 8,800 pages) and 131,072 do not fit with headroom.
- **Checks of the served model** (endpoint `openai_server.py`, greedy, thinking off):
  - Chat replies are coherent, and the same text with and without chained steps
    (`TLALOC_CHAIN_STEPS=0`).
  - A code word placed at 10, 50 or 90% of a document of 1,555, 3,055 or 7,055 tokens is
    answered correctly 9 times of 9. These contexts pass 2,048, so the indexer selects.
  - Four prompts sent together: one of four replies equals the reply sent alone. The others
    part at a near tie after 215–458 characters, and the same way in two runs ("red and
    yellow" against "red and orange"). Batches of 1 and 4 run different programs.
  - No independent reference: transformers needs the 124 GB checkpoint dequantized (about
    350 GB in bf16) for a CPU forward, and vLLM keeps the 51 GB FP8 table on the GPU. The
    tiny fixture is the reference: transformers' ids in the interpreter and on the GB10,
    and the served fixture gives the interpreter's ids alone and together.
- **First readings** (`streams.py fixed`, 512 tokens, median of 3 rounds;
  [flash-next-results.jsonl](flash-next-results.jsonl), labels `flash-next-fixed-*`):

| | tokens/s per stream | aggregate | TTFT | tokens per step |
|---|---|---|---|---|
| prose, 1 stream | 26.1 | 26.2 | 0.31 s | 2.13 |
| prose, 4 streams | 24.1 | 96.4 | 0.80 s | 2.13 |
| code, 1 stream | 41.9 | 42.0 | 0.31 s | 3.48 |
| code, 4 streams | 36.4 | 145.9 | 0.79 s | 3.26 |

  At four streams this is below the published 27.7–28.4 (vLLM) and 37.2 (TensorFold) on
  prose and 49.6 (TensorFold) on code.
- **Where a step goes** (`profile.sh` without chaining, 256-token context):
  - Server compute per verify step: 88 ms at one stream, 127 ms at four; 95 and 131 ms at
    6,000 tokens of context.
  - nsys, one stream, 80.6 ms of kernels: the Gated DeltaNet in-projection (FP8
    `[2560, 16384]` per layer, 1,440 MiB in all, XLA's Triton GEMM with the e4m3 widening
    fused, 4 rows) 26.0 ms, about 58 GB/s; the routed experts 19.2 ms; widening the
    attention layers' FP8 query projections (`[2560, 12288]`) to bf16 in a kernel of their
    own every step, 4.8 ms.
  - nsys, four streams, 119 ms of kernels: experts 53.0 ms, in-projection 22.5 ms.
  - Prefill: 566 ms per 512-token call at 2,000 tokens of context (900 tokens/s).
- Left:
  - the in-projection and the attention projections as kernels that read the FP8 codes at
    the memory rate for a few rows;
  - longer contexts (more pages, fewer entries, smaller prefill temporaries);
  - prefill calls of 2,048 tokens and of 4 sequences (their temporaries);
  - the measurement against the published table (step 7, below).

## Step 7: Qwen3.8-Flash-Next measured (2026-10-24)

The step-6 export and server unchanged (`main` at `ac123041`: context 8,192, 3 drafts,
chained steps, `TLALOC_PJRT_MEMORY_FRACTION=0.045`), `openai_server.py` in front,
`streams.py` as "How they measure" says: greedy, thinking off, one warm-up request. Raw
results: [flash-next-results.jsonl](flash-next-results.jsonl), labels
`flash-next-categories-c1`, `-c4`, `flash-next-fixed-*-step7`, `flash-next-fixed-*-4-distinct`.

| | Tlaloc | vLLM, NVFP4 | TensorFold, NVFP4 |
|---|---|---|---|
| prose 512, 1 stream | 26.4 | | |
| prose 512, 4 streams: per stream / aggregate | 24.7 / 99.1 | 27.7–28.4 | 37.2 |
| code 512, 1 stream | 42.3 | | |
| code 512, 4 streams: per stream / aggregate | 36.9 / 147.7 | | 49.6 |
| 4 streams, a prompt of its own per stream: prose / code per stream | 17.7 / 26.7 | | |
| 40 prompts, 1 stream: decode median / TTFT | 39.4 / 0.31 s | | |
| 40 prompts, 4 streams: decode median / TTFT / batch aggregate median | 23.9 / 0.78 s / 47.2 | 26.4 (40 prompts, tonyd2wild) | |
| tokens per step: prose / code / 40 prompts | 2.13 / 3.26–3.48 / 2.90 | | 2.1 (fresh prose) / 5 (code) |

- **Target not met.** At four streams prose is 24.7 tokens/s per stream (target 37, vLLM
  27.7–28.4, TensorFold 37.2) and code 36.9 (target 50, TensorFold 49.6). The fixed rows are
  the median of 3 rounds and repeat the step-6 readings within 2.5%.
- **One prompt for all streams flatters a batch.** `fixed` sends the same prompt to every
  stream; greedy, the four streams emit the same tokens in step and every routed expert is
  read once for all four rows. A verify step at four streams (timeline, `TLALOC_TIMELINE=1`):
  - 85 ms with the four streams in step (80 ms at one stream);
  - 101–107 ms when the fourth prompt was prefilled one call later, so its stream ran one
    step behind (the first round of each 4-stream run: 82 against 98–99 tokens/s aggregate
    on prose, 121 against 148 on code);
  - 115 ms (prose) and 123 ms (code) with a prompt of its own per stream
    (`fixed --distinct`): 17.7 and 26.7 tokens/s per stream, 69.7 and 104.3 aggregate.

  The published recipes do not say whether their streams share a prompt. The 35B baseline
  above sent one prompt to both engines.
- **Acceptance:** 2.13 tokens per step on prose, 3.26 (4 streams) to 3.48 (1 stream) on code,
  2.06 / 3.31 with distinct prompts, 2.90 over the 40 prompts. TensorFold reports 2.1 on fresh
  prose and 5 on code with confidence-set draft depth up to 15; vLLM's acceptance is not
  published for this model.
- **Per category, 1 / 4 streams (decode median tok/s):** coding 41.8 / 25.5, reasoning 43.5 /
  33.8, json 40.4 / 24.1, html 44.1 / 32.2, prose 26.9 / 20.7, narrative 26.5 / 23.1, summary
  31.3 / 18.0, format 27.6 / 13.3. All 80 replies end at the model's stop; 24 of 40 replies at
  four streams equal their one-stream reply (the rest part at near ties, as in step 6).
- **TTFT:** 0.30–0.31 s at one stream: a 128-token prefill call of one sequence takes
  197–233 ms. 0.78 s at four streams: the export's largest prefill call holds 2 sequences,
  so four prompts arriving together run as two calls of 375–378 ms one after the other (a
  128-token call reads nearly all the weights). A 4-sequence × 128-token entry would make it
  one call; its temporaries (512 rows, as one sequence × 512 tokens: 1.5 GiB) were not
  tried.
- **Memory:** low-water MemAvailable 6 GiB in every run (watchdog at 4 GiB).
- **Left:** the step time (step 6: the Gated DeltaNet's FP8 in-projection through XLA's GEMM
  for a few rows, the experts at four distinct streams), and draft depth (acceptance 2.1 on
  prose against 5 on code for TensorFold with deeper drafts).
