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

   Left for serving: the MTP head (vLLM's `qwen4_exp/nvidia/mtp.py`: `fc_embedding` of the
   normed token embedding added to `fc_hidden` of each normed stream of the target's
   streams before the last mix, one indexed attention layer with FP8 block experts, its
   own last mix; a draft step feeds its streams to the next), and the indexer and the
   masked attention in the CUDA kernels (contexts past 2,048).
6. Export and serve with the n-gram table as NVFP4 in memory (weights about 72 GiB + table
   27 GiB + 4 × 128K FP8 KV about 9 GiB, of 121 GiB), chained steps kept.
7. Measure against the table above. Target: above 37 tokens/s per stream on prose and 50 on
   code at four streams.
