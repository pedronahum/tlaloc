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
3. Acceptance: find what the NVFP4 draft head and the reduced vocabulary cost against
   vLLM's 3.02; then draft depth by confidence and copy drafts. `tlaloc_fp4_gemm` takes 16
   rows (4 × (1 + 3)); more drafts need it wider.
4. TTFT at four streams (0.28 s against 0.14 s).
5. `qwen4_exp` (transformers 5.20 `models/qwen4_exp`), a 4-layer parity fixture:
   - four residual streams joined by a low-rank gated mix;
   - the n-gram memory layer (splitmix-hashed ids of the last 2 and 3 tokens, 16 rows of 160
     bytes a token, gated key and value projections, a dilated depthwise conv, a
     token-history state per sequence, which the state slots and speculative slots must
     also hold);
   - query-selective sparse attention: an indexer scores blocks of 4 keys and the layer
     attends to the top 2,048 tokens, with its own key cache;
   - 512 experts, top 10, width 640, hidden 2560, 48 layers (36 Gated DeltaNet with 48
     value heads, 12 full attention); MTP.
6. Export and serve with the n-gram table as NVFP4 in memory (weights about 72 GiB + table
   27 GiB + 4 × 128K FP8 KV about 9 GiB, of 121 GiB), chained steps kept.
7. Measure against the table above. Target: above 37 tokens/s per stream on prose and 50 on
   code at four streams.
