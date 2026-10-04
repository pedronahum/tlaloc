# Qwen3.5 hybrid models in the serving path: progress log

Plan: [qwen35-plan.md](qwen35-plan.md). Branch `feat/qwen35` from `main` at `0fc7581`.

## Summary

Built:

1. the `qwen3_5` family;
2. mixture of experts;
3. FP8 weights, with quantized checkpoints read;
4. the FP8 KV cache;
5. decode attention as exact f32 dot algorithms;
6. MTP speculative decoding;
7. a fused paged-attention kernel;
8. an FP8 head for the drafts;
9. NVFP4 MLP weights with a 4-bit GEMM kernel.

Decode, tokens/s per user (follow-up turn), through Triton on the GB10, FP8 KV, 3 MTP
drafts. Qwen3.6-35B-A3B: FP8 weights, before the fused attention kernel. Qwen3.8-27B: the
NVFP4 checkpoint (NVFP4 MLPs, FP8 elsewhere), fused attention, FP8 draft head:

| | Qwen3.6-35B-A3B | Qwen3.8-27B |
|---|---|---|
| 1 user, 2K | 54.3 | 21.8 |
| 4 users, 2K | 17.0 | 13.1 |
| 1 user, 30K | 34.9 | 22.3 |
| 4 users, 30K | 15.0 | 11.2 |
| vLLM 0.29 (NVFP4, FP8 KV, MTP): 4 users at 2K / 100K | 63–68 / 28–38 | 23–24 / 12–14 |

The vLLM figures are from [spark-4user-serving.md](spark-4user-serving.md). Without MTP
see [MTP speculative decoding](#mtp-speculative-decoding).

The gap to vLLM, largest first:

- **Long-context attention:** XLA's form gathers each row's whole context bucket and
  writes it out before the dots. The fused kernel (`-PcudaKernels=true`) reads
  the pages in place and cuts a four-stream step at 30K by 17% (27B) and 13% (35B); see
  [the fused paged-attention kernel](#the-fused-paged-attention-kernel-in-the-serving-path).
  The throughput table above predates it.
- **MoE experts at four sequences** are read once per row-expert pair, so a verify step
  reads about four times a plain step's expert weights.
- **4-bit weights:** the MLPs of the NVFP4 checkpoints are served as NVFP4 by
  `tlaloc_fp4_gemm`; the 35B's routed experts are still FP8.
- **LM head:** the drafts read an FP8 copy; the verified rows read the bf16 head.
- **Prefill** is 500–800 tokens/s for the 27B, against about 1,200 in vLLM.

Open items: [FOLLOWUPS.md](../FOLLOWUPS.md) section 8.

## Part 1: the qwen3_5 family

### Ops (2087447, 0d0058d)

`CAUSAL_CONV1D` and `GATED_DELTA_RULE` are inference-only op kinds with two results each:
the output and the updated per-sequence state pool.

- **Row rules** (`LinearStateRows`):
  - padding first in a row;
  - one slot per row, and no slot shared between rows;
  - a row whose first live token is at position 0 starts from zero state.

  The interpreter refuses a block that breaks them, by name.
- **Interpreters:** one walk (`LinearStateWalk`) for both, summing in double. The f32
  interpreter rounds the carried state to f32 after every token.
- **StableHLO:**
  - the conv has no loop: each row's live tokens are moved to the front with a gather,
    follow the row's K-1 history values, and are moved back after the K
    multiply-adds;
  - the delta rule for one token per row (decode) is a `stablehlo.while` over the
    tokens;
  - the delta rule for several tokens per row is the chunked form: 64-token chunks
    computed with matmuls and one unit lower triangular solve each, and a while over
    the chunks;
  - in both ops, row states are gathered from the pool before the computation and
    scattered back after it, and a padding row's slot of -1 is dropped by the scatter.

| Check | Result |
|---|---|
| Both interpreters against transformers' `causal_conv1d_fn` / `causal_conv1d_update` / `torch_recurrent_gated_delta_rule` (fixture: a new sequence, a padding row, a continuation behind padding, then a decode call) | within 2e-6 |
| transformers' recurrent against its chunked form on the same fixture | 1.2e-7 |
| GB10 against the interpreter: T=5, T=64 with grouped heads, T=200 (four chunks, the last padded) | within 1.1e-7 |

### Family and graph (e3b214b)

- **`HfModelFamily.Qwen3_5`**:
  - the text decoder of `Qwen3_5ForConditionalGeneration`;
  - unknown keys are refused, as are `attn_output_gate = false`, a non-f32
    `mamba_ssm_dtype` and a non-empty `mlp_only_layers`;
  - image and video placeholder tokens are refused.
- **`DecoderLayerSpec`** gains:
  - `mixer` (attention or Gated DeltaNet);
  - `queryGate` (`q_proj` gives `[query | gate]` per head).
- **`HfDecoderConfig`** gains:
  - `linearAttention`;
  - `partialRotaryFactor`;
  - `qkNormGainPlusOne` and `finalNormGainPlusOne`.
- **Weight slots:** `A_log`, `dt_bias` and the gated norm's gain are staged f32
  whatever the weight dtype; the checkpoints store the first and last of these in f32.
- **Signature** (`DecodeGraphSpec`):
  - with a `LinearStatePool` there is a `stateSlots` `[B]` input after the slot
    mapping;
  - each linear layer's two pools are `convState$l` `[S, K-1, C]` and
    `recurrentState$l` `[S, Hv, Dk, Dv]`, roles `STATE_POOL_IN` and `STATE_POOL_OUT`;
  - the cache key gains `ls<layers>n<slots>`;
  - other families' signatures are unchanged.

| Check | Result |
|---|---|
| Qwen3.5-0.8B (24 layers: 18 Gated DeltaNet, 6 gated attention), weights f32, interpreter against transformers' greedy fixture, 2 prompts | ids exact, logits within 1.2e-6 (4 tokens each; `TLALOC_QWEN35_FULL=1` checks 16) |
| Same, on the GB10 through PJRT | 16 of 16 ids for both prompts |
| Tiny random model (HfQwen35GraphTest): a prompt in one prefill call, in two, or prefilled then decoded token by token | logits within 2e-5 |
| A row alone and in a batch of two; a reused state slot | bit-identical |

### Serving (e748b60)

- **`tlaloc-serving-v4`:**
  - publishes `model.linearState` (layers, slots, pool dims);
  - the exporter takes `-PstateSlots` (default 8);
  - older readers refuse v4 by version, and `ServingModel` refuses it by name.
- **Triton backend:**
  - each sequence takes a state slot at its first request and holds it to its end;
  - slots are reclaimed from idle sequences as pages are, and a sequence that finds
    none free is refused by name (UNAVAILABLE);
  - every call is handed each row's slot;
  - `max_candidate_sequences` is capped at the slot count.
- **Result:** Qwen3.5-0.8B (bf16 weights, batches up to 4, contexts 512 and 2,048,
  prefill chunks of 512) through Triton. Four concurrent sequences each decode
  transformers' 16 ids, on a fresh server and again with the slots reused.

### Performance on the GB10 (Qwen3.5-0.8B through Triton)

**Prefill:** chunked delta rule, 2,000-token prompt.

| | Time | Rate |
|---|---|---|
| Chunked | 195 ms | ~10,300 tokens/s |

**Decode, after fusing the input projections (9100b53):**

| | Client step | GPU kernels |
|---|---|---|
| 1 sequence | 13 ms | 8.1 ms |
| 4 sequences | 20 ms | 10.0 ms |

- **Fused input projections (9100b53):**
  - staged apart, XLA's dot merger concatenated the layer's four input projection
    weights on every call at batch > 1 (16.8 MB per layer per step);
  - they are now one staged weight and one matmul, which took four-sequence kernels
    from 11.6 to 10.0 ms.
- **Where single-sequence decode goes:** close to weight bandwidth. 1.4 GB of weights
  in 8.1 ms of kernels; the head alone is 2.1 ms.
- **The four-sequence step is half host time** (9.3 ms between executions): each
  response carries the logits row (248,320 floats, 1 MB) to a Python gRPC client that
  samples. Sampling in the backend, returning token ids, is the fix; not done.

### Qwen3.8-27B through Triton (bf16 weights)

Export:

- batches up to 4 and contexts 2K, 8K and 32K;
- prefill chunks of 2,048 and 128 tokens, one prompt per prefill call;
- 8,200 KV pages (f32) and 8 state slots;
- `TLALOC_PJRT_MEMORY_FRACTION=0.72`.

Load: 51.3 GB of weights, 17.6 GB of KV pools, 26 GiB of memory left free. Greedy,
`NEXT_TOKEN`, 128 tokens per turn.

| | 1 user, 2K | 4 users, 2K | 1 user, 30K | 4 users, 30K at once |
|---|---|---|---|---|
| First token, new document | 2.5 s | 10.1 s | 60 s | 238 s |
| Decode, tokens/s per user | 3.7 | 3.2 | 3.2 | 1.9 |
| First token, 13-token follow-up | 0.4 s | 1.4 s | 5.3 s (before 128-token chunks) | 10.8 s (same) |

vLLM 0.29 on the same machine with the NVFP4 checkpoint, an FP8 KV cache and MTP decodes
22–24 tokens/s for one user and 12–14 per user for four at 100K
([spark-4user-serving.md](spark-4user-serving.md)).

- **Decode is the weights' bandwidth.** Each step reads 51 GB of bf16 (at most 5.3
  steps/s at 273 GB/s); one user gets 3.7. FP8 or NVFP4 weights (Part 3) are the fix.
- **Four users at 30K also read 15.7 GB of f32 KV per step.** An FP8 cache (Part 4) is a
  quarter of that.
- **Prefill:**
  - 800 tokens/s at 2K and 500 at 30K, against vLLM's ~1,200 for this model in NVFP4;
  - the four 30K prompts of the last column are prefilled in rounds of one 2,048-token
    chunk per prompt, so all four wait for all four;
  - the attention of a prefill call is f32 over an f32 pool.
- **Follow-ups:** a sequence keeps its KV and state on the server between turns, so a
  follow-up prefills only its own tokens. With a 128-token prefill entry a 13-token
  follow-up takes 0.4 s, not the 4.2 s of a 2,048-token chunk.
- **Triton's idle timeout:** `max_sequence_idle_microseconds` (default 60 s) is counted
  from a request's arrival, and a 30K prompt runs longer than that. The repository was
  written with `-PmaxSequenceIdleMicros=600000000`.

### Open in Part 1

- **Sampling in the backend:** only greedy (`NEXT_TOKEN`); temperature and top-p need
  the logits on the client.
- **No prefix cache across sequences:** a new sequence with a known prefix prefills it
  again.
- **State slots** are a fixed budget per artifact (`-PstateSlots`).
- **Not run in the other runtimes:** the vLLM plugin, `tlaloc_serve.py` and
  `ServingModel` refuse v4 artifacts.
- **No vision tower;** image and video tokens are refused.

## Part 2: mixture of experts (qwen3_5_moe)

### MOE_EXPERTS (2a9bea2)

One inference-only op does the routing and the routed experts:

- `x [R,H]` and `routerLogits [R,E]` in;
- `gateUp [E,2I,H]` and `down [E,H,I]`, the checkpoint's own layout;
- attribute `top_k`.

Per row it takes a softmax, the top-k (ties to the lower expert) and the renormalized
weights, then sums `w · down[e] (silu(g) · u)`.

StableHLO form:

- top-k is a stable sort of the probabilities;
- the `R·k` (row, expert) pairs are sorted by expert;
- pair counts and offsets are exact integer sums, taken as f32 dots against
  triangular matrices;
- a while over tiles of one expert's rows: a gather of the tile's rows, two dots
  against a dynamic slice of the expert's weights, and a scatter-add of the weighted
  output.

Each selected expert is read once per tile: a few experts for a decode step, all of
them about once per prefill chunk.

| Check | Result |
|---|---|
| Both interpreters against transformers' `Qwen3_5MoeTopKRouter` and `Qwen3_5MoeExperts` (fixture) | within 2e-5 |
| GB10 against the interpreter: decode-sized (R=4) and prefill-sized (R=300) blocks, 16 experts, top 4, f32 and bf16 weights | within 3e-7 |

### Family (f946f55)

`HfModelFamily.Qwen3_5Moe` reuses Qwen3.5's layer reading and model settings
(`qwen35Layers`, `qwen35Refine`):

- every layer's MLP is `MlpKind.MOE`, with `MoeConfig` from `num_experts`,
  `num_experts_per_tok`, `moe_intermediate_size` and `shared_expert_intermediate_size`;
- `intermediate_size` may be absent (the shared expert's size stands in).

The graph's MoE MLP is `MOE_EXPERTS` plus the shared expert times
`sigmoid(x · shared_expert_gate)`. The router, the shared expert's gate and up, and its
output gate are one staged weight (`moeIn$l`) and one matmul.

Experts load in either layout:

- stacked (`mlp.experts.gate_up_proj`, the Hub layout);
- one by one (`mlp.experts.N.gate_proj/up_proj/down_proj`, what `save_pretrained`
  writes and what NVIDIA's NVFP4 checkpoint holds), stacked at staging.

| Check | Result |
|---|---|
| Tiny random checkpoint written by transformers (3 Gated DeltaNet + 1 attention layer, 8 experts, top 2, shared expert; committed with its oracle), interpreter | ids exact, every logit within 6e-8 |
| Same, GB10 | ids exact, every logit within 1.7e-5 |

## Part 3 groundwork: what this XLA does with narrow weights (d53b5dd)

- **The types work:**
  - f8e4m3fn bytes widen exactly;
  - packed f4e2m1fn pairs widen exactly, low nibble first.
- **Benchmark:** `PjrtQuantGemvBenchTest`, a `[17408, 5120]` projection (27B MLP size),
  times per call including about 0.3 ms of call overhead:

  | Weights | 1 row | 4 rows |
  |---|---|---|
  | bf16 | 1.31 ms | 1.24 ms |
  | f8, widened in front of the dot, scale after it | 0.70 | 0.80 |
  | int8 the same way, scale per output channel | 0.75 | 0.68 |
  | NVFP4 through `__op$block_scaled_dot` (activations rounded to e2m1 per 16 too) | 0.66 | 0.63 |
  | NVFP4, int4 or f8 with any scale in front of the dot (per 16, per tensor, in f32 or bf16) | 2.3–2.7 | 2.4–2.7 |

  - XLA fuses a plain widening of the weight into the matmul.
  - It does not fuse a multiply on the weight, so a group scale, which sits on the
    contracted axis, materializes the bf16 weight on every call.
  - 4-bit gains almost nothing over 8-bit here, and only by also rounding the
    activations to 4 bits.
- **Decision for Part 3:** stage quantized weights as f8 e4m3 with one f32 scale per
  output channel, applied after the dot. Sources:
  - a bf16 checkpoint, quantized at export;
  - NVIDIA's NVFP4/FP8 checkpoints and Qwen's block-FP8 ones, dequantized exactly and
    requantized.

  f8 rather than int8 because its log spacing keeps the small groups of a row whose
  other groups are large (an NVFP4 row's per-16 scales vary widely). A real 4-bit
  speedup needs a kernel that reads the codes in the GEMM; this XLA has none.

### Decode without the loop (6efaf7d)

A decode step's while loop over expert tiles cost more than its arithmetic: each
iteration paid a round trip and a copy of its dynamic slice. XLA flags
(`dynamic_slice_fusion`, command-buffer while loops) did not change that.

For blocks of at most 64 (row, expert) pairs, `MOE_EXPERTS` now:

1. gathers each pair's expert weights;
2. multiplies them into the pair's row and reduces;
3. weights and sums over the top k.

XLA fuses each projection into one kernel that reads each selected expert once.
Larger blocks (prefill) keep the tiled loop.

`PjrtMoeBenchTest` (`TLALOC_MOE_BENCH=1`), one Qwen3.6-35B-A3B MoE layer (256 experts,
top 8, bf16):

| Rows | Loop | Gathered |
|---|---|---|
| 1 | 0.96 ms | 0.74 ms |
| 4 | 2.29 ms | 1.59 ms |

### Qwen3.6-35B-A3B through Triton (bf16 weights)

Export as for the 27B. Greedy, 128 tokens per turn.

| | Loop | Gathered |
|---|---|---|
| 1 user, 2K: decode, tokens/s | 20.9 | 26.6 |
| 4 users, 2K: decode, tokens/s per user | 7.4 | 11.4 (45 in all) |
| 4 users, 30K: decode, tokens/s per user | 4.8 | 6.6 |

- First token, gathered form:
  - 1.7 s for one 2K document;
  - 6.5 s for four 2K documents;
  - 136 s for four 30K documents;
  - 0.5–1.3 s for a 13-token follow-up.
- vLLM decodes 28–38 tokens/s per user for four users at 100K.

## Part 3: FP8 weights

### Quantized checkpoints read (236ab32)

- **Dtypes:**
  - `F8E4M3FN` and `U8`;
  - safetensors `F8_E4M3` and `U8` tensors are kept as bytes.
- **Dequantized to f32 on load (`HfCheckpoint`):**
  - ModelOpt FP8, per tensor or per row;
  - block FP8 (128 × 128);
  - ModelOpt NVFP4: packed e2m1 codes, an e4m3 scale per 16, a global scale.
- **Check:** NVIDIA's NVFP4 Qwen3.8-27B reads back within 2.7% (FP8 layers) and 9.3%
  (NVFP4 MLP) of the bf16 checkpoint's weights.

### FP8 projections (e907751, 0088c79)

`WeightQuant.FP8` (`-PweightQuant=fp8`) stages the large projections as e4m3fn codes
with an f32 scale per output channel. The codes are widened in front of the dot and the
scale is applied after it, as for int8.

- **What is quantized:**
  - the attention and Gated DeltaNet projections;
  - the MLP and the shared expert;
  - the routed experts, per expert and channel.
- **Fused groups stay fused** (q/k/v with z, the shared expert's gate with its up), with
  their scales concatenated.
- **Rows come from** a bf16 checkpoint, or from a quantized checkpoint's dequantized
  values.

| Check | Result |
|---|---|
| Qwen3.5-0.8B, FP8 or int8 projections, GB10, text prompt | 16 of 16 ids are transformers' |
| Same, chat prompt | first 5 ids (a margin of 0.03 at the sixth) |
| Same, first-step logits | within 1–3% of the largest |
| Tiny Qwen3.5-MoE, int8 or FP8 projections and experts, GB10 against the interpreter | within 5e-5 of the largest logit |

### Qwen3.6-35B-A3B through Triton (FP8 weights)

The artifact holds 34 GB of weights, against 65 GB in bf16.

| | bf16 | FP8 |
|---|---|---|
| 1 user, 2K: decode, tokens/s | 26.6 | 33.6 |
| 4 users, 2K: decode, tokens/s per user | 11.4 | 13.1 (53 in all) |
| 4 users, 30K: decode, tokens/s per user (follow-up turn) | 6.6 | 7.0 |
| 4 users, 30K: first token, new document (median) | 136 s | 86 s |

- **The step is not only weights.** A step reads about 3 GB of active weights in bf16.
  At four users and 30K it also gathers each row's whole context window from an f32
  pool, in each of the 10 attention layers.

### Quantized groups and XLA's dot merger

**Problem.** The first FP8 export of Qwen3.8-27B decoded 6.7 tokens/s for one user but
only 3.5 per user for four. A profile of the four-stream step showed one fusion per
Gated DeltaNet layer, 1.1 ms each (53 ms of a 234 ms step). It concatenated the bf16 b
and a projections (`[5120, 48]` each) with the widened e4m3fn `[qkv | z]` weight into a
bf16 `[5120, 16480]` on every call. This was XLA's dot merger again, which joins dots
on one input. At one row it did not fire.

**Fix.**
- When the projections are quantized, the small projections beside a quantized group
  are staged f32 (`DecoderLayerPart.f32BesideQuantized`): b and a, a MoE layer's router
  and its shared expert's output gate.
- Their matmuls then take the f32 input and have no bf16 partner to merge with.

**Result.**
- Qwen3.8-27B: four users went from 3.5 to 5.3 tokens/s each, and the four-stream
  step from 234 to 163 ms.
- Qwen3.6-35B-A3B with FP8 KV: four users at 2K went from 13.0 to 14.3 tokens/s each
  (57 in all), and at 30K from 7.9 to 8.2. One user decodes 34.4.

**What remains.**
- **Recurrent states:** at four rows, each step transposes the Gated DeltaNet
  recurrent states (`[4, 48, 128, 128]` f32 per layer), about 7 ms in all.
- **Prefill:** a 2,048-token chunk takes 3.1 s with FP8 weights against 2.5 s in bf16.
  About 0.23 s of that is the e4m3fn → bf16 widening, which XLA does not fuse into a
  large GEMM.

### Qwen3.8-27B through Triton (FP8 weights, FP8 KV)

The artifact holds 28 GB of weights and 5.3 GB of KV and state pools.

| | bf16 weights, f32 KV | FP8 weights, FP8 KV |
|---|---|---|
| 1 user, 2K: decode, tokens/s | 3.7 | 6.7 |
| 4 users, 2K: decode, tokens/s per user | 3.2 | 5.3 (21 in all) |
| 1 user, 30K: decode, tokens/s | 3.2 | 5.2 |
| 4 users, 30K: decode, tokens/s per user (follow-up turn) | 1.9 | 2.9 |
| 1 user, 2K: first token | 2.5 s | 2.8 s |
| 1 user, 30K: first token | 60 s | 63 s |

A 13-token follow-up's first token takes 0.3–1.6 s with FP8 weights and KV (the bf16
column's 30K follow-ups were measured before the 128-token prefill chunk).

The first request after the server loads pays one-time costs: its first token took
10 s at 2K. The table is from a server that had served one request first.

## Part 4: FP8 KV cache

`-PkvDtype=fp8` (`DecodeModelShape.kvDtype = F8E4M3FN`, with no `kvQuant`) makes the KV
pools e4m3fn.

- **Write:** `KV_CACHE_WRITE` takes f32 or bf16 keys and values, clamps them to ±448
  and converts them, rounding to nearest even.
- **Read:** `PAGED_ATTENTION` gathers the pages a block table names as codes and widens
  the gathered window to the compute dtype. The pool itself is never widened.
- **Interpreters:** both store the rounded values (`saturateToF8e4m3fn`).
- **No scale:** a value past ±448 saturates.
- **Size:** a pool is a quarter of an f32 pool.

| Check | Result |
|---|---|
| Interpreter: hand-worked roundings and clamps read back through attention | exact |
| GB10: a decode step over e4m3fn pools holding a context, with values past ±448 and between codes written | within 3.6e-7 of the interpreter |
| Qwen3.5-0.8B, f32 weights, e4m3fn KV, against transformers | 16 of 16 ids on both prompts |
| Same with FP8 weights | same as FP8 weights alone (16 of 16 text, 5 chat) |

Qwen3.6-35B-A3B with FP8 weights, KV pools 5.6 GB → 1.8 GB:

| | f32 KV | FP8 KV |
|---|---|---|
| 1 user, 2K: decode, tokens/s | 33.6 | 33.3 |
| 4 users, 2K: decode, tokens/s per user | 13.1 | 13.0 |
| 4 users, 30K: decode, tokens/s per user (follow-up turn) | 7.0 | 7.9 |

**Long contexts are still slow.** Each row's whole context bucket is gathered and
widened before attention: for the 27B's four rows at 32K, about 0.5 GB of f32 keys
per attention layer. A paged attention that reads pages in place is the next lever.

## Decode attention: the f32 dots as an algorithm (9ae0a61)

**Problem.** `PjrtDecodeAttentionBenchTest` (`TLALOC_ATTN_BENCH=1`) times one decode
step's `PAGED_ATTENTION`: four rows, e4m3fn pools of 8,200 pages, every row 30 tokens
short of its bucket. At a 32K bucket a Qwen3.8-27B layer took 13.1 ms, reading the live
keys and values at 20 GB/s. Its two dots asked for `precision = HIGHEST`, which XLA
runs as a SIMT kernel.

**Candidate forms over the same inputs:**

| Form, 27B layer at 32K | ms | Largest difference from HIGHEST |
|---|---|---|
| f32 dots, `precision = HIGHEST` (before) | 13.1 | 0 |
| f32 dots, exact f32 algorithm (`F32_DOT_ALGORITHM`, as prefill already used) | 5.2 | 2e-8 |
| f32 dots, default precision | 5.1 | 1.4e-5 |
| TF32 ×3, bf16 ×3 and bf16 ×6 algorithms | 5.1–5.4 | 6e-8 to 2e-7 |
| bf16 window and query | 5.0 | 1.2e-4 |
| bf16 window, query and weights split into three bf16 pieces | 5.8 | 1.1e-7 |
| keys and values multiplied and reduced, no dot (the gather fuses in) | 9.0 | 1.4e-5 from default-precision dots |
| window kept e4m3fn behind an `optimization_barrier`, widened into the GEMM | 5.3 | 2e-8 |

- **The exact f32 algorithm** is as fast as any of the rounded forms. Decode now uses
  it, and `portableF32Dots` still writes HIGHEST for a TPU.
- **What remains is the gathered window being written.** The gather and widening of
  one pool alone take 4.3 ms, against about 1 ms to read the live codes. No XLA form
  above avoids it; a fused paged-decode kernel would.
- **Rounded forms need care.** With `xla_allow_excess_precision` (XLA's default), a
  split by f32 → bf16 → f32 round trips is folded away, so the split must mask bits.

**Step times.** `triton/profile.sh` (`MODE=time`, three runs) on Qwen3.8-27B with FP8
weights and KV, old and new forms served back to back:

| Workload | HIGHEST | f32 algorithm |
|---|---|---|
| 1 stream at 256 tokens (2K bucket) | 141.6 ms | 137.3 ms |
| 4 streams at 256 tokens | 161.0 ms | 152.6 ms |
| 4 streams at 30,000 tokens (32K bucket) | 340.0 ms | 219.2 ms |

Four users at 30K decode 4.4 tokens/s each instead of 2.9.

Qwen3.6-35B-A3B (FP8 weights and KV), new form only:
- **Step times:** 1 stream 28 ms, 4 streams 51 ms, 4 streams at 30K 74 ms.
- **Multi-user client:** four users decode 17.7–18.8 tokens/s each at 2K (14.3 before)
  and 13.1 at 30K (8.2 before).
- **Run-to-run variation:** attention accounts for about 4 ms of the 2K gain; the rest
  is within the variation between runs on different days of this machine.


## MTP speculative decoding

Plan: [mtp-plan.md](mtp-plan.md). The Qwen3.5 checkpoints' MTP head (one full-attention
layer, `mtp.*`) drafts k tokens per step, and the target verifies them in one call.

### Pieces

- **Per-token state writes (172c213).** `CAUSAL_CONV1D` and `GATED_DELTA_RULE` take an
  optional `writeSlots [B, T]`. A verify call reads its sequence's slot and writes the
  state after each token to its own slot; the next step reads the slot of the last
  accepted token.
  - The emitter unrolls the recurrence over the few tokens of such a call.
  - The GB10 agrees with the interpreter within 3e-8.
  - One call over three tokens leaves the states a chain of one-token calls reaches.
- **Speculative entries (12a0f1d).** `HfDecoderConfig.mtpDraftTokens`; the head's tensors
  are roles, and its layer goes through the layer code as one more layer. A decode entry
  takes the pending token and k drafts and runs, in one call:
  1. the target over them;
  2. its greedy token at every position;
  3. the count of drafts that match;
  4. the head over the step's positions, whose token at each verified position is the
     target's own greedy token;
  5. k new drafts from the last accepted position.

  The target's final hidden state is carried per state slot, so prefill chunks chain
  the head across chunks.
- **Serving (e332944, 585b3d5).**
  - Artifacts are `tlaloc-serving-v5` (`-PmtpDraftTokens=k`).
  - The Triton backend keeps per sequence k + 2 state slots, the pending token and the
    drafts.
  - A one-token request with the pending token is a verify step and answers with 1 to
    k + 1 tokens in `NEXT_TOKENS`.
  - The decode cohort counts verify steps. Without that, four streams split into
    alternating batches.
- **MoE at verify sizes.** `MOE_EXPERTS` keeps its gathered form up to 160 (row, expert)
  pairs. Four sequences verifying 3 drafts are 128 pairs: 4.6 ms per Qwen3.6-35B-A3B
  layer against 6.5 in the tiled loop.

### Correctness

| Check | Result |
|---|---|
| Speculative prefill of the 0.8B, interpreter and GB10, against transformers' `Qwen3_5DecoderLayer` with the `mtp.*` tensors | the reference drafts exactly (" Paris" then ".", newline, "The") |
| Speculative greedy decoding of the 0.8B, interpreter and GB10 | transformers' 16 ids on both prompts; 4 tokens a step on the text prompt, 2.5 on the chat prompt |
| Through Triton, 0.8B, four users, 64 tokens, against the non-speculative artifact | three users' ids equal; the fourth differs where its top two logits are 0.006 apart, where the non-speculative model also differs between runs |

### Throughput through Triton (FP8 weights and KV, 3 drafts)

Tokens/s per user, 128 tokens a turn, follow-up turn and first turn:

| | Qwen3.8-27B, no MTP | 27B, MTP | Qwen3.6-35B-A3B, no MTP | 35B, MTP |
|---|---|---|---|---|
| 1 user, 2K | 6.7 | 15.5 / 12.6 | 34.4 | 54.3 / 56.3 |
| 4 users, 2K | 5.3 | 9.3 / 9.8 | 17.7–18.8 | 17.0 / 20.9 |
| 1 user, 30K | 5.2 | 13.3 / 16.2 | not measured | 34.9 / 43.2 |
| 4 users, 30K | 3.7–4.4 | 7.5 / 9.5 | 13.1 | 15.0 / 15.8 |

- **Acceptance:** 2.5 to 3.8 tokens per step. vLLM's MTP on these models reaches 2.5 to
  3.3.
- **The dense 27B** gains 1.8 to 2.6 times.
- **The MoE model** gains 1.6 times for one user. With four users it gains 1.2 times at
  30K and is level at 2K: a four-user verify step makes every MoE layer read about four
  times the expert weights of a plain step (128 pairs, one expert read per pair).
- **A verify step's costs** (27B, one stream: 181 ms of kernels against 141 for a plain
  step):
  - four evaluations of the LM head, 12 ms each (the verify rows and three drafts);
  - the head's own attention passes over the window;
  - the target over k + 1 tokens.
- **Two fixes found by measuring:**
  - **Batching:** the decode cohort did not count verify steps, so four streams split
    into alternating batches (585b3d5).
  - **Attention form:** rows sharing a block table took the blockwise attention form,
    whose loop over 2,048-position key blocks pays a host round trip per block. A
    verify step (4 rows per sequence) and the head's pass (5) now keep the one-pass
    form, which needs 8 rows per table or more for blockwise. That took the 35B at four
    users and 30K from 10.3 to 15.0 tokens/s each, and the 27B from 4.7 to 7.5.
- **One draft** (35B): one user 44.8, four users at 2K 19.0, four at 30K 9.1 (measured
  before the attention fix), at 1.7 to 1.9 tokens per step.

## The step budget, and lessons from a TPU inference study

Zimbres, *From 1,540 to 19,511 Tokens per Second on a Single TPU v5e Chip*
([zenodo.org/records/21221952](https://zenodo.org/records/21221952)), measures Gemma 2B
under vLLM on one TPU v5e. The model and chip differ from ours, but its method carries
over:

- **Profile and account first.** Attribute every compiled operation to its tensor shapes
  and occurrence counts, and place each family against the roofline (arithmetic
  intensity against the chip's FLOPs per byte).
- **Padding is waste.** Their attention scored 2,048-position blocks for 140-token
  sequences; sizing the block to the workload gave 24%.
- **Speculation depends on the regime.** It cost them six times at batch 32.
- **Verify that a change took effect.** Check a trace, an allocation or a kernel name,
  not the flag.
- **Never quote speed from a profiled run.**

### Qwen3.8-27B, 3 MTP drafts, four sequences at 30K

`triton/profile.sh` (nsys, `XLA_DUMP=1`). Each kernel is attributed to its fusion in the
module of the verify entry that ran (`decode_b4_c32768`), and classed by the tensors that
fusion touches. Times are each kernel's median duration times its launches:

| Family | Share |
|---|---|
| FP8 weight GEMMs | 41% |
| Gated DeltaNet state handling (per-token states stacked, transposed, scattered) | 17% |
| Attention: the page gather | 15% |
| LM head (about 15 ms an evaluation, four a step) | 13% |
| Elementwise, reductions, copies | 7% |
| Other dots | 6% |

- **Weights are at the memory rate.** The weight GEMMs read about 28 GB per step at
  about 200 GB/s; only fewer bytes make them faster.
- **The first budget was wrong.** It keyed kernels by name and put the LM head at 29%:
  XLA reuses fusion names across the compiled entries, and a name's first definition was
  another entry's. Attributing within the module that ran fixed it.
- **Stalls.** The kernels' raw times sum to 51% more than their medians: single launches
  stall for up to 336 ms, the head's ordinary 15 ms among them. A likely cause, not yet
  verified, is other GPU clients on the machine (the desktop) time-slicing the GPU; a
  serving box would run headless.

### A change that failed its prediction

The budget's Gated DeltaNet line stacks each verify step's per-token states, one
`[4, 4, 48, 128, 128]` f32 tensor per layer, and XLA transposes it before the scatter
into the pool. Scattering each token's state on its own avoids the stack. Predicted: a
few percent faster. Measured (`profile.sh` `MODE=time`, three runs each, the two
artifacts served back to back):

| Four streams | Stacked | Per-token scatters |
|---|---|---|
| 256 tokens | 261.1 ms | 273.5 ms |
| 30,000 tokens | 363.1 ms | 379.6 ms |

It was reverted. Four scatters into a 3 GB pool cost more than one transpose of the
stack.

## The fused paged-attention kernel in the serving path

`triton/kernels/paged_attention.cu` (`tlaloc_paged_attention`, a typed-FFI custom call in
`libtlaloc_kernels.so`). Each block of 256 threads takes one 256-position slice of one
table and KV head, for every query of the rows sharing that table:

- the slice's pool offsets are read from the block table once, into shared memory;
- one thread per position scores the key against every query;
- one warp per query takes the softmax over the slice;
- one thread per head dimension accumulates the values, sixteen loads in flight;
- a second kernel combines the slices.

Only the slices under a row's length are read, so a 30K context in a 32K bucket reads 30K
positions.

The first versions lost to XLA's form on verify rows. ncu showed long-scoreboard stalls:
the value loop read the block table before every pool load, two dependent global reads
per position. With the offsets in shared memory, at 32K for four sequences
(`PjrtFusedPagedAttentionTest.fusedAgainstXlaTimings`, one call through PJRT):

| | XLA's form | Fused kernel |
|---|---|---|
| 27B decode (4 rows) | 5.7 ms | 3.6 ms |
| 27B verify (16 rows) | 7.0 ms | 4.5 ms |
| 35B decode | 3.4 ms | 2.0 ms |
| 35B verify | 4.4 ms | 2.7 ms |

`triton/kernels/bench/paged_attention_bench.cu` times the kernel alone and checks it
against a CPU reference.

The export flag `-PcudaKernels=true` marks every `PAGED_ATTENTION` of the graphs
`fused_kernel`. It is written to the manifest as `model.cudaKernels`. The Triton
backend registers `libtlaloc_kernels.so` with the PJRT plugin when it loads the plugin and
logs `CUDA kernels registered`. It refuses an artifact that needs the kernel when the
registration failed. Prefill entries keep XLA's form: their queries per table exceed the
kernel's 64.

Speculative step, 3 drafts, four streams (`profile.sh` `MODE=time`, three runs each, the
two artifacts served back to back):

| | XLA's form | Fused kernel |
|---|---|---|
| 27B, 256 tokens | 267.7 ms | 250.4 ms |
| 27B, 30,000 tokens | 365.6 ms | 304.0 ms |
| 35B, 256 tokens | 114.3 ms | 114.2 ms |
| 35B, 30,000 tokens | 156.8 ms | 137.1 ms |

The 35B's runs at 256 tokens spread from 106 to 122 ms in both artifacts.

## An FP8 head for the drafts

`-PmtpDraftHeadQuant=fp8` stages an e4m3fn copy of `lm_head`, with one scale per row (slots
`draftHead` and `draftHeadScale`). Only the argmax of each MTP draft reads it. The
target's tokens keep the full head, so outputs do not change; drafts can, and with them
how many are accepted. On Qwen3.5-0.8B in the interpreter, the greedy ids equal
transformers' and every verify step accepts as many tokens as with the bf16 head.

A step reads the head four times: once for the verified rows and once per draft. Three of
the four now read half the bytes. Speculative step with fused attention, four streams, 3 drafts:

| | bf16 draft head | FP8 draft head |
|---|---|---|
| 27B, 256 tokens | 254.1 ms | 232.0 ms |
| 27B, 30,000 tokens | 301.2 ms | 285.8 ms |
| 35B, 256 tokens | 112.9 ms | 107.7 ms |
| 35B, 30,000 tokens | 131.0 ms | 127.3 ms |

The 35B's runs spread by up to 17 ms in both artifacts.

## NVFP4 MLP weights, served as stored

NVIDIA's NVFP4 checkpoints (`nvidia/Qwen3.8-27B-NVFP4`, `nvidia/Qwen3.6-35B-A3B-NVFP4`) store
the MLPs, the routed and shared experts, and `lm_head` as NVFP4. NVFP4 is e2m1 codes with an
e4m3 scale per 16 values and an f32 tensor scale. The attention and Gated DeltaNet
projections are FP8, and the MTP layer is bf16. Until now these checkpoints were widened and
requantized to FP8 per channel, which dropped the per-16 scales.

- **The op.** `NVFP4_MATMUL(x, codes, scales, scale2)`: `y = (bf16(x) W'^T) * scale2`, where
  `W'` is a code times its group scale. That product has at most 6 significant bits, so it
  is exact in bf16. The tensor scale is given per output row, so gate and up stack into one
  weight with their own scales.
- **The layout.** Codes and scales are packed in the order the `mma.sync m16n8k16` A
  fragment reads them, so each lane loads 16 contiguous bytes per four k-steps. The packing
  is a transpose of the checkpoint's `[N, K/2]` bytes, so the XLA form is reshape,
  transpose, unpack, scale, and a bf16 dot.
- **The kernel** (`triton/kernels/fp4_gemm.cu`, `tlaloc_fp4_gemm`) handles up to 16 rows:
  - eight warps per block, one 16-row tile each, over a K chunk of 2,048 (1,024 above 8 rows);
  - x's chunk in shared memory as bf16;
  - the code-to-bf16 table in shared memory (in `__constant__` memory, lanes reading
    different entries serialized it: 70 against 230 GB/s);
  - partial sums per chunk, reduced in a second kernel.

  At the 27B's shapes (`triton/kernels/bench/fp4_gemm_bench.cu`, rate of codes and scales
  read):

  | | 4 rows | 16 rows |
  |---|---|---|
  | gate+up `[34816, 5120]` | 0.43 ms, 234 GB/s | 0.52 ms, 194 GB/s |
  | down `[5120, 17408]` | 0.19 ms, 262 GB/s | 0.22 ms, 225 GB/s |

  FP8 at the same rate would take 1.8 times as long.
- **Prefill** rows use the XLA form, which widens the weight on every call: 5 to 9 ms per
  projection.
- **Staging** (`-PweightQuant=nvfp4`): the decoder layers' gate+up, down and shared-expert
  projections are NVFP4, as stored, or rounded from bf16 (`Nvfp4Quantizer`, ModelOpt's
  formulas). The other quantized projections, the routed experts and the MTP layer are
  FP8.

Qwen3.8-27B-NVFP4 against the bf16 checkpoint with FP8 MLPs. Both have fused attention, the
FP8 draft head, FP8 KV and 3 drafts. Speculative step for four streams:

| | FP8 MLPs | NVFP4 MLPs |
|---|---|---|
| 256 tokens | 239.0 ms | 208.6 ms |
| 30,000 tokens | 286.5 ms | 256.0 ms |

Tokens/s per user (follow-up turn) for the 27B, 3 drafts, FP8 KV, both with fused attention
and the FP8 draft head (`tri_users.py`, two turns of 128 tokens):

| | FP8 MLPs | NVFP4 MLPs | before both, FP8 MLPs |
|---|---|---|---|
| 1 user, 2K | 17.5 | 21.8 | 15.5 |
| 4 users, 2K | 12.5 | 13.1 | 9.3 |
| 1 user, 30K | 15.9 | 22.3 | 13.3 |
| 4 users, 30K | 9.2 | 11.2 | 7.5 |

The two artifacts are different checkpoints with different numerics: the first turn of a
30K prompt continues identically for 400 characters and then diverges. Each verify step
emits about 3 tokens with either.

## Qwen3.6-35B-A3B toward 50 tokens/s per user

The step budget of the four-stream verify step at 30K, with FP8 experts, fused attention
and the FP8 draft head, was 118 ms:

| Family | ms per step |
|---|---|
| Routed experts (FP8, read once per row-expert pair) | 50.3 |
| Gated DeltaNet states (stack, transpose 12.0, scatter 8.9) | 24.1 |
| Attention (`tlaloc_paged_attention`, about 90 GB/s on verify rows) | 20.7 |
| LM head (bf16) and drafts' head | 12.1 |
| Other dots, elementwise | 11.2 |

**NVFP4 experts and heads.** `MOE_EXPERTS` with packed NVFP4 experts runs as
`tlaloc_moe_fp4` (`triton/kernels/moe_fp4.cu`):

1. The routed pairs are listed per expert.
2. Each expert used is read once per 8 of its rows, on `tlaloc_fp4_gemm`'s tiles.
3. `silu(g) * u` is rounded to bf16, and the slots are combined in order.

At 16 rows it reads the 100 experts used of a layer at 213 GB/s, in 0.83 ms.
`-PheadQuant=nvfp4` and `-PmtpDraftHeadQuant=nvfp4` give both heads NVFP4. NVIDIA's
NVFP4 checkpoint stores the head that way. A step went from 106.7 to 73.8 ms at 256
tokens and from 127.9 to 94.3 ms at 30K.

**The Gated DeltaNet recurrence as a kernel.** `tlaloc_gated_delta`
(`triton/kernels/gated_delta.cu`) keeps a row's state slice in registers across its tokens
and writes it to the pool in place. In a verify step it writes after every token, at
`writeSlots`. The custom call's pool result aliases its operand. It takes 0.20 ms a
layer, against about 0.8 ms in XLA's stacked form. A step went from 79.4 to 57.9 ms at
256 tokens and from 98.7 to 78.9 ms at 30K.

The export flag for Tlaloc's kernels is `-PcudaKernels=true` (attention and the
recurrence). NVFP4 weights always use their kernels.

Tokens/s per user (follow-up turn, `tri_users.py`), 3 drafts, FP8 KV:

| | FP8, before | NVFP4 experts and heads | + Gated DeltaNet kernel |
|---|---|---|---|
| 1 user, 2K | 52.6 | 66.7 | 77.6 |
| 4 users, 2K | 22.4 | 36.4 | 42.4 |
| 1 user, 30K | 45.3 | 57.9 | 61.6 |
| 4 users, 30K | 17.2 | 25.5 | 31.0 |

The "before" column already has fused attention and the FP8 draft head.

### Ornith 1.5

`ornith-ai/Ornith-1.5-35B-A3B-NVFP4` has Qwen3.6-35B-A3B's architecture and is served the
same way. Its config repeats `bos_token_id`, `eos_token_id`, `pad_token_id` and
`hidden_size` outside `text_config`; the parser now accepts these and checks that
`hidden_size` agrees. With NVFP4 experts and heads, before the Gated DeltaNet kernel:

| | Ornith 1.5 35B-A3B | Qwen3.6-35B-A3B |
|---|---|---|
| step, 4 streams, 256 tokens | 73.3 ms | 73.8 ms |
| step, 4 streams, 30K | 94.6 ms | 94.3 ms |
| 1 user, 2K | 92.5 | 66.7 |
| 4 users, 2K | 33.6 | 36.4 |
| 1 user, 30K | 57.5 | 57.9 |
| 4 users, 30K | 24.2 | 25.5 |

At 2K for one user, Ornith's greedy continuation of the test prompt repeats one line,
which the drafts predict (3.7 tokens a step), so that figure overstates it.
