# Qwen3.5 hybrid models in the serving path: progress log

Plan: [qwen35-plan.md](qwen35-plan.md). Branch `feat/qwen35` from `main` at `0fc7581`.

## Summary

All four parts are built:

1. the `qwen3_5` family;
2. mixture of experts;
3. FP8 weights, with quantized checkpoints read;
4. the FP8 KV cache.

Decode, tokens/s per user, through Triton on the GB10:

| | Qwen3.6-35B-A3B | Qwen3.8-27B |
|---|---|---|
| 1 user, 2K | 34.4 | 6.7 |
| 4 users, 2K | 17.7–18.8 | 5.3 |
| 4 users, 30K (follow-up turn) | 13.1 | 4.4 |
| vLLM 0.29 (NVFP4, FP8 KV, MTP): 4 users at 2K / 100K | 63–68 / 28–38 | 23–24 / 12–14 |

Tlaloc's figures are with FP8 weights and an FP8 KV cache. The vLLM figures are from
[spark-4user-serving.md](spark-4user-serving.md).

The gap to vLLM, largest first:

- **Long-context attention** gathers each row's whole context bucket and writes it out
  before the dots (about 5 ms per 27B layer for four rows at 32K, against about 1 ms
  to read the codes).
- **No speculative decoding (MTP).**
- **MoE decode at four rows** reads its experts at about half the memory rate.
- **Weights are FP8, not 4-bit.** This XLA has no fused 4-bit GEMM.
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

