# Qwen3.5 hybrid models in the serving path: progress log

Plan: [qwen35-plan.md](qwen35-plan.md). Branch `feat/qwen35` from `main` at `0fc7581`.

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
