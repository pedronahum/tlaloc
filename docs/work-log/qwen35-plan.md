# Qwen3.5 hybrid models in the serving path: plan

Branch `feat/qwen35` from `main` at `0fc7581` (2026-10-03).

Goal: serve `Qwen3.8-27B` (`qwen3_5`) and `Qwen3.6-35B-A3B` (`qwen3_5_moe`) on the GB10
for 4 users at long context. Measured target: vLLM 0.29 on the same machine
([spark-4user-serving.md](spark-4user-serving.md)). Four parts, done in order, each merged
on its own:

1. the `qwen3_5` family (Gated DeltaNet layers, gated attention, partial RoPE);
2. MoE (`qwen3_5_moe`);
3. FP8 and NVFP4 weights;
4. an FP8 KV cache.

## What exists

- `HfModelFamily` (`ir/.../inference/HfDecoder.kt`): Llama, Qwen3, Muse Glimmer. An
  unknown architecture or config key is refused by name.
- `HfDecoderGraph` builds decode and prefill entries: every layer is paged attention
  plus a SwiGLU MLP. The signature (`DecodeGraphSpec`) carries two KV pools per layer.
- `PAGED_ATTENTION` and `KV_CACHE_WRITE` are inference-only op kinds with interpreter
  arms and StableHLO emission. Multi-result ops exist (`DxirOpResult`); the emitter keeps
  one SSA name per result.
- Weights are staged f32 (Llama, Qwen3) or bf16 (Muse Glimmer), optionally int8.
- Runtimes: `tlaloc_serve.py`, the vLLM plugin, the Triton backend (the only one that
  batches concurrent requests), and `ServingModel` (JVM).
- Oracles run in `~/.local/venvs/vllm` (transformers 5.17, which has `qwen3_5` and
  `qwen3_5_moe`).

## The model (transformers `modeling_qwen3_5.py`)

Text path only. `layer_types` alternates three `linear_attention` layers and one
`full_attention` layer.

- **RMSNorm**: every norm multiplies by `1 + w` (input, post-attention, q/k, final).
  The gated norm inside a linear layer multiplies by `w`.
- **Full-attention layer**:
  - `q_proj` gives `[heads, 2 * headDim]` per token: the first half of each head is the
    query, the second half the gate.
  - The query and key go through a per-head q/k norm, then RoPE on the first
    `headDim * partial_rotary_factor` channels (64 of 256); the rest pass through.
  - The attention output is multiplied by `sigmoid(gate)` before `o_proj`.
  - Scale is `headDim^-0.5`.
  - For text tokens the three mRoPE position grids are equal, so the interleaved mRoPE
    reduces to plain RoPE. Image and video tokens are refused by name.
- **Linear-attention layer (Gated DeltaNet)**, for input `x`:
  - Projections: `qkv = x Wqkv` (`2 Hk Dk + Hv Dv` channels), `z = x Wz`,
    `b = x Wb`, `a = x Wa`.
  - `qkv` goes through a causal depthwise conv1d (kernel 4, no bias), then SiLU, then
    is split into q, k, v.
  - `beta = sigmoid(b)` and `g = -exp(A_log) * softplus(a + dt_bias)`.
  - q and k are L2-normalized (eps 1e-6 inside the rsqrt), and q is scaled by `Dk^-0.5`.
  - Value head `j` reads key head `j / (Hv / Hk)`.
  - Per value head, with state `S` `[Dk, Dv]`, each token does:
    ```
    S = S * exp(g)
    S = S + k ⊗ (beta * (v - kᵀS))
    o = qᵀS
    ```
  - `o` goes through the gated RMSNorm (`w * norm(o) * silu(z)`, per head), then
    `out_proj`.
  - The state and conv state are f32 and carried across calls.

## Part 1 design

### Ops

Two inference-only kinds with two results each (the output and the updated pool), so
a prefill chunk runs its scan once.

```
CAUSAL_CONV1D(x [B,T,C], weight [K,C], convState [S,K-1,C], tokenSlots [B,T], positions [B,T])
  -> (y [B,T,C], convState')

GATED_DELTA_RULE(q [B,T,Hk,Dk], k [B,T,Hk,Dk], v [B,T,Hv,Dv], g [B,T,Hv], beta [B,T,Hv],
                 state [S,Hv,Dk,Dv], tokenSlots [B,T], positions [B,T])
  -> (o [B,T,Hv,Dv], state')
```

Token rules, shared by both ops:

- A token is live when its slot is `>= 0`. A dead token (padding) neither reads nor
  writes state, and its output row is 0.
- A live token at position 0 starts from zero state. Any other live token continues from
  the state the previous live token of its row left, or, for the row's first live token,
  from the pool at its slot.
- After the call, each row's slot holds the state after its last live token.
- The live tokens of a row share one slot, and different rows use different slots.

The conv weight is staged `[K, C]` (the file's `[C, 1, K]` reordered). The conv output
is pre-activation; SiLU is a separate op. L2 norms and the q scale are primitive ops
before `GATED_DELTA_RULE`.

The interpreter computes in f32, token by token, as transformers' recurrent form does.
The first StableHLO form is a `stablehlo.while` over the T tokens, with the row states
gathered from the pool before the loop and scattered back after it. A chunked form for
long prefill comes once the recurrent form is certified.

### Signature

- `DecodeModelShape.linearState`: which layers are linear, the slot count `S`, and the
  state dims.
- A scheduler input `stateSlots [B]` I32 (role `STATE_SLOTS`) after the slot mapping and
  window inputs. The graph turns it into per-token slots, `-1` wherever `slotMapping` is
  `-1`.
- A linear layer's two pools are `convState$l` `[S, K-1, C]` and `recurrentState$l`
  `[S, Hv, Dk, Dv]`, both f32, with roles `STATE_POOL_IN` and `STATE_POOL_OUT`, in the
  place of that layer's K/V pools. The pools stay two per layer, in layer order, and
  donated.
- A padding row's `stateSlots` is 0, and its tokens are dead through `slotMapping`.

### Family

`HfModelFamily.Qwen3_5`:

- architectures `Qwen3_5ForConditionalGeneration` and `Qwen3_5ForCausalLM`;
- the decoder under `text_config`, tensors under `model.language_model.`;
- `mtp.*` and `model.visual.*` are not read.

`DecoderLayerSpec` gains:

- `mixer` (`ATTENTION` or `GATED_DELTA_NET`);
- `queryGate` (the gate interleaved in `q_proj`).

`HfDecoderConfig` gains:

- the linear-attention dims;
- `rotaryDim`;
- `finalNormGainPlusOne` and `qkNormGainPlusOne`.

New layer parts are `IN_PROJ_QKV`, `IN_PROJ_Z`, `IN_PROJ_B`, `IN_PROJ_A`, `CONV1D`,
`DT_BIAS`, `A_LOG`, `LINEAR_NORM` and `OUT_PROJ`. Weights are staged bf16, as Muse
Glimmer's are.

### Checks

1. **Ops, interpreter:** against a plain Kotlin recurrence, and against transformers'
   `torch_recurrent_gated_delta_rule` and `causal_conv1d_*`. Padding, reset at position
   0, continuation across calls, and GQA heads.
2. **Ops, GB10:** PJRT against the interpreter.
3. **A tiny random `qwen3_5`** written by transformers: config parse and prefill/decode
   logits in the interpreter.
4. **Qwen3.5-0.8B** (same architecture as the 27B), a greedy fixture from transformers:
   interpreter parity, then PJRT greedy parity on the GB10.
5. **Serving:**
   - an artifact with state pools, run through the Triton backend's sequence mode
     (4 concurrent sequences);
   - Qwen3.5-0.8B tokens equal transformers';
   - Qwen3.8-27B (bf16, 54 GB staged) decode and prefill rates measured against vLLM.

## Parts 2-4 (designed when reached)

- **MoE**:
  - router softmax, top-8, renormalized weights;
  - 256 experts plus a sigmoid-gated shared expert;
  - decode gathers the selected experts' weights; prefill needs a grouped form;
  - checks against a tiny random `qwen3_5_moe` from transformers.
- **FP8 / NVFP4 weights:**
  - FP8 is e4m3 with 128x128 block scales, the Qwen FP8 checkpoints.
  - NVFP4 is the ModelOpt format: packed e2m1, fp8 scales per 16, and a global f32
    scale.
  - Both need new DTypes, a loader, and dequantization feeding the GEMM. Measure
    whether XLA keeps the bytes narrow in the GEMM.
- **FP8 KV cache:** an f8e4m3 pool dtype; `KV_CACHE_WRITE` converts on write and
  `PAGED_ATTENTION` on read, with per-tensor scales.
