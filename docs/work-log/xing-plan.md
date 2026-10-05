# Xing 4.0 in the serving path: plan

`XingChen-AGI/Xing4.0-29B-A4B` (`Xing4_0ForCausalLM`, `model_type` `xing4_0`, custom
modeling code `modeling_xing4_0.py`, transformers 5.14). 62 GB in bf16; an FP8 copy is
published as `Xing4.0-29B-A4B-FP8`.

## The architecture

- **40 decoder layers**, hidden size 3584, vocabulary 131,072, `lm_head` untied.
- **Attention is DeepSeek-V3's MLA**:
  - queries: `q_a_proj` (3584 → 768), `q_a_layernorm`, `q_b_proj` (768 → 32 heads × 192);
  - a head's query is 128 "nope" dims and 64 rope dims;
  - `kv_a_proj_with_mqa` gives a 512-wide latent plus 64 rope dims, and the latent goes
    through `kv_a_layernorm`;
  - `kv_b_proj` (512 → 32 × (128 + 128)) gives each head's key-nope and value;
  - RoPE is YaRN (factor 64 over 4,096 positions) on interleaved pairs of the 64 rope dims;
  - the scale is `192^-0.5 · mscale²`, with `mscale = 0.1 · ln(64) + 1`.
- **4 residual streams, mixed by hyper-connections** (`attn_hc`, `ffn_hc`), one before each
  sublayer. From the 4 streams flattened to 14,336 and RMS-normalized without a gain, one
  linear map `hc_fn` (24 outputs) gives:
  - 4 pre weights (sigmoid): the sublayer reads `Σ pre_i · stream_i`;
  - 4 post weights (2 · sigmoid);
  - a 4×4 mixing matrix (clamped to ±30, exponentiated, then 20 Sinkhorn iterations).

  The new streams are `post_i · out + Σ_j comb_ij · stream_j`. The embedding starts all
  4 streams; the model averages them before the final norm.
- **MLP**: layers 0 and 1 are dense SwiGLU (9,216). Layers 2–39 are MoE:
  - 64 experts of 1,024 plus a shared expert of 1,024;
  - the router scores with a sigmoid, and selects the top 4 by score plus
    `e_score_correction_bias`;
  - the selected weights are the scores without the bias, normalized and times 2.0.
- **One MTP layer** (index 40, DeepSeek style): `enorm`, `hnorm`, `eh_proj`, its own
  `embed_tokens`, and `shared_head` (norm and head).

## How it maps onto Tlaloc

- **The KV cache holds MLA's latent:**
  - 576 values per token and layer: `kv_a_layernorm(latent)` and the rotated 64 rope dims;
  - one pool is passed as both key and value of `PAGED_ATTENTION` (one KV head, D = 576);
  - the first 512 dims of the output are the latent-weighted sum;
  - FP8 that is 576 bytes per token and layer, 2.8 GB for four users at 30K.
- **Absorbed attention:**
  - each head's query nope part is mapped into the latent through its block of
    `kv_b_proj` (`W_uk`), a batched matmul [heads, rows, 128] · [heads, 128, 512];
  - the attention output is mapped back through `W_uv`;
  - both blocks are staged from `kv_b_proj` per head.
- **Hyper-connections are ordinary ops:** RMSNorm, a matmul, sigmoids, and an unrolled
  Sinkhorn loop on [rows, 4, 4]. Twenty iterations at two places per layer is a lot of
  small reductions, so a kernel may come later.
- **`MOE_EXPERTS` needs a sigmoid routing:** a selection bias operand, and a scale.
  `tlaloc_moe_fp4` and the gathered XLA form take the routing's ids and weights from
  the emitter, so only the routing changes.

## Steps

1. The family:
   - config keys (refused by name when unknown);
   - the new layer parts and their checkpoint names;
   - the layer specs: dense or MoE, MLA.
2. A transformers fixture of a 4-layer truncation (2 dense, 2 MoE) in f32 on the CPU, from
   the real checkpoint (`harness/python`, the vLLM venv, `trust_remote_code`): greedy ids
   and logits.
3. The graph:
   - MLA with the latent pool;
   - the residual streams and hyper-connections;
   - the sigmoid routing.

   Then interpreter parity against the fixture.
4. Export with FP8 weights and KV; then NVFP4 experts (rounded: there is no NVFP4
   checkpoint).
5. Serve through Triton and measure for 1 and 4 users at 2K and 30K.
6. MTP speculative decoding with layer 40.

## First measurements

Exported from the bf16 checkpoint with FP8 KV and `-PcudaKernels=true`, without MTP (one
token per step). Through Triton, tokens/s per user (follow-up turn):

| | FP8 projections and experts | NVFP4 experts (rounded), FP8 elsewhere |
|---|---|---|
| step, 4 streams, 256 tokens | 74.4 ms | 59.0 ms |
| 1 user, 2K | 21.7 | 22.9 |
| 4 users, 2K | 10.3 | 12.9 |
| 1 user, 30K | 15.6 | 16.1 |
| 4 users, 30K | 6.1 | 6.4 |

The text is coherent. Two things stand out:

- **Prefill is about 180 tokens/s.** The MLA attention takes XLA's form, with D = 576
  over 32 query heads and one KV head. A 30K prompt then takes about 170 s, longer
  than the profile client's sequence idle limit (so `profile.sh` has no 30K step time).
- **A four-stream step at 2K is about 75 ms after NVFP4.** That is far more than its
  weight bytes (about 6 GB) need.

### Where a four-stream step goes (NVFP4 experts, 256 tokens)

nsys, kernel medians times launches, per step:

| | ms |
|---|---|
| NVFP4 experts (`tlaloc_moe_fp4`, about 16 used per layer, near the memory rate) | 13.5 |
| LM head, bf16 (131,072 × 3,584) | 4.2 |
| `kv_b_proj` sliced and transposed into its key and value blocks every step | 3.5 |
| q/kv projections, o_proj, the latent matmuls | about 10 |
| all kernels | 42.7 |

- **4,342 kernel launches a step**, most of them the hyper-connections' Sinkhorn
  iterations (20 each, at two places in each of 40 layers).
- **13.8 ms of host gap between steps.** XLA launches every one of those kernels from the
  host.

Next, by expected gain:

1. one kernel per hyper-connection (map, sigmoids, Sinkhorn, combination);
2. `kv_b_proj` staged as its key block and its transposed value block;
3. an NVFP4 or FP8 head;
4. MLA attention for prefill rows (a kernel, or a chunked XLA form);
5. MTP speculative decoding from layer 40.
