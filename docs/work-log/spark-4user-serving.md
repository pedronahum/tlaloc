# Four users, 100K context, one DGX Spark (2026-10-03)

Question: can one GB10 serve a Qwen model of about 30B parameters to 4 concurrent
users with 80K–100K-token contexts, and what limits the experience?

There is no "Qwen3.8 37B". The two candidates of that size are
`Qwen3.8-27B` (dense) and `Qwen3.6-35B-A3B` (MoE, 256 experts, 8 active, about
3B active parameters per token). Both were measured with NVIDIA's NVFP4
checkpoints on vLLM 0.29.0 (`~/.local/venvs/vllm`). Scripts and raw output:
[scripts/spark-4user/](../../scripts/spark-4user/).

## Result

| | Qwen3.6-35B-A3B (MoE) | Qwen3.8-27B (dense) |
|---|---|---|
| 4 users, 2K context, tok/s per user | 63–68 | 23–24 |
| 1 user, 100K document, time to first token (cold) | 29–36 s | 81–83 s |
| 1 user at 100K, tok/s | 82–97 | 23–24 |
| 4 users at 100K, follow-up time to first token (median) | 2.3–6.1 s | 3.2–11.9 s |
| 4 users at 100K, tok/s per user | 28–38 | 12–14 |
| 4 users paste 100K at the same moment, last one's first token | 92–113 s | 256–259 s |

(NVFP4 weights, FP8 KV cache, prefix caching, MTP with 3 draft tokens; ranges cover
the default and the 16K prefill-chunk configurations.)

Use the MoE. It decodes 2.5–3 times faster and prefills 2.5 times faster than the
dense 27B. With 4 users at 100K context it streams 28–38 tok/s to each user (reading
speed is about 5–8 tok/s). Follow-up turns start in 2–8 s.

The experience is not uniform:

- **A fresh 100K document costs about 30 s before the first token.** This is
  prefill compute (about 3,400 tok/s on the GB10). No setting changed it by more
  than 20%.
- **While one user's document is prefilling, the other users' streams drop to about
  10 tok/s.** The GPU is shared, and each prefill chunk delays every decode step.
  Smaller chunks (1K) did not raise the others' rate and made time to first token
  worse (table below).
- **Follow-ups recompute 2–5K tokens.** The recurrent (Gated DeltaNet) state is
  cached only at 2,144-token block boundaries reached during prefill. Tokens
  generated in earlier turns are not cached.

## Configuration

```bash
scripts/spark-4user/serve.sh nvidia/Qwen3.6-35B-A3B-NVFP4 moe \
  --kv-cache-dtype fp8 --enable-prefix-caching --mamba-cache-mode align \
  --max-num-batched-tokens 16384 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

`serve.sh` adds `--max-model-len 131072 --max-num-seqs 4 --gpu-memory-utilization 0.60
--language-model-only`. The KV pool then holds 3.4M tokens, 26 times a 131K
request. Model load takes 21 GiB.

| Setting | Effect measured (MoE) |
|---|---|
| MTP, 3 draft tokens | 1 user at 100K: 97 tok/s against 57 without; 4 users at 100K: 28–36 against 22–25. Mean accepted length 2.5–3.3 |
| FP8 KV cache | Selects the FlashInfer attention backend. With a bf16 cache vLLM picks FLASH_ATTN, and 1 user at 100K drops from 97 to 40 tok/s |
| Prefix caching (`align`) | Follow-up time to first token 1–3 s instead of re-prefilling 100K (29–36 s) |
| `--max-num-batched-tokens 16384` | Cold time to first token 29 s against 36 s; with staggered arrivals 30 s against 49–58 s. Speculative decoding otherwise caps it at 2,048 |
| `--mamba-cache-mode all` | No change against `align` |

## Staggered arrivals

4 users, each with their own 100K document, arriving 30 s apart, 4 turns, 10 s between
turns, 512 output tokens per turn.

| Prefill chunk | First turn, time to first token (median / worst) | tok/s during first turn | Follow-up time to first token (medians, turns 2–4) | Follow-up tok/s (medians) |
|---|---|---|---|---|
| 1,024 | 61.5 / 78.0 s | 10.4 | 8.3, 4.9, 2.5 s | 38, 38, 59 |
| 2,048 (default) | 48.7 / 58.1 s | 9.6 | 3.2, 5.1, 2.7 s | 35, 43, 70 |
| 16,384 | 30.2 / 30.4 s | 9.8 | 4.0, 8.1, 5.0 s | 38, 38, 45 |
| 16,384, `mamba-cache-mode all` | 30.2 / 30.4 s | 9.8 | 3.8, 7.8, 4.8 s | 36, 39, 45 |

The follow-up columns depend on how the four users' turns happen to overlap, so
differences under about 30% between rows are not significant.

## Memory and bandwidth

Only one layer in four keeps a KV cache. The other layers hold a fixed-size
recurrent state per sequence.

| | KV per token (FP8) | KV, 4 users × 100K | Recurrent state per user |
|---|---|---|---|
| 35B-A3B | 10 KiB | 3.8 GiB | 61 MiB |
| 27B | 32 KiB | 12.2 GiB | 147 MiB |

Memory is not the limit: 21–22 GiB of weights plus the cache use under a third of the
unified memory. The limits are:

- **Decode:** memory bandwidth (273 GB/s). Every step reads the active weights and
  every user's KV. At 4 × 100K the MoE reads about 3.8 GiB of KV per step, so long
  contexts slow everyone.
- **Prefill:** compute. The dense 27B has 9 times the active parameters of the MoE,
  so it prefills 2.5 times slower and decodes about 3 times slower.

## What did not work

- **FlashInfer's FP4 MoE kernel** (`--moe-backend flashinfer_cutlass`): refused. The
  experts in `nvidia/Qwen3.6-35B-A3B-NVFP4` are weight-only NVFP4 (W4A16). Only
  Marlin runs them, with bf16 arithmetic. A W4A4 checkpoint is needed for FP4 tensor
  cores in the experts.
- **NVFP4 KV cache** (`--kv-cache-dtype nvfp4`): refused. vLLM 0.29 has no attention
  backend for a 4-bit cache at head size 256.
- **`VLLM_USE_FLASHINFER_MOE_FP4`**: not a vLLM 0.29 variable (logged as unknown).
  The recipes that set it got Marlin from `auto`.
- **First start**: FlashInfer JIT-builds kernels and picked the HPC SDK `nvcc`, which
  has no `libcudart` (link failure). The fix is `CUDA_HOME=/usr/local/cuda-13.0`.
  Building the fused MoE kernels at the default parallelism while the model held
  60% of memory OOM-killed `cicc`. The fix is `MAX_JOBS=3`. Both are set in
  `serve.sh`.

## Improvements, by effect on the user

1. **Cold 100K prefill (30 s).**
   - A W4A4 NVFP4 expert checkpoint running on FP4 tensor cores would cut the
     expert GEMM time.
   - Attention prefill over 100K is quadratic; FP8 attention compute would speed it
     up.
   - In the client, prefill a document when it is attached rather than when the
     question is sent. The 30 s then overlaps the user typing.
2. **Follow-up time to first token (2–8 s).** Save the recurrent state at the end
   of each turn, not only at prefill block boundaries. A follow-up would then prefill
   only the new message (tens of tokens instead of 2–5K). vLLM 0.29 does not do this.
3. **Streams slowing during someone else's prefill.** This is shared compute on one
   GPU; scheduling only moves the cost between the user waiting and the users
   reading. A second Spark (ConnectX-7 link) allows prefill on one and decode on the
   other, or tensor parallelism with twice the bandwidth.
4. **Decode at 100K.** A 4-bit KV cache for head size 256 would halve the KV bytes
   read per step, about half of a step's traffic at 4 × 100K.

Not measured: output quality of NVFP4 weights with an FP8 KV cache. Run a long-context
evaluation before choosing this for users.

## Tlaloc's serving path for this model

Tlaloc cannot load either model today. `HfModelFamily` has Llama, Qwen3 and Muse
Glimmer, and refuses `qwen3_5` / `qwen3_5_moe` by name. What is missing, in the order
it would be needed:

1. **A `qwen3_5` family:**
   - a Gated DeltaNet op: causal conv1d plus a recurrent state per sequence, with a
     state pool and prefill and decode forms;
   - the output gate fused into `q_proj`;
   - interleaved mRoPE over a quarter of the head dimension;
   - head size 256 at long context (the largest context run so far is 32,768).
2. **MoE:** a router, top-8 dispatch, a grouped GEMM and a shared expert.
3. **FP8 and NVFP4 weights:** dtypes, a ModelOpt/compressed-tensors loader, and dots
   that dequantize inside the GEMM. Only int8 weights exist today.
4. **FP8 KV cache.** The int8 KV contract exists; its bytes are deferred.
5. **Batching, prefix caching and speculative decoding:**
   - continuous batching with prefix caching in the Triton backend, including state
     snapshots (Triton is the only Tlaloc server that batches concurrent requests);
   - MTP speculative decoding.

For scale, Tlaloc today serves Muse Glimmer 30B (dense, bf16 weights) through Triton
at 9.6 tok/s in total for 4 sequences at 32K context. It prefills a 31,744-token
prompt in 62.6 s, about 510 tok/s ([SERVING_ARCHITECTURE.md](../SERVING_ARCHITECTURE.md)).
vLLM serves the MoE above at 28–38 tok/s per user at 100K, with prefill at about
3,400 tok/s.
