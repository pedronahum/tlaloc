# vLLM-TorchTPU lessons: plan and results

Branch `feat/vllm-torchtpu-lessons`, from `feat/deepinfer-lessons` at `c4d016ca` (2026-10-22).
The source of every item is [vllm-torchtpu-lessons.md](vllm-torchtpu-lessons.md). Built and
measured on the GB10 (XLA:GPU); nothing here has run on a TPU or on more than one device.

## Results

| Item | Status | Result |
|---|---|---|
| Router top-k without a sort | done | The XLA form of MOE_EXPERTS (every target without `-PcudaKernels`, and the TPU's ragged-dot form) chooses the top k by k rounds of a row max, the lowest expert holding it and a mask, instead of a stable sort over every expert: the same experts, ties to the lower id (`PjrtMoeRouteTest`, `PjrtMoeExpertsTest`). 256 experts, top 8: 0.114 against 1.578 ms at 2,048 rows, 0.114 against 0.155 ms at 4 (`PjrtMoeBenchTest.routingTopKTime`, `TLALOC_MOE_BENCH=1`; 0.114 ms is the dispatch floor). |
| Sampler without the CUDA kernel | done | SAMPLE_TOKENS without `kernel = true` lowers to StableHLO: candidates by `chlo.top_k` of order-preserving integer keys (logit descending, then id), weights by the fixed exp2 steps, top-k by rank, top-p and the draw by Hillis-Steele prefix sums in i64, splitmix64 uniforms, and a `stablehlo.if` over the whole vocabulary when a row keeps more than the 4,096 candidates and the last one weighs something. It draws exactly the reference's tokens on XLA:GPU, both branches, ties and signed zeros included (`PjrtSampleTokensXlaTest`). 3.5 to 3.8 ms for 16 rows of 248,320 logits on the GB10 (the kernel: 0.082 ms), where `top_k` of 4,096 sorts; the graph uses the kernel with `-PcudaKernels`, the StableHLO form otherwise, and `-Psampling` no longer needs the CUDA kernels. |
| Prefix-aware admission | done (as same-batch deferral) | Every prompt of a batch runs in the same rounds, so an admission order alone changes nothing in this backend. What does: two new prompts of a batch that share an uncached first grain of tokens both prefill it. With the prefix cache the second now waits one batch and starts from the first's cached prefix (`a prompt waits one batch for another's prefix to be cached`); `speculative_checks.py` sends the pair together and compares with a server without the cache. |
| ALL_GATHER and REDUCE_SCATTER | done | Real ops (`GatherScatterAttrs`): ALL_REDUCE's replica groups plus a dimension, interpreter arms under the same single-process semantics (identity with one replica; unit tests for groups of two), forward tangents, `stablehlo.all_gather` / `stablehlo.reduce_scatter` in the flattened-id form. They compile and run on one device under the single-device options and `CompileOptions.spmd` (`PjrtCollectiveCompileTest`). No adjoints: the per-replica transpose pair could not be checked by the single-process interpreter. |
| Expert-parallel MOE_EXPERTS | done | `expert_parallel = true`: the expert weights hold a slice, its first global id the last operand (i32 `[1]`, so every rank runs one program); routing chooses among all experts, the op adds only the chosen experts it holds. Two slices of 8 experts sum to the whole op within 9e-8 on XLA:GPU (decode rows, prefill rows, ragged dot, sigmoid routing) and within one f32 rounding in the interpreter (`PjrtMoeExpertParallelTest`). Not with NVFP4 experts. |
| Decode context parallel | done | PAGED_ATTENTION `return_lse = true` returns each row's and head's log-sum-exp (-inf, with an output of 0, for a row with no position) from all three XLA forms (per row, rows sharing tables, blockwise); `ContextParallel.merge` combines slices. Pages striped over two slices, merged, equal attention over the whole context within 9e-8 on XLA:GPU and 5e-8 in the interpreter (`PjrtContextParallelTest`). Full-attention tables only; not the fused CUDA kernel. Found on the way: a non-finite scalar constant was emitted as `-Infinity`, which MLIR does not parse; it is now its bit pattern. |
| Compile once across ranks | already there, now tested | The executable cache holds a lock per entry around load-or-compile (in-process and a POSIX record lock between processes, JVM and backend), so ranks that start together compile once and the others load: three sessions on one cache, 1 compile and 2 loads (`PjrtExecutableCacheTest.sessionsStartingTogetherCompileOnce`). |
| One-hot permute for few rows | done | `permute = "onehot"` in the ragged-dot MoE form (set by `raggedDotMoe` up to 1,024 pairs): rows into expert order and the weighted results back by one-hot matmuls instead of a gather and a scatter. Equal to the interpreter on XLA:GPU, f32 and bf16 (`PjrtMoeExpertsTest`, `MoeRaggedDotEmitTest`). The crossover is for a TPU to measure. |
| Ragged mixed-batch entry | not built | Measured first. A prompt's prefill already runs first in its batch, and with chained steps off the four-stream time to first token is the same (0.20 s against 0.20 s chained; 0.10 s at one stream both ways), so the step in flight is not what a prompt waits for. One call for a prefill and the others' decode step would spare the decoders a stall per arriving prompt, not shorten the prompt's wait. |
| First tokens before the decode calls | tried, reverted | The backend sent a batch's responses after its decode calls, so a prompt's first tokens also waited for them. Sending them as soon as the prefill returned left the four-stream first step at 192.1 ms (194.3 before; p90 246.1 against 245.5): with chaining, the decode step of the batch was already queued on the GPU ahead of the prefill, and the call after it only reads its results. |
| TPU forms of the remaining items | recorded | TPU_BRINGUP question 7 lists what a TPU must measure: the sampler's StableHLO form, the router, the one-hot crossover, the collectives across chips. |
| Not built | — | A TPU KV layout (K/V interleaved and packed to 32-bit words) waits for a Mosaic attention kernel to read it; W4A8 NVFP4 experts on a TPU need a requantization whose accuracy is unmeasured; DP lockstep, vocab-parallel argmax and PP/PCP need several devices; structured output needs a grammar engine in the endpoint. |

## Where the four-stream time to first token goes

`openai_server.py --verbose` logs, per request, when it sent the tokens to Triton and when the
first step came back. Qwen3.6-35B-A3B, the 40 prompts of 32 to 123 tokens (median 52):

| | sent | first step |
|---|---|---|
| one stream | 3.1 ms | 95.9 ms |
| four streams, before | 13.2 ms | 194.3 ms (p90 245.5) |
| four streams, chaining off | — | (TTFT median 0.20 s, as chained) |
| four streams, first tokens sent before the decode calls | 13.3 ms | 192.1 ms (p90 246.1) |

Four 55-token prompts prefill in about 126 ms of server compute, as one prompt nearly does: a
prefill of a few hundred (row, expert) pairs reads most of the 35B's experts (about 18 GB of
NVFP4 weights, 66 ms at 273 GB/s). The endpoint sends a request's tokens to Triton within 3 ms
of receiving it (13 ms at four streams); the rest of the wait is the backend's. At four
streams a prompt waits about 100 ms more than alone, which fits two decode steps (about 50 ms
each) ahead of its prefill: the one running when it arrives and the next, chained or, unchained,
the call its batch runs after the prefill before responding. Not measured step by step.
