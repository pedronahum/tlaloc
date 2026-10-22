# DeepInfer lessons: plan

Branch `feat/deepinfer-lessons`, from `feat/flash-next` at `6d9c95c2` (2026-10-21). The
source of every item is [deepinfer-lessons.md](deepinfer-lessons.md); the numbers below are
its item numbers.

Measured on Qwen3.6-35B-A3B NVFP4 (3 drafts, FP8 KV, `-PcudaKernels=true`) on the GB10,
with the Qwen3.5-0.8B fixture's transformers ids as the correctness gate for anything that
changes served arithmetic.

## Facts found before starting

- The GB10 runs block-scaled MMA: `mma.sync.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X`
  (e2m1 × e2m1, an e4m3 scale per 16 values: NVFP4's own format) and
  `m16n8k32.kind::mxf8f6f4.block_scale` (e4m3 × e2m1, ue8m0 per 32), compiled for `sm_121a`
  or `compute_120f`. Plain `sm_121`, which `build_backend.sh` uses, rejects them.
- The 35B checkpoint's experts and shared experts are `W4A16_NVFP4` (ModelOpt 0.37): the
  weights were calibrated for 16-bit activations. FP8 × FP4 needs ue8m0 weight scales per 32,
  which NVFP4's e4m3 per 16 do not convert to exactly. FP4 activations (W4A4) keep the
  weights exact and quantize the activations to NVFP4.

## Order of work

| # | Item | Proof |
|---|---|---|
| 2 | Router top-k as a kernel (`tlaloc_moe_route`) | equal to the sort's indices and weights; ms per 2K chunk |
| 3 | FP4 × FP4 block-scaled MMA for prefill experts (opt-in) | kernel against the reference; fixture ids; 35B acceptance; ms per chunk |
| 1 | Packed prefill (per-token sequence ids, no padded rows) | interpreter parity; fixture ids; four-stream TTFT |
| 7 | In-graph sampler (top-k / top-p, counter RNG) | distribution tests; greedy unchanged |
| 5 | Chaining across batch changes (`remap`) | chained inputs equal the host's; dropped steps counted |
| 4 | Draft depth by confidence, on the device | acceptance and tokens/s at one and four streams |
| 6 | Prefix cache with state snapshots | a hit gives the same ids as a full prefill |
| 10 | Fewer execute arguments | host time per step |
| 11 | Scaled FP8 KV | saturation test; fixture ids |
| 12 | Deterministic mode | the 35B gives the same ids twice |
| 9 | Prefill interleaved with decode under a budget | decode gaps p50/p95/p99 under a 64K prefill |
| 13 | Timers inside the program | per-entry times without a profiler |
| 14 | Shared expert in the routed MoE kernel | ms per layer |
| 15 | Merged epilogues | only if the profile shows the passes |
| 8 | Flash-Next selective attention past 2,048 positions | kernel against XLA's form |

## Results (2026-10-21)

Qwen3.6-35B-A3B NVFP4 on the GB10 unless marked otherwise. "Categories" is the 40-prompt
harness (`scripts/spark-4user/streams.py categories`) behind the OpenAI endpoint, greedy;
"prefill chunk" is one 2,048-token call after 16,000 tokens (`profile.sh prefill:16000:3`).

| # | Item | Status | Result |
|---|---|---|---|
| 2 | Router top-k kernel | done | `tlaloc_moe_route` (`route_kernel` on MOE_EXPERTS, on with `-PcudaKernels=true`): one warp per row, softmax or sigmoid + bias, K rounds of a warp argmax (ties to the lower id). Indices and weights equal the sort's (`PjrtMoeRouteTest`). 10.6 µs for a 2,048-row chunk against about 1.1 ms for XLA's sort, per layer; in a prefill chunk's profile the sort's 45 ms becomes 0.94 ms. The end-to-end chunk time moves within XLA's autotuning spread (±20 ms between exports). |
| 3 | FP4 × FP4 prefill experts (W4A4) | done, opt-in | `-PprefillActQuant=nvfp4`: prefill rows are quantized to NVFP4 (e4m3 scale per 16, global scale amax/2688) and the experts run `mma.sync.m16n8k64.kind::mxf4nvf4.block_scale` (`ExpertGemmFp4Kernel`, `sm_121a`; `build_backend.sh` now compiles for it). Decode keeps W4A16. Kernel equals the CPU reference; the interpreter quantizes the same way (`Nvfp4ActivationsTest`, `PjrtNvfp4MatmulTest`). Experts per layer at 2,048 rows 5.24 → 3.12 ms; prefill chunk 885 → 733 ms. Acceptance unchanged (2.90 tokens per step). Replies: 15–17 of 41 identical to W4A16's, against 29 of 41 between two W4A16 runs, so it changes text; it stays opt-in because the checkpoint was calibrated for 16-bit activations. |
| 1 | Packed prefill | done, opt-in | `-PprefillPacked=P`: prefill entries whose row-wise ops (projections, norms, MLP, experts, head) run on P live rows gathered by `PACK_INDEX` / `UNPACK_INDEX`; attention and Gated DeltaNet stay on batch × tokens. Interpreted, the logits and every pool equal the padded entry's bit for bit (`HfPackedPrefillTest`); the 0.8B fixture gives transformers' ids through Triton ("2 prompts were prefilled packed (29 tokens in 32 rows)"). Categories at 4 streams: TTFT median 0.25 → 0.20 s (vLLM 0.14). |
| 7 | Sampler in the graph | done, opt-in | `-Psampling=true` (needs `-PcudaKernels=true` and MTP): SAMPLE_TOKENS / `tlaloc_sample`, temperature, top-p, top-k and a 24-bit seed per sequence; a counter RNG (splitmix64 of seed and position), weights from a fixed exp2 polynomial, so the kernel and `SampleTokensAttrs.reference` agree exactly. Drafts are greedy and verified as a point mass (accept with probability p(d), else resample without d). χ² checks pass; 0.082 ms for 16 rows of the 35B's vocabulary. Device draws the interpreter's tokens at T 0.8 and transformers' greedy ids at T 0 (`PjrtSampledSpeculativeTest`). Greedy categories unchanged (137.7 / 79.4 tok/s); T 0.7 at 4 streams 51.2 tok/s per stream, 2.13 tokens per step. The endpoint passes the parameters with `--sampling`. |
| 5 | Chained steps across batch changes | done (partial take) | A chained step whose next batch keeps some of its generations is taken for those rows (`a chained step was taken in part`); before, the whole step was dropped. Checked by `speculative_checks.py` ("ending apart"). Remapping rows into a different batch shape is not built. |
| 4 | Draft depth by confidence | not built | Needs control flow in the verify program (no IF / WHILE emission), and the verify entry would have to be packed for a variable draft count to save work; at 2.9 tokens per step of 4 the measured headroom is small. |
| 6 | Prefix cache with state snapshots | done, opt-in | `TLALOC_PREFIX_CACHE=1` (or model parameter `prefix_cache`): after a prompt's prefill the backend keeps its pages (refcounted) and a copy of its recurrent-state slot at a multiple of the prefill chunk; a new sequence whose prompt starts with a cached prefix shares those pages, copies the slot and prefills only the rest. `prefix_cache_entries` (default 8), LRU, evicted when pages run short. Not for windowed pools. On the 0.8B fixture a 96-token shared prefix is restored and both prompts give the ids of a server without the cache (`verify.sh` step 9). |
| 10 | Fewer execute arguments | not built | Host time of a 35B decode step is 1.69 ms launch + 0.29 ms inputs, and chaining issues the next step before the current one is read, so it is hidden. |
| 11 | Scaled FP8 KV | not built | The FP8 KV is unscaled e4m3, as vLLM's default; no saturation was found in the fixtures. |
| 12 | Deterministic mode | done, opt-in | `TLALOC_DETERMINISTIC=1` (`run_server.sh` passes it and `--xla_gpu_deterministic_ops=true`): attention partial sums and the W4A4 down projection add in fixed point (2^-24 and 2^-32 units) through integer atomics. Two 35B runs: 41 of 41 replies identical (29 of 41 without). Off by default: attention verify +40 %, W4A4 +0.7 ms a layer. |
| 9 | Prefill under a budget | done | While other sequences decode, a prompt runs at most `prefill_budget_tokens` per batch (default one call of the largest prefill entry; `TLALOC_PREFILL_BUDGET`, 0 turns it off), and the decoding sequences step between its calls (`a prompt runs in parts while other sequences decode`). One stream decoding 600 tokens while a prompt of 80,000 characters (about 20,000 tokens) arrives, fused export: gaps between that stream's chunks p50 / p95 / p99 / max 22.3 / 265.8 / 775.2 / 801.3 ms with the default budget, 22.2 / 26.5 / 50.9 / 9,048.7 ms without; the long prompt's first token after 9.55 s against 9.19 s. |
| 13 | Timers inside the program | not built | The backend's clock already gives per-entry compute times (`profile.sh MODE=time`), and nsys gives per-kernel times. |
| 14 | Shared expert in the routed MoE kernel | done | NVFP4 experts and an NVFP4 shared expert with the experts' intermediate size become one 15-operand MOE_EXPERTS, `sigmoid(shared gate)` as the shared expert's weight; blocks past the routed experts spread the shared expert's tiles. Per layer at 4 rows +9 µs against about 86 µs for the separate shared MLP; the 35B decode program runs 4 `Fp4GemmKernel` launches where it ran 88. Equal to the interpreter (`PjrtNvfp4MatmulTest`). |
| 15 | Merged epilogues | not built | No epilogue pass is worth merging: in the fused export's profiles no elementwise fusion takes more than 1.5 % of a prefill chunk's kernel time or 1 % of a decode step's. |
| 8 | Flash-Next selective attention | not built | Flash-Next is not served by Triton yet, and its selection is per row, which the paged kernel does not take. |

The endpoint now reuses its gRPC clients (`ClientPool`) instead of opening one per request.
Categories TTFT with it: 0.10 s at one stream and 0.20 s at four, as without, so the 30–65 ms the endpoint adds to a request is not the client's setup.

All opt-ins together (`-PprefillActQuant=nvfp4 -PprefillPacked=… -PcudaKernels=true`, fused
shared expert), against the W4A16 export of the same build:

| | W4A16 | W4A4 | + packed | + fused shared |
|---|---|---|---|---|
| prefill chunk after 16K (ms) | 885 | 733 | — | 761 |
| decode, 4 streams at 256 (ms a step) | — | — | — | 51.9 |
| categories, 1 stream: tok/s, TTFT | 138.5, 0.10 s | 139.8, 0.10 s | 134.7, 0.10 s | 144.4, 0.10 s |
| categories, 4 streams: tok/s, TTFT | 75.6, 0.25 s | 77.5, 0.23 s | 79.7, 0.20 s | 81.2, 0.20 s |
| categories, 4 streams: aggregate tok/s | 147.3 | 152.9 | 155.1 | 163.0 |

The fused export's prefill chunk profiles at 690 ms of kernels: the W4A4 expert GEMM 148 ms,
the prefill attention kernel 121 ms, XLA's dense dots and the Gated DeltaNet triangular
solves most of the rest.
