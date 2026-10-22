# DeepInfer: what Tlaloc can take from it

Source: [deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer) at `774ce244`
(2026-10-20), read only (not built or run), together with its
`doc/inference_deployment_qa.md`. Compared against Tlaloc `feat/flash-next` at `7d409556`.

## What DeepInfer is

DeepSeek's production engine for DeepSeek-V4.1-Flash, on NVIDIA SM100 and Ascend 950. It
began as a vLLM fork. About 70K lines: a C++ scheduler and sequence state (`csrc/engine`),
a Python worker, and about 60 TileLang kernels. FlashMLA, DeepGEMM, DeepEP and DeepSelect
are submodules, and they are empty in the release. It is built for wide expert parallelism
(EP64–128 decode, EP2–8 prefill), with prefill, decode and the vision encoder on separate
machines.

The release leaves out the KV-cache store and the load balancers. The Q&A doc describes
them, but the code does not contain them. That includes the quadratic `cost(q, c)` prefill
chunk solver: `grep` finds no `compute_quota` anywhere in the tree.

**Its benchmark numbers are not comparable to ours.** `examples/benchmark.py` reports decode
tokens/s as `steps/s × requests × (1 + nextn)`, which assumes every draft is accepted.

## Mechanisms worth knowing

### Engine

- **Zero-overhead scheduling.** While the GPU runs forward(N) and sample(N), the CPU does
  four things: schedules N+1 (C++, GIL released), launches forward(N+1), post-processes N
  (waiting only on N's sample event), and launches sample(N+1) (`worker.py:128-229`).
  - Only the forward pass is speculative; sampling sees exact host state.
  - Step N's tokens and accept counts never come back to the host before forward(N+1).
    The host builds a `remap[i]` (each row's index in the previous batch, or −1), and two
    small kernels gather the previous step's sampled and draft tokens and correct
    `seqlens_k` on the device (`gather_input_ids_zos.py`, `compute_seqlens_k_zos.py`).
  - Fresh rows come from a pinned host buffer, so the batch can change between steps
    without losing the overlap.
  - A sequence that hit EOS costs one wasted forward row.
  - Blocks are reserved for `nextn + 1` in-flight tokens and trimmed when the step commits
    (`sequence.hpp:216-293`).
- **Decode metadata built on the GPU.** The host sends compact per-sequence data
  (`cu_seqlens_q`, state ids, flattened block tables). One TileLang kernel expands it into
  every per-token index: positions, SWA and compressed slots, and the block table per
  ratio (`prepare_v4_decode_metadata.py`). `forbid_cuda_sync()` guards the launches.
- **CUDA graphs keyed by total query tokens, not by sequence count.** The buckets are
  1, 2, 4, then multiples of 8 up to `max_num_seqs × (1 + nextn)`. All graphs share one
  pool, and smaller ones use prefix slices of the largest one's buffers. Padding rows are
  placeholder sequences with one query each (`cuda_graph_runner.py:41-148`).
- **Prefill is packed varlen** (`cu_seqlens`) and runs eagerly. There are no padded rows.
- **Prefill and decode never share a step.** A rank attempts prefill every 70 decode steps
  when all ranks want it, and every 100 when any rank does (`scheduler.hpp:295-341`).
- **Preemption kills the request with the longest output** ("insufficient_system_resource")
  rather than recomputing it.
- **One `cuMemAlloc` holds every KV block,** with all layers' entries interleaved inside a
  block (`cache_engine.py:376-477, 910`).
- **Timers inside replayed graphs.** Events are recorded during capture as event-record
  nodes and swapped before each replay with `cudaGraphExecEventRecordNodeSetEvent`
  (`csrc/graph_timer`). Kernel times are read with no sync and no profiler.
- **Small CPU details:** `gc.disable()` with a collect every 1000 steps, NUMA binding, and
  object pools with deferred release.

### Speculative decoding (DSpark / DFlash)

- **A block-parallel drafter.** One forward pass of the drafter layers produces all 5
  drafts:
  - The input is `[sampled token, noise, noise, noise, noise]` at the next 5 positions.
  - Attention inside the block is bidirectional. Every row reads the same key list: an SWA
    window plus the block (`dflash.py:230-269`, `get_kv_index.py:151-230`).
  - The drafter's keys are projections of the target's attention inputs at chosen layers.
    They are written inside the target's CUDA graph, so the drafter has no prefill and no
    rollback.
  - A rank-r bigram correction `logits += W2·W1[prev draft]` is the only sequential part.
  - It needs a trained drafter. The Qwen MTP heads are autoregressive and cannot be used
    this way.
- **A confidence head:** a sigmoid GEMV over the draft hidden plus the bigram embedding.
  - Its per-draft outputs are multiplied cumulatively into P(prefix accepted).
  - Every (request, draft) pair is sorted by that probability, and pairs are added
    greedily while expected tokens divided by step time keeps rising.
  - Step time comes from a per-graph-bucket curve profiled at start-up (trimmed mean,
    cached on disk for 1 h), plus an online median slope correction
    (`mtp_varlen.hpp:19-267`).
  - Because cumulative products only fall within a request, the global top-k is a prefix
    of each request's drafts.
- **Async scheduling under speculation.** The host fixes only the total query budget. A GPU
  kernel splits it across requests with an O(N²) rank count that equals a stable sort
  exactly (`schedule_mtp_varlen.py`).
- **Measured gain over a fixed 5 drafts:** −1.5% at 16 sequences per rank, +1.6% at 32,
  +5.4% at 60, +11% at 96. Fewer drafts are verified per step (5 to 3.2–4.4) for almost the
  same accepted length (3.29 to 3.20).
- **Rejection sampling.**
  - Draft logprob rows are stored per slot. A draft is accepted iff `u < exp(min(lp_t − lp_d, 0))`.
  - A rejected position resamples from `max(p_t − p_d·e^{−5e-5}, 0)`; the small penalty
    keeps the residual from being empty.
  - Σ min(p, q) comes out as a by-product, and it measures the confidence head's
    calibration online.
- **Rollback costs nothing.** SWA and compression rings are sized `window + nextn + 1`
  (128 + 6), so writes at rejected positions never clobber the live window
  (`cache_engine.py:88-133`).

### Sampler

- **Top-k/top-p without a sort.** A radix select over the logits (as sortable uint32,
  11-bit digits) also accumulates fixed-point probability mass in its histogram, so one
  pass finds the exact top-p and top-k threshold. A tie scan fixes the cut index
  (`cuda/fused_sample.py:78-369`).
- **Sampling is a two-level fixed-point inverse CDF** with a counter-based RNG keyed by
  `(seed, 3·position + case)`.
  - The integer sums are associative, so the result does not depend on thread order or
    batch composition (`random_sampling.py`).
  - No RNG state is carried between steps.
- **Batch-invariant mode** (`DEEPINFER_BATCH_INVARIANT=1`): deterministic DeepGEMM and
  TileLang kernels, batch-invariant FlashMLA, and the fused path off in prefill.
- **Structured output.** xgrammar masks are built on the CPU while the logits GEMM runs.
  The draft rows' masks come from walking the grammar along the drafts and rolling back.
  The copy runs on a side stream.

### Kernels and model

- **Fusions:**
  - norm → RoPE → Hadamard → FP8/FP4 group cast → paged store in one kernel, with an
    optional bf16 side output at a chosen stage (`fused_transform.py:189-360`);
  - attention with q-norm/RoPE in front and inverse RoPE plus FP8 output cast behind;
  - router GEMM + scoring + bias + top-k + replica map in one kernel (`bf16_mega_gate`);
  - the MoE dispatch, both GEMMs, SwiGLU, combine and the shared expert in one kernel
    (Mega MoE);
  - mHC post + next pre + RMSNorm + FP8 cast, written straight into the MoE input layout;
  - GEMMs that share an input are merged at load time.
- **Formats:**
  - activations are MXFP8 (per-32 ue8m0), quantized inside the norms;
  - experts run FP8 × FP4 MMA;
  - main KV is FP4 with per-16 e4m3 scales (288 B/token);
  - SWA KV is MXFP8 (528 B/token for 512 dims);
  - indexer K is MXFP4 (68 B/token).
- **Sparse attention:**
  - Decode runs one softmax over two caches: the SWA ring plus the top-k compressed tokens.
  - Prefill gathers `[SWA | chunk | selected]` into one buffer and makes one sparse call.
  - Indexer top-k uses DeepSelect with `end = valid_len` (logits are never cleared) and an
    output offset (indices land in buffer coordinates).
  - Two-stage selection: block max-pool, top-k blocks, then tokens within them, always
    keeping the last block. Selections are reused across neighbouring layers.
- **YOCO bounded replay.** The upper ("consumer") layers take their KV from one shared
  hidden state. Prefill therefore projects that KV for every token once, and runs the rest
  of the upper half only on the last 140 tokens, which nearly halves prefill FLOPs. It
  depends on the model's architecture.
- **Engram tables** sit in host memory, sharded. The fetch by hash is issued before layer 0
  and overlaps the early layers.

## What applies to Tlaloc, ranked

These rankings come from Tlaloc's measured gaps, taken from FOLLOWUPS §8 and the progress
logs. On Qwen3.6-35B-A3B at four streams, the step time is already level with vLLM
(42.4 against 41.2 ms). What remains:

- time to first token at four streams: 0.26 against 0.14 s;
- long prefill: 53 against 29–36 s at 100K;
- acceptance: 2.87 against 3.02 tokens per step;
- greedy only;
- no prefix cache across sequences.

### Tier 1: direct hits on measured gaps

1. **Packed prefill entries keyed by token count.**
   - DeepInfer packs prefill varlen and keys graphs by total tokens. Tlaloc pads every
     prompt row to the entry's width: four short prompts become 4 × 128 = 512 rows, about
     250 ms.
   - The static-shape version is a prefill entry of T tokens with per-token sequence ids,
     positions, slot mappings and per-sequence block tables. The pieces that need it:
     - the attention prefill kernel and `KV_CACHE_WRITE` need the per-token ids;
     - the MoE pair lists already work per token;
     - the GDN/conv prefill form needs segment boundaries, chunked per sequence.
   - Expected: four-stream TTFT from 0.26 s to about the one-stream 0.10 s. The same entry
     shape lets a decode cohort's rows ride along later (item 9).
2. **A routing top-k kernel.**
   - `emitMoeExperts` sorts every expert per row with a stable `stablehlo.sort`
     (`StablehloEmitter.kt:5004/5025`). XLA runs that as a segmented radix sort: 45–47 ms
     of a 2,048-token chunk, about 2.3 s of a 100K prefill.
   - A `tlaloc_moe_route` FFI kernel would do scoring + bias + top-k, with one warp per row
     over 256 or 512 experts. It could also emit the per-expert pair lists that
     `tlaloc_moe_fp4`'s `ListsKernel` builds today.
   - This is the `bf16_mega_gate` idea without the GEMM.
3. **FP8 activations for prefill experts and MLPs.**
   - Tlaloc's prefill experts are W4A16 on `mma.sync` bf16 at about 18 TFLOP/s: 265 ms of
     the 760 ms of non-attention time per 2K chunk. Dense NVFP4 MLPs above 16 rows widen
     the weights on every call (5–9 ms per projection).
   - DeepInfer quantizes activations to MXFP8 inside the preceding norm and runs FP8 × FP4
     block-scaled MMA.
   - On the GB10 (SM121), check that `mma.sync ... kind::mxf8f6f4.block_scale` is available.
     Then the expert prefill tile would take FP8 A with per-32 scales written by the RMSNorm
     epilogue, and it would also lift the 16-row limit for dense projections.
   - Accuracy gate: transformers' ids on the f32 Qwen3.5-0.8B fixture, and acceptance
     unchanged on the 35B.
   - This is the largest prefill lever and also the largest job.

### Tier 2: clear gains, more design work

4. **Variable draft depth decided on the device, so it works with chained steps.**
   - flash-next-plan defers dynamic depth because chaining needs the host out of the
     decision. DeepInfer's answer fits that constraint: the host fixes the total verify
     rows, which is the entry's size, and a device kernel splits them by cumulative draft
     confidence.
   - Without a trained confidence head, use the draft head's own top-1 probability.
     TensorFold's `--mtp-confidence 0.70` does the same.
   - At four streams the total stays at 16 rows (the `tlaloc_fp4_gemm` limit), but a code
     stream can take 6 rows while a prose stream takes 2.
   - A device-side `while` that stops drafting once every row's cumulative probability is
     below a threshold also saves head passes, at about 2 ms each.
   - DeepInfer's gain at low concurrency is small (−1.5 to +1.6% at 16–32 sequences per
     rank). Expect single-digit percent unless one stream is code-heavy.
5. **Chaining across batch changes.**
   - Today a chained step is dropped when a generation ends or a prompt arrives, and an
     unchained step pays the 3.3 ms execute and the host read.
   - Add a `remap` input (each row's index in the previous step's outputs, or −1). The next
     inputs become a device gather from the previous outputs plus fresh host rows. A row
     that hit its end token then wastes one row of compute instead of the whole chain.
6. **A prefix cache for hybrid models, with state snapshots at a fixed grain.**
   - DeepInfer keeps SWA state only at 128-token multiples, and the hit length is the last
     snapshot (`L − 134` at most).
   - The Gated DeltaNet and conv states are the same problem. Copy a sequence's state into
     a spare slot at every prefill-chunk boundary, which prefill already produces. The
     paged KV pools are already shareable by page.
   - A cross-sequence hit then costs one state copy plus the suffix prefill. It matters for
     agent traffic, where long system prompts repeat; the endpoint today reuses only a live
     sequence's own KV.
7. **An in-graph sampler.**
   - Copy DeepInfer's design whole: radix-select thresholds for top-k/top-p, a fixed-point
     inverse CDF, and a counter RNG keyed by (request seed, position).
   - Counter RNG suits Tlaloc: no state crosses steps, chained steps stay valid, and
     results are batch-invariant by construction.
   - Speculative sampling then needs the draft probabilities kept per draft slot, plus the
     residual resample with a small penalty.
   - This removes the "greedy only" limit in the endpoint at little cost to the step.
8. **Flash-Next selective attention past 2,048 positions.**
   - Today the keep mask sends `PAGED_ATTENTION` to XLA's whole-bucket gather.
   - DeepInfer's shape is one softmax over a merged index list (the dense recent window
     plus the selected chunks), read straight from the paged pool. `INDEXER_TOPK`'s two
     `stablehlo.sort`s become a radix select with `end = valid_len`.
   - This is the CUDA-kernel item already queued in flash-next-plan.

### Tier 3: smaller or situational

9. **Interleave long prefill with decode under a time budget.**
   - Today a long prefill stalls every decode stream for the whole chunk.
   - The Q&A's idea, which is absent from the released code, is to choose the chunk size
     from a fitted `cost(q, c)` so a step stays under a budget. On one GPU, alternate decode
     steps with prefill chunks sized that way, and measure decode gaps p50/p95/p99 under a
     64K prefill, as the community harnesses do.
10. **Fewer execute arguments.** DeepInfer keeps every KV block in one allocation, with
    layers at offsets inside it. For Tlaloc's 3.3 ms of host time over about 1,300 arguments,
    pack the pools and the weights into a few buffers each. FOLLOWUPS already proposes this
    for the weights.
11. **Scaled FP8 KV.** Per-32 ue8m0 scales cost 16 B per 512 values and remove the ±448
    saturation hazard of today's unscaled e4m3 pools. They may also bear on the acceptance
    gap, which is attributed to FP8 in the target's attention path.
12. **A deterministic mode.** 35B ids differ from run to run today (XLA autotune, atomics in
    attention), so backend changes are validated on the 0.8B fixture. Pinned autotune
    results, deterministic XLA ops and an atomic-free combine path would let the 35B be its
    own oracle, the way DeepInfer uses batch-invariant mode.
13. **Timers inside the graph.** Swappable event nodes give per-entry step times with no
    profiler. That is needed for item 4's cost curve, and it replaces some nsys runs.
14. **Shared expert fused into the routed MoE kernel.** This saves one read of the
    activations and one launch per layer. It is small at decode, where the experts are
    already at the memory rate.
15. **Merged attention epilogue** (inverse RoPE + FP8 cast) and **a merged
    norm → RoPE → quant → store**, if the nsys budget shows those elementwise passes.

### Not applicable to one GB10

- Wide EP, EPLB, DP attention, the DeepEP and Mega MoE communication, and EP failover.
- Prefill/decode/encoder disaggregation, the 3FS KV store, and autoscaling.
- YOCO bounded replay, which needs a YOCO model.
- Engram tables in host memory: on unified memory, host and device are the same LPDDR.
  The Flash-Next problem is capacity, not placement.
- The DFlash block drafter as such, unless a trained block drafter for a served model
  exists or Tlaloc trains one. Tlaloc has the training pieces (HfCausalLm fine-tuning,
  AdamW), so a block drafter distilled from the target's hidden states is possible later.
  Seven drafts in one pass would change the four-stream draft-depth trade-off.
