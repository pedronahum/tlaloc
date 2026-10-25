# vLLM on TorchTPU: what Tlaloc can take from it

Sources: the announcement ([vllm.ai/blog/2026-10-20-vllm-torchtpu](https://vllm.ai/blog/2026-10-20-vllm-torchtpu))
and [vllm-project/vllm-torchtpu](https://github.com/vllm-project/vllm-torchtpu) at `20ca0e7e`
(2026-10-22), read only (about 171K lines of Python; nothing built or run). Paths below are
under `src/vllm_torchtpu/` unless they start with `docs/` or `scripts/`. Compared against
Tlaloc `feat/deepinfer-lessons`.

The stack it pins: torch 2.13, torch_tpu 0.2.0, jax 0.11.2, libtpu 0.0.50. The older JAX
backend (`tpu-inference`) is now called "vLLM JAX"; this plugin is the recommended path for
PyTorch model definitions.

## What it is, from the code

- **An out-of-tree vLLM platform.** Upstream vLLM's scheduler, KV cache manager and API
  server run unmodified. `TPUModelRunner` subclasses `GPUModelRunner` and aliases
  `torch.cuda.*` to `torch.tpu` (`runner/tpu_runner.py`).
- **One compiled graph per token bucket.** `torch.compile(backend="tpu")` (Dynamo → FX →
  MLIR → PJRT, no Inductor). `TpuPlatform.check_and_update_config` sets `splitting_ops = []`:
  the whole model is one graph, not vLLM's piecewise split. Outside the graph (input prep)
  TorchTPU runs in defer-and-fuse eager mode.
- **Kernels are JAX/Pallas functions called from torch.** `torch_tpu._internal.pallas.jax_op`
  appears 66 times; `distributed/sharded_jax_op.py` patches it so that `shard_map` outputs
  are sized per shard.
- **One worker process per TPU device**, talking through `torch.distributed` (backend
  `tpu_dist`). A v7x chip is two logical devices of 96 GB.

## Parallelism

Six dimensions, composable, single- and multi-host: TP, DP, PP, EP, prefill context parallel
(PCP), decode context parallel (DCP). Devices = DP × TP × PP × PCP.

- **No GSPMD or Shardy over the model.** Each worker's JAX mesh is one device
  (`_create_mesh_for_parallelism`). TP, DP and PP use vLLM's `GroupCoordinator` collectives
  over `tpu_dist`; the only `mark_sharding` in the tree is commented out. Multi-device SPMD
  exists only inside single ops: `shard_map` over meshes built for one op (EP:
  `distributed/ep_mesh.py`; PCP: `distributed/pcp.py`; DCP: `distributed/dcp.py`), with
  Pallas `make_async_remote_copy` (device-to-device DMA) inside the kernels.
- **MoE expert parallelism is all-gather, local experts, reduce-scatter.**
  `TpuDeviceCommunicator.dispatch` all-gathers hidden states, top-k weights and ids over the
  DP group; `combine` reduce-scatters. Each rank subtracts a *traced* `experts_start` from
  the global ids and masks experts it does not own (`fused_moe_gmm.py`), so every rank
  compiles the same program and the in-graph collectives line up. The docs call the DP + EP
  case "all-to-all"; the default code does not.
- **A fused EP kernel (opt-in)** does routing, the push of each row to the expert's owner,
  the expert matmuls and the push back in one SPMD Pallas program
  (`kernels/fused_moe/v2/`). On Qwen3.5-397B DP8 + EP, 8k/1k at 256 concurrent: 33,304 against
  30,888 tok/s; the reduce-scatter it replaces cost 1,190 µs per MoE layer, fully exposed.
  On the 35B, TPOT was 3.9 % worse, so it is armed only above a token count.
- **The best MoE layout they report is DP attention + EP** (e.g. DP 8, TP 1 on a v7x host).
  It needs lockstep: `_dp_coordinated_step` all-reduces (MAX) the token count, chunk count
  and request count over a CPU group, every rank pads to the same bucket, and idle ranks
  run dummy forwards so the collectives match. `core/tpu_scheduler.py` aligns prefill
  admission across DP ranks.
- **Mesh order matters.** `_ep_mesh_order` sorts the EP mesh by device id: rank-ordered
  all-gathers offloaded to SparseCore took 321 µs instead of 178 µs on a 2x2x2 slice
  (DeepSeek-V4-Pro, −4.5 % end to end). With PCP or DCP the mesh stays in rank order,
  because every Pallas kernel in one compiled module must share one device assignment.
- **Pipeline parallel** hands a stage over with one collective-permute over ICI that all
  stages run together, sending `hidden + residual` summed in the model dtype (half the
  bytes; bounded at 2^-8 relative in bf16) (`distributed/pp_wave.py`). A time-budget chunk
  scheduler keeps stages busy (`core/pp_chunk_scheduler.py`).
- **Prefill context parallel** is ring attention: page p lives on rank p mod PCP, query
  tokens are packed rank-major, KV pages stream round the ring inside a `shard_map` kernel
  with an online softmax (`kernels/experimental/pcp_streaming_rpa/`).
- **Decode context parallel** all-gathers the log-sum-exp and reduce-scatters the weighted
  outputs (`distributed/dcp_comm.py`); the Helix-style all-to-all was "no benefit over AG-RS
  on TPU topologies" and is off by default.
- **Intra-chip.** Process groups over a v7x chip's two cores (`distributed/intra_chip.py`):
  EP between chips and TP inside one (`moe_hierarchical.py`); Kimi K3's replicated latent
  projections column-split across the two cores, "~22 % of decode device time at TP=32".
- **Multi-host.** One `vllm serve` per host with a `TCPStore` rendezvous that fills
  `TORCH_TPU_SLICEBUILDER_ADDRESSES` and `TORCH_TPU_TOPOLOGY`, or Ray. Weights stream from
  GCS, and with EP each rank downloads only its experts (`--enable-ep-weight-filter`).

### Against Tlaloc

Tlaloc's multi-device design ([MULTIHOST_DESIGN.md](../design/MULTIHOST_DESIGN.md)) is the other
model: one program over N devices (`CompileOptions.spmd`, `num_partitions = N`), sharding as
Shardy annotations in DXIR (`DxirMesh`, `sdy.manual_computation`), `ALL_REDUCE` a real op,
`ALL_GATHER` / `REDUCE_SCATTER` still verification-only. The serving graphs are single-device.

What the code settles for Tlaloc's serving path when it gets several devices:

1. **For MoE serving, the program is per rank, not one SPMD program.** vLLM-TorchTPU keeps
   each rank's model graph single-device and puts the cross-device work in explicit
   collectives and per-op `shard_map`s. That is what Tlaloc's `sdy.manual_computation` already
   expresses. EP needs `ALL_GATHER` and `REDUCE_SCATTER` as real ops (emission, interpreter,
   adjoint-free INFERENCE_ONLY is enough), and MOE_EXPERTS with an expert offset operand
   (traced, so all ranks share one program text and one executable cache key).
2. **DP attention + EP needs a lockstep protocol in the backend:** agree on the step's
   bucket across ranks, pad to it, run a dummy step on idle ranks. The Triton backend's batch
   former is per instance; this is a new cross-instance barrier.
3. **Two collective forms are enough to start:** all-gather + reduce-scatter for EP, all-gather
   of the LSE + reduce-scatter for decode context parallel. The fused remote-DMA EP kernel is
   a later step whose own measurements say it pays only at large token counts.
4. **Device ids and mesh order:** sort by device id for the EP mesh, keep one device
   assignment for every kernel of a module. This matches the TorchTPU note in
   MULTIHOST_DESIGN §4a (ranks follow the physical topology, not launch order).
5. **PP's summed hand-off** (one tensor, model dtype) halves the transfer and is a pattern for
   any Tlaloc stage split.

None of this can be measured on one GB10; it is design input for G4.

## KV cache

- **Layout.** `[pages, page_size, ceil(2 × kv_heads / packing), packing, align(head_dim, 128)]`
  (`kernels/ragged_paged_attention/v3/kernel.py`, `get_kv_cache_shape`): token-major in a
  page, K and V interleaved along the head axis, narrow dtypes packed into 32-bit words (two
  bf16, four fp8), head dim padded to 128 lanes. One cache per layer by default;
  `VLLM_TPU_BLOCK_MAJOR_KV=1` stores all layers of one block together, so a block moves as
  one DMA for offload and transfer.
- **The batched RPA backend** (used by every CI recipe) has two layouts, heads along
  sublanes or sequence along lanes, and block sizes 128 or 256. The default backend uses
  16-token pages above 8,192 positions of model length.
- **Dtypes and scales.** bf16, fp8 e4m3 and e5m2. Per-tensor static `k_scale` / `v_scale`, and
  in batched RPA per-token dynamic scales, packed into extra head-dim channels; a packed FP4
  KV path exists. The benchmark harness forces fp8 KV.
- **One kernel for mixed batches.** Ragged paged attention v3 takes decode, prefill and mixed
  segments in one call (`distribution = [i, j, k]`, each case compiled with its own block
  sizes) and writes the new KV inside the kernel. The batched variant computes its DMA
  schedule on the device with a Pallas metadata kernel (`schedule.py`), runs a
  triple-buffered manual DMA pipeline and returns the LSE for context parallelism. The runner
  orders each batch decode first, then verify windows, then prefill.
- **Hybrid models and prefix caching.** GDN conv and recurrent state live inside
  attention-shaped pages of one unified pool (`TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL`); the block
  size is derived so the state regions fit. "Seed copies" move a sequence's state to its new
  block at chunk boundaries and on prefix hits (`_collect_mamba_state_seed_copies`).
  `PrefixAwareScheduler` admits the requests with the fewest uncached tokens first, looking
  64 requests ahead. Kimi K3 on the unified pool is slower until optimized (TTFT 995 →
  1,573 ms in its baseline).
- **Sizing.** A profile run at the largest token count, then the HBM budget less a reserve for
  KV connectors, minimum across EP/DP ranks (`worker/tpu_worker.py`).
- **Offload and disaggregation.** A C++ path that reads TPU buffers directly backs a vLLM
  offloading connector and a P/D connector that reshards KV between prefill and decode
  layouts (e.g. PCP4/TP2 → DP8/TP1).

### Against Tlaloc

Tlaloc's pools are per layer, K and V separate, `[numBlocks, blockSize, kvHeads, headDim]`
(`DecodeGraphSpec.poolType`), f32 / bf16 / unscaled e4m3; windowed rings for sliding layers;
Gated DeltaNet state in separate state pools with slots per sequence; the prefix cache added
on this branch copies a state slot on a hit (`CopyStateSlot`), the same move as their seed
copy.

1. **A ragged mixed-batch attention entry is the largest structural difference.** Tlaloc runs
   prefill and decode as separate entries keyed by batch × context × tokens (48 programs for
   the 35B export), and the budget on this branch interleaves them across batches. One
   ragged entry keyed by total tokens, with per-sequence query lengths and the KV write
   inside the kernel, would let a decode step and a prefill chunk share a call, drop the
   context ladder from the program key, and remove verify padding. On the GB10 this is a
   change to `tlaloc_paged_attention` (it already takes per-row lengths and tables) and to
   the backend's batch former. Measured afterwards ([vllm-torchtpu-plan.md](vllm-torchtpu-plan.md)):
   it would not shorten the four-stream TTFT (0.20 s against vLLM's 0.14 s), since a prompt's
   prefill already runs first in its batch and the wait is for the decode steps ahead of it.
2. **Their prefix-aware admission order** (fewest uncached tokens first) is a small backend
   change on top of the prefix cache.
3. **Per-token dynamic KV scales** are the scaled-FP8 option Tlaloc set aside (the not-built
   item 11 of the DeepInfer plan). They store the scale in padded head-dim lanes, which is free
   on TPU because head_dim pads to 128 anyway; on the GB10 it would be a separate scale pool.
   Worth it only if long-context accuracy at FP8 shows a loss.
4. **For a TPU KV layout**, K/V interleaved per head and dtype-packed into 32-bit words is the
   shape a Mosaic kernel wants (sublane packing). Tlaloc's separate K and V pools are fine for
   XLA and the CUDA kernel, and would be re-laid out for a Mosaic attention kernel.
5. **A unified pool** (state inside attention pages) is not needed for Tlaloc: its state slots
   are already separate and the backend already reclaims them with pages.
6. **Sizing at load** does not fit Tlaloc's model, where the pool shapes are compiled into the
   exported programs; `numBlocks` stays an export decision.

## Batching, buckets, compilation

- **Token buckets:** doubling from 16, the last exactly `max_num_batched_tokens`
  (`_get_token_paddings`), optional linear steps; one executable per bucket. The request-count
  dimension is fixed during warm-up. Sampling, logits, rejection sampling and structured
  output compile separately over power-of-two request counts. Each bucket costs host memory:
  "about 13 GB per bucket on TP8 Qwen3.5-397B with MTP".
- **Compile cache:** PJRT binaries under `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`, Dynamo /
  AOTAutograd artifacts under `VLLM_CACHE_ROOT` (the blog names `VLLM_XLA_CACHE_PATH`). One
  rank compiles a bucket and the others replay it by graph fingerprint
  (`TPU_PRECOMPILE_PEER_REPLAY`). `VLLM_XLA_CHECK_RECOMPILATION` fails on a compile after
  warm-up.
- **Host overhead:** non-blocking uploads from reused staging tensors behind an event fence;
  an upload skipped when its bytes did not change (`_upload_unless_unchanged`); placeholder
  tokens replaced on the device under async scheduling; the MTP draft loop as one compiled
  program.

### Against Tlaloc

Tlaloc already has the persistent executable cache, a recompile check
(`StepBenchmark`), chained verify steps with the next step's inputs made on the device, and
the MTP head inside the verify program. Two items are new:

1. **Peer replay by fingerprint**, for several ranks on one host. Tlaloc ranks run the same
   program text, so the cache key already agrees; what is missing is a lock so that one rank
   compiles and the others wait (the cache already takes POSIX record locks).
2. **Skip an input upload when unchanged.** Tlaloc writes step inputs into pinned mapped
   memory with no copy, so there is little to skip on the GB10.

## Speculative decoding and sampling

- MTP, EAGLE-3, DFlash (block-diffusion drafts, up to 16 tokens), DSpark, n-gram, suffix.
  Async scheduling with speculation is refused unless the method drafts on the device.
- **Sampling on the device:** Gumbel-max (argmax of logits / T − log(−log u)), top-k / top-p by a
  31-step binary search over float32 bit patterns (`top_k_top_p.py`), one generator shared
  across TP ranks. **Per-request seeds are refused** ("JAX does not support per-request
  seed"). A vocab-parallel greedy argmax avoids gathering logits under TP.
- Structured output: an xgrammar bitmask applied on the device, including to verify rows.

### Against Tlaloc

Tlaloc's sampler (this branch) has per-sequence seeds and is exact against its reference, which
vLLM-TorchTPU does not offer. Two items transfer:

1. **The bit-pattern binary search for top-p / top-k** needs no sort and no shared memory; it
   is the form for a TPU sampler (Tlaloc's `tlaloc_sample` is CUDA only, and SAMPLE_TOKENS
   has no XLA form).
2. **Vocab-parallel greedy argmax** for a tensor-parallel head (each rank's max and index, one
   small all-gather) when Tlaloc shards the head.

Structured output (a token bitmask applied before the argmax or draw) is a gap Tlaloc has not
listed.

## Quantization and MoE kernels

- Formats: fp8 (per tensor, per channel, 2D block), compressed-tensors (W8A8, W4A8, W4A16,
  mxfp4), mxfp4, nvfp4, awq, gptq. The stated rule on TPU: run 4-bit weights on the fp8 path
  (W4A8), since the MXU has no FP4.
- **MoE is a megablox grouped matmul** (`kernels/megablox/gmm_v2.py`), not `ragged_dot`; the
  block size selects W4A16 or W4A8. Permute and unpermute run on SparseCore; below a row
  count they run as one-hot matmuls on the TensorCore instead; tiny batches take a dense-expert
  path.
- **Router top-k avoids the sort:** k rounds of max-and-mask, because `torch.topk` lowers to a
  full `stablehlo.sort` (a 45-stage bitonic network at 512 experts) (`kernels/router_topk.py`,
  default `rowmax`).

### Against Tlaloc

1. **Router top-k without a sort in the XLA lowering.** Tlaloc's CUDA `tlaloc_moe_route` does
   K rounds of a warp argmax; the XLA form of MOE_EXPERTS (used on TPU and without
   `-PcudaKernels`) still emits `stablehlo.sort`, which cost about 1.1 ms a layer on the
   GB10. K rounds of `reduce max` + mask in StableHLO is the same choice (ties to the lower
   id) with no kernel. This applies on both targets now.
2. **For TPU MoE, a grouped-matmul kernel rather than `chlo.ragged_dot`.** Tlaloc's TPU MoE
   form from the TorchTPU lessons is two ragged dots; vLLM-TorchTPU, by the same team's stack,
   chose a Pallas grouped matmul with tuned block sizes. Measure the ragged-dot form on a
   TPU first, and keep a Mosaic grouped matmul as the fallback.
3. **One-hot permute for small row counts** (decode): the gather/scatter of a few rows as a
   small matmul, on TPU where a gather is slow.
4. **W4A8 on TPU:** NVFP4 checkpoints would run with their e2m1 weights widened to fp8 for the
   MXU. Tlaloc's NVFP4 lowering on TPU would need that form.

## Numbers they publish

The blog has none. The repo's regression baselines (v7x, `vllm bench serve`, random data,
temperature 0, closed-loop concurrency), selected:

| Model | Slice, layout | In / out, concurrency | Result |
|---|---|---|---|
| Qwen3.5-35B-A3B FP8 | v7x-8, TP4 + EP | 1k / 1k, 64 | TPOT 9.72 ms, 11,509 tok/s |
| Qwen3.5-35B-A3B FP8 | v7x-8, DP4 / TP2 + EP | 1k / 1k, 64 | TPOT 10.33 ms, TTFT 95 ms |
| Qwen3-30B-A3B NVFP4 | v7x-8, TP8 + EP | 1k / 1k, 64 | TPOT 12.5 ms, 8,764 tok/s |
| Qwen3.5-397B-A17B FP8 | v7x-8, DP8 + EP | 8k / 1k, 512 | TPOT 111 ms, 29,351 tok/s |
| DeepSeek-V4-Pro | v7x-16, DP16 + EP | 8k / 1k, 1,024 | TPOT 215 ms, 38,709 tok/s |
| Kimi-K3 | v7x-32, TP32 + EP | 8k / 1k, 1 | TPOT 19.2 ms, TTFT 1,573 ms |

These are eight v7x chips (or more) against one GB10; they are not comparable with Tlaloc's
numbers and are listed as the targets a TPU run would be measured against.

## What applies to Tlaloc, ranked

### Tier 1: applies now, on the GB10

1. **Router top-k without a sort in the XLA form of MOE_EXPERTS** (K rounds of max and mask).
   Small, testable against the interpreter, and the TPU path needs it.
2. **A ragged mixed-batch attention entry** (decode, verify and prefill tokens in one call,
   keyed by total tokens). The largest item here; measured afterwards, not what Tlaloc's
   TTFT waits for.
3. **Prefix-aware admission** (fewest uncached tokens first) on the new prefix cache.

### Tier 2: needed for the TPU bring-up (G2b)

4. **A TPU sampler form** (Gumbel-max, top-p / top-k by bit-pattern binary search) so
   SAMPLE_TOKENS lowers without the CUDA kernel.
5. **The TPU MoE choice measured:** ragged dot against a Mosaic grouped matmul; one-hot
   permute for small row counts; W4A8 for NVFP4 weights.
6. **A Mosaic-friendly KV layout** (K/V interleaved per head, packed to 32-bit words, head dim
   padded to 128) for a TPU attention kernel.

### Tier 3: needed for several devices (G4)

7. **`ALL_GATHER` and `REDUCE_SCATTER` as real ops**, MOE_EXPERTS with a traced expert offset,
   and the all-gather / local experts / reduce-scatter EP form.
8. **DP lockstep in the Triton backend** (agree on the bucket, pad, dummy steps).
9. **Decode context parallel** as all-gather of the LSE + reduce-scatter (the attention kernel
   would return the LSE, as theirs does).
10. **Peer replay of compiled executables** across a host's ranks.

### Not applicable to one GB10, or already covered

- Pipeline parallel, prefill context parallel (ring attention), intra-chip process groups,
  SparseCore offload, multi-host rendezvous, KV offload and P/D transfer: need several
  devices or TPU hardware.
- Async scheduling with device-side token substitution, the fused MTP loop, the persistent
  compile cache: Tlaloc has them (chained steps, the head in the verify program, the
  executable cache).
- A unified KV / state pool and KV sizing at load: Tlaloc's separate state pools and
  export-time pool shapes already cover what they solve.

## Doc / code differences found

- The docs describe DP + EP MoE as all-to-all; the default code all-gathers and
  reduce-scatters.
- The blog and README name `VLLM_XLA_CACHE_PATH`; the compile-cache doc names
  `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`.
- TorchTPU itself (how `tpu_dist` collectives lower, whether XLA overlaps them) was not read;
  it is a binary dependency of this repo.
