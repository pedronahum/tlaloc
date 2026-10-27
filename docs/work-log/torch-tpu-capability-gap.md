# TorchTPU against Tlaloc on TPU: capabilities and the plan to pass it

Sources:

- TorchTPU: [google-pytorch/torch_tpu](https://github.com/google-pytorch/torch_tpu).
  - Release `v0.2.0` (tag at `005903d0`, 2026-10-17; open-sourced at PyTorch Conference on 2026-10-20).
  - Read at HEAD `36329316` (2026-10-24, pyproject `0.2.1`). It was read only, never built or run.
- Tlaloc: `main` at `0d8aa3d9`.

Earlier notes, which this one does not repeat:

- [torch-tpu-lessons.md](torch-tpu-lessons.md): internals and the ranked lessons. The results are in [torch-tpu-plan.md](torch-tpu-plan.md).
- [torch-tpu-testing-assessment.md](torch-tpu-testing-assessment.md): tests.
- [vllm-torchtpu-lessons.md](vllm-torchtpu-lessons.md): serving.

## 1. Summary

- **Tlaloc has never run a program on a TPU.**
  - It has also never run on more than one real device. The only multi-device runs are on 4 virtual XLA:CPU devices.
  - Every Tlaloc TPU item is written and pinned on the GB10 or the CPU host, and marked unverified on TPU.
  - So today TorchTPU can do everything on a TPU, and Tlaloc nothing that has been shown.
- **Most of the gap is unbuilt work, not a structural limit.** A Tlaloc program is one StableHLO module, compiled whole, with no Python at run time. That is the form XLA:TPU compiles best. TorchTPU spends much of its code getting PyTorch eager code into that form:
  - deferred ops;
  - defer-and-fuse;
  - graph breaks around collectives;
  - SPMD handshakes;
  - functionalization by view inversion.
- **The structural gaps are ecosystem and scale, not compiler.** They are:
  - PyTorch's user base;
  - about 755 ATen overloads against Tlaloc's 115 `OpKind`s;
  - 178 reused upstream test modules;
  - Hugging Face, TorchTitan, vLLM and SGLang;
  - Google's TPU CI (v5e-8 and v7x-4 presubmit, every 4 h on v5/v6/v7);
  - runs at 8K chips (28% MFU, Qwen3 235B, 8,192 v7x, per the conference talk).

  Tlaloc cannot match these by building. It can pass TorchTPU where whole-program compilation and typed Kotlin give it an advantage. Section 4 lists those areas.

## 2. What TorchTPU can do that Tlaloc cannot

Kinds of gap:

- **U**: unbuilt. Tlaloc's design allows it; nobody has built it.
- **H**: needs hardware Tlaloc has not had: a TPU, or several devices.
- **S**: structural. It follows from what Tlaloc is, and building cannot fully close it.

### 2.1 Running at all

| Capability | TorchTPU | Tlaloc | Gap | Why |
|---|---|---|---|---|
| Execute on TPU | v5e, v5p, v6e, v7x validated; v8i/v8t specs in code (`hardware.py:237-284`) | `PjrtTarget.Tpu`, libtpu 0.0.50 pin, `TpuClientOptions`, 12 TPU tests that skip, `scripts/tpu/session.sh`; never run | H | No TPU VM. The bring-up kit is packed (`build/tpu-session.tar.gz`) and has not been run. |
| Hardware detection, per-generation specs | PCI-ID detection; HBM, FLOP/s, SparseCore count, devices per chip (`hardware.py`) | `vmemLimitFor(kind)` only | U | Not needed until a TPU is available; MFU reporting needs it. |
| Megacore / 2 devices per chip (v7x) | handled; rank ≠ device id (0,1,2 → 0,1,4) | noted in MULTIHOST_DESIGN §4a; no code | U+H | — |

### 2.2 Distribution

| Capability | TorchTPU | Tlaloc | Gap | Why |
|---|---|---|---|---|
| Multi-process / multi-host runtime | `tpu_dist` c10d backend; slice-builder discovery; GKE JobSet example | `PjrtFfm.requireKvStoreForMultiNode` refuses more than one node; KV-store callbacks zeroed (G4a) | U+H | Never had a second host. |
| Multislice over DCN | validated to 8K chips; `topology_aware_mesh` (DCN outer, ICI inner) | `MEGASCALE_*` not in code | U+H | — |
| Collectives | allreduce (8 reduce ops), broadcast, all-gather, reduce-scatter, all-to-all, gather, scatter, send/recv, barrier, coalesced forms, `ragged_all_to_all` | `ALL_REDUCE` (sum only, with VJP), `ALL_GATHER`, `REDUCE_SCATTER` (no adjoints) | U | No all_to_all, collective_permute or ragged all-to-all; AG/RS have no gradients. |
| FSDP | SimpleFSDP validated; FSDPv2 experimental | G4e designed | U | — |
| Tensor parallel / DTensor | examples, DTensor tests | `sdy.mesh`, `sdy.sharding`, `manual_computation` emitted; text pins only; 12 `sdy-opt` tests skip (no binary) | U+H | Tlaloc emits the annotations; nothing has been partitioned on real devices. |
| Expert parallel | `ragged_all_to_all` + `ragged_dot` + `ragged_dot_weight_grad` | `MoeExpertsAttrs.EXPERT_PARALLEL` (slices summed on one GPU); `raggedDotMoe` | U+H | No all-to-all dispatch; no `ragged_dot` gradient. |
| Pipeline parallel | upstream `torch.distributed.pipelining` tests run on TPU | absent | U | — |
| Context parallel | doc row only (ring attention suggested as a Pallas kernel) | `ContextParallel.merge` + LSE on one device | — | Neither has run it across chips. |
| Distributed checkpointing | DCP save/load, async save | single-file checkpoints; safetensors writer not sharded | U | — |
| Fault tolerance / elastic | absent (composes with torchrun) | absent | — | Neither has it. |

### 2.3 Kernels

| Capability | TorchTPU | Tlaloc | Gap | Why |
|---|---|---|---|---|
| User kernels | Pallas through `torch.tpu.pallas.jax_op` (JAX at trace time only), Tokamax bridged, Helion experimental | `MOSAIC_KERNEL` with prebuilt payloads (5 from Pallas, 2 from Kotlin); `MosaicWriter`, `MosaicRmsNorm`, `MosaicTiledMatmul` emitters | U | No Kotlin kernel language for Mosaic. The PTX side has one (`KernelScope`); the Mosaic side has three hand-written emitters. |
| Flash attention on TPU | Kiln forward and backward (migrated after the release, `d7be622f`); splash SDPA | descriptor name `tpu_pallas_flash_attention` with no payload | U | — |
| Ragged paged attention | not in this repo (Tokamax / vLLM-TorchTPU) | absent | U | The TPU KV layout waits on it. |
| Remote-DMA kernels (in-kernel collectives) | Pallas `make_async_remote_copy`; Kiln has remote copies and semaphores | absent | U+H | — |
| Grouped matmul / MoE | `ragged_dot`, `grouped_mm`, `_scaled_grouped_mm` (FP8 rowwise Pallas) | `ragged_dot` MoE, forward only | U | — |
| FP8 matmul | `_scaled_mm`: tensorwise, rowwise, mxfp8 blockwise 1x32; Pallas 128-blockwise | FP8 KV cache and an FP8 draft head; no FP8 tensors in general | U | Tlaloc's quantized matmuls are CUDA kernels (NVFP4) or portable unpack-to-bf16. |
| SparseCore | embedding ops with fused optimizer updates; stream pinning to a SparseCore | absent | S | Niche (recommender embeddings); custom calls into libtpu internals. |

### 2.4 Execution

| Capability | TorchTPU | Tlaloc | Gap | Why |
|---|---|---|---|---|
| Eager execution on device | default mode, op by op, compiled at `-O1` | none; the JVM interpreter is the eager mode | S | Tlaloc compiles whole programs by design. The cost: no op-by-op device debugging. |
| Dynamic shapes | bounded buckets (≤ 8, 128 grid) under `torch.compile`; not with Pallas, not persisted | `BoundedProgram` (masks), `DecodeBucketPolicy`; padded `set_dimension_size` mode is a spike only | U | — |
| Compile cache | memory → `/dev/shm` with flock → GCS tier | memory → disk dir with POSIX locks (JVM and Triton, one key); no remote tier, no eviction | U | — |
| Async transfers, pinned host, batch transfer | `batch_transfer_{h2d,d2h}`, `pinned_host` memory space | `PJRT_Event` await; no raw-buffer extension, no memory kinds | U | "No consumer" (torch-tpu-plan item 15). |
| Activation offload to host | validated (`torch.ops.ao.offload`) | absent | U | — |
| Streams | experimental; TensorCore/SparseCore pinning | absent | U | Low value for whole programs. |

### 2.5 Training

| Capability | TorchTPU | Tlaloc | Gap | Why |
|---|---|---|---|---|
| Activation checkpointing with barriers | FX pass inserts variadic `optimization_barrier`s | absent: no `optimization_barrier` anywhere | U | Listed as not done in torch-tpu-plan item 8. |
| On-device optimizers | fused AdamW, Adam, SGD, Adagrad; functional AdamW, Muon, SGD; `SingleTraceTrainer` (one graph for the whole step) | `withDeviceAdamW` only (1.51 against 17.97 ms a step on the GB10); SGD, RMSprop and Adam host-side | U | — |
| Gradient accumulation | plain PyTorch | no helper | U | — |
| Chunked LM head + CE | in the 30B training study | absent | U | — |
| Large-scale recipes | TorchTitan adapters; fine-tuning "to ~8K chips" | fine-tune / LoRA examples run AdamW on the host | U+H | — |

### 2.6 Numerics

| Capability | TorchTPU | Tlaloc | Gap | Why |
|---|---|---|---|---|
| Dtypes | fp8 (e4m3fn/fnuz, e5m2/fnuz, e8m0), fp4 e2m1 storage, f16, bf16, f32, f64, complex64/128, all ints; no int4 | f32, f64, bf16, fp8 for KV, NVFP4 (CUDA kernel or unpack), int8 weights | U | — |
| Matmul precision default | `medium` = 1 bf16 pass; `high` 3; `highest` 6 | `portableF32Dots` on for TPU, HIGHEST by default | U | The fast path is one setting away. The default needs to be decided (section 3, phase 0). |
| Excess precision off | default, locked at the first compile | `STRICT_PRECISION` in the TPU test lanes only | — | Equal. |
| Determinism | no mode | deterministic serving mode (deepinfer arc) | — | Tlaloc ahead. |

### 2.7 Tooling, ecosystem, process

| Capability | TorchTPU | Tlaloc | Gap | Why |
|---|---|---|---|---|
| Profiler | `torch.profiler` with TPU activity, XProf, multi-host session id, hardware counters on v7+ | absent; `triton/profile.sh` is nsys (CUDA) | U | No PJRT profiler-extension binding. |
| Memory API | `torch.accelerator.memory_*` | `PjrtSession.memoryStats` | — | Equal in kind. |
| Model coverage | ~755 ATen overloads; "5,000 HF models" claimed (~300 in repo); TorchTitan; vLLM/SGLang (separate repos) | 115 `OpKind`s; 7 `HfModelFamily`s (Llama, Qwen3, Qwen3.5 dense and MoE, Muse Glimmer, Xing4, Qwen4Exp) | S | PyTorch's op surface is the price of running arbitrary user code. Tlaloc covers families, not arbitrary graphs. |
| Hardware CI | v5e-8, v7x-4 presubmit; v5/v6/v7 every 4 h; nightly TPU, TSan, upstream; perflab TPU against B200 | GitHub runners without accelerators; GB10 device lane by hand | H+S | — |
| Packaging | PyPI wheel, Python 3.11 to 3.14t, torch 2.11 to nightly | Maven Central alpha03 | — | Different ecosystems. |
| Agent tooling | TPU agent suite (separate repo, preview) | — | S | — |

## 3. Why the gap exists, in one line each

- **No TPU access.** Every H row, and every row marked unverified on TPU, waits on one Cloud TPU VM.
- **Tlaloc built serving on CUDA first.**
  - The Triton backend links `cuda_runtime_api.h` and `cudaMalloc`s weights outside the PJRT pool.
  - NVFP4 MoE always emits the CUDA kernel.
  - The 35B and Flash-Next served configurations depend on both.
  - The portable forms (sort-free router, StableHLO sampler, `raggedDotMoe`, the ctypes runtime) exist, but have not run on a TPU.
- **Distribution was designed, not built** (MULTIHOST_DESIGN §6, G4a–G4e). The first missing piece is the PJRT KV store.
- **TorchTPU has Google's kernel authors and Tokamax.** Tlaloc's Mosaic side is three emitters and seven payloads.

## 4. Where Tlaloc can be better than TorchTPU

These follow from TorchTPU's own code and stated limitations.

1. **Sharded whole programs, compiled once (SPMD by construction).**
   - TorchTPU emits no sharding: "partitioning will be a no-op" (`compilation.cc:364-369`).
   - Its model is one process per device, with explicit collectives.
   - Collectives break the graph by default ("skinny"). Fusing across them needs `spmd_safe` or a cross-rank handshake.
   - MPMD and FSDPv2 are marked "low perf".
   - Tlaloc already emits Shardy annotations into one program that is the same text on every rank. Shardy and XLA:TPU's SPMD partitioner can then place the collectives and overlap them, as in JAX.
   - This is the largest structural advantage, and nothing has exercised it.
2. **Kernels with gradients, in Kotlin, without JAX.**
   - TorchTPU user kernels need JAX at trace time; Kiln is internal C++.
   - Tlaloc can give `KernelScope` a Mosaic back end, so one Kotlin kernel DSL targets PTX and Mosaic.
   - The coarsener's `gradient_body` already gives claimed kernels an analytic backward.
   - Bounded buckets can take kernels; TorchTPU's cannot ("no Pallas custom kernels" under dynamism).
3. **Dynamic shapes that persist.**
   - TorchTPU's bounded dynamism has no AOTAutograd persistence and is not production-ready (`dynamic_shapes.md:225-256`).
   - Tlaloc's bounded dims are typed at compile time and its executable cache is persistent: 520 s → 9.8 s on the 35B.
   - The padded `set_dimension_size` mode removes the mask operands.
4. **Serving in the framework.**
   - TorchTPU's repo has a contiguous-cache example: 23.6 ms/token at batch 1 for a 30B model on one v7x, about 32% of HBM bandwidth.
   - Tlaloc has paged KV, bucketed decode, speculative decoding, prefix cache, det mode and a framework-free runtime.
   - On a TPU, this needs the Mosaic ragged paged-attention kernel and the portable runtime.
5. **Numerics checked against mathematics.**
   - F64 oracle, central differences, readable gradient source that compiles.
   - TorchTPU checks TPU against B200 goldens.
   - On TPU, where the default matmul is one bf16 pass, an F64-referenced precision report per op is something TorchTPU does not have.
6. **No Python, no per-op dispatch.**
   - TorchTPU measures about 50 µs per op-dispatch bubble in eager, and an `O2` floor of about 0.6 ms.
   - A Tlaloc decode step is one executable call from the JVM or from the ctypes driver.

## 5. Plan

Phases are in dependency order. "TPU" marks an item that needs the VM.

### Phase 0: first TPU run (TPU, single host, v6e-1 or v5e-1 is enough)

1. Run `scripts/tpu/session.sh`, lanes g2b, kernels, training, serving, finetune and suite. Record every item in TPU_BRINGUP "What the first TPU run must record":
   - `premapped_buffer_size`;
   - convert pairs under `STRICT_PRECISION`;
   - `ragged_dot` going to the grouped matmul;
   - serialized-text Mosaic bodies and `MosaicTiledMatmul`;
   - threefry;
   - tiled bf16 sizes;
   - donation and memory kinds;
   - the sampler at a 248K vocabulary;
   - the sort-free router;
   - the one-hot crossover.
2. Settle the f32 dot default for a performance session: HIGHEST costs 6 passes. Keep HIGHEST for parity lanes, and make DEFAULT or HIGH the serving and training default, measured.
3. Qwen3-0.6B greedy decoding with token parity against HF, and the fine-tune lane, both on TPU.
4. Time one Muse-Glimmer-30B-class decode at batch 1 against TorchTPU's 23.6 ms/token (v7x) with the same contiguous cache. This is the first like-for-like number.

### Phase 1: single-chip parity (mostly buildable on the GB10, proven on TPU)

5. **A Mosaic back end for `KernelScope`** (grid, block specs, VMEM scratch, async copy, semaphores), checked against Pallas exports with `check_kmosaic_forms.py`.
6. **Mosaic flash attention, forward and backward**, written in that DSL and claimed through the recognizers, not only by op kind (the `TPU_MEGAKERNELS` "Not done" list).
7. **Mosaic ragged paged attention**, then the TPU KV layout (vllm-torchtpu-plan, not built).
8. **The portable serving path on TPU.**
   - The Python ctypes runtime and the vLLM plugin on a `tpu` platform.
   - Weights inside the PJRT pool.
   - No NVFP4: int8 weights or FP8.
9. **FP8 matmul as a portable op** (`dot_general` on f8 operands with scales). XLA:TPU v7 has native FP8; NVFP4 stays CUDA.
10. **Training step completion:**
    - `OPTIMIZATION_BARRIER` OpKind plus activation checkpointing;
    - chunked LM head + CE;
    - device SGD and Muon;
    - gradient accumulation inside the step;
    - fine-tune and LoRA examples moved onto `withDeviceAdamW`.
11. **Profiler:** bind the PJRT profiler extension and write XProf traces.
12. **i64 hot spots** (`NGRAM_HASH`, sampler prefix sums): s32 forms where the value range allows.

### Phase 2: multi-chip (TPU, v5e-8, v6e-8 or v7x-8, single host)

13. **G4a:** PJRT KV store callbacks (distributed runtime), then multi-device `PjrtSession` with `CompileOptions.spmd(ids)`.
14. **Collectives:**
    - `ALL_TO_ALL` and `COLLECTIVE_PERMUTE` OpKinds;
    - `RAGGED_ALL_TO_ALL`;
    - adjoints for `ALL_GATHER` and `REDUCE_SCATTER`;
    - `ragged_dot` gradient.
15. **Shardy-propagated training** (section 4.1): data parallel, FSDP as parameter sharding plus propagation, and TP. Each is measured against the same model run with explicit collectives.
16. **Expert parallel with all-to-all dispatch** for the Qwen3.5 MoE.
17. **Sharded checkpoints:** sharded safetensors, read per shard.

### Phase 3: multi-host and the "better than" claims (TPU pod slice)

18. G4b–G4e:
    - multi-process launch;
    - `MEGASCALE_*` and DCN topology;
    - a topology-aware mesh (DCN outer, ICI inner);
    - a fingerprint check of the program text before the first collective.
19. Published numbers, measured per the bench registry:
    - MFU on a dense and an MoE fine-tune against TorchTPU / TorchTitan on the same slice;
    - decode tok/s against vLLM-TorchTPU on the same model and chip;
    - compile and cold-start time with the persistent cache.
20. A scheduled TPU lane: nightly on a reserved or spot v5e-8, running the device suite and the bench registry. TorchTPU tests on hardware every 4 h; Tlaloc tests on hardware never.

### Not planned

- Eager device mode.
- SparseCore.
- ATen-scale op breadth.
- PyTorch interop.

These are TorchTPU's ground; matching them does not make Tlaloc better.

## 6. Tlaloc docs that disagree with the code

Fixed on `feat/tpu-gap` ([tpu-gap-plan.md](tpu-gap-plan.md), item 16).

- `docs/audits/TPU_READINESS_AUDIT.md` (2026-09-21) predates the TorchTPU lessons work.
- `TPU_MEGAKERNELS.md` says "no grid"; `MosaicTiledMatmul` has one.
- `DISTRIBUTED.md` says the collectives are differentiated; `ALL_GATHER` and `REDUCE_SCATTER` have no adjoints.
- `vllm-torchtpu-lessons.md` calls AG/RS verification-only; that arc built them.
- `TPU_BRINGUP.md` counts 5 skipped TPU tests; `CAPABILITIES.md` counts 12 (5 smoke + 7 Mosaic).
