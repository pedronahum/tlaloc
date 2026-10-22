# TorchTPU: what Tlaloc can take from it

Source: [google-pytorch/torch_tpu](https://github.com/google-pytorch/torch_tpu) at `418c9810`
(2026-10-20), read only (not built or run). Compared against Tlaloc `feat/flash-next` at
`7d409556`.

[TPU_READINESS_AUDIT.md](../TPU_READINESS_AUDIT.md) compared the two from the April
announcement and the conference talks. This note works from the released code. It corrects
some of the audit's statements and answers several of the open questions in
[TPU_BRINGUP.md](../TPU_BRINGUP.md).

## What TorchTPU is, from the code

TorchTPU is a PyTorch backend for TPUs: about 470K lines of C++ and Python, built with
Bazel, on PJRT through `xla::GetCApiClient("tpu", ...)`.

- **Every ATen op becomes a deferred op.** A deferred op is an op enum plus a closure that
  emits StableHLO on demand, holding its input buffer refs, parameter keys and output shapes
  (`csrc/eager/device_buffer.h:415-531`). The eager mode decides when the queue becomes a
  program:
  - **Strict eager (default):** each op is materialized at once as its own executable,
    compiled at XLA `O1` with autofdo off. Views are never materialized alone, so they
    always fuse into their consumer.
  - **Defer-and-fuse:** the stream queue builds up until a sync or a host read, or until
    the queue's tail repeats a 2–128-op pattern twice. From then on every repetition is cut
    into one cached executable (`events_queue.cc:289-333`). The queue holds weak pointers,
    so ops whose results nobody holds are dropped as dead code.
- **`torch.compile` does not use torch-mlir or Inductor.** The FX graph is replayed through
  the same eager kernels in a tracing mode, which yields one StableHLO module compiled at
  `O2` (`compile/compiler.py:370-494`, `export/export.py:431-620`).
  - The graph is always split at CPU/TPU boundaries and around collectives
    (`split_compiler.py`).
  - Collectives stay inside the graph only when every rank proves it traced the same graph
    (a fingerprint exchange).
- **The cache key is computed without building the MLIR.** It combines:
  - argument dims, dtypes and layouts;
  - each op's enum, parameter keys and input indices;
  - the donated-input set;
  - a fingerprint of the compile options, including the raw `XLA_FLAGS` and
    `LIBTPU_INIT_ARGS` strings.
  - (`traversal.cc:289-418`, `compilation.cc:465-501`)

  The module is built only on a miss.
- **Three cache tiers** (`compilation_cache.cc`, `tier2_compilation_cache.cc`):
  - an in-process map of futures, so concurrent requests share one compile;
  - serialized executables in `/dev/shm`, with a `flock` per key so only one process
    compiles;
  - a remote filesystem or GCS tier, raced against a local backup compile.
- **Donation for in-place ops.** The input is donated when nothing else holds it
  (`live_data_ptrs == 1`). The argument carries `jax.buffer_donor = true`, and the
  donation set is part of the key (`op_dispatcher.cc:243-294`).
- **Scalars become device operands**, except 0, 1, −1 and 2, which are baked in so the
  compiler can simplify them. A learning rate therefore never forces a recompile
  (`cache_key.h:97-210`).
- **Bounded dynamism exists, but only under `torch.compile`.**
  - Up to 8 buckets, aligned to 128.
  - Bounds are `128k − c`, so that *outputs* (for example a KV cache after `cat`) land on a
    multiple of 128.
  - The input is padded by a small separate program; inside, `set_dimension_size` is
    applied, or the argument is typed `#stablehlo.bounds`.
  - Padded regions are reset by XLA's dynamic padder, and outputs keep their padding at the
    tail, so returning one only changes the shape metadata.
  - The next bucket is compiled ahead, asynchronously.
- **Custom kernels** are `tpu_custom_call`s carrying a Mosaic module (details below).
  - The kernels that ship are flash attention (C++ MLIR builders) and `nonzero` (Kiln, a
    C++ DSL shaped like Pallas).
  - The FP8 `scaled_mm` and grouped-mm Pallas kernels are generated at build time, but the
    C++ ops in this snapshot lower to StableHLO instead.
  - MoE uses `chlo.ragged_dot`, which XLA:TPU lowers to its internal grouped-matmul
    kernel.
  - There is no paged or ragged attention; users are pointed to Tokamax.
- **Its performance case studies are thin.**
  - The 30B decode is batch 1, greedy, in a Python loop on one v7x chip: 23.6 ms per token,
    roughly 32% of memory bandwidth by my arithmetic.
  - Training is about 2,600 tok/s per TensorCore, roughly 18% MFU by my arithmetic. No MFU
    figure appears anywhere in the repo.
  - Only the FLUX study measures each optimization separately: 1.46× total, of which Splash
    attention gives 1.25× over `torch.compile`, and only with 2,048-token blocks (512 is 10%
    slower).

## Answers to Tlaloc's open TPU questions

1. **Client create options.**
   - libtpu accepts `premapped_buffer_size` (int64). TorchTPU always passes it, forcing 0
     when `TPU_PREMAPPED_BUFFER_SIZE` is unset, so libtpu does not reserve its default
     4 GiB (`pjrt_client.cc:31-42`, `pjrt_state.cc:264-270`).
   - Tlaloc passes `create_options = NULL` for TPU, so it gets the 4 GiB.
   - `PjrtTarget.Tpu` should be allowed this one option, and keep refusing the GPU
     allocator options.
2. **Compile options for more than one device.**
   - TorchTPU uses `num_replicas = 1`, `num_partitions = N`, an explicit device assignment,
     and `use_spmd_partitioning` plus `use_shardy_partitioner` set to true, even when the
     module carries no sharding (`compilation.cc:275-312`). It needs them for Mosaic
     remote-DMA kernels and for `ragged_dot`.
   - Tlaloc's 6-byte `CompileOptionsProto` is single-device and must grow these fields for
     G4.
3. **Collectives.**
   - TorchTPU's `stablehlo.all_reduce` carries `channel_handle = {1, 0}` and
     `use_global_device_ids`, and its `replica_groups` lists every subgroup, including ones
     the rank is not in (`allreduce.cc:52-67`). Tlaloc emits no `channel_handle`.
   - `stablehlo.collective_broadcast` is not implemented on TPU; broadcast is an all-reduce
     of zeros plus the data.
   - Device ids are not ranks. On v7x, ranks 0, 1, 2 are devices 0, 1, 4. Order ranks by
     physical topology.
4. **Precision.**
   - f32 `dot_general` with DEFAULT precision is one bf16 pass on TPU. HIGH is 3 passes,
     about 3× the MXU cost; HIGHEST is 6 passes, about 6×.
   - `portableF32Dots` makes every f32 dot HIGHEST on TPU. That is right for the parity
     lanes and costs 6× in a performance lane. The bring-up should measure both, and the
     serving and training lanes need a stated policy: bf16 compute, or HIGH where f32 is
     wanted.
5. **Excess precision.**
   - TorchTPU sets `xla_allow_excess_precision = false` by default, locked at the first
     compile, so fused bf16 graphs round every intermediate as eager mode does.
   - This is the TPU switch for the two XLA-CUDA findings recorded in the audit: XLA folds a
     convert pair, and it rounds a bf16 dot at the op boundary. Set it for the parity lanes
     so the interpreter's round-each-op rule holds.
6. **64-bit integers are emulated on TPU** as pairs of u32, and there is no native s64
   division.
   - Loop counters must be s32, or XLA cannot see the trip count (`scan_builder.cc:98-99`).
     Tlaloc's loops already use i32 or the index operand's dtype.
   - `NGRAM_HASH` is i64 throughout (64-bit products, XORs, a remainder by a prime), so it
     would be slow on TPU. That matters only if Flash-Next is ever served there.
   - In `examples/muse_glimmer_30b` the ring index avoids `%` for the same reason.
7. **What a Mosaic payload must be.**
   - The module must go through `tpu::createMosaicSerdePass({serialize = true, target_version = 13})`.
     After that pass, **MLIR text is accepted as well as bytecode**
     (`precompiled_kernels/mosaic_custom_call.h:30-33`).
   - The JSON also carries `scoped_memory_configs` (memory space 1 is VMEM, sized here to
     50% of capacity: 128 MiB on v5e/v6e, 64 MiB on v5p/v7).
   - The calling convention, from Kiln's `kernel.cc:183-360`:
     - arguments in order: grid indices (i32), scalar-prefetch SMEM refs, input blocks,
       output blocks, scratch;
     - function attributes: `iteration_bounds`, `dimension_semantics`, `window_params`
       (each an affine map or an index-map function, plus window bounds) and
       `scratch_operands`.
   - That is the specification for the G5 gap "no grid or pipelining".
8. **Paged attention on TPU** has no reference here. Serving on TPU will need a Mosaic
   ragged paged-attention kernel. Otherwise, start with a contiguous cache per slot, as the
   case study does.
9. **MoE on TPU.** Emit `chlo.ragged_dot` and pad the token dimension to a multiple of 256.
   Otherwise XLA falls back to an expander that TorchTPU records as giving wrong results
   (`grouped_mm_aten_kernels.cc:101-202`).

## What applies to Tlaloc, ranked

### Tier 1

1. **A persistent executable cache.**
   - Tlaloc keeps executables only in memory: `PjrtSession` is keyed by StableHLO text, and
     the Triton backend compiles every bucket serially at load (`tlaloc_backend.cc:1013-1035`).
     Recorded costs:
     - 16 compiles take 2 minutes when Muse Glimmer loads;
     - the LoRA step compiles in 20.4 s;
     - small programs compile in 1.3–1.6 s, mostly XLA autotuning the dots.
   - The PJRT C API has `PJRT_Executable_Serialize` and `PJRT_Executable_DeserializeAndLoad`.
     Copy TorchTPU's key: the StableHLO fingerprint, compile options, the raw `XLA_FLAGS`,
     the plugin path and version, and the device kind. Use a directory with a `flock` per
     key.
   - A loaded executable keeps the autotuning choices it was compiled with. That should
     also remove one of the two causes of the 35B's run-to-run id differences (the other is
     the atomics in attention).
   - Works on CUDA today, and it is the TPU compile-time answer too.
2. **Compile the buckets in parallel at load,** with futures so concurrent requests for one
   key share a compile. PJRT compile is thread-safe, and TorchTPU sizes its pool at
   `(cores − 1) × hyperthreads`. On the GB10's 20 cores this should cut load time several
   times over, before any persistent cache.
3. **Run the whole training step on the device.**
   - Tlaloc's fine-tune and LoRA examples upload weights or adapters every step, read the
     gradients back, and run AdamW in Kotlin. TorchTPU compiles the step, optimizer
     included, into one program.
   - Emit AdamW into the step graph, with parameters and moments as donated inputs that
     alias their outputs (`tf.aliasing_output`, already used for the KV pools). Promote the
     step count and learning rate to scalar operands, so a schedule never recompiles.
   - Two more recipes from the 30B training study:
     - activation checkpointing per layer, with an optimization barrier after the
       recomputed values so XLA's CSE does not merge them back into the forward;
     - the LM head fused with cross-entropy, chunked along the sequence, with the backward
       pass inside the compiled chunk, so `[B, T, V]` logits are never materialized.
   - This is the "move the training-loop helper into `:runtime-pjrt`" FOLLOWUPS item, made
     larger.

### Tier 2

4. **Padded mode for bounded dims.**
   - The bounded-dims spike found that a `#stablehlo.bounds` parameter compiles, but the
     PJRT C API cannot create a dynamic buffer. TorchTPU's alternative avoids that: pass a
     static padded buffer plus an i32 size, apply `stablehlo.set_dimension_size` inside,
     and let XLA's dynamic padder neutralize the tail.
   - That would let `BoundedProgram` drop the valid-mask operands, which is the FOLLOWUPS
     §3 item "set_dimension_size not pursued". The spike showed `set_dimension_size` works
     on GPU for one reduction; the full op set needs checking.
   - It does not save FLOPs: work still scales with the bound.
5. **Compile repeated layers once.**
   - Emit one private `func.func` per distinct layer body, called at each site, and cache
     bodies by their text with locations stripped.
   - On TPU, `xla_tpu_enable_deduplicated_calls` keeps the calls deduplicated. XLA:GPU
     inlines calls, so on CUDA this saves only Tlaloc's emission time.
   - Saving XLA compile time on GPU needs a `stablehlo.while` over stacked layer weights.
     That costs the weight layout, so measure the compile share first.
6. **StableHLO rewrites before PJRT** (`csrc/mlir/passes`):
   - softmax over `concat(scores, small tail)` is partitioned into two reductions, so the
     concatenation is never materialized (attention sinks);
   - a user-written `x − max(x)` before softmax is dropped;
   - `x · (1 + w)` norms are rewritten as `x + x·w`;
   - a separable multi-axis gather becomes one-hot `dot_general`s.

   Each is a small pattern on Tlaloc's DXIR before emission. Check first which ones match
   graphs Tlaloc actually builds: the `(1 + w)` RMSNorm of the Qwen3.5 family, and sinks in
   models that have them.
7. **A benchmark discipline, per perflab** (`perflab/measure.py`):
   - warm up until the compile count is the same two steps in a row (at least 10, at most
     20);
   - a recompile inside the timed steps is an error;
   - Python's cyclic GC is off while timing;
   - check results by synchronizing on the outputs, because a failed TPU execution does
     not fail `synchronize()`;
   - report separately: first-step time, warmup overhead, steady step time, device time
     from the profiler, and peak memory;
   - subtract the profiler's own time;
   - say whether caches were cold;
   - latency runs synchronize every step, throughput runs once at the end;
   - every benchmark runs on TPU and on a B200 in the same job.

   Tlaloc should add what perflab lacks: medians and spread rather than a plain mean of 10
   steps, and MFU/MBU.

### Tier 3

8. **Raw-buffer batched transfers.** `PJRT_RawBuffer_Extension` (an alias plus a copy at a
   byte offset) moves N same-shaped tensors, or a slice of each, in one call, and a D2H
   copy can land directly in caller memory. This is useful if KV offload or state snapshots
   ever cross the host.
9. **Validate argument layouts before execute.** PJRT checks only byte size, so two buffers
   of the same size with different tilings pass silently. This matters on TPU, where tiled
   layouts are the norm.
10. **Optimization level per program.** TorchTPU measured an `O2` floor of about 0.6 ms
    forward and 2 ms train on TPU, from instruction-memory paging. Decode-sized programs
    should be timed at `O1` and `O2` during bring-up.

### Not applicable

- Eager modes, defer-and-fuse loop detection, view decomposition and inversion, and
  mutation by storage swap. Tlaloc compiles whole programs and has no eager device mode.
- The graph splitting and the MPMD handshake. They exist because Dynamo graphs can differ
  across ranks; a Tlaloc program is the same text on every rank. A fingerprint check before
  the first multi-host run is still cheap insurance, because a mismatch hangs a TPU rather
  than failing.
- SparseCore embedding custom calls.
