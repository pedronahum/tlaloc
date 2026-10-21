# TorchTPU lessons: plan

Branch `feat/torch-tpu-lessons`, from `feat/flash-next` at `7d409556` (2026-10-20). The
source of every item is [torch-tpu-lessons.md](torch-tpu-lessons.md).

Items that only a TPU can prove are built with local tests and marked **unverified on
TPU**.

| # | Item | Proof |
|---|---|---|
| 1 | `CompileOptions` builder: multi-device fields (partitions, device assignment, SPMD, Shardy), `env_option_overrides`, optimization level; the 6 bytes stay the default | byte pins; an override observed on GB10 XLA |
| 2 | Persistent executable cache (`PJRT_Executable_Serialize` / `DeserializeAndLoad`), JVM and Triton backend | GB10: second load skips compile, same outputs |
| 3 | Triton backend compiles buckets in parallel at load | GB10 load time before and after |
| 4 | TPU client option `premapped_buffer_size` | marshalling pins; **unverified on TPU** |
| 5 | `all_reduce` with `channel_handle` and `use_global_device_ids`; notes on broadcast and device ids | emission pins; GB10 compiles a one-device program |
| 6 | Precision policy for f32 dots on TPU (parity vs performance) and `xla_allow_excess_precision=false` for parity lanes | unit pins; **unverified on TPU** |
| 7 | Argument checks before execute (dtype, dims) | refusal tests |
| 8 | Training step on the device: AdamW in the graph, donated parameters and moments, scalar step and learning rate | GB10: equal to the host-AdamW loop |
| 9 | DXIR rewrites: partitioned softmax over a small concat tail, redundant max-subtract, `x·(1+w)` | interpreter parity per rewrite |
| 10 | Benchmark discipline (warm until compiles stop, recompile in timed region fails, separate metrics) | used on a real run |
| 11 | MoE as `chlo.ragged_dot` for TPU (tokens padded to 256) | GB10 parity against the existing forms; **unverified on TPU** |
| 12 | Mosaic: `scoped_memory_configs`, grid / `window_params` in the Kotlin Mosaic emitter | text pins against Pallas exports; **unverified on TPU** |
| 13 | Bounded dims padded mode (`set_dimension_size`) | GB10 spike over the bounded fixtures |
| 14 | Repeated layers compiled once (`func.call`) | measured compile share first |
| 15 | Raw-buffer batched transfers | decide: build only with a consumer |

## Results (2026-10-21)

Measured on the GB10 (XLA:GPU) unless marked otherwise.

| # | Item | Status | Result |
|---|---|---|---|
| 1 | `CompileOptions` | done | `CompileOptions` (runtime-pjrt `ffm`): partitions, device assignment, SPMD, Shardy, optimization level, `env_option_overrides`; `DEFAULT` is the six bytes; `spmd(ids)`; `STRICT_PRECISION`. Byte pins; XLA:GPU compiles `spmd([0])` and `O1`; `xla_allow_excess_precision = false` stops XLA folding an f32→bf16→f32 pair (the bf16 smoke test's finding), so overrides reach XLA. |
| 2 | Persistent executable cache | done | JVM `PjrtExecutableCache` (`TLALOC_PJRT_CACHE_DIR`, `PjrtSession(diskCache = ...)`) and Triton `executable_cache.cc` (`executable-cache-dir`), one key (pinned golden, same hex on both sides), one file format, POSIX record locks both sides. Qwen3.6-35B-A3B: 36 programs 520 s compiled → 9.8 s loaded; model ready 564 s → 57 s; decode 53.7 / 53.2 ms (256 / 8,000 context) compiled, 54.1 / 52.8 ms loaded. 475 MB on disk. |
| 3 | Parallel compile at load | done, default off | `PjrtClient::CompileAll` with `compile-threads`. 19 at once: 1,271 s for the 36 programs, each ~300 s instead of ~13 s. XLA:GPU compiles in parallel inside a program and autotuning shares the GPU, so the default is 1. Decode unchanged (52.0 / 52.8 ms). Progress is logged per program. |
| 4 | TPU `premapped_buffer_size` | done, **unverified on TPU** | `TpuClientOptions` (0 unless `TPU_PREMAPPED_BUFFER_SIZE`); marshalling pinned. |
| 5 | `all_reduce` form | done | `channel_handle` + `use_global_device_ids`; runs on one device under both compile forms. Broadcast and device-id notes in MULTIHOST_DESIGN §4a. Multi-device execution not available here. |
| 6 | Precision policy | done, **unverified on TPU** | `F32DotPrecision` (HIGHEST / HIGH / DEFAULT) for `portableF32Dots` (`PjrtSession.portableF32DotPrecision`, HIGHEST by default); TPU test lanes compile with `STRICT_PRECISION`, `TestBackend.foldsConvertPairs` false there. |
| 7 | Argument checks | done | `PjrtSession.executeOn` checks each buffer's PJRT type and dims against the parameter (`validateArguments`, on by default); a [2, 3] buffer for a [3, 2] parameter is refused by name, and runs with the check off. |
| 8 | Training step on the device | done (AdamW) | `CapturedStep.withDeviceAdamW(AdamW)`: one program for loss, gradients and the update; parameters and moments donated; learning rate and bias corrections as scalar operands. Interpreted, it equals the host AdamW loop bit for bit (3 steps). On the GB10 (CausalLM, 2 layers, d 128): losses within 5e-3 of the host loop over 20 steps, parameters within 2.3e-4; **1.51 ms a step against 17.97 ms** (median of 20, `StepBenchmark`). Not done: activation checkpointing with optimization barriers, the chunked LM head + cross-entropy, and the examples (pinned to the 0.1.0-alpha02 artifacts) using it. |
| 9 | StableHLO rewrites | not built | None of the four matches a Tlaloc graph: the `(1 + w)` gains add in f32 on an `[n]` vector before the multiply (HF's arithmetic, nothing to gain); no decoder graph has attention sinks; the one explicit max-subtract (Sinkhorn) feeds no softmax op; no separable gathers. |
| 10 | Benchmark discipline | done | `StepBenchmark`: warm until the compile count is stable (10–20), a compile in the timed steps throws, median / p10 / p90 / mean / min / max, first step and warm-up overhead, device peak bytes (`PjrtSession.memoryStats`). Used for item 8. |
| 11 | MoE as `chlo.ragged_dot` | done, **unverified on TPU** | `lowering = "ragged_dot"` on MOE_EXPERTS (`raggedDotMoe`, `PjrtSession.moeRaggedDot`, on for a TPU): pairs sorted by expert, rows padded to 256, two ragged dots, scatter-add. On XLA:GPU it matches the interpreter within 7.5e-7 (f32, bf16, sigmoid routing). |
| 12 | Mosaic grid and serialized text | done, **unverified on TPU** | `MosaicWriter` (generic form, `stable_mosaic` serialized form), `MosaicTiledMatmul` (grid, `window_params`, index maps, VMEM scratch): equal to Pallas's module after `cse` (29 ops) and, serialized, to `mosaic-serde`'s output at version 9 (`harness/python/check_kmosaic_forms.py`). `MosaicKernel.vmemLimitBytes` writes `scoped_memory_configs`; `vmemLimitFor(kind)` is TorchTPU's half-of-VMEM table; `MosaicKernel.ofSerializedText` makes a body without jaxlib. |
| 13 | Bounded dims padded mode | spike | `set_dimension_size` over a padded buffer and an i32 size: sum, max, elementwise + reduce, dot and softmax honour the size on XLA:GPU with garbage in the padding (`PjrtPaddedBoundSpikeTest`). Not wired into `BoundedProgram`; dynamic outputs untested. |
| 14 | Repeated layers compiled once | not built | XLA:GPU inlines calls, so `func.call` saves only Tlaloc's emission; compile time needs a `while` over stacked weights, a layout change to every decoder graph. Item 2 removes the compile from every load after the first. |
| 15 | Raw-buffer batched transfers | not built | No consumer: nothing moves many same-shaped buffers between host and device per step. |
