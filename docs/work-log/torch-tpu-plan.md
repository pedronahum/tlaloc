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

