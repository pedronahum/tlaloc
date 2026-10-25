# Measuring and profiling

How to time a Tlaloc program, see what XLA compiled, and find where a step's time
goes. The numbers below were measured on the GB10 (DGX Spark); each names its source.

## Timing a step

`StepBenchmark` (`:runtime-pjrt`) times a repeated step:

- the first step is timed on its own, because it compiles;
- warm-up runs until the compile count is the same on two consecutive steps (at least
  `minWarmup`, at most `maxWarmup` steps), and fails if compiling never stops;
- each of `timedSteps` steps is synchronized and timed alone; a compile among them
  throws `RecompileDuringTimingException`, because the number would include it;
- the result has the first-step time, the warm-up overhead, the median, the 10th and
  90th percentiles, the mean, the minimum, the maximum and the device's peak bytes.

<!-- doc-test: perf-step-benchmark -->

```kotlin
val bench = StepBenchmark(minWarmup = 10, maxWarmup = 40, timedSteps = 50)
val result = bench.run(
    compileCount = { session.cacheSize },          // programs compiled so far
    peakBytes = { session.memoryStats().second },  // the device's peak, when it reports one
) { session.runOn(step.gradient, inputs) }
println(result.toJson("mlp_gradient"))             // one JSON line, for a results file
```

It measures latency: a step starts after the previous one's result is on the host.

### On the GB10

- An idle GB10 runs its clocks at a few hundred MHz. Run the step for two seconds
  before measuring, or the first timed steps are slower than the rest.
- Millisecond steps move by about 20 % between runs (host dispatch), so a single run
  says little. Compare medians over several runs, or ratios within one session.
- A process holding the GPU slows every run: with a desktop keeping it 16–22 % busy,
  serving runs were 27–42 % slower (`docs/SERVING_ARCHITECTURE.md`).

### The benchmark registry

`scripts/bench-registry.sh` times the registered steps and compares each median with
its history:

```bash
scripts/bench-registry.sh                      # measure, compare, record
scripts/bench-registry.sh --no-record          # compare only
scripts/bench-registry.sh --out results.jsonl  # also keep this run's results
scripts/bench-registry.sh --perturb            # 3 ms added to every step: every case must be flagged
```

The cases are in `runtime-pjrt/src/jvmTest/kotlin/io/tlaloc/runtime/pjrt/BenchRegistry.kt`;
each is measured, run once in the suite, or skipped, with the reason. A case is
flagged when its median exceeds `max(1.25 m, m + 4 · 1.4826 · MAD)` of the last ten
recorded medians `m` (`1.5 m` with fewer than three), and never for less than 0.1 ms.
The history is `~/.local/share/tlaloc/device-lane/bench-history.jsonl`.
`scripts/device-lane.sh` runs it after the suites, the examples and the Triton checks.

## Seeing what was compiled

| To see | Use |
|---|---|
| the gradient the compiler wrote, as Kotlin | `tlaloc { dumpGradSource.set(true) }` ([READABLE_REVERSE.md](READABLE_REVERSE.md)), or `step.gradSource()` |
| the StableHLO a program becomes | `fn.toStablehlo()` (`:stablehlo`) |
| the HLO XLA ran, before and after its passes | `XLA_FLAGS="--xla_dump_to=DIR --xla_dump_hlo_as_text"` |
| how many programs a session compiled | `PjrtSession.cacheSize` |
| the persistent cache's hits and misses | `PjrtExecutableCache.hitCount`, `missCount`, `rejectedCount` |

The session's executables are keyed by the program's StableHLO text, so a new shape
or a changed constant is a new compile. `PjrtSession.prepare(fn)` compiles ahead of a
timed loop.

`TLALOC_PJRT_CACHE_DIR` keeps compiled executables on disk. The key covers the plugin
(path, size, modification time), the platform and its version, the device kind,
`XLA_FLAGS`, the compile options and the program. Loading the 36 programs of
Qwen3.6-35B-A3B took 9.8 s instead of 520 s compiling them
(`docs/work-log/torch-tpu-plan.md`). Nothing evicts entries.

## Profiling a served model

`triton/profile.sh MODEL_DIR MODEL_NAME WORKLOAD...` starts the Triton server, waits
until the GPU is idle (it labels the run "contended" when it never is), and runs the
workloads: `decode:PREFIX:STEPS[:STREAMS]`, `prefill:PREFIX:CHUNKS`,
`prompts:TOKENS:ROUNDS:STREAMS`.

- `MODE=time` (the default) times `REPEAT` runs and reads the server's own queue and
  compute times from Triton's statistics.
- `MODE=nsys` runs the server under Nsight Systems, captures the first repeat, and
  `triton/profile_report.py` cuts the capture into executions, with each one's
  kernels, kernel time and the host gap before it. With `XLA_DUMP=1` it ties each
  kernel to its HLO instruction and classes it (weights, KV write, KV gather,
  attention), with the bandwidth each class implies.
- `TLALOC_TIMELINE=1` (`triton/run_server.sh`) logs every backend call with its rows,
  their arrival times, and when the call ran and finished.

> [!TIP]
> Two things to know about the server's logs: XLA ends some lines with a NUL byte, so
> `grep` needs `-a`; and `--log-verbose=1` breaks the paged attention custom call.

## Where the time goes

What made steps slow before, measured:

| Cause | Measured | Remedy |
|---|---|---|
| Parameters and gradients cross the host every step | a 2-layer causal LM step: 25.78 ms with AdamW on the host, 2.22 ms with `withDeviceAdamW` on the device | keep state on the device, donate it ([the case study](case-studies/training-step-on-device.md)) |
| A compile per input length | 200 requests of 163 lengths: 163 compiles in 264 s; with bounded buckets, 6 in 7.6 s (execution 13 % slower from padding) | [bounded dimensions](DYNAMIC_SHAPES.md) |
| Compiling programs in parallel | 36 programs, 19 at once: 1,271 s, against 520 s one at a time (XLA already compiles in parallel inside a program, and autotuning shares the GPU) | `TLALOC_COMPILE_THREADS=1`, the default |
| Dispatch | 40–416 µs per execution across sessions on one machine; 40–70 % of a small measurement | open; to measure past it, compare floors within one session |
| Host time per execution with many buffers | about 3.3 ms for the ~1,300 arguments of a 35B entry | chained steps hide it |
| A `while` loop's condition read back per iteration | about 0.6 ms per block, 208 blocks per call at 31,744 positions | open |
| `precision = HIGHEST` on a dot | a 27B decode attention layer at 32K: 13.1 ms (a SIMT kernel), 5.2 ms with the f32 dot algorithm, 2e-8 apart | the dot algorithm attribute |
| XLA merging differently typed weights | a bf16 concatenation per call, 53 ms of a 234 ms step | stage the small weights in f32 |

Sources: `examples/training-step-on-device`, `docs/work-log/torch-tpu-plan.md`, `docs/work-log/bounded-dims-progress.md`,
`docs/design/KPTX_PAGED_PERF.md`, `docs/FOLLOWUPS.md`, `docs/SERVING_ARCHITECTURE.md`,
`docs/work-log/qwen35-progress.md`.

On the GB10, a plain f32 `MATMUL` runs as TF32, and XLA folds an f32 → bf16 → f32
convert pair; both change numbers, not only time. [NUMERICAL_PRECISION.md](NUMERICAL_PRECISION.md)
has what each backend computes.

### Ceilings

Measured on the GB10: a kernel streams about 233 GB/s against 273 GB/s nominal; cuBLAS
reaches about 70 TFLOPS in bf16, 33 in TF32 and 19 in f32. Decode of a large model is
bound by reading its weights: the 53 GB Muse Glimmer decoder took 243 ms of kernels a
step against a 228 ms floor (`docs/SERVING_ARCHITECTURE.md`).
