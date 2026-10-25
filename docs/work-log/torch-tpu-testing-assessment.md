# TorchTPU tests and examples against Tlaloc's

Source: [google-pytorch/torch_tpu](https://github.com/google-pytorch/torch_tpu) at `3632931`
(2026-10-24), read only. Tlaloc at `165db610`. Counts are test functions as written; parametrization
multiplies TorchTPU's more than Tlaloc's.

## 1. Size

| | TorchTPU | Tlaloc |
|---|---|---|
| Test functions | about 6,000: `tests/` 4,990, colocated Python 844, C++ gtest 1,058 | 3,457 JVM (`@Test`), 101 Python `unittest`, 4 Triton C++ checkers, `verify.sh` 516 checks |
| Reused suites | 178 upstream PyTorch test modules (device-generic runner); upstream `pytorch/examples` nightly | none |
| Op surface swept | 373 ATen ops through an OpInfo harness, every dtype, base / in-place / `out=` | 115 `OpKind`s through hand-written catalogues |
| Examples | 7 (FSDP, tensor parallel, Muse Glimmer 30B training and inference, FLUX, simple) | 17 standalone builds |
| Benchmarks | perflab: hand-written models plus 263 generated HF models, forward / training / decode, TPU and B200 GPU, nightly and every 4 h, dashboards, auto-bisect | `StepBenchmark`, the `benchmarks` module (155 tests), `triton/profile.sh`, CUDA kernel benches; numbers kept in docs |
| Hardware in CI | presubmit on CPU, v5e-8, v7x-4; postsubmit every 4 h on v5/v6/v7; nightly TPU, TSan, upstream | none: GitHub runners without a GPU, PJRT plugin, oracle venv or `stablehlo-translate` |

## 2. Area by area

| Area | TorchTPU | Tlaloc | Verdict |
|---|---|---|---|
| Per-op forward values | OpInfo sweep against recorded B200 goldens, per-op per-dtype tolerance tables | interpreter (F32 and F64) and device against the interpreter, per test | TorchTPU broader and systematic |
| Gradients | `ops_test_grad_tpu_vs_gpu`: TPU grads against GPU grads | central differences (35 files), F64 oracle, JVP against VJP, vmap against loops, gradient source printed and recompiled bit-identical, JAX / PyTorch parity | Tlaloc deeper: it checks against the math, TorchTPU against another implementation |
| IR / compiler | substring and op-count checks on StableHLO; MLIR pass tests checked numerically with the StableHLO reference interpreter | 150 full-text StableHLO pins, 77 compiler diagnostics, ABI dumps; 88 round trips that always skip | Tlaloc pins more of the text; TorchTPU checks pass semantics numerically |
| Compile caching / recompiles | compile counters in 37 files; bucket reuse with no recompiles | `RecompileDuringTimingException` in `StepBenchmark`; executable cache tests | TorchTPU tests recompile counts in ordinary tests |
| Dynamic shapes | 254 tests; every op swept again with a random bounded dimension | 18 `Bounded*` files | TorchTPU broader |
| Distributed | 344 tests on real multi-chip slices: collectives, FSDP, DTensor, DCP, handshakes | collectives and context parallelism on a group of one device; sharding checked in the IR | gap |
| TPU | the product | never executed; CPU-host fixture checks only | gap |
| Error behaviour | 1,180 error-message tests run on TPU and GPU; when the reference fails the TPU must too; fault injection | compiler diagnostics (77); refuse-by-name emitter pins | TorchTPU covers runtime errors, Tlaloc compile-time ones |
| Precision | `numerical_precision.md` (MXU passes, excess precision, a bit-level demo); 32 precision tests | tolerances in `TestBackend` and per test; TF32 behaviour noted in CAPABILITIES | gap in documentation and policy |
| Memory | peak-memory deltas in 13 files; ZeRO-3 retained bytes asserted | peak bytes in `StepBenchmark`; KV pool checks in `verify.sh` | gap |
| Determinism / stress | randomized test order by default, seed printed for repro, compile-cache and RNG stress tests, daily TSan | scattered deterministic-loss asserts | gap |
| Property tests | Hypothesis in 2 files | none (kotest declared in `stablehlo`, `io.kotest.property` unused) | both thin |
| Whole models | module registry: models instantiate and run; 30B inference prints text, no golden tokens | token and logit parity with HF: TinyLlama 6/6, Qwen3-0.6B, Qwen3.5-0.8B, tiny MoE / qwen4_exp fixtures | Tlaloc deeper; neither checks its largest models end to end |
| Serving | not in scope | `verify.sh`: KServe HTTP / gRPC, buckets, ragged batches, KV pools, speculative, quant; every check re-run with `--perturb`, which must fail | Tlaloc only |
| Examples | distributed-heavy, each distributed example checked against a CPU reference; README blocks run as doc-tests; case studies with methodology and numbers (FLUX doc and example numbers have drifted) | wider in kind (readable gradients, physics, GP, Java, Spark, Triton); only `quickstart` runs in CI | TorchTPU tests its examples, Tlaloc does not |
| Benchmarks | registry with a required presubmit policy and reason, storage-free smoke of every model in presubmit, typed known failures, metrics checked against the dashboard registry, history | same measurement rules in `StepBenchmark` (warmup to stable compile count, recompile is an error, p10/p50/p90, peak bytes); no registry, history or thresholds | same method, no tracking |
| Tests of the test harness | `op_testing_test`, `error_testing_test`, CI filter tests | `OracleVenvIntegrityTest`, the `--perturb` pass of `verify.sh` | comparable |

## 3. Where Tlaloc is ahead

- Gradients checked against mathematics (finite differences, F64, closed forms), not against a second
  framework's gradients.
- Full StableHLO text pins and gradient source that compiles and runs.
- Mutation-style checks: `verify.sh --perturb` and `pjrt_device_test --perturb` must fail, which
  shows the checks can fail. TorchTPU has no equivalent.
- Real-checkpoint token parity for the small models.
- Compile-time error tests (named indices, bounded dims, vmap).

## 4. Improvements, in order

### Tier 1

1. **An `OpKind` sweep.** One test iterates `OpKind.entries`. For each op a registry entry gives
   a sample generator (shapes, attrs, dtypes) and a status per surface: F32 interpreter, F64
   interpreter, VJP, JVP, vmap, emitter, renderer, host twin. Each status is either supported or
   an exclusion with a reason. The modes run as separate test classes:
   - F32 against F64;
   - VJP against central differences;
   - JVP against VJP;
   - vmap against a loop;
   - emit and parse;
   - the device against the interpreter, on the GB10.

   A new op without an entry fails the build. This replaces the catalogue in
   `GradientEmissionCoverageTest` and the KDoc exclusion lists. It is TorchTPU's OpInfo
   (`tests/op_testing.py`, `ops_test.py`) on Tlaloc's surfaces.
2. **Recorded goldens, pinned by hash.** Run the oracle venv (JAX / PyTorch) and the GB10 once.
   Write each input/output pair to `fixtures/goldens/` (or release assets) and pin it by sha256
   in a committed file. CI replays the goldens against the interpreter and emitter, so the oracle
   parity tests (`LinalgJaxParityTest`, `TransformerVsPytorchTest` and others) also run on GitHub
   runners, where today they skip. TorchTPU: `tests/goldens.pin`, `gen_gpu_golden`.
3. **A device lane that runs without being asked.** Use a self-hosted runner, or a nightly cron on
   the GB10. It runs `:runtime-pjrt:jvmTest`, the GPU examples, `triton/verify.sh` and a
   benchmark set, and writes results to a dated file. It must follow the rules already in place:
   - one Gradle build at a time;
   - Triton ports 8100 and up;
   - no backend rebuild while a server runs;
   - PJRT `create_options` always.

   Today every device result depends on someone running the tests by hand.
4. **A precision policy.**
   - `docs/NUMERICAL_PRECISION.md` covers:
     - TF32 is the default for plain `MATMUL` on the GB10;
     - what `portableF32Dots` does and does not change;
     - `xla_allow_excess_precision`;
     - the 4e-5 floor between GPU and interpreter;
     - bf16 rules;
     - the TPU's 3- and 6-pass f32 dots.
   - `TestBackend` gets a per-op, per-dtype tolerance table, checked to be positive, with a
     switch that forces strict tolerances.
   - A closeness failure prints the tightest atol and rtol that would have passed (×1.2).
     TorchTPU: `compute_optimal_tolerances`.

### Tier 2

5. **Multi-device without TPUs.** Run collectives, sharded steps and context parallelism on N
   virtual devices of the XLA CPU client (its device-count create option). Check numerically
   against the one-device result, in CI. This turns `PjrtCollectiveCompileTest` and
   `GradShardingVerifyTest` from compile and IR checks into executed ones, and it is the
   precondition for FSDP and tensor-parallel examples.
6. **A benchmark registry with history.**
   - Each `StepBenchmark` case is registered with a presubmit policy (run, smoke, skip) and a
     reason.
   - Results are written as JSON keyed by commit, and a check compares them with the last N runs
     on the device lane.
   - Known failures are typed records with reasons.
   - A storage-free smoke builds every registered model in CI (trace and emit only).

   TorchTPU: `perflab/registry.py`, `known_failures.py`, `benchmark_registry_test.py`.
7. **Examples and READMEs as tests.**
   - CI builds all 17 examples and runs the six that need no device (readable-gradients,
     differentiable-physics, named-indices, per-example-gradients, gaussian-process,
     internals/*).
   - Fenced README commands carry a marker and run (extend `ReadmeSnippetsTest` and
     `onboarding-smoke.sh`).
   - Each number in a README is printed by the example and compared, so docs and examples cannot
     drift.
8. **Make the 88 round trips run, and check the emitter numerically.**
   - Fetch a pinned `stablehlo-translate` / `sdy-opt` in CI.
   - Run emitted modules through the StableHLO reference interpreter
     (`stablehlo-translate --interpret`) against `DxirInterpreter`. This checks emitter
     semantics without a device, as TorchTPU does for its MLIR passes.
9. **Property tests.** kotest is already a dependency. Generate random DXIR graphs over
   elementwise, broadcast, reduce and shape ops, and check:
   - F32 against F64;
   - VJP against finite differences;
   - vmap against loops;
   - JVP against VJP;
   - render, compile and rerun;
   - emit and parse.

   Shrinking gives minimal failing graphs.
10. **Runtime error parity.** For invalid inputs (shape mismatch, wrong dtype, out-of-range
    gather indices, a closed session), the interpreter and the device reject them the same way
    with the same message. Add fault injection in `PjrtSession` for compile and execute failures.

### Tier 3

11. **Test conventions.** One seed per test, printed on failure, with a property to replay it.
    Randomized test order in a nightly lane. A flake runner that repeats one test N times
    (`BGDHyperOptTest`).
12. **Memory tests.** Peak-byte bounds for buffer donation and the device AdamW. A leak test over
    many executes and session closes (host and device bytes flat). KV pool capacity at its limit.
13. **Larger-model parity on slices.** The first N layers of Qwen3.6-35B-A3B and Qwen3.8-27B with
    real weights against HF, as perflab benchmarks 4-layer slices. Today these models have
    benchmarks and served measurements, but no logit parity.
14. **Profiler.** The PJRT profiler extension, so a test can assert kernel counts and device time
    instead of relying on `nsys` outside the process.
15. **Distributed examples and a case study** (FSDP and tensor parallel on virtual CPU devices,
    then on the GB10) with the methodology written out: hardware, versions, shapes, warmup and
    timed counts, sync point, cold and warm compile.

### Not worth copying

- Test count as a target: most of TorchTPU's count is ATen's surface and upstream reuse.
- Substring checks on emitted IR; Tlaloc's full-text pins are stronger.
- Timing without a device sync, and case-study numbers that are not regenerated by the example.

## 5. Status of tiers 1 and 2

| Item | State |
|---|---|
| 1. Op sweep | done: `OpCatalogue` (115 entries, 106 samples), 9 surfaces, 809 checks pass and 145 are excluded with reasons. It found that `RSQRT` had no VJP and `ARGMAX` no F64 interpreter arm (both added), and the gaps in FOLLOWUPS section 15 |
| 2. Recorded goldens | done: `Goldens`, pinned by SHA-256; 11 oracle tests that skipped in CI now replay; a live run must equal its recording |
| 3. Device lane | done: `scripts/device-lane.sh` (suites, GPU examples, `verify.sh` on 8100-8102, benchmarks), dated summary and `history.tsv`; scheduling it is left to the machine's owner |
| 4. Precision policy | done: docs/NUMERICAL_PRECISION.md and `Tolerances` (positive defaults, overrides with reasons, `TLALOC_STRICT_TOLERANCES`, suggested atol/rtol on failure) |
| 5. Multi-device | done on XLA:CPU with four virtual devices through jaxlib, recorded. It found that parameter shardings were never emitted and that the reverse transform dropped meshes and parameter shardings (both fixed) |
| 6. Benchmark registry | done: `BenchRegistry` (measured, smoke-only, skipped with reasons), `scripts/bench-registry.sh` with history and a `--perturb` control. Millisecond steps move by about 20 % between runs on the GB10, so the bound follows each case's spread |
| 7. Examples in CI | done: the `examples` job runs eight examples and checks their output against their READMEs, and compiles seven GPU ones |
| 8. Round trips | done for the 76 `stablehlo-translate` ones (jaxlib's serializer, recorded verifications); found that `stablehlo.tan` needs StableHLO 1.4.0. The 12 `sdy-opt` ones still need the binary. Checking emitted modules numerically with the StableHLO reference interpreter is not done (jaxlib does not ship it) |
| 9. Property tests | done: `RandomGraphPropertyTest`, 150 graphs per property, shrinking |
| 10. Error parity | done: `PjrtErrorParityTest`; it found the unchecked buffer count of a first execute (fixed) and pins the gather-index clamp |

## 6. Documentation and examples

TorchTPU's documentation is a site (MkDocs, built with `strict: true`, so a broken link
fails it) in four parts: getting started (intro, VM and GKE setup, feature set, API
reference, FAQ), in depth (how it works, the execution model, the memory model,
distributed, custom kernels, dynamic shapes), case studies (FLUX optimization in
milestones with before and after code and numbers; FSDP; a 30B model's training and
inference), and analysis (numerical precision, profiling). The pages are generated from
Marimo notebooks, and `ci/tools/run_doc_commands.py` runs a marked command block from an
example's README (`<!-- doc-test: name -->`) to check that it resolves its imports. Its
examples are fewer than Tlaloc's (simple, FSDP, tensor parallel, FLUX, a 30B model), and
most need a TPU.

Before this work, Tlaloc had:

- **Examples ahead of TorchTPU's:** 17 standalone builds, run for real, whose READMEs
  print real output; CI runs eight and checks their output.
- **No index:** `docs/` held 34 Markdown files (33 committed, plus NUMERICAL_PRECISION.md from tier 1), of which 21 were plans, audits and design notes.
- **No guides for the in-depth topics:** how programs run and compile, buffers and
  memory, dynamic shapes, distributed, custom kernels, profiling. The facts were in
  plans, work logs and KDoc.
- **No case study.**
- **Snippets checked by copying:** of the README's and GETTING_STARTED's 25 Kotlin
  blocks, two were copied into `ReadmeSnippetsTest` and the Gradle files were built by
  the onboarding smoke; the rest were not compiled anywhere.
- **A link checker CI did not run.**

What was done:

| Item | State |
|---|---|
| Doc snippets | `scripts/check-doc-snippets.py`: TorchTPU's marker, but the block is Kotlin and is compiled and run against the published artifacts, with hidden setup and checks in `docs/snippets/`. A snippet can be compile-only or required to fail with a named compiler error. 26 marked; in CI. Writing the wrappers changed one README block (`= ...` replaced by real values) |
| Links | `scripts/check-doc-links.py` in CI, over the new guides |
| Layout | plans to `docs/plans/`, audits to `docs/audits/`, design notes to `docs/design/`; references rewritten; `docs/README.md` routes by task |
| Guides | HOW_IT_WORKS, DYNAMIC_SHAPES, DISTRIBUTED, CUSTOM_KERNELS, PERFORMANCE, FAQ |
| Case study | a training step on the device, with `examples/training-step-on-device` regenerating its numbers |

Writing them found:

- **The forward transform dropped meshes.** It kept parameter shardings but not their
  meshes, so the JVP of a sharded function failed to build (fixed, with a test).
- **Docs and comments that were wrong:**
  - `NUMERICAL_PRECISION.md` said TPU sessions compile with `STRICT_PRECISION`; only
    the tests' do;
  - the `emitAllReduce` KDoc predated `channel_handle`;
  - `OpCatalogue` named a test that does not exist.
- **Gaps:** GETTING_STARTED's environment table lacked four variables, and nothing
  said that `runOn` never donates.

Not done: a rendered site (GitHub renders the Markdown, and the link check covers what
`strict` would); notebooks; a distributed example, which waits for a multi-device
runtime.
