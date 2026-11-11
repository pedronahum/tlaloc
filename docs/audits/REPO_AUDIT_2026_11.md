# Repository audit, 2026-11-11

Commit `869950a6` (`0.1.0-alpha04`). Scope: every first-party module, CI, release, licensing,
docs, and position against JAX, PyTorch and TorchTPU. Items already in
[FOLLOWUPS.md](../FOLLOWUPS.md) are left out unless there is new evidence.

Evidence labels:

- **Run**: a program was compiled and executed, and gave the stated output.
- **Source**: traced through the code; not executed.
- **API**: read from public GitHub or Maven Central endpoints.

Nothing here ran on a TPU, on a second GPU type, or against a live Triton server. Statements
about other projects' behaviour come from their documentation or from memory, not from reading
their code in this audit.

## Baseline

Measured on the GB10 (CUDA plugin) on 2026-11-11 with the fixes on branch `fix/audit-2026-11`:
`./gradlew test` plus `:maestro:jvmTest` ran 3,581 first-party tests, none failed and 50 were
skipped (runtime-pjrt 33, stablehlo 12, endpoint 3, ir 1, maestro 1). The vendored Maestro lane
was not rerun. The scripts/prepush.sh gates pass. Without the AD fixes, 4 of the 6
new control-flow tests fail.

## Fix first

| # | Item | Why | Size |
|---|---|---|---|
| 1 | AD-1, AD-2, AD-3 | `grad {}` returns wrong numbers with no diagnostic | Small: refuse at compile time, guard `n < 0` |
| 2 | SV-1, SV-2, SV-6 | Open, unauthenticated listeners; a stalled client holds a seat forever | Small to medium |
| 3 | SV-3 | Tokens after an end token enter the next turn's context | Medium: touches chained steps |
| 4 | RT-1, RT-2, RT-3, RT-4 | Device-memory leak on failed steps; cache throws under the documented threading; cache deletes valid entries; a CUDA client can be created without options | Small each |
| 5 | NN-1, NN-3, NN-4 | One dropout mask for a whole training run; Muon scales the wrong matrices; labels silently clamp on a device | Small each |
| 6 | IR-1, IR-2 | NaN logit yields a token id outside the vocabulary; `ARGMAX` is order-dependent on NaN | Small |
| 7 | CI-2 | Non-commercial text is in a published Apache-2.0 jar | A decision, then a release |
| 8 | CI-1, CI-3, CI-4 | Release workflow has never reached its upload; the signing job has no pinned supply chain | Small to medium |
| 9 | DOC-1 to DOC-9 | Headline claims the code does not meet | Small |
| 10 | CI-6, CI-8 | No PJRT, CUDA or C++ backend test runs in CI | Medium |

## Automatic differentiation

Paths: `ir/.../passes/`, `compiler-plugin/.../plugin/`.

| ID | Sev | Evidence | Finding | Where |
|---|---|---|---|---|
| AD-1 | High | Run | Silent wrong gradient. An `IF` whose region holds ops outside `SAFE_LIFT_OPS` (DIV, EXP, LOG, SQRT, GATHER, SUM, others) is not cloned, so the primal node appears in the gradient body and its id collides. `grad { x -> val y = if (x > 0f) 1f/x else x; val s = y.sin(); val t = x*2f; val u = t*3f; val v = u+x; s+v }` at 2 returns 7.1634; the derivative is 6.7806. A code comment calls this a known leak; it is not in FOLLOWUPS. | `DxirReverseTransform.kt:270-305`, `PhiCalculus.kt:340-344` |
| AD-2 | High | Run | `return@grad` inside a branch is lowered as the branch's value. `val scale = if (x > 1f) { return@grad x*x } else { 2f }; x*scale` at 3 returns 27; correct is 6. A `return` in a loop body is evaluated and dropped (Source). | `FirLambdaToDxirLowering.kt:357,403,941,1147` |
| AD-3 | High | Run | The closed form for a symbolic trip count accepts any scalar parameter, with no `n >= 0` guard. `for (i in 0 until n) s = s*2f` at n = -2 gives gradient 0.25; Kotlin runs zero iterations, so it is 1. | `PhiCalculus.kt:2223-2233` |
| AD-4 | High | Run | An unused tensor parameter's zero gradient has a guessed shape. `grad2 { a: V, b: M -> a.sum().toFloat() }` with `b` 2×2 and `a` size 3 returns 6 zeros for `b`. | `DxirReverseTransform.kt:505`, `DxirToIrSynthesis.kt:1697` |
| AD-5 | Med | Run | `MAX`/`MIN` ties are not a subgradient: `[1,1,0].max()` has gradient `[1,1,0]`, and `jvp` at dx = `[1,1,0]` is 2 (true directional derivative 1). JAX and PyTorch split ties. Same in max-pool. | `Vjp.kt:516-534,1450`, `DxirForwardTransform.kt:743-769` |
| AD-6 | Med | Run | `POW` has no zero guard: `a^n` at a = 0, n = 0 gives NaN; `x^x` at 0 gives NaN. | `Vjp.kt:1890-1901`, `DxirForwardTransform.kt:572-589` |
| AD-7 | Med | Run | The JVP of a float-to-int `CAST` truncates the tangent instead of zeroing it: `jvp { x.toInt().toFloat() }` at (1.5, 2.5) gives 2; the VJP correctly gives 0. Returning null for a non-float source also clears AD-8. | `DxirForwardTransform.kt:141,887` |
| AD-8 | Med | Run (by the auditing pass) | `hessian` through `where`, and by the same path `maximum`/`minimum`/`clip`, stops with "irConstFor returned null". | `DxirForwardTransform.kt:887` |
| AD-9 | Med | Run (by the auditing pass) | Reverse mode runs both branches' adjoints: `grad { a -> if (k < 3) a[k] else a[0] }` at k = 5 throws `ArrayIndexOutOfBoundsException`; the primal is defined. | `DxirReverseTransform.kt:1855-1929` |
| AD-10 | Med | Run (by the auditing pass) | Forward-mode shape defects: `MAX`/`MIN` tangent has no run-time shape template, so `jvp2` throws in `stretchTo`; `MAXPOOL2D` tangent bakes sentinel dims into a rank-6 reshape and is refused. | `DxirForwardTransform.kt:416-439,759` |
| AD-11 | Med | Run (by the auditing pass) | Loops unroll into one JVM method. `for (i in 0 until 400) s = (s*1.01f).sigmoid()+s` ends in `MethodTooLargeException` after the plugin's guard, with no diagnostic; an `if` body at n = 3000 ends in `OutOfMemoryError` at 4 GB. | `PhiCalculus.kt:558`, `DxirReverseTransform.kt:1853,878` |
| AD-12 | Low | Run (by the auditing pass) | `BF16` missing from zero and seed: an unused BF16 parameter throws "cannot zero-seed". | `DxirReverseTransform.kt:2052-2067` |
| AD-13 | Low | Source | `SymbolicEngines.kt:118` swallows the failure, so incomplete Symja jars report "WHILE has regions". An assignment in an `if` branch reports "id 7 not in nodeMap". | `SymbolicEngines.kt:118`, `FirLambdaToDxirLowering.kt:830-837` |

Latent, blocked today: the trailing-break hoist tests before the first iteration
(`FirLambdaToDxirLowering.kt:912-930`), and result indices of multi-result ops are dropped at
`DxirReverseTransform.kt:1540` and `PhiCalculus.kt:576-578`. Each becomes silent if the refusal
upstream of it is lifted.

Clean: accumulation order, aliased operands, CSE signature, softmax and soft cross-entropy,
concat and slice adjoints, linear-algebra rules, nested `grad` with an outer capture, the vmap
rule table. Conv adjoints, the recognizer coarsener and Mosaic vjps were not reviewed in depth.

## IR, emitter and host ops

| ID | Sev | Evidence | Finding | Where |
|---|---|---|---|---|
| IR-1 | High | Source | `SAMPLE_TOKENS`' XLA form returns id `vocab` on any NaN logit: `maximum` gives NaN, the `EQ` mask is all false, and the `minimum` over `sel` returns its `vocab` init. The reference skips NaN. | `StablehloEmitter.kt:5717-5719,5823-5827`, `SampleTokensAttrs.kt:99` |
| IR-2 | Med | Source | The `ARGMAX` reducer is not commutative on NaN: `[1,5,NaN,2]` gives 3 in linear order and depends on reduction order. The interpreter gives 1; JAX gives 2. | `StablehloEmitter.kt:3219-3226`, `DxirInterpreter.kt:2423` |
| IR-3 | Med | Source | The host and interpreter skip zero left entries in `MATMUL`; `dot_general` does not. `[[0,1]]·[[NaN],[2]]` is 2 in the oracle and NaN on a device, so a diverging weight is hidden from the oracle. | `HostOps.kt:3422`, `DxirInterpreter.kt:756`, `BatchedOps.kt:49`, `NamedOps.kt:161,232` |
| IR-4 | Med | Source | NaN disagreements: `RELU` and `SIGN` (host 0, device NaN) and host `max`/`min` (skip NaN; interpreter and device propagate). The comment at `StablehloEmitter.kt:317-323` says `-0.0` is the only divergence. | `HostOps.kt:138,172-176,2171-2189`, `DxirInterpreter.kt:386,395-401,1373,1394` |
| IR-5 | Med | Source | Interpreter `CAST` does not implement several conversions: F32 3e9 to I64 clamps through `toInt`, I32 300 to I8 stays 300, 2.7 to U8 stays 2.7, 0.1 to F8 stays 0.1, 0.5 to Bool to F32 stays 0.5. | `DxirInterpreter.kt:1222-1245,356` |
| IR-6 | Med | Source | Conv and pool extents use truncating division: `(2-3)/2+1 = 1` passes the guard for a 2×2 input, window 3, stride 2, and the host returns a partial window. StableHLO's count is 0. | `HostOps.kt:977,1511`, `DxirInterpreter.kt:2843,3286`, F64 twins |
| IR-7 | Med | Source | `DTensor`'s constructor does not check storage against dims or rank, and `size` multiplies in Int. A `[2,2]` tensor over 3 floats constructs; `65536×65536` over an empty array passes. | `core/.../DTensor.kt:55-61` |
| IR-8 | Med | Run (`iree-opt`) | Float literals printed for non-float dtypes: SUM/MEAN init, PAD, ZEROS_LIKE on I8/U8/Bool, FloatArray consts. Loud at parse time. Bool consts run in the interpreter and throw in the emitter. | `StablehloEmitter.kt:241,253-258,385,408,1618,8168,8195,8330` |
| IR-9 | Med | Run (`iree-opt`) | The Mosaic DSL emits rejected forms (`arith.divi`, `arith.maximumi`, rank-1 `sum`) and, silently, `splat`/`where` on an I32 tile as raw float bits: `-1f` becomes `0xBF800000 : i32`. | `MosaicKernelDsl.kt:160,186,229,282` |
| IR-10 | Low | Source | `COMPARE` writes `SIGNED` for BF16, F8E4M3FN, U8 and Bool. Zero-size `MEAN` is 0 on device and NaN in the interpreter. RNG shape arithmetic overflows Int at 1.2e9 elements. | `StablehloEmitter.kt:8129-8132,1625,896,1070` |
| IR-11 | Low | Source | Host elementwise and reduce paths box through non-inline `(Float, Float) -> Float`. | `HostOps.kt:29-33,80-84,187,2101-2108` |

Plausible, not executed: scratch or multi-result custom calls with a sharding emit one
`sharding_per_value` entry (`StablehloEmitter.kt:2806,2840`); `RNG_NORMAL` emits f64 math, so an
F32 program may need f64 on a TPU (`:1069-1105`); named `MATMUL` picks the wrong axis on a
square `[K,K]` left operand (`:1735-1755`).

Clean: threefry schedule, narrowing rounding, constant printing, integer DIV, reductions and
sort, broadcast and negative-axis handling, shared caches. The emitter's attention, GDN and MoE
forms and QR/eigh emission were not read closely.

## Native runtime

Paths: `runtime-pjrt/.../ffm/`.

| ID | Sev | Evidence | Finding | Where |
|---|---|---|---|---|
| RT-1 | High | Source | A failed execution leaks. `awaitEvent` throws before `destroyEvent`, and the populated output buffers are dropped. `ServingModel` catches the failure and continues, leaking device memory per failed step. Same in `bufferToHost*`. | `PjrtFfm.kt:1419-1424,1660-1668,1952-1961,2002-2011` |
| RT-2 | High | Source | `OverlappingFileLockException` from the executable cache. `keyLocks` is per instance, each `PjrtSession` builds an instance, and file locks are per JVM. Two threads loading one artifact with `TLALOC_PJRT_CACHE_DIR` set (the documented one-model-per-thread use) throw out of the compile. `evict` also hits this inside one instance. | `PjrtExecutableCache.kt:117-119,190-192`, `PjrtSession.kt:127` |
| RT-3 | Med-high | Source | Any failed read or deserialize deletes the entry, including a transient RESOURCE_EXHAUSTED or heap OOM. At the 0.045 fraction serving uses, the fallback compile cannot fit, so one transient failure disables the cache for later processes too. | `PjrtExecutableCache.kt:148-156` |
| RT-4 | Med-high | Source | A CUDA client can be created with no options: public `createClient(null)`, or `PjrtSession(plugin = cuda, target = Metal or Cerebras)`, or an env override pointing at the CUDA plugin. This is the configuration that reserved 75 % of unified memory. Refuse null options when the plugin exposes the GPU extension. | `PjrtFfm.kt:986`, `PjrtSession.kt:120-122,1005-1009`, `PjrtBinaries.kt:278-281,350-353` |
| RT-5 | Med | Source | `PJRT_Plugin_Initialize`, required by the header before other calls, is never called. Matters most on libtpu. | absent from `PjrtFfm.load` (`PjrtFfm.kt:65-75`); `pjrt_c_api.h:281` |
| RT-6 | Med | Source | Check-then-act on buffers and executables outside the session lock: `buf.toFloatArray` racing `session.close()` is a use-after-free of the client. | `PjrtFfm.kt:2230-2242,2303-2307` |
| RT-7 | Med | Source | An exception thrown by a `registerExecuteHandler` handler crosses the upcall boundary and aborts the JVM. | `PjrtFfiRegistry.kt:115-129` |
| RT-8 | Med | Source | XLA compiles run inside `computeIfAbsent`; a lookup that collides in that bin blocks for the whole compile. | `PjrtSession.kt:216,507,575,736,859` |
| RT-9 | Med | Source | The shared cache directory loads native code authenticated by an unkeyed hash. Anyone able to write to it can plant an executable. | `PjrtExecutableCache.kt:125-129,256-268` |
| RT-10 | Low | Source | Partial staging leaks in `runOnF64`, `runOnBf16`, `stage`. `(n * elementSize).toLong()` overflows above 2 GiB. Metal and Cerebras plugin search walks ancestors of the working directory. | `PjrtSession.kt:250,644,692`, `PjrtFfm.kt:1205-1502`, `PjrtBinaries.kt:301,358` |

Plausible: a null `host_layout` on read-back permutes a non-descending TPU layout;
`runOnPjrt` does not pin its plugin.

Clean: all PJRT offsets and enums (checked by compiled `offsetof` against the vendored header,
and against the shipped plugin's disassembly for the two short structs), host-buffer semantics,
cache atomicity and identity, KPTX register and memory bounds in the paged kernels, CUDA driver
signatures.

Runtime gap: execution is synchronous on the caller thread, `PJRT_Error_GetCode` is not bound
(so OOM and cancellation are only distinguishable by parsing a string), and `close()` has no
timeout.

## Serving

No memory-safety defect was found on the request path. The defects are in the layer around the
engine.

| ID | Sev | Evidence | Finding | Where |
|---|---|---|---|---|
| SV-1 | Critical if network-reachable | Source | No authentication at any layer; both endpoints default to `0.0.0.0`; `run_server.sh` publishes 8000-8002 on all interfaces. `Admit` has no owner check, sequence ids are predictable, and the page-exhausted error returns other sequences' ids. A gRPC call naming a live id continues that user's context. `POST text/plain` passes, so a web page can drive a local endpoint. | `OpenAiEndpoint.kt:36,142`, `Main.kt:78`, `openai_server.py:428`, `run_server.sh:93`, `sequence_mode.cc:1858-1870,2002-2016,2143`, `Sequences.kt:31` |
| SV-2 | High | Source | A client that stops reading a stream holds its seat forever: `Output.send` blocks without a timeout and `s.busy` stays set. Waiters queue without limit, and `g.next()` has no deadline. | `OpenAiEndpoint.kt:680-683`, `Sequences.kt:97`, `TritonClient.kt:21` |
| SV-3 | High | Source | On a chained step (the default), the end-token truncation is skipped (`!issued`), so `q.length` runs past the end token. The frontends record the KV as ending before it, and `Admit` sets `position = seq.length`, so the next turn's context is `... EOS y EOS <user turn>`. The comment above the check says the KV matches what the client receives. | `sequence_mode.cc:2862-2879,2966-2969,2069`, `OpenAiEndpoint.kt:486-489,548`, `openai_server.py:358-362,382` |
| SV-4 | High (Python server) | Source | `chat_template_kwargs` passes to `apply_chat_template` unfiltered, so a request can supply a `chat_template` that runs under the GIL. The Kotlin endpoint whitelists keys. | `openai_server.py:240-243,285-287` |
| SV-5 | High (multi-tenant) | Source | One long prompt blocks everyone: with no decode in the batch the prefill budget is 0 and `RunPrompts` loops on the single worker. Cancellation is checked afterwards. | `sequence_mode.cc:3097-3195,3475,3515` |
| SV-6 | Med | Source | `readAllBytes` with no size limit, tokenization before the length check, and JSON recursion with no depth limit. `StackOverflowError` escapes `catch (e: Exception)`. | `OpenAiEndpoint.kt:113,159,349,419`, `Json.kt:128-187`, `openai_server.py:236` |
| SV-7 | Med | Source | A conversation that outlives Triton's idle timeout gets a 500: the "must specify START" answer is not retried. | `Sequences.kt:73`, `OpenAiEndpoint.kt:501-516`, `TritonClient.kt:139` |
| SV-8 | Med | Source | Requests near the context limit fail: the endpoint allows `prompt + max_tokens == max_context`; the backend adds speculative reach. | `OpenAiEndpoint.kt:423`, `sequence_mode.cc:2091-2097` |
| SV-9 | Med (multi-tenant) | Source | Prefix-probing oracle: live sequences and the prefix cache are shared across callers and `cached_tokens` is returned. No per-tenant salt. | `Sequences.kt:71-77`, `OpenAiEndpoint.kt:534`, `sequence_mode.cc:2020-2027` |
| SV-10 | Med | Source | A disconnect is seen only on a failed stream write, so an abandoned `stream:false` request runs to `max_tokens`, against the KDoc. | `OpenAiEndpoint.kt:87-88,517` |
| SV-11 | Med | Source | `StopStrings.feed` cuts by UTF-16 unit; a split emoji is written as two `?`. | `OpenAiEndpoint.kt:583-585,682` |
| SV-12 | Low-med | Source | A false reply match duplicates context in a multi-turn chat. Bad parameters return 500 and drop the sequence (`seed` over 2^31, `top_p: 0`). `/health` is a constant through a GPU fault. `response_format` and penalties are ignored without error. Python `Detokenizer.add` is quadratic and `stop` is ignored. | `OpenAiEndpoint.kt:154,212,364,502-516`, `openai_server.py:171-177,228,288` |
| SV-13 | Low | Source | `Safetensors` element count overflows in unchecked `Long`: `[65536,65536,65536,65536]` passes over a zero-byte range. | `Safetensors.kt:73,302` |

Plausible: a non-IO exception after `acquire` leaves a sequence busy; two endpoint replicas
collide on sequence ids (`Sequences.kt:31`); `Parse` copies `shape[1]*4` bytes without checking
`bytes.size()` (`sequence_mode.cc:1895`).

Clean: KV page refcounts and prefix pinning, chained-step lifetimes, stale KV masked by
`seqLens`, sampler bounds, tool-call parsing on malformed output, safetensors offsets and shard
paths, no pickle or `eval`.

Production gaps against vLLM and SGLang: API keys, latency histograms, OpenTelemetry, a
multi-node router, queue limits and priorities, vision, structured outputs, per-request LoRA.

## Training layer

| ID | Sev | Evidence | Finding | Where |
|---|---|---|---|---|
| NN-1 | High | Source | `Dropout` draws its mask at capture, so a `CapturedStep` uses one mask for every step, with no error, and the same captured primal keeps it during evaluation. The KDoc says it re-keys per step. FOLLOWUPS records this for LoRA only. | `nn/.../Dropout.kt:57-62`, `Training.kt:104-107` |
| NN-2 | High | Source | `BatchNorm` statistics cannot update inside `Sequential` or a capture, and `Sequential` has no eval mode. A checkpoint writes `runningN = 0`; evaluation normalizes with the batch's own statistics. | `BatchNorm.kt:161-165,246`, `Components.kt:205` |
| NN-3 | High | Source, script | `Muon` applies PyTorch's scale to Tlaloc's `[in, out]` layout. For d = 512, ff = 2048, vocab = 32,000 (Tlaloc vs torch): gate 1.0 vs 2.0, down 2.0 vs 1.0, head 1.0 vs 7.9. The test passes because it feeds raw matrices in torch's layout. | `Muon.kt:70-72`, `DeviceOptimizers.kt:237`, `Layers.kt:79`, `DeviceOptimizersTest.kt:107` |
| NN-4 | High | Source | Label ids at or above the class count silently use the last class on a device: `TAKE_ALONG` lowers to `gather`, which clamps. The interpreter refuses, so host tests pass. | `Losses.kt:109-114`, `StablehloEmitter.kt:1303`, `SelectionKernels.kt:49` |
| NN-5 | Med | Source | Adam's global step gives a new key (resume, phased unfreeze) a first step of about 2.5·lr. `Checkpoint` validates neither keys nor dims. | `Optimizers.kt:292-294,333-345`, `Checkpoint.kt:212-231` |
| NN-6 | Med | Script | `BatchNorm` variance `Σx²/n − μ²` goes negative in f32: 98 of 200 batches NaN at mean 1000, std 0.01, n = 256. `LayerNorm` already centers. | `BatchNorm.kt:76,252-254,303` |
| NN-7 | Med | Source | `MixtureOfExperts` and `BatchNorm` refuse `MIXED_BF16` at capture through F32 leaves. | `MixtureOfExperts.kt:103`, `BatchNorm.kt:252` |
| NN-8 | Med | Source | `CausalLM` bypasses `Embedding.paddingIndex`. | `Transformer.kt:702` |
| NN-9 | Low | Source | Qwen3 with `use_sliding_window: true` and no `sliding_window` becomes full attention; transformers defaults to 4096. `fanOf` applies torch's layout to Dense's `[in, out]`. | `HfDecoder.kt:868-872`, `Initializers.kt:40-46` |

Clean: Adam and AdamW arithmetic, global-norm clipping, schedules, LayerNorm and RMSNorm axes,
RoPE, masks, initializer gains, LoRA, gradient accumulation, `HfCausalLm` refusals.

## Build, CI, licensing

| ID | Sev | Evidence | Finding | Where |
|---|---|---|---|---|
| CI-1 | High | API | All 4 tag runs of `release.yml` failed at "Version is not on Central yet", because releases are uploaded locally first. Test, smoke and upload have never run from a tag. Every release used the workstation's key. | `release.yml:73-84`, `docs/RELEASING.md` |
| CI-2 | High | Run (jar inspected) | `TWIL_TOOL_INSTRUCTIONS` is in `tlaloc-tokenizer-jvm-0.1.0-alpha04.jar` on Central, under an Apache-2.0 POM with no licence copy or NOTICE entry; the upstream licence is non-commercial and requires a copy with distribution. Whether the text is copyrightable is a legal question. | `ChatTemplate.kt:369`, `NOTICE` |
| CI-3 | High | Source | No dependency verification, no first-party lockfiles (0 of 17 modules), no `distributionSha256Sum`. | `gradle/`, `gradle-wrapper.properties` |
| CI-4 | High | Source | 28 of 28 third-party action references are mutable tags, three in the job that receives the signing key. `release.yml` has no `environment:`. Three workflows have no `permissions:`. | `.github/workflows/*` |
| CI-5 | High | API | `main` is unprotected; 15 of the last 70 pushes were red. 8 Dependabot PRs have been open since 2026-10-01, including Kotlin 2.4.21, all failing in about 2 minutes. | repository settings |
| CI-6 | High | Source, counts | CI never executes PJRT or CUDA. 383 of 3,648 tests are gated on hardware or tools (202 of 315 in `runtime-pjrt`). The device lane's history has one run (2026-10-25, failed), before Ripple, TwIL and alpha04. | `history.tsv` under `~/.local/share/tlaloc/device-lane/` |
| CI-7 | Med | Source | About 21 tests `return` instead of aborting, so they pass vacuously on a runner. `Goldens.kt` already throws `TestAbortedException`. | `BoundedArtifactRunTest.kt:79-263`, `SafetensorsWriterOracleTest.kt:156-227`, `VllmPluginContractTest`, `HfLlamaServingArtifactTest:72,125` |
| CI-8 | High | Source | 9,585 lines of `triton/backend` are in no workflow, with no sanitizer, clang-tidy or fuzzing; `verify.sh` is manual. | `triton/backend/` |
| CI-9 | Med | Source | No lint or format on first-party code. No `.editorconfig`, no `allWarningsAsErrors`. | `build.gradle.kts` |
| CI-10 | Med | Source, API | Symja: the parent POM lists GPL-3.0 and LGPL-3.0, the core POM LGPL, the repository GPL-3.0. `SymjaEngine` ships in `tlaloc-ir-jvm`. No general licence check. | `libs.versions.toml:10-14`, `ir/build.gradle.kts:72-77`, `build.gradle.kts:418` |
| CI-11 | Med | Source | F64 twins and `VmapIntrinsics.kt` state they are generated; no generator is in the repository. | `HostOpsF64.kt:26`, `DxirInterpreterF64.kt:30`, `VmapIntrinsics.kt:1` |
| CI-12 | Med | Source | The test-count step ends in `|| true` with no per-lane floor. `kotlin-next` is green by construction (34 of 34 runs), and nothing reports a failing probe. | `build.yml:109-111`, `kotlin-next.yml` |
| CI-13 | Med | Source | Unpinned bootstrap: curl into bash, "latest" JDK piped to tar, unhashed libtpu wheel, `--extra-index-url`, images by tag, and the first plugin found under `~/.local/venvs/*` loaded with no hash. | `scripts/setup-*.sh`, `scripts/tpu/session.sh:125-153`, `PjrtBinaries.kt:120` |
| CI-14 | Med | Counts | Tests are thin where untrusted input arrives: endpoint test/main 0.25, tokenizer 0.24. `Http2.kt`, `Protobuf.kt` and `Sequences.kt` have no direct test. One property-based test file; Kotest is declared in 15 builds and imported by 1 file. No fuzzing of safetensors, HPACK or tokenizer JSON. | `endpoint/`, `tokenizer/`, `ir/.../RandomGraphPropertyTest.kt` |
| CI-15 | Low | Counts | 1,009 `§0.4.NNN` references in main sources, published in `-sources.jar`. `/home/pedro` in 4 tracked files; a personal email at `ALPHA_PLAN.md:691`; a sibling project named in `docs/vendoring.md:12,29` and an audit. | `git grep` |
| CI-16 | Low | Source | `build.gradle.kts` is 1,141 lines with no convention plugin and no configuration cache. `./gradlew test` needs `python3`, which CONTRIBUTING does not list. 38 test files hard-code `~/.local/venvs`. | root build |

Clean: no `pull_request_target`, no credentials in tree or history, vendored Maestro matches
`vendoring.md` exactly, XLA headers byte-identical to upstream, ABI dumps cover every module, no
`@Ignore`.

## Docs and claims

| ID | Claim | Evidence against | Where |
|---|---|---|---|
| DOC-1 | "no tape and no runtime tracing", pipeline from `grad { f }` to PJRT | `grad {}` runs on host tensors. Every GPU path uses `capture`, which traces at run time. **The compile-time derivative does not reach the GPU.** | `README.md:50,503-507`, `HOW_IT_WORKS.md:11-13` |
| DOC-2 | RNG "bit-exact with JAX" | Only the non-default stream (`jax_threefry_partitionable=False`). Normal draws are Box-Muller, not JAX's `erfinv`. Cauchy, exponential and chi-square are claimed too. | `README.md:196`, `CAPABILITIES.md:56`, `Random.kt:24-33,168-178` |
| DOC-3 | Each model "generates the same greedy token ids as Hugging Face transformers" | Ripple and TwIL compared on tiny checkpoints only; real TwIL 99.5 % against vLLM; Muse 31 of 32 ids. | `README.md:253-254` |
| DOC-4 | The derivative prints as Kotlin source | Tensor `grad {}` bodies print a SKIPPED note; the example is a scalar lambda. | `README.md:155`, `TlalocIrGenerationExtension.kt:70-72,1328` |
| DOC-5 | "The same programs run in F64, end to end" | `capture`, `:nn` training, bounded programs and IREE refuse F64. | `README.md:122`, `CAPABILITIES.md:59` |
| DOC-6 | Linear algebra "differentiable to any order" | Tests stop at second order. | `README.md:195`, `CAPABILITIES.md:54` |
| DOC-7 | "Three runtimes load it without a JVM" | Python runtime and vLLM plugin refuse v3 artifacts; `ServingModel` refuses windowed and quantized KV. As served, Muse, Ripple and TwIL are Triton-only. | `README.md:243`, `tlaloc_serve.py:540` |
| DOC-8 | "runs on the Cerebras Wafer-Scale Engine", in the opening paragraph | Simulator only; the plugin repositories return 404; one commit message is the evidence. | `README.md:29-31,338`, `GETTING_STARTED.md:447` |
| DOC-9 | "whole suite passes on Metal" | 147 passed, 66 skipped, on 2026-10-12; no CI lane; the runtime-pjrt suite has grown by 108 tests since. | `README.md:313-315,349` |

Stale: `CAPABILITIES.md` test count (3,346), CI commit (`d0ca85b`), "no hosted site", FP8 and
int8 row (line 60), LoRA timing, Qwen3 timing, "never run" for distributed; three-versus-four
experimental surfaces across three docs; README's "Maturity" table references; example counts;
JDK tables omit `endpoint`; README decode table has no date, commit or conditions and four rows
predate the XLA upgrade; GSM8K drops the repo's own one-standard-error note; `TPU*` asterisk is
undefined. 38 broken links, all in `DIFFKTX_SPEC.md`, `docs/audits` and `docs/plans`.

Doc-tests: 30 snippets compile from mavenLocal; every device, Metal and Cerebras snippet is
compile-only. `GETTING_STARTED`, `READABLE_REVERSE`, and all Java blocks have unchecked snippets.

## Position against JAX, PyTorch and TorchTPU

"Better than" has no support today. Sources for competitor state are the JAX and PyTorch
changelogs and Google's TorchTPU material as of this date; torch-xla's repository is scheduled
to archive on 2026-11-15, so TorchTPU is the competitor.

| Axis | Position | Basis |
|---|---|---|
| Static tensor types | Ahead | Rank, dtype and axis names in the type; errors at file:line:col |
| Readable derivatives | Ahead, partial | Recompilable Kotlin; scalars and the capture route only |
| JVM-native serving and in-process use | Ahead | No competitor has an equivalent; DJL is the one to beat |
| Serving on one box | Level | GSM8K 0.92 with vLLM 0.916; long prefill behind |
| Custom derivative rules | Level | |
| F64 discipline | Mixed | No implicit mixing; capture and `:nn` are F32 only |
| AD breadth | Behind | Nested `hessian` rank-1; no `in_axes`; no `scan` or `while_loop` equivalent; JVM method size caps unrolling |
| vmap | Behind | Refuses conv, pooling, RNG, run-time gather, attention |
| Compile model | Behind on dynamic shapes | One program per shape or bucket |
| Devices proven | Behind | One GB10, one Mac, a simulator, CPU; TPU compile-only; x86 NVIDIA untested |
| Distributed | Far behind | 4 virtual CPU devices; multi-host refused |
| Op coverage | Behind | 120 `OpKind`; no FFT, SVD, pinv, cumsum, argsort, inverse trig, conv1d/3d, LSTM |
| Training stack | Behind | No data pipeline, trainer, model-wide eval mode, RNG-resume, or anomaly detection |
| Ecosystem, install | Behind | Single maintainer; JDK 25, Kotlin plugin, plugin from a pip wheel |

The architectural limits behind these:

- `capture` runs the forward in the JVM on real values, so weights must sit on the heap (a 65B
  checkpoint needs about 260 GB). JAX's trace is abstract.
- The `-1` shape sentinel plus axis-matching in synthesis is the root of AD-4 and AD-10.
- `StablehloEmitter.kt` is 8,405 lines of string concatenation; IR-8 is the class of bug a typed
  builder prevents. `DxirOp` trusts its declared type, and attrs are `Map<String, Any>`.
- A new `OpKind` touches about ten places. `OpCatalogue` is the safety net; an op-definition
  table would remove the need for it.
- `FloatArray` as interpreter storage prevents exact integer semantics (IR-5).

### Moves, in order

| # | Move | Bar |
|---|---|---|
| 1 | Correctness: everything in "Fix first" | AD-1 to AD-3 are compile errors or correct; a fuzz over `if` and loop shapes compares against finite differences |
| 2 | Truth pass; generate the test count; date, commit and conditions on every number | No undated figure in README or CAPABILITIES; CI fails on a stale table |
| 3 | Compile `grad {}` to the device, with no tracer at run time | CausalLM trains via `grad {}` on a GPU; gradients equal capture within 1e-6. This is what would make the headline true |
| 4 | Abstract capture: shapes without values | A 65B checkpoint captures with weights off-heap |
| 5 | Print tensor gradients | No SKIPPED across the plugin corpus; identical bits at 20+ tensor bodies |
| 6 | Public head-to-head in CI against JAX and PyTorch with `torch.compile` on a GPU | Within 10 % of JAX, then ahead on at least one workload. No such comparison exists: the only table is uncommitted, PyTorch on CPU |
| 7 | One real TPU session | All `session.sh` lanes on v6e-1 and v5e-4; the skip list published |
| 8 | Self-hosted runner for the device lane; second GPU and x86 | `verify.sh` and the PJRT suite green per commit |
| 9 | Serving hardening | SV-1 to SV-10 closed; API keys, histograms, queue limits |
| 10 | Two GPUs, then two hosts | At least 1.8× on 2 GPUs for FSDP; loss within 1e-4 of unsharded |
| 11 | Training stack: `DataSource`, trainer loop, model-wide eval, RNG in checkpoints, `detect_anomaly` | The LoRA and fine-tune examples shrink to config |
| 12 | Spark on a cluster, against DJL | 3+ executors with GPU scheduling; survives an executor loss |

Fights to skip: an op-count race, a Python API, pod-scale TPU, mobile, eager mode, KPTX replacing
XLA codegen, and Cerebras until its plugin is public.
