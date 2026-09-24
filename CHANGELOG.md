# Changelog

All notable changes to Tlaloc are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) **with the alpha
carve-outs in [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)**. Read that before
depending on a coordinate.

This file starts at `0.1.0-alpha01`. Earlier work is recorded in the commit history
and in [`DIFFKTX_SPEC.md`](DIFFKTX_SPEC.md).

## [Unreleased]

### Changed

- **Symja 3.2.0.** The build compiles and tests `tlaloc-ir` against
  `org.matheclipse:matheclipse-core:3.2.0`, and the compiler's refusal for a
  symbolic-trip-count loop now names that coordinate. Symja stays `compileOnly`
  (not in a consumer's dependency graph). The 0.1.0-alpha01 `tlaloc-ir` jar calls
  the same 36 Symja methods and fields, so it also runs with 3.2.0 on the
  compiler plugin classpath.
- **Gradle 9.7.1.** The repository's wrapper moves from 9.5.0 to 9.7.1; the
  Tlaloc Gradle plugin is tested with it.
- **`tlaloc { dumpGradSourceDir }` writes one subdirectory per compilation.** The
  Gradle plugin writes each Kotlin/JVM compilation's gradient sources into
  `<dumpGradSourceDir>/<source set>` (`main`, `test`; `jvmMain`, `jvmTest` in a
  Multiplatform build) instead of into `dumpGradSourceDir` itself, and declares
  that subdirectory an output of the compile task. The build cache now stores and
  restores the dumped files, and the directory's absolute path no longer enters
  the compile task's cache key, so a project that sets it gets cache hits from a
  checkout at another path. A build that reads the files from
  `dumpGradSourceDir` directly now reads `dumpGradSourceDir/main`. The raw
  `-P plugin:io.tlaloc.plugin:dumpGradSourceDir=<dir>` option is unchanged.
- **API documentation.** KDoc in every published module describes the code
  without internal changelog numbers, plan phases or planning-document names;
  `./gradlew test` fails if one comes back (`scripts/check-kdoc-internal-refs.py`).

### Fixed

- **Vendored Maestro server builds.** `third-party/maestro/maestro-server`
  compiles: it declares its dependency on `maestro-tlaloc`, and its Spring
  configuration builds the Tlaloc step runtime with the pod-spec builder and the
  cluster accelerator from `TLALOC_CLUSTER_VENDOR` / `TLALOC_CLUSTER_ARCH`. The
  vendored build's Spotless, Checkstyle and PMD checks run on JDK 25 with Gradle 9
  (Spotless 8.10.2, google-java-format 1.30.0, Checkstyle 9.3).

## [0.1.0-alpha01] — 2026-09-24

The first published version. [docs/CAPABILITIES.md](docs/CAPABILITIES.md) lists
what it contains, what is certified and on what hardware.

### Contents

- **Automatic differentiation at compile time.** A K2 compiler plugin rewrites
  `grad`, `grad2`, `grad3`, `valueAndGrad*`, `jvp`, `jvp2`, `vjp`, `vjp2`,
  `jacobian`, `jacobianReverse` and `hessian` calls into synthesized gradient
  code. Custom rules through `customVjp`, `customJvp` and `customVjpJvp`. Loops
  and branches are differentiated through φ-calculus coarsening. One reverse
  transform serves the intrinsics, the Tracer-capture API and `:nn` training.
- **Readable gradients.** `dumpGradSource` and `dumpGradSourceDir` print the
  derived gradient as Kotlin that compiles without the plugin and is bit-identical
  to the compiled gradient; `CapturedStep.gradSource()` prints captured tensor
  gradients.
- **Typed tensors.** Rank, dtype and named axes in the Kotlin type;
  `NAMED_INDEX_MISMATCH`, `TENSOR_SHAPE_MISMATCH` and `NOT_DIFFERENTIABLE` compile
  errors at the offending call. dtypes F32, F64, I32 and BF16.
- **Captured values.** A `grad { }` body may reference compile-time constants
  declared outside it (folded as literals) and immutable runtime values of type
  `Float`, `Double`, `Int` or `Long` (bound at the call site).
- **`:nn`.** Immutable layers (Dense, Conv2d, pooling, BatchNorm, Dropout,
  Embedding, EmbeddingBag, GRU, Flatten), SGD, Momentum, RMSprop and Adam, learning
  rate schedules, gradient clipping, and checkpoints: a model and its optimizer
  state in one safetensors file, reloaded bit for bit.
- **Execution.** StableHLO and Shardy emission; PJRT from Kotlin through FFM and
  from Python through ctypes; IREE through its command-line tools; KPTX, a PTX DSL
  with kernel claiming (no kernel is registered by default).
- **Serving.** Paged attention, KV-cache writes, decode bucketing, safetensors
  reading and writing, a serving artifact that runs in a Python process with no
  JVM and no ML framework, and a vLLM platform plugin. A real TinyLlama-1.1B
  produces the same tokens as HuggingFace transformers.
- **`@ExperimentalTlalocApi`**, a `@RequiresOptIn(ERROR)` marker on the
  provisional surfaces: the four-worlds scopes, `AllReduceAttrs`, and the kernel
  choice and cost-model packages.
- **Tooling.** The Gradle plugin `io.github.pedronahum.tlaloc`
  (`tlaloc-gradle-plugin`), which applies `tlaloc-compiler-plugin` of the same
  version to every Kotlin/JVM compilation and takes the plugin options in a
  `tlaloc { }` block; `tlaloc-bom`; a committed ABI baseline checked by
  `apiCheck`; `./gradlew apiDocs` for a local API reference.
- **Requirements.** Kotlin 2.3.20–2.3.29 (other versions are refused by name at
  compile time). JDK 25 to build code that uses `grad { }`; the library modules are
  Java 21 bytecode, so JDK 21 runs them. PJRT, CUDA, KPTX and IREE need JDK 25.
  Symja (LGPL-3.0) is optional and not in the published dependency graph.

### Changed since earlier source builds

These matter only if you built Tlaloc from source before this version.

- **Group id `io.tlaloc` → `io.github.pedronahum`.** `io.tlaloc` could not be
  verified on Maven Central. Package names are unchanged (`io.tlaloc.*`), and so is
  the compiler plugin id used in `-P plugin:io.tlaloc.plugin:<option>`.
- **Version `0.0.1-SNAPSHOT` → `0.1.0-alpha01`.**
- **Every artifact id carries a `tlaloc-` prefix**: `tlaloc-core`, `tlaloc-ir`,
  `tlaloc-autograd`, `tlaloc-nn`, `tlaloc-stablehlo`, `tlaloc-maestro`,
  `tlaloc-runtime-pjrt`, `tlaloc-runtime-iree`, `tlaloc-runtime-cuda`,
  `tlaloc-kptx`, `tlaloc-compiler-plugin` (JVM artifacts `tlaloc-core-jvm` and so
  on), plus the new `tlaloc-gradle-plugin` and `tlaloc-bom`.
- `tlaloc-ir`, `tlaloc-autograd`, `tlaloc-stablehlo`, `tlaloc-maestro`,
  `tlaloc-runtime-iree` and `tlaloc-runtime-pjrt` expose the Tlaloc modules their
  public signatures use as `api` dependencies: `tlaloc-autograd` alone is enough to
  write `grad { }` over a `DTensor`.
- **Every compile-time refusal is an error by default.** An unlowerable
  `grad { }` lambda, and every call the IR phase leaves as written, stops the
  build at the call site. `strictLowering = false` turns these into warnings; the
  call then throws `IllegalStateException` when it runs.
- **A working build is silent.** The lowered-IR dumps are INFO messages behind
  `dumpLoweredIr`, so a correct program compiles under `-Werror`.
- `grad(::f)` and other arguments that are not a lambda written at the call site
  are refused by name.
- `dumpGradSource` refuses a value other than `true` or `false`.
- The runtime message for a `grad { }` that was not rewritten lists its three
  possible causes.
- Error messages no longer cite internal work-item numbers.
- `PjrtSession` can be used from several threads. `executeOn` serialises calls on
  one executable, and `close()` waits for calls in flight. A `PjrtBuffer`,
  `PjrtLoadedExecutable` or `PjrtClient` used after its client is closed, and a
  closed `PjrtBuffer` passed to `executeOn` or `execute`, throw instead of touching
  freed memory. Closing twice, or after the client, does nothing.
- `IreeModule` is `AutoCloseable`; closing it deletes its compiled VMFB.
  `runOnIree` and `IreeRuntime.invoke` delete their temporary files.
- The PJRT CUDA plugin is searched for under `$VIRTUAL_ENV`, `~/.local/venvs/*`,
  `~/.venv`, `~/venv`, `~/.local`, `/usr/local` and `/usr`, from the JVM and from
  Python alike; IREE tools under `$TLALOC_IREE_BIN`, `$VIRTUAL_ENV/bin`,
  `~/.local/venvs/*/bin` and `PATH`. A failed search lists every place it looked.
- In the Python serving runtime (`harness/python`, not published to Maven),
  `PjrtApi.load(plugin_path=None, platform="cuda")` takes a `platform` and, with no
  path and no `TLALOC_PJRT_PLUGIN_PATH`, searches for a plugin; when none is found
  it raises `FileNotFoundError` with the search report, where it used to raise
  `ValueError`.
- `CosineDecay` holds its final rate past `decaySteps`, where PyTorch's
  `CosineAnnealingLR` rises again.

### Added since earlier source builds

- The Gradle plugin and the BOM (see *Contents*).
- `KotlinVersionGuard`: an unsupported Kotlin version is a compile error naming
  the version found and the supported range; `unsafeAllowUnsupportedKotlin` turns
  it into a warning.
- "Tlaloc internal error" diagnostics: an unexpected exception inside the plugin,
  or a lowered call the IR phase did not find, is reported at the call site with
  the issue-tracker address instead of crashing the compiler or compiling green.
- Model checkpoints, learning-rate schedules and gradient clipping in `:nn`; a
  safetensors writer in `:core`.
- `LICENSE` (Apache-2.0), complete Maven Central POM metadata, sources and javadoc
  jars for every publication, and signing.
- `./gradlew releaseToCentralPortal` and `.github/workflows/release.yml`.
- CI on x86_64 Linux, aarch64 Linux and arm64 macOS, a JDK 21 lane for the library
  modules, and a probe against the next Kotlin release.
- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) sections on running on a GPU,
  configuration (every environment variable and system property Tlaloc reads) and
  troubleshooting.

### Removed

- `io.tlaloc.maestro.MaestroDescriptor` and `io.tlaloc.maestro.StubExecutor`. The
  first-class Maestro step type that replaces them lives in the vendored Maestro
  build under `third-party/maestro`, which is not published: a Maven Central user
  has no replacement.
- The `main` entry points in `tlaloc-maestro`; the exporters run through
  `./gradlew :maestro:exportServingArtifact` and
  `:maestro:exportLlamaServingArtifact`.
- The unused `timeoutSeconds` parameter of `runOnPjrt`.

### Fixed

- Two `grad { }` calls at the same character offsets in two different files could
  compile each other's gradient, and in a shared Kotlin daemon one compilation
  could clear another's pending gradients.
- An f64 `grad { }` body with a literal constant crashed the compiler with a
  `ClassCastException`, and an f64 constant close to 1 or 0 could be folded as if
  it were exactly 1 or 0.
- A `grad { t: Tracer<…> -> … }` call (the plugin-free tape overload) drew a
  spurious lowering warning.
- The Python serving runtime did not search for a PJRT plugin, so `serve.py`
  reported no plugin on a machine whose JVM lane ran on CUDA.
- Both PJRT bindings check the plugin's `PJRT_Api` size and API version before
  reading a function pointer and refuse an incompatible plugin by name.
- A malformed `TLALOC_PJRT_MEMORY_FRACTION`, `TLALOC_PJRT_PREALLOCATE`,
  `TLALOC_PJRT_NODE_ID` or `TLALOC_PJRT_NUM_NODES` is refused with the variable's
  name and value.
- A `PjrtSession` whose client creation fails releases the memory it allocated.
- Symbolic simplification no longer swallows JVM errors such as
  `OutOfMemoryError`.
- The KPTX kernel caches and the kernel-resolver registry are safe to use from
  several threads.

[0.1.0-alpha01]: https://github.com/pedronahum/tlaloc/releases/tag/v0.1.0-alpha01
