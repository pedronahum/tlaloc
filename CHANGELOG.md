# Changelog

All notable changes to Tlaloc are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) **with the alpha
carve-outs written down in [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)** —
read that before depending on a coordinate.

A note on where the history actually lives: this file starts at
`0.1.0-alpha01`. The 497 sections of work before it are recorded in commit
titles (`§0.4.NNN <area>: <what turned out to be true>`) and, up to §0.4.311, in
[`DIFFKTX_SPEC.md`](DIFFKTX_SPEC.md). This file does not attempt to
retro-summarise them; it is the record from the first named version forward.

## [Unreleased]

### Added

- **`@ExperimentalTlalocApi` — an opt-in marker on the part of the surface that is
  provisional.** `docs/COMPATIBILITY.md` promised that every alpha API may change
  without a deprecation cycle, which graded `grad` (thousands of oracle tests) and
  a scope taxonomy nothing has ever executed exactly the same. A
  `@RequiresOptIn(ERROR)` marker in `:core` now separates them. Three surfaces
  carry it, each for a stated reason: the **four-worlds scope taxonomy**
  (`KernelScope`, `OrchestrationScope`, `ProgramScope`, `ClusterScope`, `Tlaloc`,
  `BufferHandle`, `HandleRef` — its own KDoc scopes it to "v1 keeps each op
  single-scope"); **`io.tlaloc.ir.AllReduceAttrs`** (distributed execution is 📐 and
  has never run on two hosts; v1 is `"sum"` only); and the **kernel-choice and
  cost-model packages** (`io.tlaloc.ir.recognizer.kernel`,
  `io.tlaloc.ir.recognizer.cost` — unit-certified machinery whose purpose is
  picking a kernel, and the one kernel measured against XLA lost at small shapes).
  `grad`, the op surface and `:nn` are deliberately **not** marked. Two tests pin
  it: `ExperimentalTlalocApiTest` reads the marker and the marked/unmarked sets out
  of the class **files** (BINARY retention is invisible to reflection, which the
  first attempt at that test discovered the hard way), and
  `ExperimentalApiOptInTest` runs a real `K2JVMCompiler` with no `-opt-in` and
  asserts the refusal, the `@OptIn` fix, and that a certified surface is unaffected.
- **A binary-compatibility baseline.** `api/<module>.api` is committed and
  `./gradlew apiCheck` (wired into `check`, so `./gradlew test` covers it) fails on
  any difference; `./gradlew apiDump` re-baselines. Negative-tested by adding a
  public function to `:stablehlo` and watching the check fail with the diff.
  **It covers six of eleven modules**: binary-compatibility-validator 0.18.2's ABI
  reader refuses Java 25 bytecode (`Unsupported class file major version 69`), so
  the six modules §0.4.503 lowered to Java 21 are validated and the five that stay
  at 25 are not. The ignore list is derived from `tlalocJvmTargets`, so lowering a
  module to 21 later starts validating it automatically.
- **An API reference you can actually read.** `./gradlew apiDocs` aggregates all
  eleven published modules into one Dokka site at `build/docs/api/index.html` —
  2,898 pages. Before this, the Dokka HTML §0.4.498 wired existed only inside
  eleven separate `-javadoc.jar` files. Not hosted. The 206 unresolved-KDoc-link
  warnings it emits (110 distinct targets, 117 of them in `:ir`) are now **counted**
  — Tier 0 recorded them as uncounted — and still unswept.
- **The compile-error position is certified.** `DiagnosticSourcePositionTest`
  asserts that `LAMBDA_NOT_LOWERABLE` and `NAMED_INDEX_MISMATCH` report at the
  offending call's own file, line and column, and that no Tlaloc error is ever
  emitted without a position. Eighteen test classes already pinned the diagnostic
  *text*; not one had looked at `location`.

### Changed

- **The "red squiggle in the IDE" claim now matches the evidence.** The README and
  `docs/GETTING_STARTED.md` both promised an IDE redline and nothing tested it.
  They now claim what is certified — a build failure at the offending call's file,
  line and column — and state the IDE behaviour as *expected, untested*, with a
  🧪 row in `docs/CAPABILITIES.md` saying so. The capability is not removed; the
  claim is.
- **`DIFFKTX_SPEC.md` §14, §17 and §18 stopped describing a repository that no
  longer exists.** §17's ladder had been swept on 2026-04-19 and then left for 400
  sections: the **IREE runtime** (step 5), the **PJRT runtime** (step 10) and the
  **whole coarsening block** (steps 13–16) were marked ⬜ NOT DONE for capabilities
  that are certified. Each is now marked with the section that shipped it, cited
  only where `git log` or the file's own §0.4 era table can establish the number.
  **One mark went the other way:** step 16 — "gate coarsening release on matching
  the paper's speedups within 20%" — is 🟡, not ✅. All six benchmarks are ported
  and all six are harness inhabitants, but the PyTorch/JAX comparison has never
  been run and no §0.4 entry has published the verdict, so coarsening's
  *correctness* is certified and its *speedup relative to the paper* is not. §14
  had four lines that were simply false (ktlint + detekt "enforced in CI" — neither
  exists; Robolectric; kotlinx-benchmark; a pinned MLIR commit) and §18 carried
  seven open questions HEAD had already answered, the licence among them.
- **Two documented counts that disagreed with each other and with the build.**
  `docs/GETTING_STARTED.md` said "eight standalone runnable projects" where the
  README said ten — there are **ten**, seven at the top level and three under
  `examples/internals/`, each with its own `settings.gradle.kts`. The same file's
  list of published modules omitted **`nn`**, which is the module a reader most
  likely wants, and called the plugin options "four" when §0.4.503 had added a
  fifth (`unsafeAllowUnsupportedKotlin`). The README's test count said 2,345 while
  `docs/CAPABILITIES.md` said 2,509 and the suite was at neither.
- **The README's own License section was contradicting its Requirements section.**
  It said `ir-jvm` "carries Symja at runtime scope, with no supported way to opt
  out yet" — which §0.4.503 had made false two commits earlier by moving Symja to
  `compileOnly`. Both places now say the same thing, and the part that is still
  true (no Symja-free `SymbolicEngine` exists, so a policy that forbids LGPL-3.0
  outright leaves you without a CAS) is stated as the residual risk it is.

- **The JDK floor is per module now, and it is 21 for the library.** Every module
  used to emit Java 25 bytecode, which made JDK 25 a hard requirement for every
  consumer of `:core`, `:nn` or `:stablehlo`. §0.4.311 adopted JDK 25 for one
  reason — the Foreign Function & Memory API went stable in JEP 454 and Tlaloc's
  PJRT and CUDA bindings are FFM — and that reason applies to three modules.
  `core`, `ir`, `autograd`, `nn`, `stablehlo` and `maestro` now emit **Java 21**
  and are compiled with `-Xjdk-release=21`, so "no JDK 22+ API" is a compile-time
  check rather than an assumption; `runtime-pjrt`, `runtime-cuda`, `kptx`,
  `runtime-iree` and `compiler-plugin` stay at 25. A new `verifyJvmTarget` task
  per module, wired into `check`, reads the class-file major version out of every
  published jar, so the table cannot drift from the bytecode.
  **Read it as two sentences:** to *run* Tlaloc — `grad { }`, `:nn`, StableHLO
  emission — a JDK 21 is enough; to *build* code containing `grad { }` you still
  need a JDK 25 on the build machine, because Kotlin loads a compiler plugin into
  the compiler's own JVM and the plugin is 25 bytecode. The supported consumer
  configuration is `jvmToolchain(25)` + `jvmTarget = JVM_21`, which
  `examples/quickstart` now is, and `scripts/jdk21-smoke.sh` runs that
  configuration's synthesized gradient on a real JDK 21.
- **Symja is an optional dependency.** `org.matheclipse:matheclipse-core` — 8.3 MB,
  **LGPL-3.0**, with its own transitive tree — was a mandatory *runtime* dependency
  of `io.tlaloc:ir`, and therefore of `:autograd`, `:nn` and `:stablehlo`, whether
  or not a program ever differentiated a loop-bearing body. It is `compileOnly`
  now and no longer appears in the published POM. Add
  `implementation("org.matheclipse:matheclipse-core:3.1.1")` only if you need the
  computer algebra system; you need it to differentiate a loop whose trip count is
  not a compile-time constant (a `for` over a `const val` bound is unrolled with
  no CAS at all), and on the day a body genuinely needs it the compiler refuses
  **by name**, naming the coordinate, the licence and the one line to add. Tlaloc
  only links Symja across the `SymbolicEngine` interface — it does not modify or
  redistribute it — which is what keeps an LGPL-3.0 dependency compatible with
  Tlaloc's Apache-2.0 licence.
- **The PJRT plugin search is no longer one developer's path.**
  `PjrtBinaries.pluginPath` fell back to the literal
  `~/.local/venvs/iree/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so`
  — one venv name, one Python minor version, one plugin package. It now globs all
  three across `$VIRTUAL_ENV`, `~/.local/venvs/*`, `~/.venv`, `~/venv`,
  `~/.local`, `/usr/local` and `/usr`, over `lib` and `lib64`, `site-packages` and
  `dist-packages`, and any `jax_plugins/*cuda*/*.so`. `TLALOC_PJRT_PLUGIN_PATH`
  still wins outright. Resolution is deterministic (every directory listing is
  sorted), and the new `PjrtBinaries.pluginSearchReport` names **every** location
  tried and what was at each — which is what the examples' "GPU lane unavailable"
  reason now prints, instead of advice that told a user who *had* installed a
  plugin nothing at all.

### Added

- **CI is four lanes now, and two of them exist to catch what one machine cannot
  see.** Every ✅ in `docs/CAPABILITIES.md` had been certified on one host — an
  NVIDIA GB10, aarch64, JDK 25 — and CI was a single `ubuntu-latest` job running
  `./gradlew test`. `.github/workflows/build.yml` now runs that suite on **x86_64
  Linux and arm64 macOS**, plus a `library-jdk21` job that RUNS the six
  Java-21-targeted modules' own suites on a JDK 21 and then compiles and runs a
  synthesized gradient there; `.github/workflows/kotlin-next.yml` probes the next
  published Kotlin (resolved from Maven Central, not pinned) and is allowed to
  fail, which is the mitigation `DIFFKTX_SPEC.md` §16 named for "K2 plugin API
  changes across Kotlin releases" and nobody had implemented. Two new build
  properties make those lanes possible and both are usable by hand:
  `-PtlalocTestJdk=21` points every `Test` task's launcher at another JDK (and
  refuses by name for a module whose own bytecode target is higher), and
  `-PtlalocKotlinVersion=<v>` swaps the Kotlin compiler the whole build runs.
  **The lanes themselves have never run** — they were written on a machine with no
  access to GitHub Actions — so `docs/ALPHA_PLAN.md` marks them 🧪 and nothing
  claims a green run. What DID run, locally: the library's 1,822 tests on OpenJDK
  21.0.2, and the Kotlin probe against 2.4.20 and 2.3.10 — which found that
  **Kotlin 2.4.20 does not compile `:compiler-plugin`** (one error, direct
  `MessageCollector` access in `TlalocCompilerPluginRegistrar.kt`) and that the
  §0.4.503 version guard refuses a foreign compiler by name from inside a real
  2.3.10 compile.

- **The compiler plugin refuses an unsupported Kotlin version by name.** The
  plugin reads 40 `org.jetbrains.kotlin.fir.*` packages of internal K2 API that
  JetBrains moves between feature releases, and nothing checked which compiler it
  was running inside: a user on 2.2.x or 2.4.x got a raw `NoSuchMethodError` from
  the middle of `compileKotlin`, naming JetBrains classes and never Tlaloc. The
  new `KotlinVersionGuard` runs before a single extension is registered and
  reports a compile ERROR naming the version it found, the version it was built
  against, the supported range and the opt-out. Supported: **2.3.20 through
  2.3.29** — the whole bugfix family of the feature release the plugin was built
  against, because Kotlin numbers feature releases by tens in the third component
  (2.3.0, 2.3.10 and 2.3.20 have different internals) and bugfixes by ones above
  them. `-P plugin:io.tlaloc.plugin:unsafeAllowUnsupportedKotlin=true` downgrades
  the refusal to a warning and registers anyway; the warning says in as many words
  that a crash below it is then the expected outcome.

- **A trained model can be saved and loaded.** Tlaloc could train on a GPU and
  could not persist the result: `:nn`'s `Components.kt` said so in a comment
  ("minus `store`/`load`, out of scope v1") and `:core`'s safetensors support was
  read-only. `:core` now has a **safetensors writer**
  (`SafetensorsWriter.encode`, plus an atomically-renaming
  `SafetensorsFileWriter` on the JVM) covering F32, F64, I32 and BF16 and
  refusing every other dtype by name, and `:nn` has `ModelCheckpoint` /
  `saveCheckpoint` / `loadCheckpoint`, which put a model's parameters, its
  non-trainable buffers and its optimizer's state into **one safetensors file**.
  Loading returns a NEW model (`ModelSnapshot.restore`), because layers are
  immutable here. The round trip is bit-identical — parameters, predictions and
  gradients — and a run resumed from a checkpoint produces bit-identical losses
  and parameters for the next 15 steps. The file is an ordinary safetensors file:
  `safetensors.torch.load_file` opens it, and the writer is certified in both
  directions against the reference Python library on raw bytes.
- **Non-trainable persistent state has a home.** New `Stateful<T>` interface in
  `:nn`, implemented by `BatchNorm` (its running statistics) and by `Sequential`
  (which forwards its children's, with the same `"<index>."` prefix it uses for
  parameters). Deliberately separate from `Trainable`, whose `parameters` list is
  the one the reverse transform returns a gradient per — running statistics have
  no gradient.
- **Learning-rate schedules**: `ConstantLR`, `StepDecay`, `ExponentialDecay`
  (continuous or staircase), `CosineDecay`, `LinearWarmup`, and a `Scheduled`
  optimizer combinator that applies any of them to any optimizer. A schedule is a
  pure function of the completed-step count, so the step count is checkpointed
  state and a resumed run continues the schedule rather than restarting it. The
  three PyTorch also ships agree with `StepLR` / `ExponentialLR` /
  `CosineAnnealingLR` to 2.8e-7 relative over 34 rates.
- **Gradient clipping**: `GradientClipping.globalNorm`, `byGlobalNorm` and
  `byValue`, pure functions on the gradient map. `byValue` agrees with torch's
  `clip_grad_value_` exactly; `byGlobalNorm` uses the exact `maxNorm / ‖g‖` ratio
  where torch uses `maxNorm / (‖g‖ + 1e-6)`, a recorded ~1.6e-7 divergence. A
  non-finite gradient norm is refused by name instead of being scaled into zeros
  or NaNs.
- **`examples/gpu-training` now saves, reloads and keeps going.** The decision
  boundary and the held-out accuracy it prints are computed by the model that came
  back off the disk; observed 337/337 parameter scalars and all 1024 predictions
  bit-identical on both the CUDA and the host lane.

- **A `grad {}` body can reference a compile-time constant declared outside it.**
  Until now a `grad {}` lambda could reference *nothing* outside itself: the
  lowering resolved property accesses against its own environment (lambda
  parameters and lambda-local `val`s) and every other reference — a top-level
  `const val` included — was refused with `reference to symbol outside the
  lowering scope`. A captured reference the compiler can resolve to a constant is
  now folded into the lowered IR as **exactly** the `DxirConst` an inline literal
  would have produced, so the reverse transform, the φ-calculus coarsening, the
  synthesized bytecode and the printed gradient source cannot tell the two
  spellings apart. What folds: any `const val` (top level, file level, or in a
  companion / named `object`), a `const val` whose own initializer is constant
  arithmetic, and a top-level or enclosing-function `val` whose initializer the
  compiler can fold — including an `Int` constant used as a `for` loop's trip
  count, which takes the same unrolled path a literal bound does rather than the
  symbolic-trip-count one. `examples/differentiable-physics` now uses its own
  `const val`s, and the derivative the compiler writes for it is byte-identical
  to the one it wrote from the inlined literals.
- **A captured RUNTIME value refuses with its own wording**, distinct from the old
  generic out-of-scope sentence: a `var`, a computed `val`, a parameter of the
  enclosing function or a non-`const` member property is named, the reason it is
  not foldable is named, and the two ways out (`const val`, or a lambda
  parameter) are named.
- **A `grad {}` body can reference a RUNTIME value declared outside it.** This is
  the other half of the same arc:

  ```kotlin
  val scale = computeScale()                     // not a compile-time constant
  val g = grad { x: Float -> f(x) * scale }      // now lowers
  ```

  The captured value becomes a trailing parameter of the lowered gradient
  function, and the IR phase binds it at the call site by reading the very
  declaration the user's own lambda closed over — so the synthesized gradient
  closes over it the same way, and reads it when it is *called*, not when it was
  derived. **The returned function's arity does not change**: a captured value is
  an input, never a differentiation target, so `grad` still returns one gradient,
  `grad2` a `Pair` and `grad3` a `Triple`. What is supported, and certified by
  equivalence against the same body written with that value as an explicit lambda
  parameter: an immutable local `val` or a parameter of the enclosing function, of
  type `Float`, `Double`, `Int` or `Long`, under the reverse-mode `grad` family.
  What still refuses, by name: a `var` (no single value to bind), a top-level or
  member property (reading one is a getter call, not a value declaration), any
  other type, and the forward / assembly / seeded-cotangent intrinsics, whose own
  parameter lists are rebuilt from the lowered one. See
  [docs/ALPHA_PLAN.md](docs/ALPHA_PLAN.md).

### Changed

- **`CosineDecay` clamps past `decaySteps`, where PyTorch's
  `CosineAnnealingLR` is periodic and climbs back toward the initial rate.** This
  is a deliberate divergence and it is certified as one: a test asserts both that
  torch climbs and that Tlaloc holds. Warm restarts would be their own schedule.
- **A working build is silent.** Every recognised `grad {}` used to emit two
  compiler **warnings**, each dumping the lowered Tlaloc IR into the consumer's
  build output: the FIR checker's `LAMBDA_LOWERED` and the IR extension's "saw
  handoff". They were developer introspection, and because they were warnings
  they also **broke the build outright** in any project compiling with
  `allWarningsAsErrors = true` (`e: warnings found and -Werror specified`) — on
  a program that was entirely correct. Both are now off by default; the IR half
  is an `INFO` rather than a `WARNING`. Turn them back on with
  `-P plugin:io.tlaloc.plugin:dumpLoweredIr=true`. `dumpGradSource` /
  `dumpGradSourceDir` are unchanged.
- **An unlowerable `grad {}` lambda is a compile-time error.** It was a warning;
  the call was then left unrewritten and `io.tlaloc.autograd`'s fallback body
  threw `IllegalStateException` at the **first call**, telling the user to add a
  compiler plugin that was already applied. The refusal now happens at the call
  site at compile time, carrying the lowering's own verbatim reason (one of the
  ~208 named `LoweringException` sites) and naming the opt-out. Opt out with
  `-P plugin:io.tlaloc.plugin:strictLowering=false`, which restores the warning
  and the late failure exactly.
- **`grad(::f)` and other non-lambda arguments are refused by name** instead of
  compiling green and throwing at the first call: the plugin lowers the body of
  a `{ }` written at the call site, and now says so.
- **The runtime message when a `grad {}` was not rewritten** no longer claims the
  compiler plugin is missing. It names the two states that can produce it — no
  plugin on the compile classpath, or a plugin that refused the body under
  `strictLowering=false` — and says where the compile-time reason is.

### Fixed

- **An f64 scalar body with a literal constant no longer kills the compiler.**
  `DxirReverseTransform`'s scalar constant folding built every folded constant
  from a `Float` projection of its operands, whatever the node's dtype was, so an
  f64 node folded to a constant typed `f64` carrying a `java.lang.Float`. Three
  phases later `DxirToIrSynthesis`'s `v as Double` threw a bare
  `ClassCastException` out of the K2 IR generation extension, with no Tlaloc
  diagnostic of any kind: `grad { x: Double -> x * 1.5 }` neither produced a
  gradient nor refused. The f64 arm of the fold now does its arithmetic in
  `Double`, so the value has the width its type claims. Found while writing the
  captured-constant tests above, on the *inlined-literal control*, and unrelated
  to captures.
- **The Tracer-capture route no longer draws a spurious refusal.**
  `io.tlaloc.autograd.grad` / `grad2` / `grad3` / `valueAndGrad*` are overloaded:
  the compile-time intrinsic and the runtime `Tracer` tape share every one of
  those names. The plugin's checker matched on the FQN alone, so a
  `grad { t: Tracer<...> -> ... }` call — the documented plugin-free route — got
  a `could not lower lambda: ... unsupported type io.tlaloc.autograd.Tracer`
  warning on every call. A Tracer-typed lambda is now recognised as the tape
  overload and not diagnosed at all. (Had it not been, promoting the refusal to
  an error would have broken that route outright.)

## [0.1.0-alpha01] — 2026-09-22

The first named version. The engine did not change in this release; the
*packaging* did, from "unpublishable" to "publishable but not yet published".

### Added

- **`LICENSE` — Apache-2.0.** The repository had no license file, which made it
  "all rights reserved" by default and contradicted its own "developed in the
  open" framing. Copyright 2026 Pedro N. Rodriguez. The README's `## License`
  section states the one dependency that needs a paragraph (Symja).
- **Maven Central metadata.** Every published POM now carries `<name>`,
  `<description>`, `<url>`, `<inceptionYear>`, `<licenses>`, `<developers>`,
  `<scm>` and `<issueManagement>`. Before this, they carried coordinates and
  dependencies only, which Central rejects.
- **`verifyPomMetadata`** — a `check`-wired Gradle task, per module, that reads
  the POMs the build actually generates and fails by name on a missing
  Central-mandatory element, a publication with no javadoc artifact, or a module
  with no sources jar. The "Central-ready" claim is a gate, not a sentence in a
  commit message.
- **Dokka + a javadoc jar per publication.** Dokka 2.2.0 (the tooling item
  `DIFFKTX_SPEC.md` §14 listed and nobody wired). The jar carries Dokka **HTML**
  under the `javadoc` classifier, because Dokka's Javadoc format does not support
  Kotlin Multiplatform projects and ten of the eleven modules are KMP.
- **A sources jar for `:compiler-plugin`.** The KMP modules got one from the
  multiplatform plugin; this module's `from(components["java"])` publication
  published the binary jar alone and could never have passed validation.
- **Signing and the Central deploy repository.** In-memory GPG key from
  `signingInMemoryKey` (Gradle property) or `SIGNING_IN_MEMORY_KEY` (environment);
  credentials from `centralUsername` / `centralPassword` or `CENTRAL_USERNAME` /
  `CENTRAL_PASSWORD`. Both are **no-ops when absent**, so a contributor's
  `publishToMavenLocal` and the GitHub build lane are unaffected.
- **`CHANGELOG.md`** (this file), **[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)**
  (what alpha promises and what it does not) and
  **[docs/RELEASING.md](docs/RELEASING.md)** (the release procedure).
- **[docs/ALPHA_PLAN.md](docs/ALPHA_PLAN.md)** — the ledger for the road to a
  usable alpha, with one row per item and what pins each.

### Changed

- **Version `0.0.1-SNAPSHOT` → `0.1.0-alpha01`**, across the root build, the
  README, `docs/GETTING_STARTED.md` and all ten standalone example projects.
  The old coordinate no longer resolves; republish with
  `./gradlew publishToMavenLocal -x test`.

### Fixed

- **Two stale license claims in the repository's own words.**
  `SymbolicEngine.kt`'s KDoc and `docs/STAGE_B_PLAN.md` §3.2.1's bullet list both
  still described Symja as Apache-2.0, five months after §0.4.13 corrected the
  rest of that document to LGPL-3.0. Both now say LGPL-3.0, and both now record
  the discrepancy §0.4.498 found: the *artifact's* POM says LGPL-3.0 while the
  *repository's* `license.txt` is plain GPL-3.0.

### Not done, deliberately

- **Nothing has been published to Maven Central.** There are no credentials on
  the machine this work was done on and a Central upload is irreversible. The
  wiring is correct and dry-runnable; it has never been run against Central, and
  `docs/ALPHA_PLAN.md` says so in the row that would otherwise claim it.

[Unreleased]: https://github.com/pedronahum/tlaloc/compare/main...HEAD
[0.1.0-alpha01]: https://github.com/pedronahum/tlaloc/releases/tag/v0.1.0-alpha01
