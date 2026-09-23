# Getting started with Tlaloc

Differentiable Kotlin, end to end: you write a plain Kotlin lambda, the
Tlaloc K2 compiler plugin rewrites `grad { }` calls into synthesized
gradient code at compile time, and misuse (mismatched named axes,
undifferentiable bodies) is a **compile error carrying the offending call's
own file, line and column** — not a runtime crash. An IDE running Kotlin's
K2 analysis executes the same FIR checkers, so you should also see a redline
there while you type; that part is expected and is the one thing here no test
covers (`DiagnosticSourcePositionTest` pins the positions, nothing drives an
IDE).

> Alpha — `0.1.0-alpha01`. Artifacts are not yet on Maven Central — consume via
> `mavenLocal()` from a repo checkout. Coordinates and APIs may change without a
> deprecation cycle; [COMPATIBILITY.md](COMPATIBILITY.md) says exactly what may
> break and what will not, and [CHANGELOG.md](../CHANGELOG.md) records what did.

## 0. What you need, per module

Since §0.4.503 the JDK floor is a **per-module fact**, not one number.

| | Bytecode | Why |
|---|---|---|
| `core` `ir` `autograd` `nn` `stablehlo` `maestro` | **Java 21** | compiled with `-Xjdk-release=21`, so "no JDK 22+ API" is a compile-time check, not a hope |
| `runtime-pjrt` `runtime-cuda` `kptx` | Java 25 | the Foreign Function & Memory API, stable in [JEP 454](https://openjdk.org/jeps/454) — this is the *original* reason Tlaloc moved to 25 in §0.4.311 |
| `runtime-iree` | Java 25 | no FFM; it *could* be lowered, but nothing certifies an IREE run on a JDK 21 and an uncertified claim is worse than a high floor |
| `compiler-plugin` | Java 25 | it reads K2 compiler internals, and Kotlin loads a plugin into the compiler's own JVM |

Two sentences follow from that, and the second one is the one people get wrong:

1. **To RUN** Tlaloc code — `grad { }` gradients, `:nn` layers and optimizers,
   StableHLO emission — a **JDK 21** is enough. PJRT / CUDA *execution* needs 25.
2. **To BUILD** code containing `grad { }` you need a **JDK 25 on the build
   machine**, because the K2 plugin is 25 bytecode and runs inside the Kotlin
   compiler's JVM. Lowering your own `jvmTarget` does not change that.

So the supported consumer configuration is:

```kotlin
kotlin {
    jvmToolchain(25)                                   // to BUILD
    compilerOptions { jvmTarget.set(JvmTarget.JVM_21) } // to RUN on 21
}
java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
}
```

[examples/quickstart](../examples/quickstart) is exactly that, and
`bash scripts/jdk21-smoke.sh` (with `JDK21_HOME` set to a JDK 21) publishes,
compiles it at target 21 and runs the synthesized gradient on a real JDK 21.

**Kotlin: 2.3.20 through 2.3.29.** The plugin refuses anything else *by name* at
compile time, naming both the version it found and the range, because it reads
40 `org.jetbrains.kotlin.fir.*` packages of unstable K2 internals. Kotlin numbers
feature releases by tens in the third component (2.3.0, 2.3.10, 2.3.20 are three
different feature releases) and bugfixes by ones above them — so the whole bugfix
family of 2.3.20 is supported and nothing else is. `-P
plugin:io.tlaloc.plugin:unsafeAllowUnsupportedKotlin=true` downgrades the refusal
to a warning if you want to try it anyway.

**Symja is optional.** `org.matheclipse:matheclipse-core` (**LGPL-3.0**, 8.3 MB)
used to be a mandatory runtime dependency of `io.github.pedronahum:tlaloc-ir`. It is `compileOnly`
now, so it is not in your dependency graph unless you put it there. You need it
only to differentiate a loop whose trip count is not a compile-time constant — a
`for` loop over a `const val` bound is unrolled with no CAS at all. On the day a
body genuinely needs it, the compiler refuses by name and prints the one line to
add. Note *where* it goes: the CAS runs inside the Kotlin compiler, not inside
your program, so the line is
`kotlinCompilerPluginClasspath("org.matheclipse:matheclipse-core:3.1.1")` — an
`implementation` dependency is on the wrong classpath and the plugin will not see
it. Tlaloc only *links* Symja across the `SymbolicEngine` interface; it does not
modify or redistribute it, which is what keeps an LGPL-3.0 dependency compatible
with Tlaloc's Apache-2.0 licence.

## 1. Publish the artifacts locally

```bash
git clone <tlaloc repo> && cd tlaloc
./gradlew publishToMavenLocal -x test
```

All **thirteen** published modules land under `io.github.pedronahum:*:0.1.0-alpha01`:
`tlaloc-core`, `tlaloc-ir`, `tlaloc-autograd`, **`tlaloc-nn`**, `tlaloc-stablehlo`,
`tlaloc-compiler-plugin`, `tlaloc-runtime-pjrt`, `tlaloc-runtime-cuda`,
`tlaloc-runtime-iree`, `tlaloc-kptx`, `tlaloc-maestro`, `tlaloc-gradle-plugin`
(plugin id `io.github.pedronahum.tlaloc`) and `tlaloc-bom`. `:benchmarks` is a
harness and publishes nothing.

## 2. Set up a consumer project

`settings.gradle.kts` needs `mavenLocal()` for plugins as well as for
dependencies. The Tlaloc Gradle plugin is published to Maven, not to the Gradle
Plugin Portal, so after a release the plugin repository is `mavenCentral()`:

```kotlin
pluginManagement { repositories { mavenLocal(); mavenCentral(); gradlePluginPortal() } }
dependencyResolutionManagement { repositories { mavenLocal(); mavenCentral() } }
```

`build.gradle.kts`:

```kotlin
plugins {
    kotlin("jvm") version "2.3.20"
    // Puts the K2 compiler plugin (same version) on every Kotlin/JVM compilation:
    // compile-time grad rewriting + compile-time errors.
    id("io.github.pedronahum.tlaloc") version "0.1.0-alpha01"
    application
}

dependencies {
    implementation(platform("io.github.pedronahum:tlaloc-bom:0.1.0-alpha01"))
    implementation("io.github.pedronahum:tlaloc-core")
    implementation("io.github.pedronahum:tlaloc-ir")
    implementation("io.github.pedronahum:tlaloc-autograd")
}

// Optional. Each property is a compiler-plugin option; the values shown are the defaults.
tlaloc {
    strictLowering.set(true)
    dumpGradSource.set(false)
    // dumpGradSourceDir.set(layout.buildDirectory.dir("gradients"))
    dumpLoweredIr.set(false)
    unsafeAllowUnsupportedKotlin.set(false)
}
```

The plugin works with `kotlin("jvm")` and `kotlin("multiplatform")`, declared in
the same `plugins { }` block as the Tlaloc plugin or in a parent project's. In a
Multiplatform build it skips Kotlin/JS, Native, Wasm and Android compilations with a
warning naming each one. A project that applies neither Kotlin plugin is refused.

**Without the Gradle plugin.** Put the compiler plugin on a compilation's plugin
classpath and pass options as `-P` arguments.
`kotlinCompilerPluginClasspath(...)` reaches every compilation of the project;
`kotlinCompilerPluginClasspathMain(...)` only `main`:

```kotlin
dependencies {
    kotlinCompilerPluginClasspath("io.github.pedronahum:tlaloc-compiler-plugin:0.1.0-alpha01")
}
tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
    compilerOptions.freeCompilerArgs.addAll("-P", "plugin:io.tlaloc.plugin:strictLowering=true")
}
```

[examples/readable-gradients](../examples/readable-gradients) uses this route,
because one of its source sets must compile without the plugin.

The complete working project is [examples/quickstart](../examples/quickstart)
— run it from the repo root with `scripts/onboarding-smoke.sh` (or
`./gradlew -p examples/quickstart run` after step 1).

## 3. Your first gradient

```kotlin
import io.tlaloc.autograd.grad
import io.tlaloc.core.*
import io.tlaloc.core.ops.*

val g = grad { a: DTensor<Rank2<Sym, Sym>, F32> -> (a matmul a).sum().toFloat() }
val gradient = g(Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f)))
// ones·Aᵀ + Aᵀ·ones — computed by code synthesized at compile time.
```

Without the plugin on the compiler classpath, `grad { }` throws at the
call site with instructions (the tape API
`io.tlaloc.autograd.gradWithScalars` works plugin-free for scalars).

### What the body may reference

The lambda's own parameters and its local `val`s / `var`s, plus — since
§0.4.500 — any captured reference the compiler can resolve to a
**compile-time constant**: a `const val` declared anywhere (top level, file
level, or in a companion / named `object`), a `const val` whose initializer is
itself constant arithmetic, and a top-level or enclosing-function `val` whose
initializer the compiler can fold. An `Int` constant works as a `for` loop's
trip count. Such a capture is folded into the lowered IR as *exactly* the
constant an inline literal would have produced, so the gradient is identical
either way.

Since §0.4.501 a captured **runtime** value works as well — an immutable
local `val`, or a parameter of the enclosing function, typed `Float`,
`Double`, `Int` or `Long`:

```kotlin
val scale = computeScale()                      // not a compile-time constant
val g = grad { x: Float -> x * x * scale }      // lowers; scale is read at the call
```

It becomes a trailing parameter of the derived gradient function, and the
plugin binds it at the call site by reading the declaration your lambda closed
over. **The returned function's arity is unchanged**: a captured value is an
input, never a differentiation target, so `grad` still returns one gradient,
`grad2` a `Pair` and `grad3` a `Triple`.

What still refuses, by name and with the reason:

```
e: Tlaloc could not lower this lambda at compile time: captured value 'gain' is
   not a compile-time constant (it is a `var`, and a captured runtime value must
   be immutable — the gradient is derived where the lambda is written but runs
   where it is called, so a mutable capture has no single value to bind. Declare
   'gain' as a `val`) — …
```

That covers a `var`; a top-level or member property (reading one is a getter
CALL, not a value declaration the gradient can close over — copy it into a local
`val` first); a captured value of any other type, including a tensor; and the
forward-mode, assembly and seeded-cotangent intrinsics (`jvp`, `jacobian`,
`hessian`, `vjp` and their arity-2 spellings), which build their own parameter
lists out of the lowered one. See [ALPHA_PLAN.md](ALPHA_PLAN.md).

### Reading the gradient the compiler derived

Tlaloc's north star is Tangent's: the derivative is code you can read.
Two plugin options print, for every `grad {}` / `valueAndGrad {}` lambda
the plugin synthesizes, the reverse-transformed gradient as Kotlin
source over the `:core` host ops — the SAME DXIR function the synthesis
then compiles to bytecode, so what you read is what runs:

```
-P plugin:io.tlaloc.plugin:dumpGradSource=true      # compiler INFO message per lambda
-P plugin:io.tlaloc.plugin:dumpGradSourceDir=<dir>  # …plus one .kt file per lambda
```

Each dump is headed by the lambda's source location, and the dumped
`.kt` file compiles and runs standalone — bit-identical to the compiled
gradient (pinned in `DumpGradSourceTest`). Scalar lambdas render today;
tensor `grad {}` lambdas (whose shapes are symbolic at compile time)
print a named "dump SKIPPED" note instead of unprintable source. The
runtime capture route (`CapturedStep.gradSource()`) renders tensor
gradients too — see `docs/AD_SINGLE_ENGINE_AUDIT.md`.

## 4. Compile-time safety

Named axes live in the type system, and the checker runs on every
compilation of the lambda:

- `contract` over operands sharing no named axis → `NAMED_INDEX_MISMATCH`
  **error**, reported at the call's own line and column, so the build fails
  there (pinned by `DiagnosticSourcePositionTest`). IntelliJ in K2 mode runs
  the same checker and should redline the call as you type — expected, and
  untested: nothing in this repository drives an IDE.
- A body the reverse-mode transform cannot differentiate →
  `NOT_DIFFERENTIABLE` error, with the transform's reason verbatim.
- Concrete-dim shape violations → `TENSOR_SHAPE_MISMATCH` error.
- A body the plugin cannot **lower** at all → `Tlaloc could not lower this
  lambda at compile time: <reason>` **error**, carrying the lowering's own
  verbatim reason (§0.4.499). Before 0.1.0-alpha01 this was a warning and
  the program then threw `IllegalStateException` at the first call to the
  returned function; the information is the same, hours earlier. Opt out
  with `strictLowering=false` below if you want that late failure back.

## 4a. Plugin options

All **five** are properties of the Gradle plugin's `tlaloc { }` block, under the
same names (`tlaloc { strictLowering.set(false) }`). Without the Gradle plugin they
are `-P plugin:io.tlaloc.plugin:<name>=<value>` on the Kotlin compile task
(`compilerOptions.freeCompilerArgs`). A
misspelled boolean value is refused by name rather than read as `false` — for
the four added since §0.4.499; `dumpGradSource` predates that and still reads
an unknown value as `false` (named in `docs/ALPHA_PLAN.md`):

| option | default | what it does |
| --- | --- | --- |
| `strictLowering` | `true` | An unlowerable `grad {}` / `jvp {}` / `vjp {}` lambda is a compile **error**. `false` restores the pre-0.1.0-alpha01 warning plus a runtime `IllegalStateException`. |
| `dumpLoweredIr` | `false` | Dump the lowered Tlaloc IR for every recognised intrinsic lambda: one WARNING from the FIR checker, one INFO from the IR extension. **Developer introspection.** It is off by default because a consumer's build log is not the place for it — and because, being warnings, the two dumps used to break every build compiling with `-Werror`. Do not combine this option with `-Werror`: K2's diagnostic DSL has no INFO severity, so the FIR half is necessarily a warning. |
| `dumpGradSource` | `false` | Print each synthesized gradient as readable Kotlin (INFO). See above. |
| `dumpGradSourceDir` | — | Like `dumpGradSource`, and also write one `.kt` per lambda into this directory. |
| `unsafeAllowUnsupportedKotlin` | `false` | Downgrade `KotlinVersionGuard`'s refusal (§0.4.503) from an ERROR to a warning and try the plugin on a Kotlin outside 2.3.20–2.3.29 anyway. It is spelled "unsafe" because it is: the plugin reads 40 packages of internal K2 API. |

A build with none of these set and a `grad {}` that lowers is **silent**:
Tlaloc says nothing at all. That is pinned by `DiagnosticNoiseTest`, which
compiles a `grad {}` consumer with `-Werror` and asserts both that it
succeeds and that no message mentions Tlaloc.

Try it: uncomment the `bad` block at the bottom of the quickstart's
`Main.kt`.

## 4b. `@ExperimentalTlalocApi` — the part of the surface that is provisional

`0.1.0-alpha01` says every API may change without a deprecation cycle, and that
is true of `grad` (thousands of oracle tests behind it) and of a scope taxonomy
nothing has ever executed, which makes it useless as a signal. Since §0.4.505
three surfaces carry a `@RequiresOptIn(ERROR)` marker, `io.tlaloc.core.ExperimentalTlalocApi`,
and touching one without opting in is a compile error naming the marker:

| Marked | Why |
|---|---|
| The four-worlds scopes — `KernelScope`, `OrchestrationScope`, `ProgramScope`, `ClusterScope`, `Tlaloc`, `BufferHandle`, `HandleRef` | Their own KDoc scopes them to "v1 keeps each op single-scope" and names Kotlin's context parameters as where a multi-scope op would send the whole design |
| `io.tlaloc.ir.AllReduceAttrs` | Distributed execution is 📐 in [CAPABILITIES.md](CAPABILITIES.md) — designed, never run on two hosts — and v1 supports `"sum"` only |
| `io.tlaloc.ir.recognizer.kernel` and `io.tlaloc.ir.recognizer.cost` | The machinery is unit-certified; what it is *for* is picking a kernel, and the one kernel of ours measured against XLA lost at small shapes ([KPTX_PAGED_PERF.md](KPTX_PAGED_PERF.md)) |

`grad`, the tensor and op surface, `:nn`'s layers, optimizers and schedules,
safetensors, StableHLO emission and the PJRT/IREE runtimes **are not marked**,
deliberately. A marker on everything teaches you to add `-opt-in=` once and stop
reading it.

```kotlin
@file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)   // per file
```

```kotlin
kotlin { compilerOptions { optIn.add("io.tlaloc.core.ExperimentalTlalocApi") } }  // per module
```

[examples/internals/four-worlds](../examples/internals/four-worlds) and
[examples/internals/layer3](../examples/internals/layer3) each carry the file-level
form with a comment saying why — they are the two examples that consume a marked
surface, and they are how the marker is checked from outside this repository.

## 5. Running on real hardware

The quickstart's gradient runs on the JVM. The same DXIR programs
dispatch to GPUs through:

- **`io.github.pedronahum:tlaloc-runtime-pjrt`** — PJRT-XLA via pure-Kotlin FFM (no JNI,
  no Python at runtime); F32 and F64. See `PjrtSession`.
- **`io.github.pedronahum:tlaloc-runtime-iree`** — IREE CPU/CUDA via subprocess facade.
- **`io.github.pedronahum:tlaloc-kptx`** — hand-written/DSL PTX kernels inside XLA
  executables (the escape-hatch tier; see `docs/KPTX_PLAN.md`).

Both GPU runtimes need their toolchains (a JAX-bundled PJRT plugin or
IREE binaries) — see the module KDocs and README "Requirements".

## 6. Where to go next

- [examples/](../examples/) — **ten** standalone runnable projects, each its own
  Gradle build resolving Tlaloc from `mavenLocal`, indexed with the verbatim
  output of every one in [examples/README.md](../examples/README.md). Seven at
  the top level — `quickstart`, `readable-gradients`, `differentiable-physics`,
  `named-indices`, `mnist`, `gpu-training`, `gpu-inference` — and three under
  `examples/internals/`: `layer3` (kernel selection), `four-worlds` (the scope
  taxonomy) and `tpu` (the program written before the hardware). Six need
  nothing but a JDK. (This line said "eight" until §0.4.505, and named seven.)
- **API reference** — `./gradlew apiDocs` from the repository root writes an
  aggregated Dokka site for all eleven modules to `build/docs/api/index.html`.
  Nothing is hosted.
- [DIFFKTX_SPEC.md](../DIFFKTX_SPEC.md) — the book of work; §0.4 is the
  session-by-session ship log.
- [README](../README.md) — architecture, layer map, benchmarks.
