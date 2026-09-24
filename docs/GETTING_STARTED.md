# Getting started with Tlaloc

You write a plain Kotlin lambda, and the Tlaloc K2 compiler plugin rewrites
`grad { }` calls into gradient code at compile time. Misuse (mismatched named
axes, bodies it cannot differentiate) is a **compile error at the offending
call's own file, line and column**, not a runtime crash. An IDE running Kotlin's
K2 analysis executes the same FIR checkers, so it should show the error as you
type; `DiagnosticSourcePositionTest` pins the positions, and no test drives an
IDE.

> Alpha — `0.1.0-alpha01`, on Maven Central as `io.github.pedronahum:tlaloc-*`.
> Coordinates and APIs may change
> without a deprecation cycle. [COMPATIBILITY.md](COMPATIBILITY.md) lists what
> may break and what will not, and [CHANGELOG.md](../CHANGELOG.md) records what
> did.

## 0. What you need, per module

The JDK floor differs per module:

| | Bytecode | Why |
|---|---|---|
| `core` `ir` `autograd` `nn` `stablehlo` `maestro` | **Java 21** | compiled with `-Xjdk-release=21`, so no JDK 22+ API can slip in |
| `runtime-pjrt` `runtime-cuda` `kptx` | Java 25 | the Foreign Function & Memory API ([JEP 454](https://openjdk.org/jeps/454)) |
| `runtime-iree` | Java 25 | no test certifies an IREE run on a JDK 21 |
| `compiler-plugin` | Java 25 | it reads K2 compiler internals, and Kotlin loads a plugin into the compiler's own JVM |

1. **To run** Tlaloc code (`grad { }` gradients, `:nn` layers and optimizers,
   StableHLO emission), a **JDK 21** is enough. PJRT and CUDA execution need 25.
2. **To build** code containing `grad { }` you need a **JDK 25 on the build
   machine**, because the K2 plugin is Java 25 bytecode and runs inside the Kotlin
   compiler's JVM. Lowering your own `jvmTarget` does not change that.

The supported consumer configuration:

```kotlin
kotlin {
    jvmToolchain(25)                                    // to build
    compilerOptions { jvmTarget.set(JvmTarget.JVM_21) } // to run on 21
}
java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
}
```

[examples/quickstart](../examples/quickstart) uses exactly that, and
`bash scripts/jdk21-smoke.sh` (with `JDK21_HOME` set to a JDK 21) publishes,
compiles it at target 21 and runs the synthesized gradient on a real JDK 21.

**Kotlin: 2.3.20 through 2.3.29.** The plugin reads 40
`org.jetbrains.kotlin.fir.*` packages of unstable K2 internals, so it refuses any
other Kotlin at compile time, naming the version it found and the supported range.
Kotlin numbers feature releases by tens in the third component (2.3.0, 2.3.10,
2.3.20 are three feature releases) and bugfixes by ones above them, so the bugfix
family of 2.3.20 is supported and nothing else is. `unsafeAllowUnsupportedKotlin`
([section 4a](#4a-plugin-options)) turns the refusal into a warning. This is the
range of `0.1.0-alpha01`; the next release is built against Kotlin 2.4.20, and
[COMPATIBILITY.md](COMPATIBILITY.md) lists the range of each version.

**Gradle:** the Tlaloc Gradle plugin is tested with this repository's Gradle 9.7
wrapper and by hand with Gradle 8.14.3.

**Symja is optional.** `org.matheclipse:matheclipse-core` (**LGPL-3.0**, 8.3 MB)
is `compileOnly` in `tlaloc-ir`, so it is not in your dependency graph unless you
put it there. You need it only to differentiate a loop whose trip count is not a
compile-time constant; a `for` loop over a `const val` bound is unrolled with no
CAS. When a body needs it, the compiler refuses by name and prints the line to
add. The CAS runs inside the Kotlin compiler, not inside your program, so the line
is `kotlinCompilerPluginClasspath("org.matheclipse:matheclipse-core:3.2.0")`; an
`implementation` dependency is on the wrong classpath and the plugin will not see
it. Tlaloc links Symja across the `SymbolicEngine` interface and does not modify
or redistribute it.

## 1. The artifacts

All **thirteen** published modules are on Maven Central under
`io.github.pedronahum:*:0.1.0-alpha01`: `tlaloc-core`, `tlaloc-ir`,
`tlaloc-autograd`, `tlaloc-nn`, `tlaloc-stablehlo`, `tlaloc-compiler-plugin`,
`tlaloc-runtime-pjrt`, `tlaloc-runtime-cuda`, `tlaloc-runtime-iree`,
`tlaloc-kptx`, `tlaloc-maestro`, `tlaloc-gradle-plugin` (plugin id
`io.github.pedronahum.tlaloc`) and `tlaloc-bom`.

To build them from source instead (the examples in this repository do this):

```bash
git clone https://github.com/pedronahum/tlaloc && cd tlaloc
./gradlew publishToMavenLocal -x test
```

and put `mavenLocal()` first in both repository blocks below.

## 2. Set up a consumer project

The Tlaloc Gradle plugin is published to Maven Central, not to the Gradle Plugin
Portal, so `pluginManagement` needs `mavenCentral()` as well:

```kotlin
// settings.gradle.kts
pluginManagement { repositories { mavenCentral(); gradlePluginPortal() } }
dependencyResolutionManagement { repositories { mavenCentral() } }
```

`build.gradle.kts`:

```kotlin
// build.gradle.kts
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    kotlin("jvm") version "2.3.20"
    // Puts the K2 compiler plugin (same version) on every Kotlin/JVM compilation:
    // compile-time grad rewriting and compile-time errors.
    id("io.github.pedronahum.tlaloc") version "0.1.0-alpha01"
    application
}

kotlin {
    jvmToolchain(25)
    compilerOptions { jvmTarget.set(JvmTarget.JVM_21) }
}
java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
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
Multiplatform build it skips Kotlin/JS, Native, Wasm and Android compilations with
a warning naming each one. A project that applies neither Kotlin plugin is
refused.

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

The complete working project is [examples/quickstart](../examples/quickstart).
Run it from the repository root with `./gradlew -p examples/quickstart run` after
step 1, or with `bash scripts/onboarding-smoke.sh`, which publishes first.

## 3. Your first gradient

```kotlin
import io.tlaloc.autograd.grad
import io.tlaloc.core.*
import io.tlaloc.core.ops.*

val g = grad { a: DTensor<Rank2<Sym, Sym>, F32> -> (a matmul a).sum().toFloat() }
val gradient = g(Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f)))
// ones·Aᵀ + Aᵀ·ones, computed by code synthesized at compile time.
```

Without the plugin on the compiler classpath, `grad { }` throws at the call site
with instructions (see [Troubleshooting](#troubleshooting)). The Tracer-capture
API, `io.tlaloc.autograd.gradWithScalars`, computes gradients without the plugin.

### What the body may reference

The lambda's own parameters and its local `val`s and `var`s, plus any captured
reference the compiler can resolve to a **compile-time constant**: a `const val`
declared anywhere (top level, or in a companion or named `object`), a `const val`
whose initializer is constant arithmetic, and a top-level or enclosing-function
`val` whose initializer the compiler can fold. An `Int` constant works as a `for`
loop's trip count. Such a capture is folded into the lowered IR as exactly the
constant an inline literal would have produced, so the gradient is identical
either way.

A captured **runtime** value works as well: an immutable local `val`, or a
parameter of the enclosing function, typed `Float`, `Double`, `Int` or `Long`:

```kotlin
val scale = computeScale()                      // not a compile-time constant
val g = grad { x: Float -> x * x * scale }      // lowers; scale is read at the call
```

It becomes a trailing parameter of the derived gradient function, and the plugin
binds it at the call site by reading the declaration your lambda closed over.
**The returned function's arity is unchanged**: a captured value is an input,
never a differentiation target, so `grad` still returns one gradient, `grad2` a
`Pair` and `grad3` a `Triple`.

What still refuses, by name and with the reason:

```
e: Tlaloc could not lower this lambda at compile time: captured value 'gain' is
   not a compile-time constant (it is a `var`, and a captured runtime value must
   be immutable — the gradient is derived where the lambda is written but runs
   where it is called, so a mutable capture has no single value to bind. Declare
   'gain' as a `val`) — …
```

That covers a `var`; a top-level or member property (reading one is a getter
call, not a value declaration the gradient can close over, so copy it into a
local `val` first); a captured value of any other type, including a tensor; and
the forward-mode, assembly and seeded-cotangent intrinsics (`jvp`, `jacobian`,
`hessian`, `vjp` and their arity-2 spellings), which build their own parameter
lists out of the lowered one.

### Reading the gradient the compiler derived

Two plugin options print, for every `grad {}` / `valueAndGrad {}` lambda the
plugin synthesizes, the reverse-transformed gradient as Kotlin source over the
`:core` host ops. It is the same DXIR function the synthesis compiles to
bytecode, so what you read is what runs:

```kotlin
tlaloc {
    dumpGradSource.set(true)                                       // an INFO message per lambda
    dumpGradSourceDir.set(layout.buildDirectory.dir("gradients"))  // and one .kt file per lambda
}
```

The files land in one subdirectory per compilation, named after its source set:
`build/gradients/main/` for `src/main/kotlin`, `build/gradients/test/` for
`src/test/kotlin` (`jvmMain`, `jvmTest` in a Multiplatform build). Each
subdirectory is an output of its compile task, so the build cache stores and
restores it with the classes.

Each dump is headed by the lambda's source location, and the dumped `.kt` file
compiles and runs on its own, bit-identical to the compiled gradient (pinned in
`DumpGradSourceTest`). Scalar lambdas render; tensor `grad {}` lambdas, whose
shapes are symbolic at compile time, print a named "dump SKIPPED" note instead.
The runtime capture route (`CapturedStep.gradSource()` in `:nn`) renders tensor
gradients too. [READABLE_REVERSE.md](READABLE_REVERSE.md) shows examples.

## 4. Compile-time safety

Named axes live in the type system, and the checker runs on every compilation of
the lambda:

- `contract` over operands sharing no named axis → `NAMED_INDEX_MISMATCH`
  **error** at the call's own line and column (pinned by
  `DiagnosticSourcePositionTest`).
- A body the reverse-mode transform cannot differentiate → `NOT_DIFFERENTIABLE`
  error, with the transform's reason verbatim.
- Concrete-dim shape violations → `TENSOR_SHAPE_MISMATCH` error.
- A body the plugin cannot **lower** at all → `Tlaloc could not lower this lambda
  at compile time: <reason>` **error**. `strictLowering = false` turns it into a
  warning, and the call then throws `IllegalStateException` when it runs.

See one fail in front of you:

```bash
./gradlew -p examples/quickstart shapeError
```

```
e: Tlaloc named-index mismatch: contract operands share no named axis:
   lhs=[Batch, SeqLen] rhs=[Hidden, Hidden]
```

The task compiles `src/shapeError/kotlin/ShapeError.kt`, a source set that is
meant to fail; the build fails with the error above.

## 4a. Plugin options

All five are properties of the Gradle plugin's `tlaloc { }` block, under the same
names. Without the Gradle plugin they are
`-P plugin:io.tlaloc.plugin:<name>=<value>` in
`compilerOptions.freeCompilerArgs`. A boolean value other than `true` or `false`
is refused by name.

| option | default | what it does |
| --- | --- | --- |
| `strictLowering` | `true` | A `grad {}` / `jvp {}` / `vjp {}` lambda the plugin cannot compile is a compile **error**. `false` makes it a warning, and the call throws `IllegalStateException` when it runs. |
| `dumpLoweredIr` | `false` | Dump the lowered Tlaloc IR for every recognized intrinsic lambda: one WARNING from the FIR checker and one INFO from the IR extension. For debugging the plugin. K2's diagnostic API has no INFO severity, so the FIR half is a warning and breaks a `-Werror` build. |
| `dumpGradSource` | `false` | Print each synthesized gradient as readable Kotlin (INFO). |
| `dumpGradSourceDir` | unset | Like `dumpGradSource`, and also write one `.kt` per lambda. The Gradle plugin writes each compilation's files into `<dir>/<source set>` (`<dir>/main`, `<dir>/test`) and declares that directory an output of the compile task; the raw `-P` option writes into `<dir>` itself. |
| `unsafeAllowUnsupportedKotlin` | `false` | Turn the Kotlin version refusal into a warning and run the plugin on a Kotlin outside 2.3.20–2.3.29. The plugin reads internal K2 API, so expect crashes from inside the compiler. |

A build with none of these set and a `grad {}` that lowers is **silent**: Tlaloc
prints nothing. `DiagnosticNoiseTest` pins that by compiling a `grad {}` consumer
with `-Werror`.

## 4b. `@ExperimentalTlalocApi`: the provisional part of the surface

Every API in `0.1.0-alpha01` may change without a deprecation cycle. Within that,
three surfaces carry a `@RequiresOptIn(ERROR)` marker,
`io.tlaloc.core.ExperimentalTlalocApi`, and using one without opting in is a
compile error naming the marker:

| Marked | Why |
|---|---|
| The four-worlds scopes: `KernelScope`, `OrchestrationScope`, `ProgramScope`, `ClusterScope`, `Tlaloc`, `BufferHandle`, `HandleRef` | Each op is limited to a single scope, and Kotlin's context parameters may change the whole design |
| `io.tlaloc.ir.AllReduceAttrs` | Distributed execution is 📐 in [CAPABILITIES.md](CAPABILITIES.md) (designed, never run on two hosts), and only `"sum"` is supported |
| `io.tlaloc.ir.recognizer.kernel` and `io.tlaloc.ir.recognizer.cost` | The machinery is unit-tested; its purpose is picking a kernel, and the one Tlaloc kernel measured against XLA lost at small shapes ([KPTX_PAGED_PERF.md](KPTX_PAGED_PERF.md)) |

`grad`, the tensor and op surface, `:nn`'s layers, optimizers and schedules,
safetensors, StableHLO emission and the PJRT/IREE runtimes are not marked.

```kotlin
@file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)   // per file
```

```kotlin
kotlin { compilerOptions { optIn.add("io.tlaloc.core.ExperimentalTlalocApi") } }  // per module
```

[examples/internals/four-worlds](../examples/internals/four-worlds) and
[examples/internals/layer3](../examples/internals/layer3) use the file-level form.

## 5. Running on a GPU

The quickstart's gradient runs on the JVM. The same DXIR programs dispatch to GPUs
through:

- **`io.github.pedronahum:tlaloc-runtime-pjrt`**: PJRT-XLA through Kotlin FFM (no
  JNI, no Python at run time). See `PjrtSession`.
- **`io.github.pedronahum:tlaloc-runtime-iree`**: IREE CPU/CUDA through the IREE
  command-line tools.
- **`io.github.pedronahum:tlaloc-kptx`**: PTX kernels inside XLA executables (see
  [KPTX_PLAN.md](KPTX_PLAN.md)).

### The PJRT CUDA plugin

Tlaloc does not ship a PJRT plugin. It uses the one JAX publishes, which is a
Linux-only `.so`. Install it into a virtual environment:

```bash
python3 -m venv ~/.local/venvs/jax
~/.local/venvs/jax/bin/pip install "jax[cuda12]"
```

The plugin lands at
`~/.local/venvs/jax/lib/python3.*/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so`.
You need an NVIDIA driver; the CUDA libraries come with the wheel.

`PjrtBinaries.pluginPath` looks for it in this order:

1. `TLALOC_PJRT_PLUGIN_PATH`, if it names a file that exists;
2. `$VIRTUAL_ENV`;
3. every directory under `~/.local/venvs/`;
4. `~/.venv`, `~/venv` and `~/.local` (a `pip install --user`);
5. `/usr/local` and `/usr`.

Under each root it checks `lib/python3.*` and `lib64/python3.*`, both
`site-packages` and `dist-packages`, and any `.so` in a `jax_plugins/` directory
whose name contains `cuda`. When nothing is found, the error lists every place it
looked.

### JVM flags

PJRT and CUDA use the FFM API, which prints a warning on first use unless native
access is enabled for the calling code. Enable it where your program starts:

```kotlin
application { applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED") }
tasks.test { jvmArgs("--enable-native-access=ALL-UNNAMED") }
```

For an executable fat jar, put `Enable-Native-Access: ALL-UNNAMED` in the
manifest instead.

### Memory

A PJRT CUDA client created without options preallocates 75% of GPU memory. On a
unified-memory machine (NVIDIA GB10, Jetson) that memory is system RAM, and
several clients can exhaust it. Tlaloc never creates a CUDA client without
options: by default it uses a memory fraction of 0.5 with preallocation off. Set
`TLALOC_PJRT_MEMORY_FRACTION` and `TLALOC_PJRT_PREALLOCATE` to change that
([Configuration](#configuration)). On unified memory, keep preallocation off and
run one GPU process at a time.

### IREE

The IREE runtime runs `iree-compile` and `iree-run-module`. Install them with
`pip install iree-base-compiler iree-base-runtime` into a virtual environment, or
point `TLALOC_IREE_BIN` at the directory that holds them.

## Configuration

Everything Tlaloc reads from the environment or from system properties.

### Environment variables (runtime)

| Variable | Default | Read by | Meaning |
|---|---|---|---|
| `TLALOC_PJRT_PLUGIN_PATH` | unset | `runtime-pjrt`, Python serving runtime | Path to a PJRT plugin `.so`, used before any search. If the file does not exist, the JVM falls back to the search and the Python serving runtime refuses. For the TPU lane the file name must contain `tpu`. |
| `VIRTUAL_ENV` | set by an activated venv | `runtime-pjrt`, `runtime-iree` | Searched first for the PJRT plugin (`lib/python3.*/…`) and for IREE tools (`bin/`). |
| `TLALOC_PJRT_MEMORY_FRACTION` | `0.5` | `runtime-pjrt`, Python serving runtime | Fraction of GPU memory the CUDA client may use, in (0, 1]. |
| `TLALOC_PJRT_PREALLOCATE` | `false` | `runtime-pjrt`, Python serving runtime | `true` or `false`: whether the CUDA client reserves its fraction up front. |
| `TLALOC_PJRT_NODE_ID` | `0` | `runtime-pjrt` | This process's rank in a multi-host group. Multi-host is designed but not implemented; creating a client with more than one node is refused. |
| `TLALOC_PJRT_NUM_NODES` | `1` | `runtime-pjrt` | Size of the multi-host group. |
| `TLALOC_PJRT_COORDINATOR_ADDRESS` | unset | `runtime-pjrt` | `host:port` of node 0; required when `TLALOC_PJRT_NUM_NODES` > 1. |
| `TLALOC_IREE_BIN` | unset | `runtime-iree` | Directory holding `iree-compile` and `iree-run-module`. Searched before `$VIRTUAL_ENV/bin`, `~/.local/venvs/*/bin` and `PATH`. |
| `TLALOC_SERVING_ARTIFACT` | unset | vLLM plugin (`vllm_tlaloc`) | Directory of the serving artifact vLLM should load. |

An invalid value is refused by name (`TLALOC_PJRT_MEMORY_FRACTION must be a number
in (0, 1], got '…'`); it is never silently replaced by the default.

### System properties (compiler plugin)

The compiler plugin runs inside the Kotlin compiler, so these are properties of
the Kotlin daemon's JVM. Set them in `gradle.properties` as
`kotlin.daemon.jvmargs=-Dtlaloc.cache.dir=/path/to/cache`.

| Property | Default | Meaning |
|---|---|---|
| `tlaloc.cache.dir` | unset (no cache) | Directory for a disk cache of loop-coarsening results, keyed by the loop's IR hash and the plugin and Symja versions. If the directory cannot be opened, the plugin warns and compiles without the cache. |
| `tlaloc.soi.enabled` | `false` | `true` routes single-return `grad { }` bodies through the size-limited coarsening pass. |
| `tlaloc.soi.size.limit` | `50` | Size limit for that pass; a positive integer, anything else falls back to 50. |
| `tlaloc.simplify.enabled` | `false` | `true` simplifies each gradient expression with Symja before synthesis. Needs Symja on the plugin classpath; bodies it cannot simplify pass through unchanged. |

### Compiler plugin options

The five `tlaloc { }` options are in [section 4a](#4a-plugin-options).

## Troubleshooting

**`Tlaloc's K2 compiler plugin is built against Kotlin 2.3.20 and the running
Kotlin compiler is …`**
Your Kotlin is outside 2.3.20–2.3.29. Use `kotlin("jvm") version "2.3.20"`, or set
`tlaloc { unsafeAllowUnsupportedKotlin.set(true) }` to try anyway.

**``Tlaloc: `grad { }` was not rewritten at compile time, so this fallback body ran``**
thrown when the returned function is called. The message lists three causes:
the compiler plugin is not applied to this module (apply
`id("io.github.pedronahum.tlaloc")`); the plugin refused the call and
`strictLowering` is `false` (the compile log has a `w: Tlaloc …` warning at the
call site with the reason); or `grad` was called without a lambda written at the
call site, for example through a function reference.

**`Tlaloc could not lower this lambda at compile time: <reason>`**
The body uses something the plugin cannot differentiate; the reason names it.
Rewrite the body, or use `io.tlaloc.autograd.gradWithScalars` for that call.

**`Tlaloc internal error while compiling this call: …`**
A bug in the compiler plugin. The message includes the exception and frame;
please report it at https://github.com/pedronahum/tlaloc/issues.

**`java.lang.UnsupportedClassVersionError: io/tlaloc/plugin/… has been compiled by
a more recent version of the Java Runtime (class file version 69.0)`**
The Kotlin compiler is running on a JDK older than 25. Set `jvmToolchain(25)`; the
Kotlin daemon then runs on that JDK. With
`kotlin.compiler.execution.strategy=in-process`, Gradle itself must run on a
JDK 25. The same error at run time from `io/tlaloc/runtime/…` or `io/tlaloc/kptx/…`
means a Java 25 module was started on an older JDK.

**`<caller>: no PJRT CUDA plugin found.`** or, from `serve.py`,
**`SKIP: no PJRT cuda plugin found, so there is nothing to run on.`**
No plugin in any searched location. Install one
([The PJRT CUDA plugin](#the-pjrt-cuda-plugin)) or set `TLALOC_PJRT_PLUGIN_PATH`.
The lines after the message list every path that was checked.

**`WARNING: A restricted method in java.lang.foreign.… has been called`**
Native access is not enabled. Add `--enable-native-access=ALL-UNNAMED`
([JVM flags](#jvm-flags)).

**`<tool> not found.`** (IREE)
`iree-compile` or `iree-run-module` is not installed where Tlaloc looks. The
message lists the searched directories and the `pip install` line.

## 6. Where to go next

- [examples/](../examples/): ten standalone runnable projects, each its own
  Gradle build resolving Tlaloc from `mavenLocal`, with the output of every one
  in [examples/README.md](../examples/README.md). Seven at the top level
  (`quickstart`, `readable-gradients`, `differentiable-physics`, `named-indices`,
  `mnist`, `gpu-training`, `gpu-inference`) and three under
  `examples/internals/`: `layer3` (kernel selection), `four-worlds` (the scope
  taxonomy) and `tpu`. Six need nothing but a JDK.
- **API reference**: `./gradlew apiDocs` from the repository root writes a Dokka
  site for the library modules to `build/docs/api/index.html`. Nothing is hosted.
- [CAPABILITIES.md](CAPABILITIES.md): what is certified, written or designed, and
  the test behind each row.
- [README](../README.md): overview, architecture and maturity.
