# Getting started with Tlaloc

Differentiable Kotlin, end to end: you write a plain Kotlin lambda, the
Tlaloc K2 compiler plugin rewrites `grad { }` calls into synthesized
gradient code at compile time, and misuse (mismatched named axes,
undifferentiable bodies) is a **compile error with an IDE red squiggle**,
not a runtime crash.

> Pre-alpha. Artifacts are not yet on Maven Central — consume via
> `mavenLocal()` from a repo checkout. Coordinates and APIs may change.

## 1. Publish the artifacts locally

```bash
git clone <tlaloc repo> && cd tlaloc
./gradlew publishToMavenLocal -x test
```

Every module lands under `io.tlaloc:*:0.0.1-SNAPSHOT` (`core`, `ir`,
`autograd`, `stablehlo`, `compiler-plugin`, `runtime-pjrt`,
`runtime-iree`, `runtime-cuda`, `kptx`, `maestro`).

## 2. Set up a consumer project

`settings.gradle.kts` needs `mavenLocal()`; `build.gradle.kts`:

```kotlin
plugins {
    kotlin("jvm") version "2.3.20"
    application
}

dependencies {
    implementation("io.tlaloc:core:0.0.1-SNAPSHOT")
    implementation("io.tlaloc:ir:0.0.1-SNAPSHOT")
    implementation("io.tlaloc:autograd:0.0.1-SNAPSHOT")
    // The K2 plugin: compile-time grad rewriting + compile-time errors.
    kotlinCompilerPluginClasspath("io.tlaloc:compiler-plugin:0.0.1-SNAPSHOT")
}
```

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

Named axes live in the type system, and the checker validates your
lambda **as you type**:

- `contract` over operands sharing no named axis → `NAMED_INDEX_MISMATCH`
  **error** (the build fails; IntelliJ redlines the call).
- A body the reverse-mode transform cannot differentiate →
  `NOT_DIFFERENTIABLE` error, with the transform's reason verbatim.
- Concrete-dim shape violations → `TENSOR_SHAPE_MISMATCH` error.

Try it: uncomment the `bad` block at the bottom of the quickstart's
`Main.kt`.

## 5. Running on real hardware

The quickstart's gradient runs on the JVM. The same DXIR programs
dispatch to GPUs through:

- **`io.tlaloc:runtime-pjrt`** — PJRT-XLA via pure-Kotlin FFM (no JNI,
  no Python at runtime); F32 and F64. See `PjrtSession`.
- **`io.tlaloc:runtime-iree`** — IREE CPU/CUDA via subprocess facade.
- **`io.tlaloc:kptx`** — hand-written/DSL PTX kernels inside XLA
  executables (the escape-hatch tier; see `docs/KPTX_PLAN.md`).

Both GPU runtimes need their toolchains (a JAX-bundled PJRT plugin or
IREE binaries) — see the module KDocs and README "Requirements".

## 6. Where to go next

- [examples/](../examples/) — eight standalone runnable projects, indexed in
  [examples/README.md](../examples/README.md): readable gradients, named
  indices, the four worlds, Layer-3 kernel selection, GPU training, GPU
  serving, and the TPU program written before the TPU.
- [DIFFKTX_SPEC.md](../DIFFKTX_SPEC.md) — the book of work; §0.4 is the
  session-by-session ship log.
- [README](../README.md) — architecture, layer map, benchmarks.
