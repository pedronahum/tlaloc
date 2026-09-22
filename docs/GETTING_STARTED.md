# Getting started with Tlaloc

Differentiable Kotlin, end to end: you write a plain Kotlin lambda, the
Tlaloc K2 compiler plugin rewrites `grad { }` calls into synthesized
gradient code at compile time, and misuse (mismatched named axes,
undifferentiable bodies) is a **compile error with an IDE red squiggle**,
not a runtime crash.

> Alpha — `0.1.0-alpha01`. Artifacts are not yet on Maven Central — consume via
> `mavenLocal()` from a repo checkout. Coordinates and APIs may change without a
> deprecation cycle; [COMPATIBILITY.md](COMPATIBILITY.md) says exactly what may
> break and what will not, and [CHANGELOG.md](../CHANGELOG.md) records what did.

## 1. Publish the artifacts locally

```bash
git clone <tlaloc repo> && cd tlaloc
./gradlew publishToMavenLocal -x test
```

Every module lands under `io.tlaloc:*:0.1.0-alpha01` (`core`, `ir`,
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
    implementation("io.tlaloc:core:0.1.0-alpha01")
    implementation("io.tlaloc:ir:0.1.0-alpha01")
    implementation("io.tlaloc:autograd:0.1.0-alpha01")
    // The K2 plugin: compile-time grad rewriting + compile-time errors.
    kotlinCompilerPluginClasspath("io.tlaloc:compiler-plugin:0.1.0-alpha01")
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

Named axes live in the type system, and the checker validates your
lambda **as you type**:

- `contract` over operands sharing no named axis → `NAMED_INDEX_MISMATCH`
  **error** (the build fails; IntelliJ redlines the call).
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

All four are `-P plugin:io.tlaloc.plugin:<name>=<value>` on the Kotlin
compile task (`kotlinOptions.freeCompilerArgs` / `compilerOptions`):

| option | default | what it does |
| --- | --- | --- |
| `strictLowering` | `true` | An unlowerable `grad {}` / `jvp {}` / `vjp {}` lambda is a compile **error**. `false` restores the pre-0.1.0-alpha01 warning plus a runtime `IllegalStateException`. |
| `dumpLoweredIr` | `false` | Dump the lowered Tlaloc IR for every recognised intrinsic lambda: one WARNING from the FIR checker, one INFO from the IR extension. **Developer introspection.** It is off by default because a consumer's build log is not the place for it — and because, being warnings, the two dumps used to break every build compiling with `-Werror`. Do not combine this option with `-Werror`: K2's diagnostic DSL has no INFO severity, so the FIR half is necessarily a warning. |
| `dumpGradSource` | `false` | Print each synthesized gradient as readable Kotlin (INFO). See above. |
| `dumpGradSourceDir` | — | Like `dumpGradSource`, and also write one `.kt` per lambda into this directory. |

A build with none of these set and a `grad {}` that lowers is **silent**:
Tlaloc says nothing at all. That is pinned by `DiagnosticNoiseTest`, which
compiles a `grad {}` consumer with `-Werror` and asserts both that it
succeeds and that no message mentions Tlaloc.

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
