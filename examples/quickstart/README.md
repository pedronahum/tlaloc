# quickstart — a gradient, derived at compile time

The smallest complete Tlaloc program, and the smallest complete demonstration of
why the compile step matters.

```kotlin
val g = grad { a: DTensor<Rank2<Sym, Sym>, F32> -> (a matmul a).sum().toFloat() }

g(Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f)))   // [7.0, 11.0, 9.0, 13.0]
```

There is no tape here and no framework object at runtime. The Tlaloc K2 compiler
plugin rewrites that call *during compilation* into synthesized gradient code —
`d/dA sum(A·A) = 1·Aᵀ + Aᵀ·1` — so `g` is an ordinary Kotlin function you call.

## …and a shape bug that never gets to run

[`src/shapeError/kotlin/ShapeError.kt`](src/shapeError/kotlin/ShapeError.kt) is a
real file in a real source set. It is **supposed to fail to compile**, and you can
watch it happen:

```bash
./gradlew -p examples/quickstart shapeError
```

```
e: file:///home/pedro/programming/tlaloc/examples/quickstart/src/shapeError/kotlin/ShapeError.kt:30:5 Tlaloc named-index mismatch: contract operands share no named axis: lhs=[Batch, SeqLen] rhs=[Hidden, Hidden]
e: file:///home/pedro/programming/tlaloc/examples/quickstart/src/shapeError/kotlin/ShapeError.kt:32:12 Cannot infer type for type parameter 'NameN'. Specify it explicitly.
```

The two operands share no axis **name** — `(Batch, SeqLen)` against
`(Hidden, Hidden)` — so there is nothing to contract over. Two independent nets
catch it: Tlaloc's own checker names the mismatch, and Kotlin's type checker
cannot resolve the overload either (the remaining `e:` lines). In NumPy, PyTorch
or JAX this is a runtime error at best and a silently wrong answer at worst.

That task is wired to **fail the build if the compile ever succeeds** — a green
run there would mean Tlaloc had stopped catching the bug. It is not part of
`build`, `check` or `run`, so the quickstart itself still runs green.

## Running it

This is a standalone Gradle project. It resolves Tlaloc from **mavenLocal**,
exactly as your own project would, so publish first:

```bash
# from the repo root
./gradlew publishToMavenLocal
./gradlew -p examples/quickstart run
```

`scripts/onboarding-smoke.sh` does both in one go.

## Expected output

Verbatim, from this machine (GB10, JDK 25, Kotlin 2.3.20):

```
[1] a gradient, derived at compile time

    val g = grad { a: DTensor<Rank2<Sym, Sym>, F32> -> (a matmul a).sum().toFloat() }

    d/dA sum(A matmul A) at A = [[1, 2], [3, 4]]
      = 1·Aᵀ + Aᵀ·1
      = [7.0, 11.0, 9.0, 13.0]

    `g` is an ordinary Kotlin function. Call it in a hot loop: nothing
    allocates a tape, because there is no tape.

[2] a shape bug the compiler will not let you ship

    fun shapeErrorThatMustNotCompile() {
        grad2 { a: DTensor<Rank2<Named<Batch, Sym>, Named<SeqLen, Sym>>, F32>,
                b: DTensor<Rank2<Named<Hidden, Sym>, Named<Hidden, Sym>>, F32> ->
            (a contract b).sum()
        }
    }

    Those two operands share no axis NAME — (Batch, SeqLen) against
    (Hidden, Hidden) — so there is nothing to contract over. In NumPy,
    PyTorch or JAX that is a runtime error at best. Here the build fails.

    Watch it happen:

        ./gradlew -p examples/quickstart shapeError

    e: ShapeError.kt:30:5 Tlaloc named-index mismatch: contract operands
       share no named axis: lhs=[Batch, SeqLen] rhs=[Hidden, Hidden]

    That task is SUPPOSED to fail. It is wired to fail the build if the
    compile ever succeeds — a green run there would mean Tlaloc had
    stopped catching the bug.

quickstart OK
```

## What to read in the source

| Look at | For |
|---|---|
| [`src/main/kotlin/Main.kt`](src/main/kotlin/Main.kt) | The `grad { }` call — three lines, one gradient, no tape |
| [`src/shapeError/kotlin/ShapeError.kt`](src/shapeError/kotlin/ShapeError.kt) | The program that must not compile |
| [`build.gradle.kts`](build.gradle.kts) | The one line that makes it all work: `kotlinCompilerPluginClasspath("io.github.pedronahum:compiler-plugin:0.1.0-alpha01")` — the `shapeError` task that expects failure — and, since §0.4.503, the **supported consumer JDK configuration**: `jvmToolchain(25)` to build, `jvmTarget = JVM_21` to run |
| [`settings.gradle.kts`](settings.gradle.kts) | `mavenLocal()` first — this project is a consumer, not part of the repo build |

## Running it on a JDK 21

§0.4.503 split Tlaloc's bytecode targets: the library modules (`core`, `ir`,
`autograd`, `nn`, `stablehlo`, `maestro`) emit Java 21, while the FFM runtime
backends and the compiler plugin stay at 25. This project is the certification
that the 21 half is real:

```bash
export JDK21_HOME=/path/to/a/jdk-21     # gradle.properties reads this
./gradlew -p examples/quickstart runOnJdk21
```

The program is compiled by a JDK 25 toolchain — the plugin needs one, because
Kotlin runs a compiler plugin inside the compiler's JVM — into Java 21 bytecode,
and then executed by a **JDK 21** launcher. A JDK 22+ API anywhere in `:core`,
`:ir` or `:autograd`, or 25 bytecode in any of them or in the gradient the plugin
synthesized into this program, fails here with `UnsupportedClassVersionError` or
`NoSuchMethodError` instead of printing a derivative. Measured on OpenJDK 21.0.2.

`bash scripts/jdk21-smoke.sh` from the repository root is the same thing including
the `publishToMavenLocal` that has to come first.

## Where to go next

1. [`readable-gradients/`](../readable-gradients/) — the derivative as a file you
   can read, recompiled without the plugin, agreeing bit for bit.
2. [`differentiable-physics/`](../differentiable-physics/) — the same `grad`, now
   differentiating a physics simulator with a loop in it, to sink a free throw.
3. [`named-indices/`](../named-indices/) — what the type system buys you before
   anything runs.
