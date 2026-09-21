# quickstart — a gradient, derived at compile time

**What it shows.** The smallest complete Tlaloc program: a `grad { }` block
over a matmul, and the derivative it produces.

```kotlin
val g = grad { a: DTensor<Rank2<Sym, Sym>, F32> -> (a matmul a).sum().toFloat() }
```

There is no tape here and no framework object at runtime. The Tlaloc K2
compiler plugin rewrites that call *during compilation* into synthesized
gradient code — `d/dA sum(A·A) = 1·Aᵀ + Aᵀ·1` — so `g` is an ordinary Kotlin
function you call. (Add the `dumpGradSource` plugin flag and it will print the
derived gradient back to you as readable Kotlin; see
`docs/READABLE_REVERSE.md`.)

The file also carries, commented out, a `grad2 { }` whose two operands have
disjoint named axes. Uncomment it and the build fails — a shape bug caught by
the type checker rather than by a stack trace.

This is the reference example: every other project under `examples/` mirrors
its shape.

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
d/dA sum(A matmul A) at A=[[1,2],[3,4]]:
[7.0, 11.0, 9.0, 13.0]
quickstart OK
```

## What to read in the source

| Look at | For |
|---|---|
| [`src/main/kotlin/Main.kt`](src/main/kotlin/Main.kt) | The `grad { }` call, and the commented-out block that must not compile |
| [`build.gradle.kts`](build.gradle.kts) | The one line that makes it all work: `kotlinCompilerPluginClasspath("io.tlaloc:compiler-plugin:0.0.1-SNAPSHOT")` |
| [`settings.gradle.kts`](settings.gradle.kts) | `mavenLocal()` first — this project is a consumer, not part of the repo build |
