# Tlaloc

**A Kotlin-native differentiable-programming framework.** Compile-time AD via a K2 compiler plugin, shape- and sharding-typed tensors enforced by the Kotlin type system, StableHLO + Shardy lowering. KMP-first: one source set targets JVM, Android, iOS, and WASM.

> Pre-alpha. Built in the open; not yet packaged for consumption. See [DIFFKTX_SPEC.md](DIFFKTX_SPEC.md) §0.4 for the session-by-session ship log.

---

## Why

PyTorch owns Python. JAX owns research. We don't try to fight on that turf.

Tlaloc targets the gap Python frameworks leave open:

- **Android / KMP on-device training** — LoRA fine-tuning, federated learning, personalization, sensor-driven models.
- **JVM-native enterprise ML** — Spring Boot / Kafka / Flink / Spark hot paths.
- **Shape-safety-obsessed teams** — fintech, aerospace; silent shape bugs that are a P0.
- **Differentiable simulation** — games, robotics, XR where Kotlin already has a foothold (libGDX, Korge, Android XR).

## What it is

- **Compile-time AD.** `grad { x: Float -> ... }` is realized by a K2 plugin — not a runtime tape. The plugin lowers the lambda body to a typed SSA IR (DXIR), runs a coarsening pass, applies the reverse-mode transform, and synthesises a forward Kotlin function that computes the gradient directly.
- **φ-calculus coarsening** from Shen, Shivers, Dea et al. [Efficient, Sound Gradient Descent in Dynamic and Dependent Control Flow, OOPSLA 2021](docs/papers/coarsening-autodiff.txt). Our pass implements F1–F5 + C1–C9 with a [Symja](https://github.com/axkr/symja_android_library)-backed symbolic engine for closed-form closure of affine / indexed-affine / variable-coefficient / power-form recurrences.
- **Shape-typed tensors.** `DTensor<Rank1<Sym>, F32>` carries shape and dtype in the Kotlin type system. Rank mismatches and shape-wrong broadcasts are compile errors, surfaced in the IDE.
- **Sharding-typed.** Mesh axis names live in the type system. Users annotate a handful of tensors with partition specs; the compiler propagates shardings through the rest of the program and hands off to [Shardy](https://github.com/openxla/shardy)'s propagation passes. DP, TP, EP, ZeRO, and context parallelism are all the same mechanism.
- **StableHLO + SDY emission.** We don't write CUDA. We lower to MLIR and let PJRT-backed runtimes (XLA, IREE) codegen.

## What it is not

- Not a PyTorch clone. No `torch.nn` parity goal.
- Not a research playground for novel AD algorithms. Boring, correct, fast.
- Not a CUDA kernel project. Not a collectives library.
- Not Python-compatible at the API level. Interop is via ONNX / StableHLO artifacts.

## Example

```kotlin
import io.tlaloc.autograd.grad

fun main() {
    // Paper-faithful BGDHyperOpt (OOPSLA 2021 §6.2), minus the convergence break.
    // Inputs packed as [r, x_0..x_{M-1}, y_0..y_{M-1}]; gradient wrt slot 0 = d(err)/dr.
    val g = grad { p: DTensor<Rank1<Sym>, F32> ->
        val r = p[0]; val Mf = 3.0f
        var Sxy = 0.0f; var Sx2 = 0.0f
        for (i in 0 until 3) {
            val xi = p[1 + i]; val yi = p[4 + i]
            Sxy = Sxy + xi * yi
            Sx2 = Sx2 + xi * xi
        }
        var w = 0.0f; var k = 0
        while (k < 50) {
            val d = 2.0f * (Sx2 * w - Sxy)
            w = w - r * d / Mf
            k = k + 1
        }
        var e = 0.0f
        for (j in 0 until 3) {
            val diff = p[4 + j] - p[1 + j] * w
            e = e + diff * diff
        }
        (e / Mf).sqrt()
    }
}
```

No tape. No `torch.tensor(..., requires_grad=True)`. No gradient type wrapping the primal type. The lambda reads as straight Kotlin — the compiler plugin does the work.

## Paper-speedup closure, demonstrated

The φ-calculus pass closes the outer affine recurrence `w_{k+1} = w_k - r · (2·(Sx2·w_k - Sxy))/M` into a constant-time closed form, regardless of T. At T=50:

| shape                               |  forward | gradient | grad / forward |
|-------------------------------------|---------:|---------:|---------------:|
| natural BGD (user-written)          |  483 ns  |  614 ns  |      **1.27×** |
| pre-simplified affine (`w = a·w+b`) |  159 ns  |  286 ns  |          1.79× |
| naive C5 unroll baseline            |  524 ns  | 5825 ns  |         11.10× |

The **9.5× gradient improvement** from the unroll baseline comes from C6's symbolic closure firing end-to-end on the natural user kernel (no hand-rewriting to affine form). See [§0.4.52](DIFFKTX_SPEC.md) for the methodology and the four coordinated changes that unlocked it.

## Status

| stage | focus                                                                 | status          |
|-------|-----------------------------------------------------------------------|-----------------|
| A     | DXIR + reverse-mode SCT + K2 handoff                                  | shipped         |
| B     | φ-calculus coarsening (F1–F5 + C1–C9) + Symja engine                  | shipped         |
| C     | StableHLO + Shardy emission; SOI splice op; coarsening cache          | shipped         |
| D.1   | Brachistochrone full port (§0.4.43)                                   | shipped         |
| D.2   | HookeanSpring full port (§0.4.47)                                     | shipped         |
| D.3   | BGDHyperOpt full source port, paper-speedup closure firing (§0.4.52)  | shipped         |
| D.4+  | HMC, CartPole, QWOP                                                   | not ported      |

Full suite green at HEAD. See [DIFFKTX_SPEC.md](DIFFKTX_SPEC.md) §0.4 for every milestone and [docs/STAGE_B_PLAN.md](docs/STAGE_B_PLAN.md) for the coarsening-pass design.

## Modules

| path                                                                             | role                                                                      |
|----------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`core/`](core/)                                                                 | `DTensor`, shape / dtype types, host ops                                  |
| [`ir/`](ir/)                                                                     | DXIR + PhiCalculus + DxirReverseTransform + Symja engine + interpreter    |
| [`autograd/`](autograd/)                                                         | runtime tape (fallback when the plugin can't lower a lambda)              |
| [`stablehlo/`](stablehlo/)                                                       | StableHLO + Shardy emission; round-trip tests                             |
| [`compiler-plugin/`](compiler-plugin/)                                           | K2 plugin: FIR-to-DXIR lowering + IR-generation extension + synthesis     |

## Build

Requires JDK 17. Uses the Gradle wrapper.

```bash
./gradlew test           # full suite
./gradlew :ir:jvmTest    # IR module only
./gradlew :compiler-plugin:test --tests "*BGDHyperOptTest*"  # a specific benchmark
```

External toolchains used by `:stablehlo` round-trip tests (`stablehlo-translate`, `sdy-opt`) are located via system properties — see [docs/STAGE_B_PLAN.md](docs/STAGE_B_PLAN.md) for setup.

## Further reading

- [DIFFKTX_SPEC.md](DIFFKTX_SPEC.md) — design spec, roadmap, and implementation log. The source-of-truth handoff document between work sessions.
- [docs/STAGE_B_PLAN.md](docs/STAGE_B_PLAN.md) — φ-calculus pass design details.
- [docs/papers/coarsening-autodiff.txt](docs/papers/coarsening-autodiff.txt) — the OOPSLA 2021 paper we inherit the coarsening technique from.
- [docs/papers/symja-bakeoff-2026-04.md](docs/papers/symja-bakeoff-2026-04.md) — Symja-vs-alternatives adequacy bake-off notes.

## Acknowledgments

The φ-calculus coarsening machinery is a direct descendant of the technique in Shen, Shivers, Dea et al., *Efficient, Sound Gradient Descent in Dynamic and Dependent Control Flow* (OOPSLA 2021). The original DiffKt showed Kotlin-native AD can be 10× faster than Python frameworks on scalar benchmarks; Tlaloc extends that result to shape-typed tensors, compile-time transformation, and StableHLO lowering.
