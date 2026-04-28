# Four-Worlds + Maestro examples (Layer 2 §0.4.243+)

Reference snippets demonstrating Tlaloc's four-worlds taxonomy and typed
buffer-handle protocol.

These files are documentation-grade — copy into a project that depends on
`io.tlaloc:core` and `io.tlaloc:maestro` to run.

| File | Purpose |
|------|---------|
| [`SingleStepProgramExample.kt`](SingleStepProgramExample.kt) | A `program { }` block with Kernel/Orchestration scope discipline. Shows that orchestration ops are not callable from inside the kernel body. |
| [`TwoStepWorkflowExample.kt`](TwoStepWorkflowExample.kt) | `workflow { }` composing two `MaestroStep`s with typed `BufferHandle`s flowing between them. |
| Type-mismatch (below) | A program that **does not compile** because a step's input handle type doesn't match the producing step's output. |

## Type-mismatch example (compile error by design)

```kotlin
import io.tlaloc.core.*
import io.tlaloc.core.ops.contract
import io.tlaloc.maestro.program
import io.tlaloc.maestro.workflow
import io.tlaloc.autograd.relu
import io.tlaloc.autograd.sum

fun main() {
    val tokens = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f))
    val activations = Tensors.f32Matrix<Sym, Sym>(2, 3, FloatArray(6))

    val encode = Tlaloc.program("encode", tokens, Mesh0) { x -> x.relu() }
    val score = Tlaloc.program("score", activations, Mesh0) { m -> m.sum() }
    //                                         ^^^^^^^^^^^^
    // 'score' takes a Rank2<Sym, Sym> input, not a Rank1<Sym>.

    Tlaloc.workflow("mismatched") {
        val activated = step(encode, seed(tokens, Mesh0))
        // Compile error at the next line: 'activated' is
        //   BufferHandle<DTensor<Rank1<Sym>, F32>, Mesh0>
        // but 'score' expects
        //   BufferHandle<DTensor<Rank2<Sym, Sym>, F32>, Mesh0>
        step(score, activated)
    }
}
```

The Kotlin compiler reports a type-mismatch error at the `step(score, activated)`
call site because `MaestroStep<In, Out>`'s `In` parameter doesn't match
the type of `activated`. **No plugin diagnostic is involved** — this is
Kotlin's native type checker doing its job at the function-call site. The
error rendering is the standard Kotlin "Type mismatch: expected ..., got
..." form, pointed at the offending argument.

## Running the examples

```bash
# From a project that depends on io.tlaloc:core + io.tlaloc:maestro:
kotlin examples/four-worlds/SingleStepProgramExample.kt
kotlin examples/four-worlds/TwoStepWorkflowExample.kt
```

Both compiling examples exercise the runtime path — the K2 plugin is
**not** required (Layer 2's `program {}` builder uses the `:autograd`
runtime tracer to capture the body, not a plugin lowering). The
plugin's responsibility is enforcing world-scope discipline at compile
time, which is captured by the test suite under `:compiler-plugin`.
