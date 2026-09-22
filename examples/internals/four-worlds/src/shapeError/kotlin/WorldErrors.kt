/**
 * THIS FILE IS SUPPOSED TO FAIL TO COMPILE.
 *
 * Real Kotlin in a real source set, not a comment. The build has a task whose
 * only job is to compile it and show you the rejection:
 *
 *     ./gradlew -p examples/internals/four-worlds shapeError
 *
 * WRONG HANDLE TYPE. `activate` produces a Rank1 handle; a step that wants a
 * Rank2 handle cannot be fed it. Kotlin reports a type mismatch at the
 * `step(...)` argument, pointing at the offending value — no plugin involved,
 * and no runtime check: what crosses a step boundary is a typed
 * `BufferHandle<T, M>`, so the wrong one does not type-check.
 *
 * ON THE VIOLATION THAT IS NOT HERE — see `Main.kt`'s act [4]. This example
 * used to claim, in a commented-out block, that calling `Tlaloc.program` from
 * inside a Kernel body "does not resolve". Turning that comment into a file the
 * build actually compiles showed it was false, and it is not in this file
 * because it compiles cleanly.
 */
// §0.4.505 — THE OPT-IN. `Tlaloc`, `program` and `BufferHandle` carry
// `@ExperimentalTlalocApi` (a `@RequiresOptIn(ERROR)` marker; see `Main.kt`'s
// header for why). It is here because THIS FILE MUST STILL FAIL FOR ITS OWN
// REASON — a Rank1 handle where a Rank2 one is required — and not for a missing
// opt-in, which would be a different error teaching a different lesson.
@file:OptIn(ExperimentalTlalocApi::class)

import io.tlaloc.autograd.sum
import io.tlaloc.core.DTensor
import io.tlaloc.core.ExperimentalTlalocApi
import io.tlaloc.core.F32
import io.tlaloc.core.Mesh0
import io.tlaloc.core.Rank1
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.Tlaloc
import io.tlaloc.maestro.program
import io.tlaloc.maestro.workflow

private val input: DTensor<Rank1<Sym>, F32> =
    Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f))

private val activate = Tlaloc.program("activate", input, Mesh0) { x -> x }

/** A Rank1 handle where a Rank2 handle is required. */
fun wrongHandleType() {
    val matrixStep = Tlaloc.program(
        "matrix", Tensors.f32Matrix<Sym, Sym>(2, 3, FloatArray(6)), Mesh0,
    ) { m -> m.sum() }
    Tlaloc.workflow("mismatched") {
        val activated = step(activate, seed(input, Mesh0))
        step(matrixStep, activated)   // <-- compile error, by design
    }
}
