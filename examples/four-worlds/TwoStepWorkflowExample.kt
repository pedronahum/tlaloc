@file:Suppress("MatchingDeclarationName", "unused")

package io.tlaloc.examples.four_worlds

/**
 * Layer 2 §0.4.243+ — two-step `workflow { }` example with typed
 * BufferHandles.
 *
 * Demonstrates the Layer 2 step-boundary contract: BufferHandles cross
 * step boundaries, never raw tensors. Step composition is by direct
 * function call — Kotlin's native type checker enforces the I/O type
 * match at the call site (no plugin involvement).
 *
 * The workflow builder records reshard metadata on each edge; cross-mesh
 * transitions and named-axis transposes produce explicit reshard steps
 * (see [io.tlaloc.maestro.WorkflowEdge]). This example uses [Mesh0] for
 * both steps, so the recorded edge is `ReshardKind.None` (pass-through).
 *
 * Copy into a project depending on `io.tlaloc:core` + `io.tlaloc:maestro`.
 */

import io.tlaloc.autograd.relu
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.core.Mesh0
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.Tlaloc
import io.tlaloc.core.hostF32
import io.tlaloc.maestro.MaestroDescriptor
import io.tlaloc.maestro.StubExecutor
import io.tlaloc.maestro.program
import io.tlaloc.maestro.workflow

fun main() {
    val seedTensor = Tensors.f32Vector<Sym>(floatArrayOf(1f, -2f, 3f, -4f, 5f))

    // Build two MaestroStep artifacts, each one a `program {}` block.
    // Each compiles its body to a content-addressed StableHLO artifact.
    val activate = Tlaloc.program("activate", seedTensor, Mesh0) { x -> x.relu() }
    val score = Tlaloc.program("score", seedTensor, Mesh0) { x -> (x * x).sum() }

    // Compose them into a workflow. The DSL's `step(...)` function
    // records each step + computes the edge metadata. Type-checking is
    // by Kotlin's native checker — `score`'s input type must match
    // `activate`'s output type or this won't compile.
    val wf = Tlaloc.workflow("activate_then_score") {
        val seed = seed(seedTensor, Mesh0)
        val activated = step(activate, seed)
        step(score, activated)
    }

    println("Workflow:       ${wf.name}")
    println("Steps:          ${wf.steps.map { it.name }}")
    println("Edges:          ${wf.edges.map { "${it.fromStep} → ${it.toStep} (${it.reshardKind})" }}")

    // Emit the Maestro JSON descriptor — what a real Maestro cluster would
    // ingest. The descriptor models each Tlaloc step as a Maestro
    // Kubernetes-step with a `tlaloc-runtime:*` image (see
    // docs/maestro_descriptor.md for the conformance approach).
    val descriptor = MaestroDescriptor.emit(wf)
    println("Descriptor (first 200 chars): ${descriptor.take(200)}...")

    // Run the workflow via the in-process stub executor. v1's stub
    // executor walks the in-memory step graph; a real Maestro executor
    // would consume the descriptor + dispatch each step's artifact in a
    // Kubernetes pod.
    val seedHandle = io.tlaloc.core.BufferHandle<io.tlaloc.core.DTensor<io.tlaloc.core.Rank1<Sym>, io.tlaloc.core.F32>, Mesh0>(
        io.tlaloc.core.HandleRef(nativeId = 1L, payload = seedTensor),
    )
    val out: io.tlaloc.core.BufferHandle<io.tlaloc.core.DTensor<io.tlaloc.core.ScalarShape, io.tlaloc.core.F32>, Mesh0> =
        StubExecutor().run(wf, seedHandle)

    @Suppress("UNCHECKED_CAST")
    val finalTensor = out.ref.payload as io.tlaloc.core.DTensor<io.tlaloc.core.ScalarShape, io.tlaloc.core.F32>
    println("Final score:    ${finalTensor.hostF32().single()}")
    // Expected: relu([1,-2,3,-4,5])² summed = 1+9+25 = 35.0
}
