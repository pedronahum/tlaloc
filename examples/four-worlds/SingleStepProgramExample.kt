@file:Suppress("MatchingDeclarationName", "unused")

package io.tlaloc.examples.four_worlds

/**
 * Layer 2 §0.4.243+ — single-step `program { }` example.
 *
 * Demonstrates the Kernel / Orchestration scope discipline:
 *
 * - The outer code runs in [io.tlaloc.core.OrchestrationScope] (provided
 *   by `with(Tlaloc) { … }` or by the receiver type).
 * - `program(name, input, mesh, body)` is an extension function on
 *   [io.tlaloc.core.OrchestrationScope] — calling it outside that scope
 *   doesn't compile.
 * - Inside the body lambda, the receiver is [io.tlaloc.core.KernelScope].
 *   Kernel-only ops (relu, sum, matmul, named contract) compile;
 *   orchestration-only ops (`dispatch`, `await`, etc.) do not.
 * - The returned [io.tlaloc.maestro.MaestroStep] is the Layer-2 typed
 *   artifact — manifest + body bytes + shim function.
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
import io.tlaloc.maestro.handleOn
import io.tlaloc.maestro.program

fun main() {
    val input = Tensors.f32Vector<Sym>(floatArrayOf(1f, -2f, 3f, -4f, 5f))

    // `Tlaloc.program(...)` requires OrchestrationScope as receiver. The
    // body lambda's receiver is KernelScope — implicit; no annotation
    // needed at the call site.
    val activate = Tlaloc.program(
        name = "activate_and_score",
        input = input,
        mesh = Mesh0,
    ) { x ->
        // Inside this lambda, `x` is a Tracer<Rank1<Sym>>. Kernel-only ops
        // are callable; orchestration ops would be flagged by the
        // DslMarker. The Tracer shadow-executes the computation while
        // simultaneously building DXIR for StableHLO emission.
        val activated = x.relu()
        (activated * activated).sum()
    }

    println("Step name:      ${activate.name}")
    println("Body hash:      ${activate.manifest.bodyHash}")
    println("Body size:      ${activate.body.size} bytes")
    println("Inputs:         ${activate.manifest.inputs}")
    println("Outputs:        ${activate.manifest.outputs}")
    println("Mesh requirement: ${activate.manifest.meshRequirement}")

    // Run the shim against a typed BufferHandle. The handle wraps the
    // tensor with mesh placement Mesh0 (singleton). v1 stores the tensor
    // in HandleRef.payload; Layer 3+ replaces with a real device buffer.
    val inputHandle = input.handleOn(Mesh0)
    val outputHandle = activate(inputHandle)

    @Suppress("UNCHECKED_CAST")
    val outputTensor = outputHandle.ref.payload as io.tlaloc.core.DTensor<io.tlaloc.core.ScalarShape, io.tlaloc.core.F32>
    println("Output value:   ${outputTensor.hostF32().single()}")
    // Expected: relu([1,-2,3,-4,5])² summed = [1,0,9,0,25].sum() = 35.0
}
