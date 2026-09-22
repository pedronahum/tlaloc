/**
 * The four worlds — where your code is allowed to be, and what may cross.
 *
 * Tlaloc splits a distributed program into four scopes, each a Kotlin type:
 *
 *   Kernel        — the maths. Elementwise ops, matmuls, reductions. No I/O,
 *                   no dispatch, no cluster.
 *   Orchestration — builds Kernel bodies into artifacts (`program { }`).
 *   Program       — composes artifacts into a graph (`workflow { }`).
 *   Cluster       — runs the graph on real hardware (Maestro, Kubernetes).
 *
 * The scopes are enforced by ordinary Kotlin receivers, so "I called a
 * dispatch op from inside a kernel body" is a compile error rather than a
 * deadlock in production. And what crosses a step boundary is never a raw
 * tensor: it is a typed `BufferHandle<T, M>`, carrying both the value's type
 * and the mesh `M` it lives on. A handle from the wrong step, or from the
 * wrong mesh, does not type-check.
 *
 * This example:
 *
 *   [1] builds ONE step with `program { }` and runs it through its shim;
 *   [2] composes TWO steps with `workflow { }`, letting a BufferHandle flow
 *       between them, and prints the recorded edge;
 *   [3] emits the Maestro JSON descriptor — the artifact a real cluster
 *       ingests;
 *   [4] points at the two programs that do NOT compile.
 */
import java.io.File
import io.tlaloc.autograd.relu
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.core.BufferHandle
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.Mesh0
import io.tlaloc.core.Rank1
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.Tlaloc
import io.tlaloc.core.hostF32
import io.tlaloc.maestro.MaestroDescriptor
import io.tlaloc.maestro.handleOn
import io.tlaloc.maestro.program
import io.tlaloc.maestro.workflow

/** The tensor every part of this example starts from: relu drops 2 of its 5 values. */
private val input = Tensors.f32Vector<Sym>(floatArrayOf(1f, -2f, 3f, -4f, 5f))

/** Pull the value back out of a handle. v1 carries it in `HandleRef.payload`. */
@Suppress("UNCHECKED_CAST")
private fun scalarOf(handle: BufferHandle<DTensor<ScalarShape, F32>, Mesh0>): Float =
    (handle.ref.payload as DTensor<ScalarShape, F32>).hostF32().single()

// ---------------------------------------------------------------------------
// [1] One step. `program { }` turns a Kernel body into a shippable artifact.
// ---------------------------------------------------------------------------

private fun singleStepProgram() {
    // `program(...)` is an extension on OrchestrationScope — the `Tlaloc`
    // singleton is one, which is why the call reads `Tlaloc.program`. Call it
    // from nowhere and it does not resolve.
    val activateAndScore = Tlaloc.program(
        name = "activate_and_score",
        input = input,
        mesh = Mesh0,
    ) { x ->
        // Inside this lambda the receiver is KernelScope, and `x` is a Tracer:
        // it shadow-executes the computation while simultaneously recording
        // DXIR for StableHLO emission. Kernel ops are callable here;
        // orchestration ops are not (the DslMarker sees to that).
        val activated = x.relu()
        (activated * activated).sum()
    }

    println("[1] one step, built by `program { }`")
    println("    name:              ${activateAndScore.name}")
    println("    body hash:         ${activateAndScore.manifest.bodyHash}")
    println("    body size:         ${activateAndScore.body.size} bytes of StableHLO")
    println("    inputs:            ${activateAndScore.manifest.inputs}")
    println("    outputs:           ${activateAndScore.manifest.outputs}")
    println("    mesh requirement:  ${activateAndScore.manifest.meshRequirement}")

    // Run the step's shim. Note what goes in and what comes out: handles, not
    // tensors. `handleOn(Mesh0)` is how a tensor from outside the system
    // enters it.
    val result = activateAndScore(input.handleOn(Mesh0))
    println("    result:            ${scalarOf(result)}   (relu([1,-2,3,-4,5])^2 summed = 1+9+25)")
}

// ---------------------------------------------------------------------------
// [2] + [3] Two steps, composed, with the handle flowing between them.
// ---------------------------------------------------------------------------

private fun twoStepWorkflow() {
    val activate = Tlaloc.program("activate", input, Mesh0) { x -> x.relu() }
    val score = Tlaloc.program("score", input, Mesh0) { x -> (x * x).sum() }

    // The output of the `workflow { }` block is the graph; the value falls out
    // of the last `step(...)` call, which returns the typed handle. We keep it
    // in a local so we can print it after the block.
    var finalHandle: BufferHandle<DTensor<ScalarShape, F32>, Mesh0>? = null

    val wf = Tlaloc.workflow("activate_then_score") {
        val seeded: BufferHandle<DTensor<Rank1<Sym>, F32>, Mesh0> = seed(input, Mesh0)
        val activated = step(activate, seeded)
        // `score` takes exactly the handle type `activate` produced. Feed it
        // anything else — a bare tensor, a rank-2 handle, a handle on another
        // mesh — and Kotlin rejects the call. No plugin diagnostic: this is
        // the stock type checker doing the work.
        finalHandle = step(score, activated)
    }

    println()
    println("[2] two steps, composed by `workflow { }`")
    println("    workflow:  ${wf.name}")
    println("    steps:     ${wf.steps.map { it.name }}")
    println("    edges:     ${wf.edges.map { "${it.fromStep} -> ${it.toStep} (${it.reshardKind})" }}")
    println("    result:    ${scalarOf(finalHandle!!)}   (same answer, two artifacts)")
    println("    (reshard kind is None because both steps sit on Mesh0 with the same")
    println("     axis names; a mesh change or a transpose records an explicit edge.)")

    // The descriptor is the hand-off to the cluster world: a real Maestro
    // instance ingests this JSON and launches each step's artifact as a
    // Kubernetes job on hardware matching the step's backend matrix.
    val descriptor = MaestroDescriptor.emit(wf)
    println()
    println("[3] the Maestro descriptor a cluster would ingest (${descriptor.length} chars, first 240):")
    println("    ${descriptor.take(240).replace("\n", "\n    ")}...")
}

fun main() {
    singleStepProgram()
    twoStepWorkflow()

    println()
    println("[4] the program that does NOT compile")
    println()
    val source = System.getProperty("tlaloc.example.shapeErrorSource")?.let(::File)
    if (source != null && source.isFile) {
        source.readLines()
            .dropWhile { !it.startsWith("fun wrongHandleType") }
            .forEach { println("    $it") }
    } else {
        println("    (source not found — run via `./gradlew -p examples/internals/four-worlds run`)")
    }
    println()
    println("    `activate` yields a Rank1 handle; `matrixStep` wants a Rank2 one. What")
    println("    crosses a step boundary is a typed BufferHandle<T, M>, so the wrong one")
    println("    does not type-check. Watch it:")
    println()
    println("        ./gradlew -p examples/internals/four-worlds shapeError")
    println()
    println("    e: WorldErrors.kt:44:26 Argument type mismatch: actual type is")
    println("       'BufferHandle<DTensor<Rank1<Sym>, F32>, Mesh0>', but")
    println("       'BufferHandle<DTensor<Rank2<Sym, Sym>, F32>, Mesh0>' was expected.")
    println()
    println("    A SECOND claim used to live here, commented out: that calling")
    println("    `Tlaloc.program` from inside a Kernel body would not resolve. Turning")
    println("    these comments into a file the build actually compiles showed that it")
    println("    DOES resolve, so the claim is gone. The @WorldScope DslMarker shadows")
    println("    an IMPLICIT outer receiver inside a nested builder; `Tlaloc.program`")
    println("    names its receiver explicitly, and DslMarker never blocks that. The")
    println("    world separation you can rely on today is the typed handle above.")
    println()
    println("four-worlds OK")
}
