package io.tlaloc.maestro

import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.capture
import io.tlaloc.core.BufferHandle
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HandleRef
import io.tlaloc.core.KernelScope
import io.tlaloc.core.Mesh
import io.tlaloc.core.OrchestrationScope
import io.tlaloc.core.Shape
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirModule
import io.tlaloc.stablehlo.toStablehlo
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicLong

/**
 * `program { }` builder. Lowers a body lambda over
 * [Tracer] into a [MaestroStep] artifact with content-addressed StableHLO
 * body bytes.
 *
 * # Why an [OrchestrationScope] receiver
 *
 * The four-worlds discipline puts
 * `program { }` in [OrchestrationScope] — that's where buffer management
 * + dispatch + futures live. Inside the body lambda, the receiver is
 * [KernelScope]: kernel-only ops compile, orchestration ops do not.
 *
 * # BufferHandle at step boundaries
 *
 * By design, "BufferHandle is the only thing that crosses a step
 * boundary — never a materialized tensor, never an untyped reference."
 * The returned [MaestroStep] is therefore typed
 * `MaestroStep<BufferHandle<DTensor<Sin, F32>, M>, BufferHandle<DTensor<Sout, F32>, M>>`
 * — the type checker enforces mesh placement [M] flowing through the step
 * graph at compose time. v1's [HandleRef.payload] carries the materialized
 * [DTensor]; there is no device buffer pool yet.
 *
 * # v1 shim semantics
 *
 * The returned [MaestroStep.shim] **re-traces** the original body lambda
 * against the materialized input tensor each invocation. Works for any
 * Tracer-supported op. It does not dispatch the compiled [MaestroStep.body].
 *
 * @param name step name; flows into the manifest and synthesised StableHLO
 *   function name.
 * @param input concrete input tensor used to seed the trace; defines
 *   the param-typing for the resulting [MaestroStep].
 * @param mesh phantom-typed mesh placement [M] for both the input and
 *   output [BufferHandle]. Only single-mesh programs are covered; cross-
 *   mesh splits are reshard ops introduced by [io.tlaloc.maestro.workflow]
 *   builder edges.
 * @param body the kernel-world lambda. Receiver is [KernelScope]; argument
 *   is a [Tracer] over the input shape.
 */
fun <Sin : Shape, Sout : Shape, M : Mesh> OrchestrationScope.program(
    name: String,
    input: DTensor<Sin, F32>,
    mesh: M,
    body: KernelScope.(Tracer<Sin>) -> Tracer<Sout>,
): MaestroStep<BufferHandle<DTensor<Sin, F32>, M>, BufferHandle<DTensor<Sout, F32>, M>> {
    val kernelScope = object : KernelScope {}

    // Trace the body once to capture a DxirFunction (for StableHLO
    // emission). The capture machinery from :autograd handles the
    // tape-to-DXIR lowering; we provide the body wrapped to inject the
    // KernelScope receiver.
    val captured: DxirFunction = capture(
        f = { tracer -> kernelScope.body(tracer) },
        input = input,
        name = name,
    )

    // Emit StableHLO+SDY for the captured DxirFunction. UTF-8 encoding
    // gives us a deterministic byte stream for hashing.
    val stableHloText = DxirModule(listOf(captured)).toStablehlo()
    val bodyBytes = stableHloText.toByteArray(Charsets.UTF_8)
    val bodyHash = sha256Hex(bodyBytes)

    // Build the manifest. v1 carries one input + one output; multi-arity
    // is a v2 surface.
    val inputDescriptor = TypeDescriptor.fromDxirType(captured.params.single().type)
    val outputDescriptor = TypeDescriptor.fromDxirType(captured.returns.single().type)
    val manifest = ProgramManifest(
        name = name,
        inputs = listOf(inputDescriptor),
        outputs = listOf(outputDescriptor),
        meshRequirement = mesh::class.simpleName ?: "Mesh0",
        bodyHash = bodyHash,
    )

    // v1 shim: materialize the input handle's payload (a DTensor), re-
    // trace the body lambda against it, and wrap the output in a fresh
    // BufferHandle. Tracer shadow-execution gives us the materialized
    // value on `out.toDTensor()`.
    @Suppress("UNCHECKED_CAST")
    val shim: (BufferHandle<DTensor<Sin, F32>, M>) -> BufferHandle<DTensor<Sout, F32>, M> = { handleIn ->
        val tensorIn = handleIn.ref.payload as? DTensor<Sin, F32>
            ?: error("BufferHandle ref ${handleIn.ref.nativeId} has no DTensor payload (v1 stub)")
        var outputValue: DTensor<Sout, F32>? = null
        capture<Sin>(
            f = { tracer ->
                val out: Tracer<Sout> = kernelScope.body(tracer)
                outputValue = out.toDTensor()
                out
            },
            input = tensorIn,
            name = "${name}_shim",
        )
        val tensorOut = checkNotNull(outputValue) { "shim trace for '$name' produced no output" }
        BufferHandle(HandleRef(nativeId = nextNativeId(), payload = tensorOut))
    }

    return MaestroStep(
        name = name,
        manifest = manifest,
        body = bodyBytes,
        mesh = mesh,
        shim = shim,
    )
}

/**
 * Convenience: wrap a [DTensor] in a fresh [BufferHandle] anchored to
 * mesh [M]. The stub buffer pool stores the tensor as the ref payload;
 * no device buffer is allocated.
 */
fun <S : Shape, M : Mesh> DTensor<S, F32>.handleOn(@Suppress("UNUSED_PARAMETER") mesh: M): BufferHandle<DTensor<S, F32>, M> =
    BufferHandle(HandleRef(nativeId = nextNativeId(), payload = this))

private val nativeIdCounter = AtomicLong(0L)

internal fun nextNativeId(): Long = nativeIdCounter.incrementAndGet()

/**
 * SHA-256 of [bytes] as a lowercase hex string. The content-
 * addressing primitive: two `program { }` invocations with the same
 * lambda body and same input shape produce identical [ProgramManifest.bodyHash]es,
 * making manifest equality structural.
 */
internal fun sha256Hex(bytes: ByteArray): String {
    val md = MessageDigest.getInstance("SHA-256")
    val digest = md.digest(bytes)
    return buildString(digest.size * 2) {
        for (b in digest) {
            val v = b.toInt() and 0xff
            append("0123456789abcdef"[v ushr 4])
            append("0123456789abcdef"[v and 0xf])
        }
    }
}
