/**
 * Layer 3 — the artifact carries the device decision, not the runtime.
 *
 * Compile an attention block for an H100 and for a TPU v6e and you get, from
 * XLA, the same StableHLO twice: the fusion decision happens inside the
 * runtime, after the bytes have shipped. Tlaloc makes the decision *upstream*.
 * The bytes that go to a GB10 name `@flash_attn_v3`; the bytes that go to a
 * TPU name `@tpu_pallas_flash_attention`; the bytes that go to a generic CPU
 * name nothing at all, because they are fully decomposed into portable
 * primitives.
 *
 * This example walks the whole pipeline, one stage per printed section:
 *
 *   [1] recognize   — find the `MATMUL -> SOFTMAX -> MATMUL` shape in DXIR.
 *   [2] coarsen     — collapse those three ops into one `OpKind.COARSENED`
 *                     carrying an analytical primal body AND gradient body,
 *                     so the fused form stays differentiable.
 *   [3] lower       — pick a kernel per target, emit StableHLO, and show that
 *                     the artifacts genuinely differ byte for byte.
 *   [4] KV-quant    — ask for an FP8 KV cache and watch some targets accept
 *                     and others decline with a reason, without failing the
 *                     compile.
 *   [5] matrix      — serialize the per-target rows a cluster scheduler reads
 *                     to place the job on the right hardware.
 *
 * Everything here is pure compilation: no GPU, no TPU, no network. It runs on
 * a laptop.
 */
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.coarsener.decomposeCoarsened
import io.tlaloc.ir.recognizer.kernel.KernelDescriptor
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.lowerKernelChoice
import io.tlaloc.ir.recognizer.quant.KvQuantConfig
import io.tlaloc.ir.recognizer.quant.applyKvQuantWithDiagnostics
import io.tlaloc.ir.recognizer.recognizeAll
import io.tlaloc.maestro.populateBackendMatrix
import io.tlaloc.stablehlo.toStablehlo

/**
 * The user's hand-written attention forward, as DXIR: `softmax(QK) · V` with
 * nothing fused. This is what the recognizers see — exactly the shape you get
 * from writing attention out longhand in Kotlin.
 *
 *     Q [8,4] · K [4,8] -> S [8,8];  softmax(S) [8,8];  · V [8,4] -> O [8,4]
 */
private fun attentionForward(): DxirFunction = DxirBuilder.function("attn") {
    val q = param("Q", DxirType(F32, listOf(8, 4)))
    val k = param("K", DxirType(F32, listOf(4, 8)))
    val v = param("V", DxirType(F32, listOf(8, 4)))
    val qk = op(OpKind.MATMUL, listOf(q, k), DxirType(F32, listOf(8, 8)))
    val sm = op(OpKind.SOFTMAX, listOf(qk), DxirType(F32, listOf(8, 8)))
    val out = op(OpKind.MATMUL, listOf(sm, v), DxirType(F32, listOf(8, 4)))
    listOf(out)
}

private fun rule(title: String) {
    println()
    println("=".repeat(76))
    println(title)
    println("=".repeat(76))
}

// ---------------------------------------------------------------------------
// [1] + [2] recognize, then coarsen.
// ---------------------------------------------------------------------------

private fun recognizeAndCoarsen(): DxirFunction {
    val user = attentionForward()

    rule("[1] + [2]  recognize the shape, then coarsen it into one op")
    println("before:   ${user.body.filterIsInstance<DxirOp>().map { it.op }}")

    val matches = recognizeAll(user)
    println("matched:  ${matches.map { it.patternName }}")

    val coarsened = coarsenRecognizedPatterns(user, matches)
    val ops = coarsened.body.filterIsInstance<DxirOp>()
    println("after:    ${ops.map { it.op }}")

    // The COARSENED op is not a black box. It carries the analytical primal
    // body (what to compute) and the analytical gradient body (how to
    // differentiate the fused form), so fusing does not cost you autodiff.
    val co = ops.single { it.op == OpKind.COARSENED }
    val primal = co.attrs["primal_body"] as DxirFunction
    val gradient = co.attrs["gradient_body"] as DxirFunction
    @Suppress("UNCHECKED_CAST")
    val readsPrimal = co.attrs["reads_primal_indices"] as Set<Int>

    println()
    println("the COARSENED envelope:")
    println("  operands:            ${co.operands.size}  (Q, K, V)")
    println("  result type:         ${co.type}")
    println("  primal_body:         ${primal.name}  ops=${primal.body.filterIsInstance<DxirOp>().map { it.op }}")
    println("  gradient_body:       ${gradient.name}  ${gradient.params.size} params (dO + 3 primal) -> ${gradient.returns.size} grads")
    println("  reads_primal:        $readsPrimal  (Q=0, K=1, V=2 — all three are needed by the backward pass)")
    return coarsened
}

// ---------------------------------------------------------------------------
// [3] lower per target and emit. This is the section that matters.
// ---------------------------------------------------------------------------

private fun perTargetArtifacts(coarsened: DxirFunction) {
    val targets = listOf(
        KernelTarget.NVIDIA_GB10,
        KernelTarget.NVIDIA_H100,
        KernelTarget.GOOGLE_TPU_V6E,
        KernelTarget.AWS_TRAINIUM2,
        KernelTarget.CPU_GENERIC,
    )

    rule("[3]  same Kotlin source, one artifact per device")
    val artifacts = LinkedHashMap<KernelTarget, String>()
    for (t in targets) {
        // `lowerKernelChoice` consults the kernel-template registry for this
        // (vendor, arch). A hit annotates the COARSENED op with a
        // KernelDescriptor; a miss leaves it un-annotated.
        val lowered = lowerKernelChoice(coarsened, t)
        val kernel = lowered.body.filterIsInstance<DxirOp>()
            .firstOrNull { it.op == OpKind.COARSENED }
            ?.attrs?.get(KernelDescriptor.ATTR_KEY) as KernelDescriptor?

        // `decomposeCoarsened` is the fallback that makes this safe: any
        // COARSENED op without a kernel is replaced by its inlined primal
        // body, so every target gets runnable StableHLO either way.
        val mlir = decomposeCoarsened(lowered).toStablehlo("")
        artifacts[t] = mlir

        val customCalls = mlir.lineSequence().filter { "stablehlo.custom_call" in it }.map { it.trim() }.toList()
        println()
        println("--- ${t.vendor}/${t.arch}")
        println("    kernel chosen:   ${kernel?.kernelName ?: "(none — decomposed to portable primitives)"}"
        )
        if (customCalls.isEmpty()) {
            println("    emitted:         no custom_call; ${mlir.lineSequence().count()} lines of plain StableHLO")
        } else {
            customCalls.forEach { println("    emitted:         $it") }
        }
    }

    // The point, stated as a fact rather than a claim: the bytes differ.
    println()
    println("pairwise artifact comparison:")
    val keys = artifacts.keys.toList()
    for (i in keys.indices) {
        for (j in (i + 1) until keys.size) {
            val a = keys[i]
            val b = keys[j]
            val verdict = if (artifacts[a] == artifacts[b]) "IDENTICAL" else "DIFFERENT"
            println("  %-22s vs %-22s -> %s".format("${a.vendor}/${a.arch}", "${b.vendor}/${b.arch}", verdict))
        }
    }
}

// ---------------------------------------------------------------------------
// [4] KV-quant: a request some targets cannot honour, and must not silently drop.
// ---------------------------------------------------------------------------

private fun kvQuantAcrossTargets(coarsened: DxirFunction) {
    val request = KvQuantConfig.FP8_PER_HEAD

    rule("[4]  ask for an FP8 KV cache across a heterogeneous fleet")
    println("requested: ${request.dtype.nameTag} / ${request.scaleStrategy}")
    println()
    for (t in listOf(
        KernelTarget.NVIDIA_H100,
        KernelTarget.NVIDIA_A100,
        KernelTarget.AWS_TRAINIUM2,
        KernelTarget.GOOGLE_TPU_V5E,
    )) {
        val lowered = lowerKernelChoice(coarsened, t)
        val (afterQuant, diagnostics) = applyKvQuantWithDiagnostics(lowered, request)
        val co = afterQuant.body.filterIsInstance<DxirOp>().single { it.op == OpKind.COARSENED }
        val tag = "${t.vendor}/${t.arch}"
        if (co.attrs[KvQuantConfig.ATTR_KEY] != null) {
            println("  %-22s  accepted — KV cache will be %s".format(tag, request.dtype.nameTag))
        } else {
            println("  %-22s  declined — %s".format(tag, diagnostics.singleOrNull()?.reason ?: "no diagnostic"))
        }
    }
    println()
    println("the compile does not fail: a mixed fleet is a first-class deployment shape,")
    println("and the diagnostic stream is where the per-target answer surfaces.")
}

// ---------------------------------------------------------------------------
// [5] the backend matrix — the rows a scheduler reads.
// ---------------------------------------------------------------------------

private fun backendMatrix() {
    // A slightly larger attention block, so the cost model has something to
    // chew on.
    val attn = DxirBuilder.function("attn") {
        val q = param("Q", DxirType(F32, listOf(64, 64)))
        val k = param("K", DxirType(F32, listOf(64, 64)))
        val v = param("V", DxirType(F32, listOf(64, 64)))
        val qk = op(OpKind.MATMUL, listOf(q, k), DxirType(F32, listOf(64, 64)))
        val sm = op(OpKind.SOFTMAX, listOf(qk), DxirType(F32, listOf(64, 64)))
        val out = op(OpKind.MATMUL, listOf(sm, v), DxirType(F32, listOf(64, 64)))
        listOf(out)
    }

    val matrix = populateBackendMatrix(
        attn,
        listOf(
            KernelTarget.NVIDIA_GB10,
            KernelTarget.NVIDIA_H100,
            KernelTarget.NVIDIA_A100,
            KernelTarget.AMD_MI300X,
            KernelTarget.GOOGLE_TPU_V4,
            KernelTarget.GOOGLE_TPU_V5E,
            KernelTarget.GOOGLE_TPU_V6E,
            KernelTarget.AWS_TRAINIUM2,
            KernelTarget.CPU_GENERIC,
        ),
        kvQuant = KvQuantConfig.FP8_PER_HEAD,
    )

    rule("[5]  the backend matrix, cost-ordered")
    println("%-22s  %-34s  %-9s  %s".format("target", "kernel", "kv_quant", "cost (us)"))
    println("-".repeat(76))
    for (row in matrix.sortedBy { it.costMicroseconds ?: Double.POSITIVE_INFINITY }) {
        println(
            "%-22s  %-34s  %-9s  %s".format(
                "${row.vendor}/${row.arch}",
                row.kernelName ?: "(decompose)",
                row.kvQuantDtype ?: "-",
                row.costMicroseconds?.let { "%.2f".format(it) } ?: "-",
            ),
        )
    }
    println()
    println("one row, as it travels (JSON inside the workflow params):")
    println("  ${matrix.first().toJson()}")
    println()
    println("at launch time the runtime matches the cluster's (vendor, arch) against these")
    println("rows and turns the winner into K8s nodeSelector + accelerator + gpu fields.")
}

fun main() {
    val coarsened = recognizeAndCoarsen()
    perTargetArtifacts(coarsened)
    kvQuantAcrossTargets(coarsened)
    backendMatrix()
    println()
    println("layer3 OK")
}
