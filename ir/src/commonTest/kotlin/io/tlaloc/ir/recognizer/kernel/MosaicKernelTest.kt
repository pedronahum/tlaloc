package io.tlaloc.ir.recognizer.kernel

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.MosaicKernelAttrs
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.mosaicKernel
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.recognizer.cost.estimateOp
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * MOSAIC_KERNEL: the op, its reference semantics, and the per-target
 * decision [lowerMosaicKernels] makes.
 */
class MosaicKernelTest {

    private val t = DxirType(F32, listOf(2, 3))
    private val kernel = MosaicKernel(kernelName = "probe", bodyBase64 = "TUzvUg==")

    /** (a, b) → (a + b, a * b): a two-result reference. */
    private val reference: DxirFunction = DxirBuilder.function("probe_ref") {
        val a = param("a", t)
        val b = param("b", t)
        listOf(op(OpKind.ADD, listOf(a, b), t), op(OpKind.MUL, listOf(a, b), t))
    }

    private fun program(fallback: Boolean, k: MosaicKernel = kernel): DxirFunction =
        DxirBuilder.function("uses_probe") {
            val a = param("a", t)
            val b = param("b", t)
            val call = mosaicKernel(listOf(a, b), k, reference, referenceFallback = fallback)
            listOf(op(OpKind.SUB, listOf(call.result(1), call.result(0)), t), call.result(0))
        }

    private val a = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
    private val b = floatArrayOf(0.5f, -1f, 2f, 0f, 3f, -2f)

    private fun expected(): List<FloatArray> {
        val sum = FloatArray(6) { a[it] + b[it] }
        val prod = FloatArray(6) { a[it] * b[it] }
        return listOf(FloatArray(6) { prod[it] - sum[it] }, sum)
    }

    @Test
    fun backendConfigJsonMatchesJaxLayout() {
        assertEquals(
            "{\"custom_call_config\": {\"body\": \"TUzvUg==\", \"serialization_format\": 1, " +
                "\"needs_layout_passes\": true}}",
            kernel.backendConfigJson(),
        )
    }

    @Test
    fun backendConfigJsonEscapesStringsAndNestsObjects() {
        val k = kernel.copy(
            customCallConfig = linkedMapOf(
                "has_communication" to true,
                "collective_id" to 7,
                "cost_estimate" to linkedMapOf("flops" to 12L, "bytes_accessed" to 48L),
                "note" to "a \"quoted\" \\ path\n",
                "flags" to listOf(1.5, false),
            ),
        )
        assertEquals(
            "{\"custom_call_config\": {\"body\": \"TUzvUg==\", \"has_communication\": true, " +
                "\"collective_id\": 7, \"cost_estimate\": {\"flops\": 12, \"bytes_accessed\": 48}, " +
                "\"note\": \"a \\\"quoted\\\" \\\\ path\\n\", \"flags\": [1.5, false]}}",
            k.backendConfigJson(),
        )
    }

    @Test
    fun kernelRefusesMalformedPayloads() {
        assertFailsWith<IllegalArgumentException> { MosaicKernel("k", "not base64!") }
        assertFailsWith<IllegalArgumentException> {
            MosaicKernel("k", "AAAA", customCallConfig = mapOf("body" to "x"))
        }
        assertFailsWith<IllegalArgumentException> {
            MosaicKernel("k", "AAAA", inputOutputAliases = mapOf(0 to 0, 1 to 0))
        }
    }

    @Test
    fun builderRefusesAReferenceWithTheWrongSignature() {
        val wrong = DxirBuilder.function("wrong") {
            val a = param("a", DxirType(F32, listOf(3, 2)))
            val b = param("b", t)
            listOf(op(OpKind.MUL, listOf(b, b), t))
        }
        val e = assertFailsWith<IllegalArgumentException> {
            DxirBuilder.function("bad") {
                val a = param("a", t)
                val b = param("b", t)
                listOf(mosaicKernel(listOf(a, b), kernel, wrong))
            }
        }
        assertTrue("operand 0" in e.message!!, e.message)
    }

    @Test
    fun aliasMustJoinEqualTypes() {
        val scalarRef = DxirBuilder.function("sum") {
            val a = param("a", t)
            listOf(op(OpKind.SUM, listOf(a), DxirType(F32, emptyList())))
        }
        val e = assertFailsWith<IllegalArgumentException> {
            DxirBuilder.function("bad_alias") {
                val a = param("a", t)
                listOf(mosaicKernel(listOf(a), kernel.copy(inputOutputAliases = mapOf(0 to 0)), scalarRef))
            }
        }
        assertTrue("in-place result must have its operand's type" in e.message!!, e.message)
    }

    @Test
    fun interpreterEvaluatesTheReferenceWithEveryResult() {
        val got = DxirInterpreter.evalFunction(program(fallback = false), listOf(a, b))
        val want = expected()
        assertEquals(2, got.size)
        for (i in want.indices) assertContentEquals(want[i], got[i])
    }

    @Test
    fun tpuTargetClaimsTheOp() {
        val k = kernel.copy(inputOutputAliases = mapOf(1 to 0))
        val lowered = lowerMosaicKernels(program(fallback = false, k = k), KernelTarget.GOOGLE_TPU_V5E)
        val op = lowered.body.filterIsInstance<DxirOp>().single { it.op == OpKind.MOSAIC_KERNEL }
        val d = assertNotNull(op.attrs[KernelDescriptor.ATTR_KEY] as? KernelDescriptor)
        assertEquals(MosaicKernel.CALL_TARGET, d.kernelName)
        assertEquals("tpu_v5e", d.targetArch)
        assertSame(k, d.mosaic)
        assertEquals(listOf(OutputOperandAlias(outputIndex = 0, operandIndex = 1)), d.outputOperandAliases)
        // The claimed program still means the same thing.
        val got = DxirInterpreter.evalFunction(lowered, listOf(a, b))
        for (i in got.indices) assertContentEquals(expected()[i], got[i])
    }

    @Test
    fun nonTpuTargetRefusesByNameWithoutAFallback() {
        val e = assertFailsWith<IllegalStateException> {
            lowerMosaicKernels(program(fallback = false), KernelTarget.NVIDIA_GB10)
        }
        assertTrue("'probe'" in e.message!! && "no reference fallback" in e.message!!, e.message)
    }

    @Test
    fun nonTpuTargetInlinesTheReferenceWhenAllowed() {
        val lowered = lowerMosaicKernels(program(fallback = true), KernelTarget.CPU_GENERIC)
        val kinds = lowered.body.filterIsInstance<DxirOp>().map { it.op }
        assertEquals(listOf(OpKind.ADD, OpKind.MUL, OpKind.SUB), kinds)
        val got = DxirInterpreter.evalFunction(lowered, listOf(a, b))
        for (i in got.indices) assertContentEquals(expected()[i], got[i])
    }

    @Test
    fun programsWithoutTheOpPassThroughUnchanged() {
        val fn = DxirBuilder.function("plain") {
            val x = param("x", t)
            listOf(op(OpKind.NEG, listOf(x), t))
        }
        assertSame(fn, lowerMosaicKernels(fn, KernelTarget.NVIDIA_GB10))
    }

    @Test
    fun reverseModeRefusesByName() {
        val fn = DxirBuilder.function("loss") {
            val a = param("a", t)
            val b = param("b", t)
            val call = mosaicKernel(listOf(a, b), kernel, reference, referenceFallback = true)
            listOf(op(OpKind.SUM, listOf(call.result(0)), DxirType(F32, emptyList())))
        }
        val e = assertFailsWith<IllegalStateException> { DxirReverseTransform.apply(fn) }
        assertTrue("MOSAIC_KERNEL is INFERENCE-ONLY" in e.message!!, e.message)
    }

    @Test
    fun costIsTheReferenceFlops() {
        val op = program(fallback = true).body.filterIsInstance<DxirOp>().single { it.op == OpKind.MOSAIC_KERNEL }
        assertEquals(12.0, estimateOp(op).flops)
        assertEquals(MosaicKernelAttrs.parse(op, "test").reference.name, "probe_ref")
    }
}
