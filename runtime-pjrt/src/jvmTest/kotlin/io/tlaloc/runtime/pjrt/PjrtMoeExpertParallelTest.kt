package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.MoeExpertsAttrs
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Expert-parallel MOE_EXPERTS ([MoeExpertsAttrs.EXPERT_PARALLEL]): 8 experts, top 2, held as
 * two slices of 4 (offsets 0 and 4, one program, the offset an operand). The two slices' results
 * sum to the whole op's within f32 rounding (each slice rounds its own sum), in the interpreter
 * and on XLA:GPU, for decode rows (the gathered form), prefill rows (the tiled loop), the
 * ragged-dot lowering and sigmoid routing.
 */
class PjrtMoeExpertParallelTest {

    private val e = 8
    private val h = 32
    private val inter = 16

    private fun fn(rows: Int, experts: Int, ep: Boolean, sigmoid: Boolean, ragged: Boolean): DxirFunction =
        DxirBuilder.function("moe_ep") {
            val x = param("x", DxirType(F32, listOf(rows, h)))
            val l = param("logits", DxirType(F32, listOf(rows, e)))
            val gu = param("gateUp", DxirType(F32, listOf(experts, 2 * inter, h)))
            val dn = param("down", DxirType(F32, listOf(experts, h, inter)))
            val bias = if (sigmoid) listOf(param("bias", DxirType(F32, listOf(e)))) else emptyList()
            val off = if (ep) listOf(param("offset", DxirType(I32, listOf(1)))) else emptyList()
            val attrs = mapOf<String, Any>("top_k" to 2) +
                (if (sigmoid) mapOf("routing" to MoeExpertsAttrs.SIGMOID_BIAS, "routed_scale" to 2.5) else emptyMap()) +
                (if (ep) mapOf(MoeExpertsAttrs.EXPERT_PARALLEL to true) else emptyMap()) +
                (if (ragged) mapOf(MoeExpertsAttrs.LOWERING to MoeExpertsAttrs.RAGGED_DOT) else emptyMap())
            listOf(op(OpKind.MOE_EXPERTS, listOf(x, l, gu, dn) + bias + off, DxirType(F32, listOf(rows, h)), attrs))
        }

    private fun case(s: PjrtSession, rows: Int, sigmoid: Boolean, ragged: Boolean) {
        val rnd = Random(rows * 7 + (if (sigmoid) 1 else 0) + (if (ragged) 2 else 0))
        val x = FloatArray(rows * h) { rnd.nextFloat() * 2 - 1 }
        val l = FloatArray(rows * e) { rnd.nextFloat() * 4 - 2 }
        val gu = FloatArray(e * 2 * inter * h) { (rnd.nextFloat() * 2 - 1) * 0.2f }
        val dn = FloatArray(e * h * inter) { (rnd.nextFloat() * 2 - 1) * 0.2f }
        val bias = FloatArray(e) { rnd.nextFloat() * 0.5f }
        val tail = if (sigmoid) listOf(bias) else emptyList()
        val whole = fn(rows, e, ep = false, sigmoid, ragged)
        val half = fn(rows, e / 2, ep = true, sigmoid, ragged)
        val ins = listOf(x, l, gu, dn) + tail
        fun slice(a: FloatArray, part: Int) = a.copyOfRange(part * a.size / 2, (part + 1) * a.size / 2)
        fun halfIns(part: Int) = listOf(x, l, slice(gu, part), slice(dn, part)) + tail + listOf(floatArrayOf(part * e / 2f))
        val want = DxirInterpreter.evalFunction(whole, ins)[0]
        val parts = (0..1).map { DxirInterpreter.evalFunction(half, halfIns(it))[0] }
        val wantMag = want.maxOf { abs(it) }
        val di = want.indices.maxOf { abs(parts[0][it] + parts[1][it] - want[it]) }
        assertTrue(di <= 1e-6f * wantMag, "interpreter, rows $rows: two slices against the whole op $di of $wantMag")
        fun intsOf(a: List<FloatArray>): List<Any> = a.mapIndexed { i, v -> if (i == a.size - 1) IntArray(1) { v[0].toInt() } else v }
        val got = s.runOnHost(whole, ins).single() as FloatArray
        val gotParts = (0..1).map { s.runOnHost(half, intsOf(halfIns(it))).single() as FloatArray }
        var mag = 0f
        var d = 0f
        for (j in want.indices) {
            mag = maxOf(mag, abs(want[j]))
            d = maxOf(d, abs(gotParts[0][j] + gotParts[1][j] - got[j]), abs(got[j] - want[j]))
        }
        println("[moe-ep] rows=$rows sigmoid=$sigmoid ragged=$ragged: two slices against the whole op $d (largest |y| $mag)")
        assertTrue(d <= 1e-5f * mag + 1e-6f, "expert-parallel slices differ from the whole op: $d of $mag")
    }

    @Test
    fun twoSlicesOfTheExpertsSumToTheWholeOp() {
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        TestBackend.session().use { s ->
            for (rows in listOf(3, 300)) for (sigmoid in listOf(false, true)) case(s, rows, sigmoid, ragged = false)
            case(s, 300, sigmoid = false, ragged = true)
        }
    }
}
