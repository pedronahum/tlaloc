package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** [OpKind.MOE_EXPERTS]: routing, tie-breaking and the refusals. Parity with transformers is MoeExpertsFixtureTest. */
class MoeExpertsTest {

    private val h = 2
    private val e = 4
    private val i = 1

    private fun fn(rows: Int, k: Int): DxirFunction = DxirBuilder.function("moe") {
        val x = param("x", DxirType(F32, listOf(rows, h)))
        val l = param("logits", DxirType(F32, listOf(rows, e)))
        val gu = param("gateUp", DxirType(F32, listOf(e, 2 * i, h)))
        val dn = param("down", DxirType(F32, listOf(e, h, i)))
        listOf(op(OpKind.MOE_EXPERTS, listOf(x, l, gu, dn), DxirType(F32, listOf(rows, h)), mapOf("top_k" to k)))
    }

    // Expert j's gate and up read x[0] with gain 1, its down writes (j + 1) to output 0 and 0 to output 1,
    // so expert j's output for x = (a, 0) is (j + 1) * silu(a) * a.
    private val gateUp = FloatArray(e * 2 * i * h) { idx -> if (idx % h == 0) 1f else 0f }
    private val down = FloatArray(e * h * i) { idx -> if ((idx / i) % h == 0) (idx / (h * i) + 1).toFloat() else 0f }
    private fun silu(a: Double) = a / (1 + exp(-a))

    @Test
    fun theTopExpertsAreWeightedByTheirRenormalizedProbabilities() {
        val logits = floatArrayOf(0f, 2f, 1f, -1f)
        val y = DxirInterpreter.evalFunction(fn(1, 2), listOf(floatArrayOf(1f, 0f), logits, gateUp, down))[0]
        // Experts 1 and 2, weights softmax-renormalized over the two: e^2 / (e^2 + e^1).
        val w1 = exp(2.0) / (exp(2.0) + exp(1.0))
        val want = (w1 * 2 + (1 - w1) * 3) * silu(1.0)
        assertTrue(abs(y[0] - want) < 1e-5, "${y[0]} vs $want")
        assertEquals(0f, y[1])
    }

    @Test
    fun equalProbabilitiesSelectTheLowerExpertFirst() {
        // Experts 0..3 tie; top 1 is expert 0.
        val y = DxirInterpreter.evalFunction(fn(1, 1), listOf(floatArrayOf(1f, 0f), FloatArray(e), gateUp, down))[0]
        assertTrue(abs(y[0] - silu(1.0)) < 1e-5, "${y[0]}")
    }

    @Test
    fun bothTransformsAndTheRendererRefuseItByName() {
        val loss = DxirBuilder.function("loss") {
            val inner = fn(1, 2)
            val ps = inner.params.map { param(it.name, it.type) }
            val y = op(OpKind.MOE_EXPERTS, ps, DxirType(F32, listOf(1, h)), mapOf("top_k" to 2))
            listOf(op(OpKind.SUM, listOf(y), DxirType(F32, emptyList())))
        }
        for (ex in listOf(
            assertFailsWith<IllegalStateException> { DxirReverseTransform.apply(loss) },
            assertFailsWith<IllegalStateException> { DxirForwardTransform.apply(loss) },
        )) {
            assertTrue("MOE_EXPERTS" in ex.message!! && "inference-only" in ex.message!!, ex.message!!)
        }
        val msg = assertFailsWith<io.tlaloc.ir.render.KotlinRenderRefusal> { io.tlaloc.ir.render.toKotlinSource(fn(1, 2)) }.message!!
        assertTrue("MOE_EXPERTS" in msg, msg)
    }

    @Test
    fun aTopKOutsideTheExpertsIsRefusedByName() {
        val ex = assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(fn(1, e + 1), listOf(FloatArray(h), FloatArray(e), gateUp, down))
        }
        assertTrue("top_k" in ex.message!!, ex.message!!)
    }
}
