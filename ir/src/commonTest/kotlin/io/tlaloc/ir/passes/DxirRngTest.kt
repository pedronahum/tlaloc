package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.normalFloats
import io.tlaloc.core.uniformFloats
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.408 — Phase D1: the RNG_UNIFORM / RNG_NORMAL IR arms.
 *
 * The interpreter arms must be BIT-EXACT against the `:core/Random.kt` host
 * kernels (they call the same functions — this test is the pin that keeps it
 * that way), and both differentiation transforms must refuse the ops loudly
 * by name: the draws are non-differentiable in D1 (piecewise-constant in the
 * key; reparameterized gradients are Phase D2).
 */
class DxirRngTest {

    private val scalar = DxirType(F32, emptyList())
    private val key = RandomKey(7, 42)

    private fun rngAttrs(dims: List<Int>): Map<String, Any> =
        mapOf("key0" to key.k0, "key1" to key.k1, "dims" to dims)

    @Test
    fun interpreterUniformBitExactVsHostKernel() {
        val r2 = DxirType(F32, listOf(2, 3))
        val fn = DxirBuilder.function("rng_uniform") {
            listOf(op(OpKind.RNG_UNIFORM, emptyList(), r2, attrs = rngAttrs(listOf(2, 3))))
        }
        val out = DxirInterpreter.evalFunction(fn, emptyList())
        assertContentEquals(uniformFloats(key, 6), out[0], "interpreter must be bit-exact vs host")
    }

    @Test
    fun interpreterNormalBitExactVsHostKernel() {
        // Odd size on purpose: exercises the end-pad counter lane through the
        // interpreter path too.
        val r1 = DxirType(F32, listOf(5))
        val fn = DxirBuilder.function("rng_normal") {
            listOf(op(OpKind.RNG_NORMAL, emptyList(), r1, attrs = rngAttrs(listOf(5))))
        }
        val out = DxirInterpreter.evalFunction(fn, emptyList())
        assertContentEquals(normalFloats(key, 5), out[0], "interpreter must be bit-exact vs host")
    }

    @Test
    fun interpreterRejectsDimsAttrDisagreeingWithType() {
        val r1 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("rng_bad_dims") {
            listOf(op(OpKind.RNG_UNIFORM, emptyList(), r1, attrs = rngAttrs(listOf(5))))
        }
        val ex = assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(fn, emptyList())
        }
        assertTrue(
            ex.message!!.contains("disagrees with result type dims"),
            "unexpected message: ${ex.message}",
        )
    }

    @Test
    fun reverseTransformRefusesRngByName() {
        // loss = Σ (uniform ⊙ x): the draw sits ON the differentiable path, so
        // the reverse walk reaches it with an accumulated upstream and must
        // refuse by name rather than silently drop it.
        val r1 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("rng_in_grad") {
            val x = param("x", r1)
            val u = op(OpKind.RNG_UNIFORM, emptyList(), r1, attrs = rngAttrs(listOf(4)))
            val p = op(OpKind.MUL, listOf(u, x), r1)
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val ex = assertFailsWith<IllegalStateException> { DxirReverseTransform.apply(fn) }
        assertTrue(
            ex.message!!.contains("no VJP rule registered for RNG_UNIFORM"),
            "unexpected message: ${ex.message}",
        )
    }

    @Test
    fun forwardTransformRefusesRngByName() {
        val r1 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("rng_in_jvp") {
            val x = param("x", r1)
            val z = op(OpKind.RNG_NORMAL, emptyList(), r1, attrs = rngAttrs(listOf(4)))
            val p = op(OpKind.MUL, listOf(z, x), r1)
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val ex = assertFailsWith<IllegalStateException> { DxirForwardTransform.apply(fn) }
        assertTrue(
            ex.message!!.contains("no tangent rule for RNG_NORMAL"),
            "unexpected message: ${ex.message}",
        )
    }
}
