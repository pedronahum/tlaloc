package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.448 — audit finding C (docs/AD_SINGLE_ENGINE_AUDIT.md): the demoted
 * kinds refuse LOUDLY AND BY NAME in every layer that does not sanction them.
 *
 * One test per kind per refusing layer. Each pin asserts the message names
 * the KIND and the sanctioned ALTERNATIVE — the whole point of the demotion
 * is that a hand-built graph using one of these fails with directions, not
 * with a generic "unsupported op" (or, worst of all pre-§0.4.448 shapes, a
 * silent index-0 skip that dropped a multi-result op's gradient to zero).
 *
 * Layer matrix (the audit's demote decision):
 * - LAYERNORM / SCALED_DOT_PRODUCT_ATTENTION: refused by
 *   DxirInterpreter, DxirReverseTransform, DxirForwardTransform.
 * (SPLIT, a §0.4.448 demotee, was DELETED in §0.4.454 — its rows left with
 * the kind. ALL_REDUCE / SHARD_CONSTRAINT were UN-DEMOTED in §0.4.460
 * Phase G3a — their refusal rows left with the demotion, replaced by the
 * real-op oracles in AllReduceTest.)
 * Recognition (cost model, recognizers) and StableHLO emission remain the
 * sanctioned roles and are deliberately NOT touched by this pin.
 */
class DemotedKindRefusalTest {

    private val f32s = DxirType(F32, emptyList())
    private val vec4 = DxirType(F32, listOf(4))

    private fun assertNamedRefusal(ex: Throwable, kind: String, alternative: String) {
        val msg = ex.message ?: ""
        assertTrue(kind in msg, "refusal must name the kind $kind; got: $msg")
        assertTrue(alternative in msg, "refusal must name the sanctioned alternative ('$alternative'); got: $msg")
    }

    // --- Builders: one tiny primal per kind. ---

    private fun layernormFn(): DxirFunction = DxirBuilder.function("ln") {
        val x = param("x", vec4)
        val y = op(OpKind.LAYERNORM, listOf(x), vec4)
        listOf(op(OpKind.SUM, listOf(y), f32s))
    }

    private fun sdpaFn(): DxirFunction = DxirBuilder.function("sdpa") {
        val ty = DxirType(F32, listOf(2, 2))
        val q = param("q", ty)
        val k = param("k", ty)
        val v = param("v", ty)
        val y = op(OpKind.SCALED_DOT_PRODUCT_ATTENTION, listOf(q, k, v), ty)
        listOf(op(OpKind.SUM, listOf(y), f32s))
    }

    // --- DxirInterpreter refusals (LAYERNORM / SDPA). ---

    @Test
    fun interpreterRefusesLayernormByName() {
        val ex = assertFailsWith<IllegalStateException> {
            DxirInterpreter.evalFunction(layernormFn(), listOf(FloatArray(4) { it.toFloat() }))
        }
        assertNamedRefusal(ex, "LAYERNORM", "as batchNorm is decomposed")
    }

    @Test
    fun interpreterRefusesSdpaByName() {
        val q = FloatArray(4) { it.toFloat() }
        val ex = assertFailsWith<IllegalStateException> {
            DxirInterpreter.evalFunction(sdpaFn(), listOf(q, q, q))
        }
        assertNamedRefusal(ex, "SCALED_DOT_PRODUCT_ATTENTION", "FlashAttention")
    }

    // --- DxirReverseTransform refusals. ---

    @Test
    fun reverseTransformRefusesLayernormByName() {
        val ex = assertFailsWith<IllegalStateException> { DxirReverseTransform.apply(layernormFn()) }
        assertNamedRefusal(ex, "LAYERNORM", "as batchNorm is decomposed")
    }

    @Test
    fun reverseTransformRefusesSdpaByName() {
        val ex = assertFailsWith<IllegalStateException> { DxirReverseTransform.apply(sdpaFn()) }
        assertNamedRefusal(ex, "SCALED_DOT_PRODUCT_ATTENTION", "FlashAttention")
    }

    // --- DxirForwardTransform refusals. ---

    @Test
    fun forwardTransformRefusesLayernormByName() {
        val ex = assertFailsWith<IllegalStateException> { DxirForwardTransform.apply(layernormFn()) }
        assertNamedRefusal(ex, "LAYERNORM", "as batchNorm is decomposed")
    }

    @Test
    fun forwardTransformRefusesSdpaByName() {
        val ex = assertFailsWith<IllegalStateException> { DxirForwardTransform.apply(sdpaFn()) }
        assertNamedRefusal(ex, "SCALED_DOT_PRODUCT_ATTENTION", "FlashAttention")
    }
}
