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
 * silent index-0 skip that dropped a multi-result SPLIT's gradient to zero).
 *
 * Layer matrix (the audit's demote decision):
 * - LAYERNORM / SCALED_DOT_PRODUCT_ATTENTION / SPLIT: refused by
 *   DxirInterpreter, DxirReverseTransform, DxirForwardTransform.
 * - ALL_REDUCE / SHARD_CONSTRAINT: non-differentiable by design — refused by
 *   both transforms (the interpreter keeps its generic unsupported-op arm:
 *   their demotion is about differentiability, not host evaluation).
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

    /** Consumes result index 1 ONLY — the exact shape the reverse walk's
     * index-0 upstream lookup would have skipped silently pre-§0.4.448. */
    private fun splitFn(): DxirFunction = DxirBuilder.function("splitting") {
        val x = param("x", DxirType(F32, listOf(6, 4)))
        val split = opMulti(
            OpKind.SPLIT, listOf(x),
            types = listOf(DxirType(F32, listOf(3, 4)), DxirType(F32, listOf(3, 4))),
            attrs = mapOf("axis" to 0, "sizes" to listOf(3, 3)),
        )
        listOf(op(OpKind.SUM, listOf(split.result(1)), f32s))
    }

    private fun allReduceFn(): DxirFunction = DxirBuilder.function("ar") {
        val x = param("x", vec4)
        val y = op(OpKind.ALL_REDUCE, listOf(x), vec4)
        listOf(op(OpKind.SUM, listOf(y), f32s))
    }

    private fun shardConstraintFn(): DxirFunction = DxirBuilder.function("sc") {
        val x = param("x", vec4)
        val y = op(OpKind.SHARD_CONSTRAINT, listOf(x), vec4)
        listOf(op(OpKind.SUM, listOf(y), f32s))
    }

    // --- DxirInterpreter refusals (LAYERNORM / SDPA / SPLIT). ---

    @Test
    fun interpreterRefusesLayernormByName() {
        val ex = assertFailsWith<IllegalStateException> {
            DxirInterpreter.evalFunction(layernormFn(), listOf(FloatArray(4) { it.toFloat() }))
        }
        assertNamedRefusal(ex, "LAYERNORM", "batchNorm desugaring")
    }

    @Test
    fun interpreterRefusesSdpaByName() {
        val q = FloatArray(4) { it.toFloat() }
        val ex = assertFailsWith<IllegalStateException> {
            DxirInterpreter.evalFunction(sdpaFn(), listOf(q, q, q))
        }
        assertNamedRefusal(ex, "SCALED_DOT_PRODUCT_ATTENTION", "FlashAttention")
    }

    @Test
    fun interpreterRefusesSplitByName() {
        val ex = assertFailsWith<IllegalStateException> {
            DxirInterpreter.evalFunction(splitFn(), listOf(FloatArray(24) { it.toFloat() }))
        }
        assertNamedRefusal(ex, "SPLIT", "SLICE")
    }

    // --- DxirReverseTransform refusals (all five). ---

    @Test
    fun reverseTransformRefusesLayernormByName() {
        val ex = assertFailsWith<IllegalStateException> { DxirReverseTransform.apply(layernormFn()) }
        assertNamedRefusal(ex, "LAYERNORM", "batchNorm desugaring")
    }

    @Test
    fun reverseTransformRefusesSdpaByName() {
        val ex = assertFailsWith<IllegalStateException> { DxirReverseTransform.apply(sdpaFn()) }
        assertNamedRefusal(ex, "SCALED_DOT_PRODUCT_ATTENTION", "FlashAttention")
    }

    @Test
    fun reverseTransformRefusesSplitByName() {
        // Pre-§0.4.448 this was the SILENT shape: the walk's index-0 upstream
        // lookup found nothing for a SPLIT consumed only at index 1 and
        // skipped it — zero gradient, no error. Now it refuses by name.
        val ex = assertFailsWith<IllegalStateException> { DxirReverseTransform.apply(splitFn()) }
        assertNamedRefusal(ex, "SPLIT", "SLICE")
    }

    @Test
    fun reverseTransformRefusesAllReduceByName() {
        val ex = assertFailsWith<IllegalStateException> { DxirReverseTransform.apply(allReduceFn()) }
        assertNamedRefusal(ex, "ALL_REDUCE", "non-differentiable by design")
    }

    @Test
    fun reverseTransformRefusesShardConstraintByName() {
        val ex = assertFailsWith<IllegalStateException> {
            DxirReverseTransform.apply(shardConstraintFn())
        }
        assertNamedRefusal(ex, "SHARD_CONSTRAINT", "non-differentiable by design")
    }

    // --- DxirForwardTransform refusals (all five). ---

    @Test
    fun forwardTransformRefusesLayernormByName() {
        val ex = assertFailsWith<IllegalStateException> { DxirForwardTransform.apply(layernormFn()) }
        assertNamedRefusal(ex, "LAYERNORM", "batchNorm desugaring")
    }

    @Test
    fun forwardTransformRefusesSdpaByName() {
        val ex = assertFailsWith<IllegalStateException> { DxirForwardTransform.apply(sdpaFn()) }
        assertNamedRefusal(ex, "SCALED_DOT_PRODUCT_ATTENTION", "FlashAttention")
    }

    @Test
    fun forwardTransformRefusesSplitByName() {
        // Pre-§0.4.448 a SPLIT here hit the generic multi-result
        // out-of-scope error; the named refusal fires first now.
        val ex = assertFailsWith<IllegalStateException> { DxirForwardTransform.apply(splitFn()) }
        assertNamedRefusal(ex, "SPLIT", "SLICE")
    }

    @Test
    fun forwardTransformRefusesAllReduceByName() {
        val ex = assertFailsWith<IllegalStateException> { DxirForwardTransform.apply(allReduceFn()) }
        assertNamedRefusal(ex, "ALL_REDUCE", "non-differentiable by design")
    }

    @Test
    fun forwardTransformRefusesShardConstraintByName() {
        val ex = assertFailsWith<IllegalStateException> {
            DxirForwardTransform.apply(shardConstraintFn())
        }
        assertNamedRefusal(ex, "SHARD_CONSTRAINT", "non-differentiable by design")
    }
}
