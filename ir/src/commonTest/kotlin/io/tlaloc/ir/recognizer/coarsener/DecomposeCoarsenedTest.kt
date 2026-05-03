package io.tlaloc.ir.recognizer.coarsener

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.kernel.KernelDescriptor
import io.tlaloc.ir.recognizer.recognizeAll
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Layer 4 §0.4.275 — DecomposeCoarsened tests.
 *
 * Pin the CPU-baseline strategy: COARSENED ops without a kernel_descriptor
 * get inlined back to their primal_body's primitive ops, so the function
 * lowers cleanly to standard StableHLO without needing a custom_call
 * dispatch for every coarsened pattern. Mirrors the structural-only
 * style of the other coarsener tests (no numerical eval).
 */
class DecomposeCoarsenedTest {

    private val xType = DxirType(F32, listOf(8, 64))
    private val mType = DxirType(F32, listOf(8, 1))

    /** Reusable: a function with one canonical RmsNorm chain (matches recognizer). */
    private fun rmsNormFn(): DxirFunction = DxirBuilder.function("rms_norm") {
        val x = param("x", xType)
        val sq = op(OpKind.MUL, listOf(x, x), xType)
        val mean = op(OpKind.MEAN, listOf(sq), mType)
        val r = op(OpKind.RSQRT, listOf(mean), mType)
        val out = op(OpKind.MUL, listOf(x, r), xType)
        listOf(out)
    }

    /** Build the coarsened form of [rmsNormFn] (one COARSENED op + nothing else). */
    private fun coarsenedRmsNormFn(): DxirFunction {
        val raw = rmsNormFn()
        return coarsenRecognizedPatterns(raw, recognizeAll(raw))
    }

    @Test
    fun functionWithoutCoarsenedReturnsUnchanged() {
        // No COARSENED ops → pass is a no-op.
        val fn = rmsNormFn()
        val result = decomposeCoarsened(fn)
        // Same body, same return — the pass should short-circuit and
        // return the input function reference unchanged.
        assertEquals(fn, result, "no-COARSENED function should be returned as-is (reference equality)")
    }

    @Test
    fun coarsenedWithKernelDescriptorIsKept() {
        // Manually attach a kernel_descriptor to the COARSENED — simulates
        // what lowerKernelChoice does on a target with a fused kernel.
        // decomposeCoarsened must leave it alone.
        val coarsened = coarsenedRmsNormFn()
        val coarsenedOp = coarsened.body.filterIsInstance<DxirOp>().single { it.op == OpKind.COARSENED }
        val withDescriptor = DxirBuilder.function(coarsened.name) {
            val nodeMap = HashMap<Int, io.tlaloc.ir.DxirNode>()
            for (p in coarsened.params) nodeMap[p.id] = param(p.name, p.type, p.sharding)
            for (n in coarsened.body) {
                if (n is DxirOp && n.id == coarsenedOp.id) {
                    val attrsWithKd = n.attrs.toMutableMap().also {
                        it[KernelDescriptor.ATTR_KEY] = KernelDescriptor("rms_norm_v1", "tlaloc", "test")
                    }
                    val newOp = op(n.op, n.operands.map { o -> nodeMap[o.id]!! }, n.type, attrsWithKd, n.sharding)
                    nodeMap[n.id] = newOp
                }
            }
            coarsened.returns.map { nodeMap[it.id]!! }
        }
        val result = decomposeCoarsened(withDescriptor)
        // COARSENED op count unchanged (still 1).
        val coarsenedAfter = result.body.filterIsInstance<DxirOp>().count { it.op == OpKind.COARSENED }
        assertEquals(1, coarsenedAfter, "COARSENED with kernel_descriptor must NOT be decomposed")
    }

    @Test
    fun coarsenedWithoutKernelDescriptorIsInlined() {
        val coarsened = coarsenedRmsNormFn()
        val before = coarsened.body.filterIsInstance<DxirOp>().count { it.op == OpKind.COARSENED }
        assertEquals(1, before, "prerequisite: one COARSENED op pre-decompose")

        val decomposed = decomposeCoarsened(coarsened)
        val coarsenedAfter = decomposed.body.filterIsInstance<DxirOp>().count { it.op == OpKind.COARSENED }
        assertEquals(0, coarsenedAfter, "COARSENED without kernel_descriptor must be inlined away")

        // The inlined RmsNorm primal_body is MUL → MEAN → RSQRT → MUL — 4 ops.
        val opCount = decomposed.body.filterIsInstance<DxirOp>().size
        assertEquals(4, opCount, "decomposed body should have the 4 primal_body ops; got ${decomposed.body.map { (it as? DxirOp)?.op }}")
    }

    @Test
    fun multipleCoarsenedAreAllInlined() {
        // Function with two RmsNorm chains in a row — recognizer + coarsener
        // produce 2 COARSENED ops; decomposeCoarsened should inline both.
        val fn = DxirBuilder.function("two_norms") {
            val x = param("x", xType)
            // First norm
            val sq1 = op(OpKind.MUL, listOf(x, x), xType)
            val mean1 = op(OpKind.MEAN, listOf(sq1), mType)
            val r1 = op(OpKind.RSQRT, listOf(mean1), mType)
            val n1 = op(OpKind.MUL, listOf(x, r1), xType)
            // Second norm (on n1)
            val sq2 = op(OpKind.MUL, listOf(n1, n1), xType)
            val mean2 = op(OpKind.MEAN, listOf(sq2), mType)
            val r2 = op(OpKind.RSQRT, listOf(mean2), mType)
            val n2 = op(OpKind.MUL, listOf(n1, r2), xType)
            listOf(n2)
        }
        val coarsened = coarsenRecognizedPatterns(fn, recognizeAll(fn))
        val coarsenedCount = coarsened.body.filterIsInstance<DxirOp>().count { it.op == OpKind.COARSENED }
        assertEquals(2, coarsenedCount, "prerequisite: 2 COARSENED ops")

        val decomposed = decomposeCoarsened(coarsened)
        val coarsenedAfter = decomposed.body.filterIsInstance<DxirOp>().count { it.op == OpKind.COARSENED }
        assertEquals(0, coarsenedAfter, "all COARSENED ops must be inlined")

        val opCount = decomposed.body.filterIsInstance<DxirOp>().size
        assertEquals(8, opCount, "two RmsNorms × 4 ops each = 8 primitive ops")
    }

    @Test
    fun mixedKeepAndDecomposeRespectsKernelDescriptor() {
        // Two-norm function. Manually attach kernel_descriptor to the FIRST
        // COARSENED but not the second. Expect: 1 COARSENED kept, 1 inlined.
        val fn = DxirBuilder.function("two_norms_mixed") {
            val x = param("x", xType)
            val sq1 = op(OpKind.MUL, listOf(x, x), xType)
            val mean1 = op(OpKind.MEAN, listOf(sq1), mType)
            val r1 = op(OpKind.RSQRT, listOf(mean1), mType)
            val n1 = op(OpKind.MUL, listOf(x, r1), xType)
            val sq2 = op(OpKind.MUL, listOf(n1, n1), xType)
            val mean2 = op(OpKind.MEAN, listOf(sq2), mType)
            val r2 = op(OpKind.RSQRT, listOf(mean2), mType)
            val n2 = op(OpKind.MUL, listOf(n1, r2), xType)
            listOf(n2)
        }
        val coarsened = coarsenRecognizedPatterns(fn, recognizeAll(fn))
        val coarsenedOps = coarsened.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.COARSENED }
        assertEquals(2, coarsenedOps.size)

        val firstCoarsenedId = coarsenedOps[0].id
        val withMixed = DxirBuilder.function(coarsened.name) {
            val nodeMap = HashMap<Int, io.tlaloc.ir.DxirNode>()
            for (p in coarsened.params) nodeMap[p.id] = param(p.name, p.type, p.sharding)
            for (n in coarsened.body) {
                if (n is DxirOp && n.id == firstCoarsenedId) {
                    val attrsWithKd = n.attrs.toMutableMap().also {
                        it[KernelDescriptor.ATTR_KEY] = KernelDescriptor("rms_norm_v1", "tlaloc", "test")
                    }
                    nodeMap[n.id] = op(n.op, n.operands.map { o -> nodeMap[o.id]!! }, n.type, attrsWithKd, n.sharding)
                } else if (n is DxirOp) {
                    nodeMap[n.id] = op(n.op, n.operands.map { o -> nodeMap[o.id]!! }, n.type, n.attrs, n.sharding)
                }
            }
            coarsened.returns.map { nodeMap[it.id]!! }
        }
        val result = decomposeCoarsened(withMixed)
        val coarsenedAfter = result.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.COARSENED }
        assertEquals(1, coarsenedAfter.size, "exactly one COARSENED kept (the one with kernel_descriptor)")
        assertTrue(
            coarsenedAfter.single().attrs[KernelDescriptor.ATTR_KEY] != null,
            "kept COARSENED must be the one with kernel_descriptor",
        )
    }
}
