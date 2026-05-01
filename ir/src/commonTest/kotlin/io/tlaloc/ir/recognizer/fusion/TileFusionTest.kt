package io.tlaloc.ir.recognizer.fusion

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Layer 3 §0.4.256+ — tile-fusion identification + annotation tests.
 *
 * Verifies:
 * - Maximal elementwise chains are grouped (multi-op windows survive).
 * - Singletons stay un-grouped (no annotation pollution for trivial code).
 * - Shape boundaries break groups (different shapes can't share a tile).
 * - Reductions / matmul / shape ops break groups (not elementwise).
 * - Annotation is additive — no other attrs are dropped.
 */
class TileFusionTest {

    private val rank2 = DxirType(F32, listOf(8, 64))
    private val rank2Other = DxirType(F32, listOf(4, 32))
    private val scalar = DxirType(F32, listOf())

    @Test
    fun fourElementwiseOpsBecomeOneTileGroup() {
        val fn = DxirBuilder.function("chain") {
            val x = param("x", rank2)
            val a = op(OpKind.ADD, listOf(x, x), rank2)
            val b = op(OpKind.MUL, listOf(a, x), rank2)
            val c = op(OpKind.SUB, listOf(b, a), rank2)
            val d = op(OpKind.RELU, listOf(c), rank2)
            listOf(d)
        }
        val groups = identifyTileGroups(fn)
        assertEquals(1, groups.size, "single connected elementwise component")
        val g = groups.single()
        assertEquals(4, g.opIds.size)
        assertEquals(rank2, g.sharedShape)
    }

    @Test
    fun singletonElementwiseIsNotAGroup() {
        val fn = DxirBuilder.function("solo") {
            val x = param("x", rank2)
            val r = op(OpKind.RELU, listOf(x), rank2)
            listOf(r)
        }
        val groups = identifyTileGroups(fn)
        assertEquals(0, groups.size, "single elementwise op shouldn't be its own tile group")
    }

    @Test
    fun differentShapesBreakTheGroup() {
        // Two separate elementwise chains with different shapes.
        val fn = DxirBuilder.function("split_shape") {
            val x = param("x", rank2)
            val y = param("y", rank2Other)
            val a = op(OpKind.ADD, listOf(x, x), rank2)
            val b = op(OpKind.MUL, listOf(a, x), rank2)
            val c = op(OpKind.ADD, listOf(y, y), rank2Other)
            val d = op(OpKind.MUL, listOf(c, y), rank2Other)
            listOf(b, d)
        }
        val groups = identifyTileGroups(fn)
        assertEquals(2, groups.size, "different shapes → different groups")
        // Each group has two ops.
        for (g in groups) assertEquals(2, g.opIds.size)
        // Distinct shapes.
        assertTrue(groups[0].sharedShape != groups[1].sharedShape)
    }

    @Test
    fun reductionInTheMiddleBreaksTheGroup() {
        // ADD → SUM (not elementwise) → BROADCAST (not elementwise) → MUL.
        // The two elementwise ops aren't directly connected through
        // elementwise edges, so no group forms.
        val fn = DxirBuilder.function("reduction_break") {
            val x = param("x", rank2)
            val a = op(OpKind.ADD, listOf(x, x), rank2)
            val s = op(OpKind.SUM, listOf(a), scalar)
            val sb = op(OpKind.BROADCAST, listOf(s), rank2)
            // MUL connects only through sb (BROADCAST is non-candidate);
            // no direct edge to ADD's result.
            val m = op(OpKind.MUL, listOf(sb, x), rank2)
            listOf(m)
        }
        val groups = identifyTileGroups(fn)
        // ADD and MUL are both elementwise + same shape, but they're
        // joined only through SUM/BROADCAST which aren't candidates.
        // Without an elementwise edge between them, they stay separate
        // singletons → no group.
        assertEquals(0, groups.size, "reduction between elementwises blocks the group")
    }

    @Test
    fun matmulBetweenElementwisesBreaksTheGroup() {
        val fn = DxirBuilder.function("matmul_break") {
            val x = param("x", DxirType(F32, listOf(4, 4)))
            val a = op(OpKind.ADD, listOf(x, x), DxirType(F32, listOf(4, 4)))
            val mm = op(OpKind.MATMUL, listOf(a, x), DxirType(F32, listOf(4, 4)))
            val r = op(OpKind.RELU, listOf(mm), DxirType(F32, listOf(4, 4)))
            listOf(r)
        }
        val groups = identifyTileGroups(fn)
        assertEquals(0, groups.size, "MATMUL is not elementwise — chain is broken")
    }

    @Test
    fun annotationAddsTileGroupAttrToCandidateOps() {
        val fn = DxirBuilder.function("annotated") {
            val x = param("x", rank2)
            val a = op(OpKind.ADD, listOf(x, x), rank2)
            val b = op(OpKind.MUL, listOf(a, x), rank2)
            val c = op(OpKind.SUB, listOf(b, a), rank2)
            listOf(c)
        }
        val annotated = annotateTileGroups(fn)
        val ops = annotated.body.filterIsInstance<DxirOp>()
        assertEquals(3, ops.size)
        // All three should share the same tile_group id.
        val groupIds = ops.map { it.attrs["tile_group"] as? Int }
        assertEquals(setOf(0), groupIds.toSet(), "all three ops share group 0")
    }

    @Test
    fun annotationLeavesNonCandidateAttrsAlone() {
        val fn = DxirBuilder.function("mixed") {
            val x = param("x", rank2)
            // Ops with custom attrs that the annotation must not drop.
            val a = op(OpKind.ADD, listOf(x, x), rank2, attrs = mapOf("custom_anno" to "k1"))
            val b = op(OpKind.MUL, listOf(a, x), rank2)
            // Non-elementwise (SUM) should not get a tile_group.
            val s = op(OpKind.SUM, listOf(b), scalar)
            listOf(s)
        }
        val annotated = annotateTileGroups(fn)
        val ops = annotated.body.filterIsInstance<DxirOp>()
        // ADD's "custom_anno" preserved + tile_group added.
        val addOp = ops.first { it.op == OpKind.ADD }
        assertEquals("k1", addOp.attrs["custom_anno"])
        assertEquals(0, addOp.attrs["tile_group"])
        // MUL: tile_group added.
        val mulOp = ops.first { it.op == OpKind.MUL }
        assertEquals(0, mulOp.attrs["tile_group"])
        // SUM: no tile_group.
        val sumOp = ops.first { it.op == OpKind.SUM }
        assertNull(sumOp.attrs["tile_group"])
    }

    @Test
    fun multipleDisjointGroupsGetDistinctIds() {
        val fn = DxirBuilder.function("two_groups") {
            val x = param("x", rank2)
            val a1 = op(OpKind.ADD, listOf(x, x), rank2)
            val a2 = op(OpKind.MUL, listOf(a1, x), rank2)
            // Reduction breaks the chain; downstream chain is separate.
            val r = op(OpKind.SUM, listOf(a2), scalar)
            val rb = op(OpKind.BROADCAST, listOf(r), rank2)
            val b1 = op(OpKind.SUB, listOf(rb, x), rank2)
            val b2 = op(OpKind.RELU, listOf(b1), rank2)
            listOf(b2)
        }
        val groups = identifyTileGroups(fn)
        assertEquals(2, groups.size, "two disjoint elementwise components")
        // IDs are distinct and zero-based.
        assertEquals(setOf(0, 1), groups.map { it.groupId }.toSet())
    }

    @Test
    fun annotateTileGroupsIsIdempotent() {
        // Running the annotation twice yields the same shape (the
        // second run sees the already-tagged ops and produces the same
        // tile_group attr values).
        val fn = DxirBuilder.function("idem") {
            val x = param("x", rank2)
            val a = op(OpKind.ADD, listOf(x, x), rank2)
            val b = op(OpKind.MUL, listOf(a, x), rank2)
            listOf(b)
        }
        val once = annotateTileGroups(fn)
        val twice = annotateTileGroups(once)
        val onceTags = once.body.filterIsInstance<DxirOp>().map { it.attrs["tile_group"] }
        val twiceTags = twice.body.filterIsInstance<DxirOp>().map { it.attrs["tile_group"] }
        assertEquals(onceTags, twiceTags)
    }

    @Test
    fun emptyBodyReturnsNoGroups() {
        val fn = DxirBuilder.function("empty") {
            val x = param("x", rank2)
            listOf(x)
        }
        assertEquals(0, identifyTileGroups(fn).size)
    }
}
