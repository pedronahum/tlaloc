package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * §0.4.27 — Stage C.1 tests for [RegionTree]. Covers structural correctness of tree
 * construction from scalar / IF / WHILE / nested primals, plus bottom-up traversal
 * ordering.
 */
class RegionTreeTest {

    private val f32s = DxirType(F32, emptyList())
    private val i32s = DxirType(I32, emptyList())
    private val boolS = DxirType(Bool, emptyList())

    @Test
    fun scalarFunctionProducesSingleNodeTree() {
        val fn = DxirBuilder.function("scalar") {
            val x = param("x", f32s)
            val c = const(2f, f32s)
            val r = op(OpKind.MUL, listOf(x, c), f32s)
            listOf(r)
        }
        val tree = RegionTree.build(fn)
        assertTrue(tree.root.isRoot)
        assertTrue(tree.root.isLeaf, "scalar function's tree has no children")
        assertEquals(2, tree.root.directOps.size) // const + mul
        assertEquals(0, tree.root.depth)
        assertEquals(1, tree.size)
    }

    @Test
    fun ifProducesRootWithTwoChildren() {
        val fn = DxirBuilder.function("if_tree") {
            val x = param("x", f32s)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val negX = op(OpKind.NEG, listOf(x), f32s)
            val result = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region { yields(x) },
                elseRegion = region { yields(negX) },
            )
            listOf(result)
        }
        val tree = RegionTree.build(fn)
        assertEquals(2, tree.root.children.size, "IF has 2 regions = 2 children")
        val thenChild = tree.root.children[0]
        val elseChild = tree.root.children[1]
        assertEquals(0, thenChild.regionIndex)
        assertEquals(1, elseChild.regionIndex)
        assertTrue(thenChild.isLeaf, "empty-body then region is a leaf")
        assertTrue(elseChild.isLeaf, "empty-body else region is a leaf")
        assertEquals(1, thenChild.depth)
        assertEquals(1, elseChild.depth)
        // The IF op itself is a direct op of the root (not hidden behind children).
        val ifOp = fn.body.filterIsInstance<io.tlaloc.ir.DxirOp>().first { it.op == OpKind.IF }
        assertTrue(ifOp in tree.root.directOps)
    }

    @Test
    fun whileProducesRootWithTwoChildren() {
        val fn = DxirBuilder.function("while_tree") {
            val x = param("x", f32s)
            val n = const(5, i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val diff = op(OpKind.SUB, listOf(n, args[1]), i32s)
                    val pred = op(OpKind.STEP, listOf(diff), boolS)
                    yields(pred)
                },
                body = { args ->
                    val two = const(2f, f32s)
                    val newX = op(OpKind.MUL, listOf(args[0], two), f32s)
                    val one = const(1, i32s)
                    val newI = op(OpKind.ADD, listOf(args[1], one), i32s)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val tree = RegionTree.build(fn)
        assertEquals(2, tree.root.children.size, "WHILE has 2 regions = 2 children")
        val condChild = tree.root.children[0]
        val bodyChild = tree.root.children[1]
        // Cond region: SUB + STEP = 2 ops.
        assertEquals(2, condChild.directOps.size)
        // Body region: 2 consts + MUL + ADD = 4 ops.
        assertEquals(4, bodyChild.directOps.size)
        assertTrue(condChild.isLeaf)
        assertTrue(bodyChild.isLeaf)
    }

    @Test
    fun ifInsideWhileBodyProducesThreeLevelTree() {
        // WHILE { body: IF ... }. Tree depth = 3: root → WHILE body → IF regions.
        val fn = DxirBuilder.function("nested") {
            val x = param("x", f32s)
            val n = const(5, i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val diff = op(OpKind.SUB, listOf(n, args[1]), i32s)
                    val pred = op(OpKind.STEP, listOf(diff), boolS)
                    yields(pred)
                },
                body = { args ->
                    val carried = args[0]
                    val innerPred = op(OpKind.STEP, listOf(carried), boolS)
                    val ifOut = ifOp(
                        cond = innerPred,
                        types = listOf(f32s),
                        thenRegion = region { yields(carried) },
                        elseRegion = region {
                            val neg = op(OpKind.NEG, listOf(carried), f32s)
                            yields(neg)
                        },
                    )
                    val one = const(1, i32s)
                    val newI = op(OpKind.ADD, listOf(args[1], one), i32s)
                    yields(ifOut, newI)
                },
            )
            listOf(w.result(0))
        }
        val tree = RegionTree.build(fn)
        // Root has 2 children (while cond + while body); body has 2 children (if then + else).
        assertEquals(2, tree.root.children.size)
        val whileBody = tree.root.children[1]
        assertEquals(2, whileBody.children.size, "nested IF's two regions are grandchildren")
        // The two IF-region grandchildren have depth 2.
        val thenGrand = whileBody.children[0]
        val elseGrand = whileBody.children[1]
        assertEquals(2, thenGrand.depth)
        assertEquals(2, elseGrand.depth)
        assertTrue(thenGrand.isLeaf)
        assertTrue(elseGrand.isLeaf)
    }

    @Test
    fun bottomUpTraversalListsChildrenBeforeParents() {
        val fn = DxirBuilder.function("nested") {
            val x = param("x", f32s)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val result = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region { yields(x) },
                elseRegion = region { yields(x) },
            )
            listOf(result)
        }
        val tree = RegionTree.build(fn)
        val order = tree.bottomUp()
        // Expected order: thenChild, elseChild, root.
        assertEquals(3, order.size)
        assertEquals(tree.root.children[0], order[0])
        assertEquals(tree.root.children[1], order[1])
        assertEquals(tree.root, order[2], "root must be last in bottom-up order")
    }

    @Test
    fun rootHasNoParent() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", f32s)
            listOf(x)
        }
        val tree = RegionTree.build(fn)
        assertNull(tree.root.parent)
        assertNull(tree.root.regionOp, "root has no source op")
    }

    @Test
    fun childKnowsItsParent() {
        val fn = DxirBuilder.function("if_parent") {
            val x = param("x", f32s)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val result = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region { yields(x) },
                elseRegion = region { yields(x) },
            )
            listOf(result)
        }
        val tree = RegionTree.build(fn)
        val thenChild = tree.root.children[0]
        assertEquals(tree.root, thenChild.parent)
        val ifOp = fn.body.filterIsInstance<io.tlaloc.ir.DxirOp>().first { it.op == OpKind.IF }
        assertEquals(ifOp, thenChild.regionOp, "child's regionOp is the IF it belongs to")
    }
}
