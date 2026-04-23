package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.27 — Stage C.1 tests for [DefUseChain]. Covers forward use counts (both
 * single-user and multi-user defs), backward reachability from sinks, and region-
 * internal edge tracking (IF / WHILE both introduce ops that reference outer-scope
 * values — those edges must be captured).
 */
class DefUseChainTest {

    private val f32s = DxirType(F32, emptyList())
    private val i32s = DxirType(I32, emptyList())
    private val boolS = DxirType(Bool, emptyList())

    @Test
    fun scalarFunctionTracksDirectUses() {
        // f(x) = x * x + 1.  Uses: x → [mul] (once — operand list dedups), mul → [add].
        val fn = DxirBuilder.function("sq_plus_1") {
            val x = param("x", f32s)
            val sq = op(OpKind.MUL, listOf(x, x), f32s)
            val one = const(1f, f32s)
            val r = op(OpKind.ADD, listOf(sq, one), f32s)
            listOf(r)
        }
        val chain = DefUseChain.build(fn)
        val x = fn.params.single()
        val mul = fn.body.filterIsInstance<io.tlaloc.ir.DxirOp>().first { it.op == OpKind.MUL }
        val add = fn.body.filterIsInstance<io.tlaloc.ir.DxirOp>().first { it.op == OpKind.ADD }
        // MUL references x twice; the use list deduplicates by consumer-id so MUL only
        // appears once as a consumer of x.
        assertEquals(listOf(mul.id), chain.usersOf(x.id))
        assertEquals(listOf(add.id), chain.usersOf(mul.id))
        // The final ADD is the return; its consumer is the sentinel -1 (function sink).
        assertEquals(listOf(-1), chain.usersOf(add.id))
    }

    @Test
    fun sharedSubexpressionShowsMultipleUsers() {
        // f(x) = (x+1) * (x+1). The `x+1` result has two distinct consumers (the two
        // MUL operand slots aren't the same op — in practice dxir builder-reuse
        // would emit one ADD and two MUL operand references to it).
        val fn = DxirBuilder.function("reuse") {
            val x = param("x", f32s)
            val one = const(1f, f32s)
            val xp1 = op(OpKind.ADD, listOf(x, one), f32s)
            val sq = op(OpKind.MUL, listOf(xp1, xp1), f32s)
            listOf(sq)
        }
        val chain = DefUseChain.build(fn)
        val xp1 = fn.body.filterIsInstance<io.tlaloc.ir.DxirOp>().first { it.op == OpKind.ADD }
        // Both MUL operands reference xp1; the consumer-id dedup means MUL appears once.
        assertEquals(1, chain.useCount(xp1.id))
    }

    @Test
    fun unusedConstHasZeroUses() {
        // A const with no consumer. DxirFunction allows this — dead code.
        val fn = DxirBuilder.function("dead") {
            val x = param("x", f32s)
            const(999f, f32s) // dead const
            val r = op(OpKind.ADD, listOf(x, x), f32s)
            listOf(r)
        }
        val chain = DefUseChain.build(fn)
        val dead = fn.body.filterIsInstance<io.tlaloc.ir.DxirConst>().single()
        assertEquals(0, chain.useCount(dead.id))
    }

    @Test
    fun backwardReachableFromSinkCoversContributors() {
        // f(x) = x * x + dead_const.  (Wait, dead_const isn't an operand — let's
        // instead have a const that IS consumed and one that isn't.)
        // f(x) = x * x (uses only x); also a dead const. backwardReachable(sink)
        // should include x, the MUL op, but NOT the dead const.
        val fn = DxirBuilder.function("reach") {
            val x = param("x", f32s)
            val dead = const(999f, f32s) // dead
            val sq = op(OpKind.MUL, listOf(x, x), f32s)
            listOf(sq)
        }
        val chain = DefUseChain.build(fn)
        val sink = fn.returns.single()
        val reached = chain.backwardReachable(sink.id, fn)
        val x = fn.params.single()
        val dead = fn.body.filterIsInstance<io.tlaloc.ir.DxirConst>().single()
        val sq = fn.body.filterIsInstance<io.tlaloc.ir.DxirOp>().single()
        assertTrue(x.id in reached, "param x must be backward-reachable")
        assertTrue(sq.id in reached, "MUL must be backward-reachable")
        assertTrue(dead.id !in reached, "dead const must NOT be backward-reachable")
    }

    @Test
    fun backwardReachableIncludesRegionBodyOps() {
        // IF primal where only the predicate + then-branch yield reach the sink.
        // For C.1, `backwardReachable` widens to include ALL ops inside any region
        // it passes through — this is a conservative over-approximation that
        // C.2 can refine if needed.
        val fn = DxirBuilder.function("if_pred") {
            val x = param("x", f32s)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val ifOp = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region { yields(x) },
                elseRegion = region {
                    val negX = op(OpKind.NEG, listOf(x), f32s)
                    yields(negX)
                },
            )
            listOf(ifOp)
        }
        val chain = DefUseChain.build(fn)
        val sink = fn.returns.single()
        val reached = chain.backwardReachable(sink.id, fn)
        val x = fn.params.single()
        val ifN = fn.body.filterIsInstance<io.tlaloc.ir.DxirOp>().first { it.op == OpKind.IF }
        val pred = fn.body.filterIsInstance<io.tlaloc.ir.DxirOp>().first { it.op == OpKind.STEP }
        assertTrue(x.id in reached)
        assertTrue(pred.id in reached)
        assertTrue(ifN.id in reached)
    }

    @Test
    fun whileBodyOpsCountAsUsers() {
        // WHILE primal: the counter condition references outer-scope const(n). The
        // def-use chain must record the SUB inside the cond region as a consumer of
        // the outer const.
        val fn = DxirBuilder.function("loop") {
            val x = param("x", f32s)
            val n = const(5, i32s)
            val counterInit = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, counterInit),
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
        val chain = DefUseChain.build(fn)
        // The top-level const `n` is referenced by the SUB inside the cond region.
        val n = fn.body.filterIsInstance<io.tlaloc.ir.DxirConst>().first { it.type.dtype == I32 && it.value == 5 }
        val nUsers = chain.usersOf(n.id)
        // The SUB is what consumes n. One use.
        assertEquals(1, nUsers.size, "outer const must have 1 recorded consumer (the inner SUB)")
    }

    @Test
    fun functionReturnsCountAsSinks() {
        // The return list is recorded as having consumer = -1 (sentinel sink).
        val fn = DxirBuilder.function("id") {
            val x = param("x", f32s)
            val r = op(OpKind.ADD, listOf(x, x), f32s)
            listOf(r)
        }
        val chain = DefUseChain.build(fn)
        val r = fn.returns.single()
        assertTrue(
            -1 in chain.usersOf(r.id),
            "return value must have -1 sentinel as consumer",
        )
    }
}
