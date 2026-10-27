package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DxirRematTest {

    private val batch = 4
    private val width = 6
    private val layers = 3

    /** Σ (tanh(...tanh(x·W₁)...·Wₙ))², a stack of [layers] tanh layers. */
    private fun mlp(): DxirFunction = DxirBuilder.function("mlp") {
        val x = param("x", DxirType(F32, listOf(batch, width)))
        val ws = (1..layers).map { param("w$it", DxirType(F32, listOf(width, width))) }
        var h: io.tlaloc.ir.DxirNode = x
        val t = DxirType(F32, listOf(batch, width))
        for (w in ws) h = op(OpKind.TANH, listOf(op(OpKind.MATMUL, listOf(h, w), t)), t)
        listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(h, h), t)), DxirType(F32, emptyList())))
    }

    private val r = Random(11)
    private fun inputs() = listOf(FloatArray(batch * width) { r.nextFloat() - 0.5f }) +
        List(layers) { FloatArray(width * width) { (r.nextFloat() - 0.5f) * 0.8f } }

    private val grad = DxirReverseTransform.apply(mlp(), includeForward = true)

    /** The forward ops the backward part of [g] reads directly (not through a barrier). */
    private fun forwardReadsOf(g: DxirFunction): Set<OpKind> {
        val forward = HashSet<Int>()
        val stack = ArrayDeque(listOf(g.returns[0]))
        while (stack.isNotEmpty()) {
            val n = stack.removeLast()
            if (forward.add(n.id) && n is DxirOp) stack.addAll(n.operands)
        }
        return g.body.filterIsInstance<DxirOp>()
            .filter { it.id !in forward && it.op != OpKind.OPTIMIZATION_BARRIER }
            .flatMap { it.operands }
            .filter { it.id in forward && it !is DxirParam && it !is DxirConst }
            .mapNotNull { (it as? DxirOp)?.op }
            .toSet()
    }

    @Test
    fun theRematerializedGradientComputesTheSameValuesBitForBit() {
        val xs = inputs()
        val want = DxirInterpreter.evalFunction(grad, xs)
        for (policy in listOf(DxirRemat.SAVE_NOTHING, DxirRemat.SAVE_DOTS)) {
            val got = DxirInterpreter.evalFunction(DxirRemat.apply(grad, policy), xs)
            assertEquals(want.size, got.size)
            for (i in want.indices) assertContentEquals(want[i], got[i], "result $i")
        }
    }

    @Test
    fun saveNothingLeavesTheBackwardReadingNoForwardActivation() {
        assertTrue(forwardReadsOf(grad).isNotEmpty(), "the plain gradient reads forward activations")
        val remat = DxirRemat.apply(grad, DxirRemat.SAVE_NOTHING)
        assertEquals(emptySet(), forwardReadsOf(remat))
        assertTrue(remat.body.any { it is DxirOp && it.op == OpKind.OPTIMIZATION_BARRIER })
    }

    @Test
    fun saveDotsKeepsOnlyMatmulResults() {
        val remat = DxirRemat.apply(grad, DxirRemat.SAVE_DOTS)
        assertTrue(forwardReadsOf(remat).all { it == OpKind.MATMUL }, "reads ${forwardReadsOf(remat)}")
        // A barrier guards a parameter, a saved matmul result, or a value recomputed for an
        // earlier backward step; never an original forward value the policy did not save.
        val original = grad.body.associateBy { it.id }
        val barriers = remat.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.OPTIMIZATION_BARRIER }
        assertTrue(barriers.isNotEmpty())
        for (b in barriers) {
            val guarded = b.operands[0]
            val kept = guarded is DxirParam || original[guarded.id] == null || (guarded as DxirOp).op == OpKind.MATMUL
            assertTrue(kept, "barrier %${b.id} guards %${guarded.id}")
        }
    }

    @Test
    fun aRecomputeMarkOverridesThePolicy() {
        // The first layer's matmul is marked: SAVE_DOTS keeps the others but recomputes it.
        val marked = DxirBuilder.function("mlp_marked") {
            val x = param("x", DxirType(F32, listOf(batch, width)))
            val ws = (1..layers).map { param("w$it", DxirType(F32, listOf(width, width))) }
            var h: io.tlaloc.ir.DxirNode = x
            val t = DxirType(F32, listOf(batch, width))
            for ((i, w) in ws.withIndex()) {
                val attrs = if (i == 0) mapOf<String, Any>(DxirRemat.RECOMPUTE_ATTR to true) else emptyMap()
                h = op(OpKind.TANH, listOf(op(OpKind.MATMUL, listOf(h, w), t, attrs = attrs)), t)
            }
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(h, h), t)), DxirType(F32, emptyList())))
        }
        val g = DxirReverseTransform.apply(marked, includeForward = true)
        val remat = DxirRemat.apply(g, DxirRemat.SAVE_DOTS)
        val forwardMarked = remat.body.filterIsInstance<DxirOp>().filter { it.attrs[DxirRemat.RECOMPUTE_ATTR] == true }
        // The forward copy and the backward's recomputed copy.
        assertEquals(2, forwardMarked.size)
        val xs = inputs()
        val want = DxirInterpreter.evalFunction(g, xs)
        val got = DxirInterpreter.evalFunction(remat, xs)
        for (i in want.indices) assertContentEquals(want[i], got[i], "result $i")
    }

    @Test
    fun theBarrierIsTheIdentityUnderTheVjpAndJvp() {
        val t = DxirType(F32, listOf(3))
        val fn = DxirBuilder.function("b") {
            val x = param("x", t)
            val a = param("a", t)
            val y = op(OpKind.OPTIMIZATION_BARRIER, listOf(op(OpKind.SIN, listOf(x), t), a), t)
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(y, y), t)), DxirType(F32, emptyList())))
        }
        val x = floatArrayOf(0.3f, -0.7f, 1.1f)
        val a = floatArrayOf(5f, 6f, 7f)
        val g = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x, a))
        for (i in x.indices) {
            val want = 2 * kotlin.math.sin(x[i]) * kotlin.math.cos(x[i])
            assertEquals(want, g[0][i], 1e-6f)
            assertEquals(0f, g[1][i])
        }
    }
}
