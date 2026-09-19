package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Phase A5c — implicit broadcasting on the elementwise binaries (DiffKT parity:
 * `broadcast(S1, S2)` under every binary op).
 *
 * Before this the three consumers disagreed about what a mixed-shape binary op
 * even meant: the interpreter required equal operand SIZES, the emitter's
 * `broadcastIfNeeded` hard-required equal RANK, and AddRule/SubRule handed
 * `upstream` straight to both operands while MulRule/DivRule typed their adjoint
 * products as `upstream.type` — so the adjoints were wrong-shaped whenever the
 * operands differed. `validateDxirShapes` was the only layer already
 * broadcast-aware (equal-rank operands: dim-for-dim equal or 1; rank-differing
 * operands explicitly left to the emitter).
 *
 * Now all four agree on NumPy semantics — operands right-align against the
 * result, a size-1 axis stretches, a rank-deficient operand gains replicated
 * leading axes — and the adjoints un-broadcast through `SUM_TO`, which reads its
 * target extents from the operand's ACTUAL runtime shape (§0.4.373). That last
 * property is what makes broadcasting sound under the -1 sentinel dims of
 * `grad {}`: which axes were size-1 (or absent) is statically unknowable there,
 * so the reduce has to be decided at execution, exactly as the in-place-stretch
 * and slice adjoints already do.
 */
class DxirBroadcastBinaryGradTest {

    private val scalar = DxirType(F32, emptyList())

    /** The value contract, pinned directly on the interpreter. */
    @Test
    fun broadcastingBinaryInterpreterPins() {
        // [3,1] ⊙ [1,4] → [3,4]: both operands stretch.
        run {
            val out = DxirType(F32, listOf(3, 4))
            val fn = DxirBuilder.function("bc_stretch") {
                val a = param("a", DxirType(F32, listOf(3, 1)))
                val b = param("b", DxirType(F32, listOf(1, 4)))
                listOf(op(OpKind.MUL, listOf(a, b), out))
            }
            val r = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(1f, 2f, 3f), floatArrayOf(10f, 20f, 30f, 40f)))[0]
            assertEquals(12, r.size)
            for (i in 0 until 3) for (j in 0 until 4) {
                val want = (i + 1) * (j + 1) * 10f
                assertTrue(abs(r[i * 4 + j] - want) < 1e-5f, "p[$i,$j] = ${r[i * 4 + j]}, want $want")
            }
        }
        // [3] ⊙ [2,3] → [2,3]: the rank-deficient operand gains a replicated leading axis.
        run {
            val out = DxirType(F32, listOf(2, 3))
            val fn = DxirBuilder.function("bc_rank") {
                val v = param("v", DxirType(F32, listOf(3)))
                val m = param("m", out)
                listOf(op(OpKind.ADD, listOf(v, m), out))
            }
            val r = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(1f, 2f, 3f), FloatArray(6) { it.toFloat() }))[0]
            assertEquals(6, r.size)
            // m = [0,1,2,3,4,5]; v broadcasts over both rows.
            for (k in 0 until 2) for (j in 0 until 3) {
                val want = (k * 3 + j).toFloat() + (j + 1).toFloat()
                assertTrue(abs(r[k * 3 + j] - want) < 1e-5f, "r[$k,$j] = ${r[k * 3 + j]}, want $want")
            }
        }
        // rank-0 ⊙ [2,2]: a scalar operand splats through the binary path.
        run {
            val out = DxirType(F32, listOf(2, 2))
            val fn = DxirBuilder.function("bc_scalar") {
                val s = param("s", scalar)
                val m = param("m", out)
                listOf(op(OpKind.SUB, listOf(m, s), out))
            }
            val r = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(5f), floatArrayOf(1f, 2f, 3f, 4f)))[0]
            assertTrue(r.indices.all { abs(r[it] - (it + 1 - 5).toFloat()) < 1e-5f }, r.toList().toString())
        }
        // Equal shapes are untouched: the flat-zip fast path, and DIV/POW included.
        run {
            val t = DxirType(F32, listOf(2, 2))
            val fn = DxirBuilder.function("bc_equal") {
                val a = param("a", t)
                val b = param("b", t)
                val q = op(OpKind.DIV, listOf(a, b), t)
                listOf(op(OpKind.POW, listOf(q, b), t))
            }
            val r = DxirInterpreter.evalFunction(
                fn,
                listOf(floatArrayOf(4f, 9f, 16f, 25f), floatArrayOf(2f, 3f, 4f, 5f)),
            )[0]
            // q = [2,3,4,5]; q^b = [2^2, 3^3, 4^4, 5^5]
            val want = listOf(4f, 27f, 256f, 3125f)
            for (i in want.indices) assertTrue(abs(r[i] - want[i]) < 1e-2f, "r[$i] = ${r[i]}, want ${want[i]}")
        }
        // Incompatible aligned extents (3 vs 4) fail loudly rather than silently
        // reading out of bounds.
        run {
            val out = DxirType(F32, listOf(4))
            val fn = DxirBuilder.function("bc_bad") {
                val a = param("a", DxirType(F32, listOf(3)))
                val b = param("b", out)
                listOf(op(OpKind.ADD, listOf(a, b), out))
            }
            assertFailsWith<IllegalArgumentException> {
                DxirInterpreter.evalFunction(fn, listOf(FloatArray(3), FloatArray(4)))
            }
        }
    }

    /**
     * loss = Σ (a[3,1] ⊙ b[1,4]) — the two-axis stretch.
     *   da[i,0] = Σ_j b[0,j]      SHAPE [3,1]
     *   db[0,j] = Σ_i a[i,0]      SHAPE [1,4]
     * MulRule's products are result-shaped ([3,4]), so each contribution must be
     * un-broadcast to its own operand — SUM_TO over the stretched axis, keeping it
     * size-1. Getting this wrong returns a [3,4] gradient for a [3,1] param.
     */
    @Test
    fun broadcastMultiplyGradientKeepsOperandShapes() {
        val fn = DxirBuilder.function("bc_mul_loss") {
            val a = param("a", DxirType(F32, listOf(3, 1)))
            val b = param("b", DxirType(F32, listOf(1, 4)))
            val p = op(OpKind.MUL, listOf(a, b), DxirType(F32, listOf(3, 4)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val a = floatArrayOf(1f, 2f, 3f)
        val b = floatArrayOf(10f, 20f, 30f, 40f)
        val out = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(a, b))

        assertEquals(3, out[0].size, "da must keep the [3,1] shape")
        for (i in 0 until 3) {
            val want = b.sum().toDouble().toFloat()
            assertTrue(abs(out[0][i] - want) < 1e-4f, "da[$i,0] = ${out[0][i]}, want $want")
        }
        assertEquals(4, out[1].size, "db must keep the [1,4] shape")
        for (j in 0 until 4) {
            val want = a.sum().toDouble().toFloat()
            assertTrue(abs(out[1][j] - want) < 1e-4f, "db[0,$j] = ${out[1][j]}, want $want")
        }
    }

    /**
     * loss = Σ (v[3] ⊙ m[2,3]) — rank extension.
     *   dv[j]   = Σ_k m[k,j]      SHAPE [3]  (the leading axis sums away)
     *   dm[k,j] = v[j]            SHAPE [2,3]
     */
    @Test
    fun rankExtendingMultiplyGradient() {
        val fn = DxirBuilder.function("bc_rank_loss") {
            val v = param("v", DxirType(F32, listOf(3)))
            val m = param("m", DxirType(F32, listOf(2, 3)))
            val p = op(OpKind.MUL, listOf(v, m), DxirType(F32, listOf(2, 3)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val v = floatArrayOf(1f, 2f, 3f)
        val m = floatArrayOf(10f, 20f, 30f, 40f, 50f, 60f)
        val out = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(v, m))

        assertEquals(3, out[0].size, "dv must stay rank-1 [3]")
        val wantDv = listOf(50f, 70f, 90f) // column sums of m
        for (j in 0 until 3) {
            assertTrue(abs(out[0][j] - wantDv[j]) < 1e-4f, "dv[$j] = ${out[0][j]}, want ${wantDv[j]}")
        }
        assertEquals(6, out[1].size, "dm must stay [2,3]")
        for (k in 0 until 2) for (j in 0 until 3) {
            assertTrue(abs(out[1][k * 3 + j] - v[j]) < 1e-5f, "dm[$k,$j] = ${out[1][k * 3 + j]}, want ${v[j]}")
        }
    }

    /**
     * A broadcasting DIV and ADD in one chain: loss = Σ ((s + a[4,1]) / b[1,3]).
     * Exercises SubRule's NEG path and DivRule's `a/(b·b)` term (whose shape is the
     * RESULT's under broadcasting) against mixed shapes.
     *   da[i,0] = Σ_j 1/b[0,j]
     *   db[0,j] = Σ_i −(s + a[i,0]) / b[0,j]²
     */
    @Test
    fun broadcastDivAddGradient() {
        val fn = DxirBuilder.function("bc_div_loss") {
            val s = param("s", scalar)
            val a = param("a", DxirType(F32, listOf(4, 1)))
            val b = param("b", DxirType(F32, listOf(1, 3)))
            val added = op(OpKind.ADD, listOf(s, a), DxirType(F32, listOf(4, 1)))
            val q = op(OpKind.DIV, listOf(added, b), DxirType(F32, listOf(4, 3)))
            listOf(op(OpKind.SUM, listOf(q), scalar))
        }
        val s = floatArrayOf(1f)
        val a = floatArrayOf(1f, 2f, 3f, 4f)
        val b = floatArrayOf(2f, 4f, 8f)
        val out = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(s, a, b))

        assertEquals(1, out[0].size, "ds is a scalar gradient")
        // s feeds every one of the 4×3 result entries: ds = Σ_i Σ_j 1/b_j.
        val wantDs = (a.size * b.sumOf { bj -> 1.0 / bj }).toFloat()
        assertTrue(abs(out[0][0] - wantDs) < 1e-4f, "ds = ${out[0][0]}, want $wantDs")
        assertEquals(4, out[1].size, "da keeps [4,1]")
        for (i in 0 until 4) {
            val want = b.sumOf { 1.0 / it }.toFloat()
            assertTrue(abs(out[1][i] - want) < 1e-4f, "da[$i] = ${out[1][i]}, want $want")
        }
        assertEquals(3, out[2].size, "db keeps [1,3]")
        for (j in 0 until 3) {
            val want = -a.sumOf { (1.0 + it) / (b[j].toDouble() * b[j].toDouble()) }.toFloat()
            assertTrue(abs(out[2][j] - want) < 1e-4f, "db[$j] = ${out[2][j]}, want $want")
        }
    }

    /**
     * A scalar seed splatted to a SYMBOLIC target shape carries that shape's node as
     * a second, shape-only operand — synthesis reads the extents off it at runtime
     * (`broadcastLike(v, template)`) instead of axis-matching static atoms against
     * the params, which is a guess that broadcasting makes unsafe. A CONCRETE target
     * must NOT get one: the template forces the summed node to be cloned into the
     * gradient body, which recomputes it and, for a rank-changing node the gradient
     * scope cannot synthesise, breaks synthesis outright.
     */
    @Test
    fun scalarSeedCarriesAShapeTemplateOnlyUnderSentinelDims() {
        val concrete = DxirBuilder.function("seed_concrete") {
            val a = param("a", DxirType(F32, listOf(2, 3)))
            val p = op(OpKind.MUL, listOf(a, a), DxirType(F32, listOf(2, 3)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val concreteSeed = DxirReverseTransform.apply(concrete).body
            .filterIsInstance<DxirOp>().single { it.op == OpKind.BROADCAST }
        assertEquals(1, concreteSeed.operands.size, "a concrete-dims seed needs no runtime template")

        val symbolic = DxirBuilder.function("seed_symbolic") {
            val a = param("a", DxirType(F32, listOf(-1, -1)))
            val p = op(OpKind.MUL, listOf(a, a), DxirType(F32, listOf(-1, -1)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val symbolicGrad = DxirReverseTransform.apply(symbolic)
        val symbolicSeed = symbolicGrad.body
            .filterIsInstance<DxirOp>().single { it.op == OpKind.BROADCAST }
        assertEquals(2, symbolicSeed.operands.size, "a sentinel-dims seed must carry its shape template")
        val template = symbolicSeed.operands[1]
        assertTrue(
            template is DxirOp && template.op == OpKind.MUL,
            "the template must be the summed primal node, got $template",
        )
        assertTrue(
            symbolicGrad.body.any { it === template },
            "the template must be CLONED into the gradient body, not left as a primal reference",
        )
    }

    /**
     * Phase A5c-3 — MAX's adjoint is the stretch-form case: its two un-reduce
     * broadcasts and its mask `1.0` splat all target `x`'s shape, which under
     * broadcasting is often an INTERMEDIATE no param has (`(v * m).max(1)`), so each
     * carries `x` as a shape-only template. Under concrete dims none of them does:
     * synthesis bakes every extent as a const and there is nothing to guess.
     */
    @Test
    fun axisMaxAdjointTemplatesItsBroadcastsOnlyUnderSentinels() {
        fun build(dims: List<Int>) = DxirBuilder.function("max_${dims.joinToString("_")}") {
            val x = param("x", DxirType(F32, dims))
            val mx = op(
                OpKind.MAX, listOf(x), DxirType(F32, dims.dropLast(1)),
                attrs = mapOf("reduction_dims" to listOf(1)),
            )
            listOf(op(OpKind.SUM, listOf(mx), scalar))
        }
        fun broadcasts(fn: io.tlaloc.ir.DxirFunction) =
            fn.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.BROADCAST }

        val concrete = broadcasts(DxirReverseTransform.apply(build(listOf(2, 3))))
        assertTrue(concrete.isNotEmpty(), "MAX's adjoint must contain broadcasts")
        assertTrue(
            concrete.all { it.operands.size == 1 },
            "concrete dims need no shape template: ${concrete.map { it.operands.size }}",
        )

        val symbolic = broadcasts(DxirReverseTransform.apply(build(listOf(-1, -1))))
        assertTrue(symbolic.isNotEmpty(), "MAX's adjoint must contain broadcasts")
        assertTrue(
            symbolic.all { it.operands.size == 2 },
            "every symbolic-dims broadcast in MAX's adjoint must carry a template: " +
                symbolic.map { it.operands.size },
        )
        assertTrue(
            symbolic.all { it.operands[1].type.dims == it.type.dims },
            "each template's shape must BE the broadcast target (x for the un-reduce " +
                "stretches and the mask splat, the max node for SUM's seed), got " +
                symbolic.map { it.operands[1].type.dims to it.type.dims },
        )
    }

    /** The interpreter reads a templated seed's shape from its own type and ignores the template. */
    @Test
    fun templatedSeedEvaluates() {
        val t = DxirType(F32, listOf(2, 2))
        val fn = DxirBuilder.function("templated_seed") {
            val a = param("a", t)
            val one = const(1.0f, scalar)
            val seed = op(
                OpKind.BROADCAST, listOf(one, a), t,
                attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
            )
            listOf(op(OpKind.MUL, listOf(seed, a), t))
        }
        val out = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(1f, 2f, 3f, 4f)))
        assertEquals(listOf(1f, 2f, 3f, 4f), out[0].toList())
    }

    /** JVP⇄VJP cross-identity through the broadcasting multiply — each mode certifies the other. */
    @Test
    fun broadcastBinaryJvpVjpCrossIdentity() {
        val fn = DxirBuilder.function("bc_chain") {
            val a = param("a", DxirType(F32, listOf(2, 1)))
            val b = param("b", DxirType(F32, listOf(1, 3)))
            val p = op(OpKind.MUL, listOf(a, b), DxirType(F32, listOf(2, 3)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val a = floatArrayOf(0.7f, -1.3f)
        val b = floatArrayOf(1.1f, -0.4f, 2.0f)
        val va = floatArrayOf(0.13f, -0.29f)
        val vb = floatArrayOf(-0.07f, 0.41f, 0.19f)

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(a, b))
        var dot = 0.0
        for (i in a.indices) dot += grads[0][i].toDouble() * va[i]
        for (j in b.indices) dot += grads[1][j].toDouble() * vb[j]

        val jvp = DxirInterpreter.evalFunction(DxirForwardTransform.apply(fn), listOf(a, b, va, vb))
        val tangent = jvp[1].single().toDouble()
        assertTrue(
            abs(dot - tangent) < 1e-5,
            "JVP⇄VJP cross-identity broken through implicit broadcasting: ⟨grad,v⟩=$dot vs tangent=$tangent",
        )
    }
}
