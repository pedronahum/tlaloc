package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.401 — Phase B4: the AD **nesting matrix**, certified honestly at IR
 * level. DiffKT supports arbitrary nesting of its forward and reverse
 * transforms; until this slice Tlaloc had pinned exactly ONE composition
 * (`forward(reverse(f))`, the §0.4.361 HVP). This file certifies the rest of
 * the 2×2 — and pins the compositions that CANNOT work as loud, named
 * refusals (the §0.4.392 pin-the-refusal precedent):
 *
 * - **forward∘forward** — the second directional derivative
 *   `d²f(x)[u, v] = uᵀ H v`. The transform composes mechanically: the second
 *   application re-tangents the jvp's params (`x, d_x` → `x, d_x, d_x,
 *   d_d_x`) and splits its returns again (`y, dy` → `y, dy, d_y, d_dy`).
 *   With the second tangent of `d_x` seeded to zero, the last return is the
 *   pure bilinear Hessian form.
 * - **reverse∘forward** — grad of the directional derivative:
 *   `∇ₓ⟨∇f(x), v⟩ = H v`. The jvp has two returns, and the reverse transform
 *   takes exactly one — the composition plumbing is a RETURN PROJECTION
 *   (same params, same body, tangent return only), after which the scalar
 *   gate holds and the reverse walk goes through untouched.
 * - **reverse∘reverse** — the seeded pullback of the gradient function IS
 *   the HVP: `R(R(f), seedAsParam)(v, x) = vᵀ ∂(∇f)/∂x = H v` (H symmetric),
 *   with no scalarization step at all. §0.4.399's runtime-extent VjpRules
 *   (SUM_TO → BROADCAST_LIKE and friends) are exactly what unblocked this
 *   cell: the first reverse pass emits those adjoint ops into its gradient
 *   body, and reverse-over-reverse must differentiate THROUGH them.
 * - **reverse∘reverse refusals** — gradient bodies containing fused adjoint
 *   ops with no VjpRule (MAXPOOL2D_GRAD, EMBEDDING_GRAD) and the
 *   symbolic-concat SLICE_LIKE (§0.4.399's documented remaining gap) fail
 *   loudly, naming the offending op.
 *
 * Every passing cell is pinned against a hand-derived analytic reference AND
 * against `forward(reverse(f))` on the same point — all four routes to the
 * same Hessian must agree numerically, so each composition certifies the
 * others.
 */
class DxirNestingMatrixTest {

    private val scalar = DxirType(F32, emptyList())
    private val vec4 = DxirType(F32, listOf(4))

    /** f(x) = Σ x³ over a 4-vector: ∇f = 3x², H = diag(6x). */
    private fun cubicFn(): DxirFunction = DxirBuilder.function("cubic") {
        val x = param("x", vec4)
        val x2 = op(OpKind.MUL, listOf(x, x), vec4)
        val x3 = op(OpKind.MUL, listOf(x2, x), vec4)
        listOf(op(OpKind.SUM, listOf(x3), scalar))
    }

    /**
     * The reverse∘forward plumbing: project a scalar-f jvp down to its
     * tangent return. Same params, same body — only the returns list
     * changes, so [DxirReverseTransform]'s single-scalar-return gate holds.
     */
    private fun tangentOnly(jvp: DxirFunction): DxirFunction =
        DxirFunction(jvp.name + "_tangent", jvp.params, jvp.body, listOf(jvp.returns.last()), jvp.meshes)

    private val xVals = floatArrayOf(1f, -2f, 0.5f, 3f)
    private val uVals = floatArrayOf(1f, 0f, -1f, 2f)
    private val vVals = floatArrayOf(0.5f, 1f, 2f, -1f)

    // ---------------------------------------------------------------- fwd∘fwd

    @Test
    fun forwardOverForwardSecondDirectionalDerivative() {
        // jvp²_f(x, u, v, w): params are the jvp's (x, u) re-tangented with
        // (v, w); returns (y, f′u, f′v, f″[v,u] + f′w). With w = 0 the last
        // return is the bilinear Hessian form: for f = Σx³ that's 6·Σ(x⊙u⊙v).
        val jvp = DxirForwardTransform.apply(cubicFn())
        val jvp2 = DxirForwardTransform.apply(jvp)

        // The mechanical-composition pins: param re-tangenting and return split.
        assertEquals(4, jvp2.params.size, "jvp params re-tangented")
        assertEquals(
            listOf("x", "d_x", "d_x", "d_d_x"),
            jvp2.params.map { it.name },
            "second application tangents the first's params, d_d_x included",
        )
        assertEquals(4, jvp2.returns.size, "the jvp's (y, dy) each get a tangent")

        val w0 = FloatArray(4)
        val out = DxirInterpreter.evalFunction(jvp2, listOf(xVals, uVals, vVals, w0))

        var y = 0.0; var fu = 0.0; var fv = 0.0; var huv = 0.0
        for (i in 0 until 4) {
            val x = xVals[i].toDouble()
            y += x * x * x
            fu += 3 * x * x * uVals[i]
            fv += 3 * x * x * vVals[i]
            huv += 6 * x * uVals[i] * vVals[i]
        }
        assertTrue(abs(out[0].single() - y) <= 1e-4 * maxOf(1.0, abs(y)), "primal survives twice")
        assertTrue(abs(out[1].single() - fu) <= 1e-4 * maxOf(1.0, abs(fu)), "inner tangent f′(x)u")
        assertTrue(abs(out[2].single() - fv) <= 1e-4 * maxOf(1.0, abs(fv)), "outer tangent f′(x)v")
        assertTrue(
            abs(out[3].single() - huv) <= 1e-4 * maxOf(1.0, abs(huv)),
            "d²f(x)[u,v] = ${out[3].single()}, want uᵀHv = $huv",
        )
    }

    @Test
    fun forwardOverForwardHessianEntriesAgreeWithForwardOverReverse() {
        // Shared body with the §0.4.394 hessian intrinsic's value-DEPENDENT
        // case: f = Σ exp(x), H = diag(exp x). Basis-seeded fwd∘fwd entries
        // H_ij must match both the analytic diagonal and the fwd∘rev HVP
        // columns — two independent compositions extracting the same matrix.
        val t = DxirType(F32, listOf(3))
        val fn = DxirBuilder.function("sum_exp") {
            val x = param("x", t)
            val e = op(OpKind.EXP, listOf(x), t)
            listOf(op(OpKind.SUM, listOf(e), scalar))
        }
        val ff = DxirForwardTransform.apply(DxirForwardTransform.apply(fn))
        val fr = DxirForwardTransform.apply(DxirReverseTransform.apply(fn))
        val x = floatArrayOf(0.3f, -1.1f, 0.8f)
        val w0 = FloatArray(3)

        for (j in 0 until 3) {
            val ej = FloatArray(3).also { it[j] = 1f }
            // fwd∘rev column j: returns (∇f, H·e_j).
            val col = DxirInterpreter.evalFunction(fr, listOf(x, ej))[1]
            for (i in 0 until 3) {
                val ei = FloatArray(3).also { it[i] = 1f }
                val hij = DxirInterpreter.evalFunction(ff, listOf(x, ei, ej, w0))[3].single()
                val want = if (i == j) exp(x[i].toDouble()) else 0.0
                assertTrue(
                    abs(hij - want) <= 1e-5 * maxOf(1.0, abs(want)),
                    "fwd∘fwd H[$i,$j] = $hij, want analytic $want",
                )
                assertTrue(
                    abs(hij - col[i]) <= 1e-5f * maxOf(1f, abs(col[i])),
                    "fwd∘fwd H[$i,$j] = $hij disagrees with fwd∘rev ${col[i]}",
                )
            }
        }
    }

    // ---------------------------------------------------------------- rev∘fwd

    @Test
    fun reverseOverForwardGradOfDirectionalDerivativeIsHvp() {
        // ∇ₓ⟨∇f(x), v⟩ = H v. The tangent-only projection makes the jvp a
        // legal reverse-transform input (one scalar return); the resulting
        // gradient function's returns are (∇ₓ dy, ∇_dx dy) = (H·d_x, ∇f(x)).
        val fn = cubicFn()
        val g = DxirReverseTransform.apply(tangentOnly(DxirForwardTransform.apply(fn)))
        assertEquals(2, g.params.size, "grads for (x, d_x)")
        val out = DxirInterpreter.evalFunction(g, listOf(xVals, vVals))

        val frOut = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(DxirReverseTransform.apply(fn)), listOf(xVals, vVals),
        )
        var maxAbs = 0f
        for (i in 0 until 4) {
            val hv = 6f * xVals[i] * vVals[i]
            val gradF = 3f * xVals[i] * xVals[i]
            maxAbs = maxOf(maxAbs, abs(out[0][i] - hv))
            maxAbs = maxOf(maxAbs, abs(out[1][i] - gradF))
            // The beautiful cross-oracle: rev∘fwd == fwd∘rev, numerically.
            maxAbs = maxOf(maxAbs, abs(out[0][i] - frOut[1][i]))
        }
        assertTrue(maxAbs <= 1e-4f, "rev∘fwd diverges from analytic Hv / fwd∘rev: max|diff|=$maxAbs")
    }

    // ---------------------------------------------------------------- rev∘rev

    @Test
    fun reverseOverReverseSeededPullbackIsHvp() {
        // The seeded pullback of the gradient function IS the HVP — no
        // scalarization needed: R(R(f), seedAsParam)(v, x) = vᵀ ∂(∇f)/∂x = H v.
        val g = DxirReverseTransform.apply(cubicFn())
        val rr = DxirReverseTransform.apply(g, seedAsParam = true)
        assertEquals(2, rr.params.size, "(__upstream__, x)")
        val out = DxirInterpreter.evalFunction(rr, listOf(vVals, xVals))

        val frOut = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(DxirReverseTransform.apply(cubicFn())), listOf(xVals, vVals),
        )
        var maxAbs = 0f
        for (i in 0 until 4) {
            maxAbs = maxOf(maxAbs, abs(out[0][i] - 6f * xVals[i] * vVals[i]))
            maxAbs = maxOf(maxAbs, abs(out[0][i] - frOut[1][i]))
        }
        assertTrue(maxAbs <= 1e-4f, "rev∘rev diverges from analytic Hv / fwd∘rev: max|diff|=$maxAbs")
    }

    @Test
    fun reverseOverReverseThroughInPlaceBroadcastStretch() {
        // The cell §0.4.399 unblocked, exercised end to end: an equal-rank
        // size-1 stretch makes the FIRST reverse pass emit SUM_TO into its
        // gradient body, and the second reverse pass must differentiate
        // through it via SumToRule → BROADCAST_LIKE. f(x:[1,3]) = Σ(b ⊙ b)
        // with b = broadcast(x → [2,3]) is 2·Σx², so ∇f = 4x and H = 4I.
        val xT = DxirType(F32, listOf(1, 3))
        val bT = DxirType(F32, listOf(2, 3))
        val fn = DxirBuilder.function("stretch_sq") {
            val x = param("x", xT)
            val b = op(
                OpKind.BROADCAST, listOf(x), bT,
                attrs = mapOf("broadcast_dimensions" to listOf(0, 1)),
            )
            val m = op(OpKind.MUL, listOf(b, b), bT)
            listOf(op(OpKind.SUM, listOf(m), scalar))
        }
        val g = DxirReverseTransform.apply(fn)
        assertTrue(
            g.body.filterIsInstance<DxirOp>().any { it.op == OpKind.SUM_TO },
            "precondition: the first reverse pass emits SUM_TO for the stretch",
        )
        val rr = DxirReverseTransform.apply(g, seedAsParam = true)
        assertTrue(
            rr.body.filterIsInstance<DxirOp>().any { it.op == OpKind.BROADCAST_LIKE },
            "SumToRule's BROADCAST_LIKE adjoint lands in the second gradient body",
        )
        val v = floatArrayOf(0.7f, -1.5f, 2f)
        val x = floatArrayOf(0.4f, 1.1f, -0.9f)
        val out = DxirInterpreter.evalFunction(rr, listOf(v, x))
        var maxAbs = 0f
        for (i in 0 until 3) maxAbs = maxOf(maxAbs, abs(out[0][i] - 4f * v[i]))
        assertTrue(maxAbs <= 1e-5f, "rev∘rev through SUM_TO diverges from 4v: max|diff|=$maxAbs")
    }

    @Test
    fun reverseOverReverseThroughConcreteConcatWindows() {
        // With concrete dims ConcatRule takes the static-SLICE branch, and
        // SLICE has a VjpRule — so rev∘rev over a concat body WORKS at IR
        // level. Only the symbolic-dims SLICE_LIKE branch is the documented
        // gap (pinned below). f(x:[2]) = Σ concat(x, k)² = Σx² + const, so
        // H = 2I and the pullback is 2v.
        val xT = DxirType(F32, listOf(2))
        val cT = DxirType(F32, listOf(5))
        val fn = DxirBuilder.function("concat_sq") {
            val x = param("x", xT)
            val k = const(floatArrayOf(0.5f, -1f, 2f), DxirType(F32, listOf(3)))
            val c = op(OpKind.CONCAT, listOf(x, k), cT, attrs = mapOf("dimension" to 0))
            val m = op(OpKind.MUL, listOf(c, c), cT)
            listOf(op(OpKind.SUM, listOf(m), scalar))
        }
        val rr = DxirReverseTransform.apply(DxirReverseTransform.apply(fn), seedAsParam = true)
        val v = floatArrayOf(3f, -0.25f)
        val x = floatArrayOf(1.2f, 0.8f)
        val out = DxirInterpreter.evalFunction(rr, listOf(v, x))
        var maxAbs = 0f
        for (i in 0 until 2) maxAbs = maxOf(maxAbs, abs(out[0][i] - 2f * v[i]))
        assertTrue(maxAbs <= 1e-5f, "rev∘rev through concrete concat diverges from 2v: max|diff|=$maxAbs")
    }

    @Test
    fun reverseOverReverseRefusesFusedPoolAdjoint() {
        // The first reverse pass over maxpool emits the fused MAXPOOL2D_GRAD
        // (§0.4.389), which deliberately has no VjpRule — the second pass must
        // refuse loudly, naming the op.
        val xT = DxirType(F32, listOf(1, 1, 2, 2))
        val yT = DxirType(F32, listOf(1, 1, 1, 1))
        val fn = DxirBuilder.function("pool_loss") {
            val x = param("x", xT)
            val y = op(OpKind.MAXPOOL2D, listOf(x), yT, attrs = mapOf("window" to listOf(2, 2)))
            listOf(op(OpKind.SUM, listOf(y), scalar))
        }
        val g = DxirReverseTransform.apply(fn)
        val e = assertFailsWith<IllegalStateException> {
            DxirReverseTransform.apply(g, seedAsParam = true)
        }
        assertTrue(
            "MAXPOOL2D_GRAD" in (e.message ?: ""),
            "refusal must name the ruleless fused adjoint; got: ${e.message}",
        )
    }

    @Test
    fun reverseOverReverseRefusesEmbeddingAdjoint() {
        // Same shape of refusal for the fused EMBEDDING_GRAD scatter (§0.4.370).
        val fn = DxirBuilder.function("embed_loss") {
            val table = param("table", DxirType(F32, listOf(3, 2)))
            val idx = const(floatArrayOf(0f, 2f, 0f), DxirType(I32, listOf(3)))
            val emb = op(OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(3, 2)))
            listOf(op(OpKind.SUM, listOf(emb), scalar))
        }
        val g = DxirReverseTransform.apply(fn)
        val e = assertFailsWith<IllegalStateException> {
            DxirReverseTransform.apply(g, seedAsParam = true)
        }
        assertTrue(
            "EMBEDDING_GRAD" in (e.message ?: ""),
            "refusal must name the ruleless fused adjoint; got: ${e.message}",
        )
    }

    @Test
    fun reverseOverReverseRefusesSymbolicConcatSliceLike() {
        // §0.4.399's deliberately-remaining gap, pinned as a loud contract:
        // under sentinel dims ConcatRule emits SLICE_LIKE, whose window
        // offset is a runtime sum of prior templates' extents — no
        // literal-low adjoint can express it, so it has no VjpRule and the
        // second reverse pass must refuse naming it. (A PAD_LIKE with
        // prior-template offsets is the recorded shape of the fix.)
        val symT = DxirType(F32, listOf(-1))
        val fn = DxirBuilder.function("sym_concat") {
            val x = param("x", symT)
            val c = op(OpKind.CONCAT, listOf(x, x), symT, attrs = mapOf("dimension" to 0))
            val m = op(OpKind.MUL, listOf(c, c), symT)
            listOf(op(OpKind.SUM, listOf(m), scalar))
        }
        val g = DxirReverseTransform.apply(fn)
        val e = assertFailsWith<IllegalStateException> {
            DxirReverseTransform.apply(g, seedAsParam = true)
        }
        assertTrue(
            "SLICE_LIKE" in (e.message ?: ""),
            "refusal must name SLICE_LIKE; got: ${e.message}",
        )
    }

    // ------------------------------------------------------------ third order

    @Test
    fun thirdOrderForwardForwardReverseOnQuartic() {
        // fwd∘fwd∘rev composes for free: F(F(R(f)))(x, u, v, 0) ends in the
        // third-derivative bilinear form ∂³f[u, v] — for f = Σx⁴ (∇f = 4x³,
        // H = diag(12x²)) that's the vector 24·x ⊙ u ⊙ v.
        val fn = DxirBuilder.function("quartic") {
            val x = param("x", vec4)
            val x2 = op(OpKind.MUL, listOf(x, x), vec4)
            val x4 = op(OpKind.MUL, listOf(x2, x2), vec4)
            listOf(op(OpKind.SUM, listOf(x4), scalar))
        }
        val t3 = DxirForwardTransform.apply(
            DxirForwardTransform.apply(DxirReverseTransform.apply(fn)),
        )
        val w0 = FloatArray(4)
        val out = DxirInterpreter.evalFunction(t3, listOf(xVals, uVals, vVals, w0))
        var maxAbs = 0.0
        for (i in 0 until 4) {
            val x = xVals[i].toDouble()
            maxAbs = maxOf(maxAbs, abs(out[0][i] - 4 * x * x * x))
            maxAbs = maxOf(maxAbs, abs(out[1][i] - 12 * x * x * uVals[i]))
            maxAbs = maxOf(maxAbs, abs(out[2][i] - 12 * x * x * vVals[i]))
            maxAbs = maxOf(maxAbs, abs(out[3][i] - 24 * x * uVals[i] * vVals[i]))
        }
        assertTrue(maxAbs <= 1e-3, "third order diverges from analytic 24·x⊙u⊙v: max|diff|=$maxAbs")
    }
}
