package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Phase A2b — `SLICE_LIKE`, the runtime-extent window slice that makes CONCAT
 * differentiable under `grad {}`.
 *
 * CONCAT's adjoint is "give each operand its window of the upstream". With concrete
 * dims the window bounds are literals and `ConcatRule` has baked them into
 * `SLICE`'s `start_indices`/`limit_indices` since §0.4.360 — but under `grad {}`'s
 * -1 sentinels operand i's offset is the cumulative sum of the PRIOR operands'
 * RUNTIME axis extents and its length is its own, so there is nothing to bake
 * (the old rule produced offsets of -1, -2, …). `SLICE_LIKE` therefore reads both
 * bounds off shape-only template operands at execution — the convention `SUM_TO`
 * (§0.4.373), `PAD_TO` (§0.4.374) and the A5c templated broadcasts all follow.
 *
 * `DxirShapePlumbingTest.concatForwardAndGradientSliceBack` keeps pinning the
 * concrete-dims path end to end; this file pins the new op's value contract, the
 * rule's concrete-vs-symbolic split, and the JVP⇄VJP cross-identity that CONCAT
 * never had.
 */
class DxirConcatGradTest {

    private val scalar = DxirType(F32, emptyList())

    /** The window contract, pinned directly on the interpreter. */
    @Test
    fun sliceLikeInterpreterPins() {
        // Trailing axis, window from the START (no priors): [2,5] → cols 0..1.
        run {
            val v = DxirType(F32, listOf(2, 5))
            val fn = DxirBuilder.function("sl_start") {
                val value = param("value", v)
                val this_ = param("t", DxirType(F32, listOf(2, 2)))
                listOf(op(OpKind.SLICE_LIKE, listOf(value, this_), DxirType(F32, listOf(2, 2)), attrs = mapOf("axis" to 1)))
            }
            // value = [[0,1,2,3,4],[10,11,12,13,14]]
            val out = DxirInterpreter.evalFunction(
                fn,
                listOf(floatArrayOf(0f, 1f, 2f, 3f, 4f, 10f, 11f, 12f, 13f, 14f), FloatArray(4)),
            )[0]
            assertEquals(listOf(0f, 1f, 10f, 11f), out.toList())
        }
        // Trailing axis, window AFTER one prior of extent 2: cols 2..4.
        run {
            val v = DxirType(F32, listOf(2, 5))
            val fn = DxirBuilder.function("sl_after1") {
                val value = param("value", v)
                val this_ = param("t", DxirType(F32, listOf(2, 3)))
                val prior = param("p", DxirType(F32, listOf(2, 2)))
                listOf(
                    op(
                        OpKind.SLICE_LIKE, listOf(value, this_, prior), DxirType(F32, listOf(2, 3)),
                        attrs = mapOf("axis" to 1),
                    ),
                )
            }
            val out = DxirInterpreter.evalFunction(
                fn,
                listOf(floatArrayOf(0f, 1f, 2f, 3f, 4f, 10f, 11f, 12f, 13f, 14f), FloatArray(6), FloatArray(4)),
            )[0]
            assertEquals(listOf(2f, 3f, 4f, 12f, 13f, 14f), out.toList())
        }
        // LEADING axis (so the copy is not one contiguous run): [6,3] → rows 2..5.
        run {
            val v = DxirType(F32, listOf(6, 3))
            val fn = DxirBuilder.function("sl_axis0") {
                val value = param("value", v)
                val this_ = param("t", DxirType(F32, listOf(4, 3)))
                val prior = param("p", DxirType(F32, listOf(2, 3)))
                listOf(
                    op(
                        OpKind.SLICE_LIKE, listOf(value, this_, prior), DxirType(F32, listOf(4, 3)),
                        attrs = mapOf("axis" to 0),
                    ),
                )
            }
            val value = FloatArray(18) { it.toFloat() }
            val out = DxirInterpreter.evalFunction(fn, listOf(value, FloatArray(12), FloatArray(6)))[0]
            assertEquals(12, out.size)
            for (i in 0 until 12) {
                // rows 2..5 of a row-major [6,3] → elements 6..17
                assertEquals((i + 6).toFloat(), out[i], "out[$i]")
            }
        }
        // Two priors accumulate: [2,7] with priors of extent 2 and 3 → cols 5..6.
        run {
            val v = DxirType(F32, listOf(2, 7))
            val fn = DxirBuilder.function("sl_after2") {
                val value = param("value", v)
                val this_ = param("t", DxirType(F32, listOf(2, 2)))
                val p0 = param("p0", DxirType(F32, listOf(2, 2)))
                val p1 = param("p1", DxirType(F32, listOf(2, 3)))
                listOf(
                    op(
                        OpKind.SLICE_LIKE, listOf(value, this_, p0, p1), DxirType(F32, listOf(2, 2)),
                        attrs = mapOf("axis" to 1),
                    ),
                )
            }
            val value = FloatArray(14) { it.toFloat() }
            val out = DxirInterpreter.evalFunction(fn, listOf(value, FloatArray(4), FloatArray(4), FloatArray(6)))[0]
            assertEquals(listOf(5f, 6f, 12f, 13f), out.toList())
        }
        // A window that does not fit fails loudly rather than reading past the end.
        run {
            val v = DxirType(F32, listOf(2, 5))
            val fn = DxirBuilder.function("sl_bad") {
                val value = param("value", v)
                val this_ = param("t", DxirType(F32, listOf(2, 4)))
                val prior = param("p", DxirType(F32, listOf(2, 2)))
                listOf(
                    op(
                        OpKind.SLICE_LIKE, listOf(value, this_, prior), DxirType(F32, listOf(2, 4)),
                        attrs = mapOf("axis" to 1),
                    ),
                )
            }
            assertFailsWith<IllegalArgumentException> {
                DxirInterpreter.evalFunction(fn, listOf(FloatArray(10), FloatArray(8), FloatArray(4)))
            }
        }
    }

    /**
     * The rule's two branches. Concrete dims keep the §0.4.360 static SLICEs and
     * clone no operand — every existing pin walks that IR unchanged. Symbolic dims
     * emit SLICE_LIKE with `thisTemplate` plus exactly the PRIOR operands, and
     * declare all operands read so the templates are cloned into the gradient body.
     */
    @Test
    fun concatRuleBakesWindowsOnlyWhenTheDimsAreConcrete() {
        fun build(dims: List<Pair<List<Int>, List<Int>>>) = DxirBuilder.function("c") {
            val a = param("a", DxirType(F32, dims[0].first))
            val b = param("b", DxirType(F32, dims[1].first))
            val c = op(
                OpKind.CONCAT, listOf(a, b), DxirType(F32, dims[0].second),
                attrs = mapOf("dimension" to 1),
            )
            listOf(op(OpKind.SUM, listOf(c), scalar))
        }
        fun ops(fn: DxirFunction, kind: OpKind) = fn.body.filterIsInstance<DxirOp>().filter { it.op == kind }

        val concrete = DxirReverseTransform.apply(
            build(listOf(listOf(2, 2) to listOf(2, 5), listOf(2, 3) to listOf(2, 5))),
        )
        assertEquals(2, ops(concrete, OpKind.SLICE).size, "concrete dims keep the static SLICE windows")
        assertEquals(0, ops(concrete, OpKind.SLICE_LIKE).size)
        assertEquals(
            listOf(0, 2),
            ops(concrete, OpKind.SLICE).map { (it.attrs["start_indices"] as List<*>)[1] },
            "the baked offsets are the cumulative axis extents",
        )

        // Symbolic dims, and concatenating INTERMEDIATES rather than params so the
        // "templates must be cloned into the gradient body" half is actually
        // observable (a param template is live by definition — it is a param).
        val symbolic = DxirReverseTransform.apply(
            DxirBuilder.function("c_symbolic") {
                val s = DxirType(F32, listOf(-1, -1))
                val a = param("a", s)
                val b = param("b", s)
                val na = op(OpKind.NEG, listOf(a), s)
                val nb = op(OpKind.NEG, listOf(b), s)
                val c = op(OpKind.CONCAT, listOf(na, nb), s, attrs = mapOf("dimension" to 1))
                listOf(op(OpKind.SUM, listOf(c), scalar))
            },
        )
        assertEquals(0, ops(symbolic, OpKind.SLICE).size, "symbolic dims must bake no window")
        val likes = ops(symbolic, OpKind.SLICE_LIKE)
        assertEquals(2, likes.size)
        // operand 0: (upstream, thisTemplate) — no priors, so the window starts at 0.
        assertEquals(2, likes[0].operands.size, likes[0].toString())
        // operand 1: (upstream, thisTemplate, prior = concat's operand 0).
        assertEquals(3, likes[1].operands.size, likes[1].toString())
        assertEquals(1, likes[1].attrs["axis"])
        assertTrue(
            likes.all { it.operands[1].type.dims == it.type.dims },
            "each thisTemplate's shape must BE its window's shape",
        )
        for (template in listOf(likes[0].operands[1], likes[1].operands[1], likes[1].operands[2])) {
            assertTrue(
                symbolic.body.any { it === template },
                "every SLICE_LIKE template must be cloned into the gradient body, " +
                    "not left as a primal reference: $template",
            )
        }
    }

    /** JVP⇄VJP cross-identity through CONCAT — absent before Phase A2b. */
    @Test
    fun concatJvpVjpCrossIdentity() {
        val aT = DxirType(F32, listOf(2, 2))
        val bT = DxirType(F32, listOf(2, 3))
        val cT = DxirType(F32, listOf(2, 5))
        val fn = DxirBuilder.function("concat_chain") {
            val a = param("a", aT)
            val b = param("b", bT)
            val c = op(OpKind.CONCAT, listOf(a, b), cT, attrs = mapOf("dimension" to 1))
            val doubled = op(OpKind.MUL, listOf(c, c), cT)
            listOf(op(OpKind.SUM, listOf(doubled), scalar))
        }
        val a = floatArrayOf(0.5f, -1.25f, 2f, 0.75f)
        val b = floatArrayOf(-0.5f, 1.5f, 0.25f, 3f, -2f, 0.125f)
        val va = floatArrayOf(0.13f, -0.29f, 0.41f, -0.07f)
        val vb = floatArrayOf(-0.19f, 0.23f, 0.11f, -0.31f, 0.17f, -0.05f)

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(a, b))
        var dot = 0.0
        for (i in a.indices) dot += grads[0][i].toDouble() * va[i]
        for (j in b.indices) dot += grads[1][j].toDouble() * vb[j]

        val jvp = DxirInterpreter.evalFunction(DxirForwardTransform.apply(fn), listOf(a, b, va, vb))
        val tangent = jvp[1].single().toDouble()
        assertTrue(
            abs(dot - tangent) < 1e-4,
            "JVP⇄VJP cross-identity broken through CONCAT: ⟨grad,v⟩=$dot vs tangent=$tangent",
        )
        // And the closed form: d/da Σ (a⊕b)² = 2a (the concat window is a's own).
        for (i in a.indices) {
            assertTrue(abs(grads[0][i] - 2f * a[i]) < 1e-5f, "da[$i] = ${grads[0][i]}, want ${2f * a[i]}")
        }
        for (j in b.indices) {
            assertTrue(abs(grads[1][j] - 2f * b[j]) < 1e-5f, "db[$j] = ${grads[1][j]}, want ${2f * b[j]}")
        }
    }
}
