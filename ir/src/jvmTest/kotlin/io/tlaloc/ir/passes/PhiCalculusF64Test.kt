package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirRegionBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.math.pow
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The loop closed forms (C7: `d ← a·d + b[i]`, C9: `d ← a·dᵇ`) on F64 loops: the constant
 * coefficients reach the symbolic engine at full precision. Each rewritten function is
 * evaluated by [DxirInterpreterF64] and compared with the loop iterated in Double.
 *
 * The coefficients (1.01, 0.3, 0.9, 1.5) are not F32-representable; rounded to F32 they
 * move the results by 1e-8 relative or more. The closed forms themselves are exact
 * rewrites, so the bound is 1e-12.
 */
class PhiCalculusF64Test {
    private val f64s = DxirType(F64, emptyList())
    private val boolS = DxirType(Bool, emptyList())
    private val engine = SymjaEngine()

    private fun countOps(fn: DxirFunction, kind: OpKind): Int =
        fn.body.filterIsInstance<DxirOp>().count { it.op == kind }

    private fun loop(n: Int, step: DxirRegionBuilder.(d: DxirNode, i: DxirNode) -> DxirNode): DxirFunction =
        DxirBuilder.function("loop") {
            val p = param("p", f64s)
            val bound = const(n.toDouble(), f64s)
            val w = whileOp(
                inits = listOf(p, const(0.0, f64s)),
                cond = { args -> yields(op(OpKind.STEP, listOf(op(OpKind.SUB, listOf(bound, args[1]), f64s)), boolS)) },
                body = { args ->
                    yields(step(args[0], args[1]), op(OpKind.ADD, listOf(args[1], const(1.0, f64s)), f64s))
                },
            )
            listOf(w.result(0))
        }

    private fun assertRel(want: Double, got: Double, what: String) =
        assertTrue(abs(want - got) <= 1e-12 * abs(want), "$what: got $got, want $want (${abs(want - got) / abs(want)} relative)")

    @Test
    fun c7ClosedFormKeepsAnF64CoefficientExact() {
        // d ← 1.01·d + 0.3·i, 40 steps.
        val fn = loop(40) { d, i ->
            op(
                OpKind.ADD,
                listOf(
                    op(OpKind.MUL, listOf(const(1.01, f64s), d), f64s),
                    op(OpKind.MUL, listOf(i, const(0.3, f64s)), f64s),
                ),
                f64s,
            )
        }
        val rewritten = PhiCalculus.apply(fn, engine)
        assertEquals(0, countOps(rewritten, OpKind.WHILE), "C7 should close the loop")
        for (p in listOf(0.7, -2.5)) {
            var d = p
            for (i in 0 until 40) d = 1.01 * d + 0.3 * i
            assertRel(d, DxirInterpreterF64.evalFunction(rewritten, listOf(doubleArrayOf(p))).single().single(), "C7 at p=$p")
        }
    }

    @Test
    fun c9ClosedFormKeepsF64CoefficientsExact() {
        // d ← 0.9·d^1.5, 3 steps.
        val fn = loop(3) { d, _ ->
            op(OpKind.MUL, listOf(const(0.9, f64s), op(OpKind.POW, listOf(d, const(1.5, f64s)), f64s)), f64s)
        }
        val rewritten = PhiCalculus.apply(fn, engine)
        assertEquals(0, countOps(rewritten, OpKind.WHILE), "C9 should close the loop")
        for (p in listOf(0.7, 1.3)) {
            var d = p
            repeat(3) { d = 0.9 * d.pow(1.5) }
            assertRel(d, DxirInterpreterF64.evalFunction(rewritten, listOf(doubleArrayOf(p))).single().single(), "C9 at p=$p")
        }
    }
}
