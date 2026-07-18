package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** §0.4.353 — conservative static shape validation tests. */
class DxirShapeValidationTest {

    @Test
    fun reportsMatmulContractDimMismatch() {
        val fn = DxirBuilder.function("bad") {
            val a = param("a", DxirType(F32, listOf(2, 3)))
            val b = param("b", DxirType(F32, listOf(4, 5)))
            listOf(op(OpKind.MATMUL, listOf(a, b), DxirType(F32, listOf(2, 5))))
        }
        val errors = validateDxirShapes(fn)
        assertEquals(1, errors.size, errors.toString())
        assertTrue(errors.single().contains("contract-dim mismatch"), errors.single())
        assertTrue(errors.single().contains("(3)") && errors.single().contains("(4)"), errors.single())
    }

    @Test
    fun reportsNonBroadcastableElementwiseMismatch() {
        val fn = DxirBuilder.function("bad") {
            val a = param("a", DxirType(F32, listOf(2, 3)))
            val b = param("b", DxirType(F32, listOf(2, 4)))
            listOf(op(OpKind.ADD, listOf(a, b), DxirType(F32, listOf(2, 3))))
        }
        val errors = validateDxirShapes(fn)
        assertEquals(1, errors.size, errors.toString())
        assertTrue(errors.single().contains("incompatible at dim 1"), errors.single())
    }

    @Test
    fun dotGradientMatchesAnalytic() {
        // §0.4.353 — DotRule: s = a·b; ds/da = b, ds/db = a (seed 1).
        val vec = DxirType(F32, listOf(3))
        val primal = DxirBuilder.function("dot") {
            val a = param("a", vec)
            val b = param("b", vec)
            listOf(op(OpKind.DOT, listOf(a, b), DxirType(F32, emptyList())))
        }
        val grad = DxirReverseTransform.apply(primal)
        assertEquals(2, grad.returns.size)
        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(1f, 2f, 3f), floatArrayOf(4f, 5f, 6f)),
        )
        assertEquals(listOf(4f, 5f, 6f), out[0].toList(), "ds/da = b")
        assertEquals(listOf(1f, 2f, 3f), out[1].toList(), "ds/db = a")
    }

    @Test
    fun staysSilentOnValidBroadcastAndSymbolicDims() {
        val fn = DxirBuilder.function("ok") {
            val a = param("a", DxirType(F32, listOf(2, 3)))
            val one = param("bias", DxirType(F32, listOf(2, 1)))     // broadcastable
            val sym = param("s", DxirType(F32, listOf(-1, 3)))       // symbolic dim
            val added = op(OpKind.ADD, listOf(a, one), DxirType(F32, listOf(2, 3)))
            val symAdd = op(OpKind.ADD, listOf(added, sym), DxirType(F32, listOf(2, 3)))
            val w = param("w", DxirType(F32, listOf(3, 5)))
            listOf(op(OpKind.MATMUL, listOf(symAdd, w), DxirType(F32, listOf(2, 5))))
        }
        assertEquals(emptyList(), validateDxirShapes(fn))
    }
}
