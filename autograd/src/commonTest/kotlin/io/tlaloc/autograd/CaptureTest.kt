package io.tlaloc.autograd

import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.ir.DxirModule
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.pretty
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CaptureTest {

    @Test
    fun capturesScalarSquareFunction() {
        val fn = capture(
            f = { x: Tracer<ScalarShape> -> x * x },
            input = Tensors.f32Scalar(3f),
            name = "square",
        )
        assertEquals("square", fn.name)
        assertEquals(1, fn.params.size)
        val mulOps = fn.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.MUL }
        assertEquals(1, mulOps.size)
        assertEquals(2, mulOps[0].operands.size)
        // Both operands are the same parameter (self-multiplication).
        assertTrue(mulOps[0].operands[0] === mulOps[0].operands[1])
    }

    @Test
    fun capturesVectorPipeline() {
        val fn = capture(
            f = { x: Tracer<Rank1<Sym>> -> (x * x).sum() },
            input = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f, 4f)),
            name = "sumOfSquares",
        )
        val ops = fn.body.filterIsInstance<DxirOp>().map { it.op }
        assertEquals(listOf(OpKind.MUL, OpKind.SUM), ops)
        val sumOp = fn.body.filterIsInstance<DxirOp>().last()
        assertEquals(0, sumOp.type.rank, "sum op output must be scalar")
    }

    @Test
    fun capturesMatmulChainWithTwoInputs() {
        val fn = capture2(
            f = { w: Tracer<Rank2<Sym, Sym>>, x: Tracer<Rank2<Sym, Sym>> ->
                (w matmul x).relu().sum()
            },
            a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f)),
            b = Tensors.f32Matrix<Sym, Sym>(2, 1, floatArrayOf(1f, 1f)),
            name = "layer",
        )
        assertEquals(2, fn.params.size)
        val kinds = fn.body.filterIsInstance<DxirOp>().map { it.op }
        assertEquals(listOf(OpKind.MATMUL, OpKind.RELU, OpKind.SUM), kinds)

        // The matmul op's output shape tracks the real dims at capture time.
        val matmul = fn.body.filterIsInstance<DxirOp>().first { it.op == OpKind.MATMUL }
        assertEquals(listOf(2, 1), matmul.type.dims)
    }

    @Test
    fun capturedFunctionRoundTripsThroughPrinter() {
        val fn = capture(
            f = { x: Tracer<Rank1<Sym>> -> x.relu().sum() },
            input = Tensors.f32Vector<Sym>(floatArrayOf(-1f, 2f, 3f)),
            name = "reluSum",
        )
        val text = DxirModule(listOf(fn)).pretty()
        assertTrue(text.contains("fn reluSum"), text)
        assertTrue(text.contains("relu"), text)
        assertTrue(text.contains("sum"), text)
        assertTrue(text.contains("return"), text)
    }

    @Test
    fun paramsPreserveLeafOrder() {
        val fn = capture2(
            f = { a: Tracer<Rank1<Sym>>, b: Tracer<Rank1<Sym>> -> (a + b).sum() },
            a = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f)),
            b = Tensors.f32Vector<Sym>(floatArrayOf(3f, 4f)),
        )
        val paramNames = fn.params.map { (it as DxirParam).name }
        assertEquals(2, paramNames.size)
        // Names are p0, p1 — reflect tape leaf ids in declaration order.
        assertEquals("p0", paramNames[0])
        assertEquals("p1", paramNames[1])
    }

    @Test
    fun returnIdsOutsideTapeRejected() {
        val tape = Tape()
        val leaf = tape.traceLeaf<ScalarShape>(Tensors.f32Scalar(1f))
        val ex = runCatching {
            tape.toDxirFunction("bad", paramIds = listOf(leaf.id), returnIds = listOf(999))
        }.exceptionOrNull()
        assertTrue(ex is IllegalArgumentException, "expected IAE, got $ex")
    }
}
