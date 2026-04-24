package io.tlaloc.autograd

import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirModule
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
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
    fun capturesFunctionWithConstantLeafAsDxirConst() {
        // §0.4.71 — `capture` / `capture2` handle non-param leaves created via
        // §0.4.65's `Tracer.constant(f)`. Pre-§0.4.71, the `e.op == null` branch
        // in `toDxirFunction` called `const("leaf${id}", type)` passing a String
        // name as the const's *value*, producing a malformed DxirConst that
        // couldn't round-trip through the interpreter. That branch was dead code
        // before §0.4.65 (only `traceLeaf` made leaves); §0.4.65 surfaced it.
        //
        // This test captures `f(x) = x + const(5)` at x = 2 and evaluates the
        // captured DxirFunction — expected output is 7, NOT a type error.
        val fn = capture(
            f = { x: Tracer<ScalarShape> -> x + x.constant(5f) },
            input = Tensors.f32Scalar(2f),
            name = "plus_const",
        )
        val consts = fn.body.filterIsInstance<DxirConst>()
        assertEquals(1, consts.size, "captured function should have exactly one DxirConst (the constant leaf)")
        // The DxirConst's value must be the actual number we stored, not the
        // placeholder "leaf${id}" string the pre-§0.4.71 code shipped.
        val constNode = consts.single()
        assertTrue(
            constNode.value is Float || constNode.value is Double,
            "DxirConst value should be a numeric type; got ${constNode.value::class.simpleName} = ${constNode.value}",
        )
        val numericValue = (constNode.value as Number).toFloat()
        assertEquals(5f, numericValue)

        // End-to-end: evaluating the captured function at x=2 yields 7.
        val outputs = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(2f)))
        assertEquals(7f, outputs.single()[0])
    }

    @Test
    fun capturesFunctionWithRank1ConstantLeaf() {
        // §0.4.72 — rank-1 capture with a per-element FloatArray constant.
        // Pre-§0.4.72 the interpreter only accepted Number scalars and would
        // crash on the FloatArray const values §0.4.71's fix ships. Uses plain
        // elementwise ADD (no reduction) to stay inside evalFunction's bridge
        // op set.
        val fn = capture(
            f = { x: Tracer<Rank1<Sym>> -> x + x.constant(floatArrayOf(10f, 20f, 30f)) },
            input = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f)),
            name = "rank1_plus_const",
        )
        val consts = fn.body.filterIsInstance<DxirConst>()
        assertEquals(1, consts.size, "exactly one DxirConst for the rank-1 constant leaf")
        val constNode = consts.single()
        assertTrue(
            constNode.value is FloatArray,
            "rank-1 const carries a FloatArray; got ${constNode.value::class.simpleName}",
        )
        assertEquals(3, (constNode.value as FloatArray).size)

        // End-to-end: interpreter evaluates the rank-1 add, producing [11, 22, 33].
        val outputs = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(1f, 2f, 3f)))
        val result = outputs.single()
        assertEquals(11f, result[0])
        assertEquals(22f, result[1])
        assertEquals(33f, result[2])
    }

    @Test
    fun capturesFunctionWithRank2ConstantLeaf() {
        // §0.4.72 — rank-2 capture. x=[[1,2],[3,4]] + m=[[10,20],[30,40]] yields
        // [[11,22],[33,44]]. Exercises the FloatArray const path at rank 2.
        val fn = capture(
            f = { x: Tracer<Rank2<Sym, Sym>> ->
                val m: Tracer<Rank2<Sym, Sym>> =
                    x.constant(floatArrayOf(10f, 20f, 30f, 40f), intArrayOf(2, 2))
                x + m
            },
            input = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f)),
            name = "rank2_plus_const",
        )
        val consts = fn.body.filterIsInstance<DxirConst>()
        assertEquals(1, consts.size)
        val constNode = consts.single()
        assertTrue(
            constNode.value is FloatArray,
            "rank-2 const carries a FloatArray; got ${constNode.value::class.simpleName}",
        )
        assertEquals(4, (constNode.value as FloatArray).size)
        assertEquals(listOf(2, 2), constNode.type.dims)

        val outputs = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(1f, 2f, 3f, 4f)))
        val result = outputs.single()
        assertEquals(11f, result[0])
        assertEquals(22f, result[1])
        assertEquals(33f, result[2])
        assertEquals(44f, result[3])
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
