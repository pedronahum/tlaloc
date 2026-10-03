package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * [OpKind.CAUSAL_CONV1D] and [OpKind.GATED_DELTA_RULE] in both interpreters
 * against transformers' Qwen3.5 functions, from the committed fixture
 * `gdn_ops_fixture.json` (`harness/python/gdn_ops_fixture.py`): a prefill
 * block with a new sequence, a padding row and a continuation behind padding,
 * then a decode step that reads the pools the first call wrote.
 */
class GatedDeltaOpsFixtureTest {

    private val fx: JsonObject by lazy {
        parseJson(javaClass.getResourceAsStream("gdn_ops_fixture.json")!!.readBytes().toString(Charsets.UTF_8)) as JsonObject
    }

    private fun JsonObject.int(k: String) = (this[k] as JsonNumber).value.toInt()
    private fun JsonObject.floats(k: String) =
        (this[k] as JsonArray).elements.map { (it as JsonNumber).value.toFloat() }.toFloatArray()
    private fun JsonObject.grid(k: String) = (this[k] as JsonArray).elements.flatMap { row ->
        (row as JsonArray).elements.map { (it as JsonNumber).value.toFloat() }
    }.toFloatArray()

    private val dims by lazy { fx["dims"] as JsonObject }
    private val b by lazy { dims.int("B") }
    private val hk by lazy { dims.int("HK") }
    private val hv by lazy { dims.int("HV") }
    private val dk by lazy { dims.int("DK") }
    private val dv by lazy { dims.int("DV") }
    private val c by lazy { dims.int("C") }
    private val k by lazy { dims.int("K") }
    private val s by lazy { dims.int("S") }

    /** Both ops over one `[B, t]` block; returns (y, convPool, out, statePool). */
    private fun graph(t: Int): DxirFunction = DxirBuilder.function("gdnOps") {
        val tX = DxirType(F32, listOf(b, t, c))
        val tConv = DxirType(F32, listOf(s, k - 1, c))
        val tQk = DxirType(F32, listOf(b, t, hk, dk))
        val tV = DxirType(F32, listOf(b, t, hv, dv))
        val tG = DxirType(F32, listOf(b, t, hv))
        val tState = DxirType(F32, listOf(s, hv, dk, dv))
        val tIdx = DxirType(I32, listOf(b, t))
        val x = param("x", tX)
        val w = param("w", DxirType(F32, listOf(k, c)))
        val conv = param("conv", tConv)
        val q = param("q", tQk)
        val kk = param("k", tQk)
        val v = param("v", tV)
        val g = param("g", tG)
        val beta = param("beta", tG)
        val state = param("state", tState)
        val slots = param("slots", tIdx)
        val pos = param("pos", tIdx)
        val cv = opMulti(OpKind.CAUSAL_CONV1D, listOf(x, w, conv, slots, pos), listOf(tX, tConv))
        val dr = opMulti(OpKind.GATED_DELTA_RULE, listOf(q, kk, v, g, beta, state, slots, pos), listOf(tV, tState))
        listOf(cv.result(0), cv.result(1), dr.result(0), dr.result(1))
    }

    private fun check(name: String, got: FloatArray, want: FloatArray, tol: Float) {
        assertTrue(got.size == want.size, "$name: ${got.size} values, oracle has ${want.size}")
        var worst = 0f
        for (i in got.indices) worst = maxOf(worst, abs(got[i] - want[i]))
        assertTrue(worst <= tol, "$name: worst |interpreter - transformers| = $worst > $tol")
    }

    private fun run(eval: (DxirFunction, List<FloatArray>) -> List<FloatArray>, tol: Float) {
        val weight = fx.floats("weight")
        var conv = fx.floats("convPool")
        var state = fx.floats("statePool")
        for ((i, call) in (fx["calls"] as JsonArray).elements.withIndex()) {
            val o = call as JsonObject
            val out = eval(
                graph(o.int("T")),
                listOf(
                    o.floats("x"), weight, conv, o.floats("q"), o.floats("k"), o.floats("v"),
                    o.floats("g"), o.floats("beta"), state, o.grid("tokenSlots"), o.grid("positions"),
                ),
            )
            check("call $i y", out[0], o.floats("y"), tol)
            check("call $i convPool", out[1], o.floats("convPool"), tol)
            check("call $i out", out[2], o.floats("out"), tol)
            check("call $i statePool", out[3], o.floats("statePool"), tol)
            conv = out[1]
            state = out[3]
        }
    }

    @Test
    fun bothOpsMatchTransformersAcrossAPrefillAndADecodeCall() {
        run({ fn, ins -> DxirInterpreter.evalFunction(fn, ins) }, 2e-6f)
    }

    @Test
    fun theF64InterpreterMatchesTransformersToF32Precision() {
        run(
            { fn, ins ->
                DxirInterpreterF64.evalFunction(fn, ins.map { a -> DoubleArray(a.size) { a[it].toDouble() } })
                    .map { a -> FloatArray(a.size) { a[it].toFloat() } }
            },
            2e-6f,
        )
    }
}
