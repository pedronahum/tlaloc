package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Invalid calls and non-finite arithmetic, on the device and in the interpreter: both
 * refuse the same mistakes by name, give the same non-finite values, and differ where
 * StableHLO's semantics differ from the interpreter's (pinned, so a change shows). After
 * a refused call the session keeps working.
 */
class PjrtErrorParityTest {

    private val v4 = DxirType(F32, listOf(4))

    private val add = DxirBuilder.function("add2") {
        val a = param("a", v4)
        val b = param("b", v4)
        listOf(op(OpKind.ADD, listOf(a, b), v4))
    }

    private fun device(block: (PjrtSession) -> Unit) {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        TestBackend.session().use(block)
    }

    @Test
    fun aWrongNumberOfInputsIsRefusedByBoth() = device { s ->
        val one = listOf(FloatArray(4))
        val host = assertFailsWith<IllegalArgumentException> { DxirInterpreter.evalFunction(add, one) }
        val dev = assertFailsWith<IllegalArgumentException> { s.runOnHost(add, one) }
        assertTrue("2" in host.message!! && "1" in host.message!!, host.message)
        assertTrue("param count 2 != input count 1" in dev.message!!, dev.message)
        assertContentEquals(floatArrayOf(2f, 4f, 6f, 8f), s.runOnHost(add, List(2) { floatArrayOf(1f, 2f, 3f, 4f) })[0] as FloatArray)
    }

    @Test
    fun aWrongSizeIsRefusedByBothNamingTheParameter() = device { s ->
        val bad = listOf(FloatArray(4), FloatArray(3))
        val host = assertFailsWith<IllegalArgumentException> { DxirInterpreter.evalFunction(add, bad) }
        val dev = assertFailsWith<IllegalArgumentException> { s.runOnHost(add, bad) }
        for (m in listOf(host.message!!, dev.message!!)) assertTrue("'b'" in m && "4" in m && "3" in m, m)
    }

    @Test
    fun aHostArrayOfTheWrongDtypeIsRefusedByName() = device { s ->
        val e = assertFailsWith<IllegalArgumentException> { s.runOnHost(add, listOf(FloatArray(4), IntArray(4))) }
        assertTrue("'b'" in e.message!!, e.message)
    }

    @Test
    fun aWrongNumberOfDeviceBuffersIsRefusedBeforeExecuting() = device { s ->
        val text = add.toStablehlo()
        val a = s.bufferFromHostF32(floatArrayOf(1f, 2f, 3f, 4f), listOf(4))
        try {
            val e = assertFailsWith<IllegalArgumentException> { s.executeStablehlo(text, listOf(a)) }
            assertTrue("2" in e.message!! && "1" in e.message!!, e.message)
            val outs = s.executeStablehlo(text, listOf(a, a))
            assertContentEquals(floatArrayOf(2f, 4f, 6f, 8f), outs[0].toFloatArray(4))
            outs.forEach { it.close() }
        } finally {
            a.close()
        }
    }

    @Test
    fun nonFiniteValuesAreTheSame() = device { s ->
        // log(−1), sqrt(−1), 1/0, −1/0, 0/0, exp(100), rsqrt(0), log(0)
        val x = floatArrayOf(-1f, -1f, 1f, -1f, 0f, 100f, 0f, 0f)
        val t = DxirType(F32, listOf(8))
        val one = DxirType(F32, listOf(1))
        val fn = DxirBuilder.function("nonfinite") {
            val p = param("x", t)
            fun at(i: Int) = op(OpKind.SLICE, listOf(p), one, attrs = mapOf("start_indices" to listOf(i), "limit_indices" to listOf(i + 1), "strides" to listOf(1)))
            val zero = op(OpKind.SLICE, listOf(p), one, attrs = mapOf("start_indices" to listOf(4), "limit_indices" to listOf(5), "strides" to listOf(1)))
            val parts = listOf(
                op(OpKind.LOG, listOf(at(0)), one),
                op(OpKind.SQRT, listOf(at(1)), one),
                op(OpKind.DIV, listOf(at(2), zero), one),
                op(OpKind.DIV, listOf(at(3), zero), one),
                op(OpKind.DIV, listOf(at(4), zero), one),
                op(OpKind.EXP, listOf(at(5)), one),
                op(OpKind.RSQRT, listOf(at(6)), one),
                op(OpKind.LOG, listOf(at(7)), one),
            )
            listOf(op(OpKind.CONCAT, parts, t, attrs = mapOf("dimension" to 0)))
        }
        val host = DxirInterpreter.evalFunction(fn, listOf(x)).single()
        val dev = s.runOnHost(fn, listOf(x)).single() as FloatArray
        fun cls(f: Float) = when {
            f.isNaN() -> "nan"
            f == Float.POSITIVE_INFINITY -> "+inf"
            f == Float.NEGATIVE_INFINITY -> "-inf"
            else -> "finite"
        }
        assertEquals(listOf("nan", "nan", "+inf", "-inf", "nan", "+inf", "+inf", "-inf"), host.map(::cls), "interpreter")
        assertEquals(host.map(::cls), dev.map(::cls), "device against interpreter")
    }

    @Test
    fun anOutOfRangeGatherIndexIsRefusedByTheInterpreterAndClampedOnTheDevice() = device { s ->
        // StableHLO's gather clamps a start index into range; the interpreter refuses it by
        // name. Pinned both ways: a change to either is a decision, not an accident.
        val fn = DxirBuilder.function("gather_oob") {
            val x = param("x", DxirType(F32, listOf(3, 2)))
            val i = param("i", DxirType(I32, emptyList()))
            listOf(op(OpKind.GATHER, listOf(x, i), DxirType(F32, listOf(2))))
        }
        val x = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val e = assertFailsWith<IllegalArgumentException> { DxirInterpreter.evalFunction(fn, listOf(x, floatArrayOf(7f))) }
        assertTrue("out of bounds" in e.message!!, e.message)
        val dev = s.runOnHost(fn, listOf(x, intArrayOf(7))).single() as FloatArray
        assertContentEquals(floatArrayOf(5f, 6f), dev, "the device clamps index 7 to the last row")
    }

    @Test
    fun invalidStablehloIsRefusedAndTheSessionKeepsWorking() = device { s ->
        val bad = add.toStablehlo().replace("stablehlo.add", "stablehlo.not_an_op")
        val a = s.bufferFromHostF32(floatArrayOf(1f, 2f, 3f, 4f), listOf(4))
        try {
            val e = assertFails { s.executeStablehlo(bad, listOf(a, a)) }
            assertTrue(e.message!!.isNotBlank(), "the refusal says why")
        } finally {
            a.close()
        }
        assertContentEquals(floatArrayOf(2f, 4f, 6f, 8f), s.runOnHost(add, List(2) { floatArrayOf(1f, 2f, 3f, 4f) })[0] as FloatArray)
    }

    @Test
    fun aClosedSessionAndAClosedBufferAreRefusedByName() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val s = TestBackend.session()
        val b = s.bufferFromHostF32(floatArrayOf(1f, 2f, 3f, 4f), listOf(4))
        b.close()
        val closedBuffer = assertFails { s.executeStablehlo(add.toStablehlo(), listOf(b, b)) }
        assertEquals("PjrtBuffer used after close()", closedBuffer.message)
        s.close()
        val closedSession = assertFailsWith<IllegalStateException> { s.runOnHost(add, List(2) { FloatArray(4) }) }
        assertEquals("PjrtSession is closed", closedSession.message)
    }

    @Test
    fun theEntryParameterCountIsReadFromTheModule() {
        assertEquals(2, entryParamCount(add.toStablehlo()))
        val sharded = "sdy.mesh @m = <[\"d\"=2]>\nfunc.func @main(%0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@m, [{\"d\"}]>, " +
            "tf.aliasing_output = 0 : i32}, %1: tensor<f32>) -> (tensor<4xf32>, tensor<f32>) {\n  return %0, %1 : tensor<4xf32>, tensor<f32>\n}"
        assertEquals(2, entryParamCount(sharded))
        assertEquals(0, entryParamCount("func.func @f() -> tensor<f32> {\n}"))
        assertEquals(null, entryParamCount("func.func @a(%0: tensor<f32>) -> tensor<f32>\nfunc.func @b(%0: tensor<f32>) -> tensor<f32>"))
    }

    private fun assertFails(block: () -> Unit): Throwable = kotlin.test.assertFails(block)
}
