package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import org.junit.jupiter.api.Assumptions.assumeFalse
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * [PjrtSession.executeOn] refuses a staged buffer whose dims or dtype differ from the parameter's,
 * by name. PJRT compares only byte sizes: a [2, 3] buffer passed for a [3, 2] parameter runs.
 */
class PjrtArgumentCheckTest {

    private val fn = DxirBuilder.function("transpose_me") {
        val x = param("x", DxirType(F32, listOf(3, 2)))
        listOf(op(OpKind.ADD, listOf(x, x), DxirType(F32, listOf(3, 2))))
    }

    @Test
    fun aBufferOfTheRightSizeAndTheWrongShapeIsRefused() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        assumeFalse(TestBackend.isMetal || TestBackend.isCerebras, "CUDA and TPU lanes")
        TestBackend.session().use { s ->
            val ok = s.bufferFromHostF32(FloatArray(6) { it.toFloat() }, listOf(3, 2))
            val wrong = s.bufferFromHostF32(FloatArray(6) { it.toFloat() }, listOf(2, 3))
            try {
                val out = s.executeOn(fn, listOf(ok))
                assertContentEquals(FloatArray(6) { 2f * it }, out[0].toFloatArray(6))
                out.forEach { it.close() }
                val e = assertFailsWith<IllegalArgumentException> { s.executeOn(fn, listOf(wrong)) }
                assertTrue("'x'" in e.message!! && "[2, 3]" in e.message!!, e.message)
                // With the check off, PJRT runs it (same byte size): the reason the check exists.
                s.validateArguments = false
                s.executeOn(fn, listOf(wrong)).forEach { it.close() }
            } finally {
                ok.close()
                wrong.close()
            }
        }
    }
}
