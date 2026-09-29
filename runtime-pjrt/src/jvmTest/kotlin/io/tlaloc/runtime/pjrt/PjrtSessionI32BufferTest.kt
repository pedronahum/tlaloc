package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.test.Test
import kotlin.test.assertContentEquals

/**
 * [PjrtSession.bufferFromHostI32]: an i32 buffer staged once and passed to
 * [PjrtSession.executeOn] carries the values [PjrtSession.runOn] passes as
 * whole-number floats, including values above 2²⁴ that the float encoding
 * cannot carry.
 */
class PjrtSessionI32BufferTest {

    @Test
    fun stagedI32BufferRunsLikeRunOn() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val ids = intArrayOf(0, 7, -3, 151_935, 1 shl 20, 42)
        val fn = DxirBuilder.function("i32_staging_probe") {
            val x = param("ids", DxirType(I32, listOf(ids.size)))
            listOf(op(OpKind.CAST, listOf(x), DxirType(F32, listOf(ids.size))))
        }
        TestBackend.session().use { session ->
            val expected = session.runOn(fn, listOf(FloatArray(ids.size) { ids[it].toFloat() })).single()
            val staged = session.bufferFromHostI32(ids, listOf(ids.size))
            try {
                val outs = session.executeOn(fn, listOf(staged))
                try {
                    assertContentEquals(expected, outs.single().toFloatArray(ids.size))
                } finally {
                    outs.forEach { it.close() }
                }
                // Beyond 2^24, where runOn's float encoding refuses: the staged
                // buffer carries it, and the cast rounds as f32 does.
                val big = intArrayOf((1 shl 24) + 1)
                val bigFn = DxirBuilder.function("i32_staging_big") {
                    val x = param("ids", DxirType(I32, listOf(1)))
                    listOf(op(OpKind.CAST, listOf(x), DxirType(F32, listOf(1))))
                }
                session.bufferFromHostI32(big, listOf(1)).use { b ->
                    val out = session.executeOn(bigFn, listOf(b))
                    try {
                        assertContentEquals(floatArrayOf(big[0].toFloat()), out.single().toFloatArray(1))
                    } finally {
                        out.forEach { it.close() }
                    }
                }
            } finally {
                staged.close()
            }
        }
    }
}
