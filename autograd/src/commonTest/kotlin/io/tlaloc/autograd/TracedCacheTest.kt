package io.tlaloc.autograd

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.Shape
import io.tlaloc.ir.passes.DxirInterpreter
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith

class TracedCacheTest {
    private val cache = FloatArray(8) { it.toFloat() } // [4, 1, 1, 2]: rows (0,1) (2,3) (4,5) (6,7)
    private val rows = floatArrayOf(10f, 11f, 20f, 21f, 30f, 31f) // [3, 1, 2]
    private val want = floatArrayOf(0f, 1f, 30f, 31f, 4f, 5f, 10f, 11f) // row 0 to slot 3, row 1 padding, row 2 to slot 1

    private fun inputs(slots: IntArray) = listOf<DTensor<*, *>>(
        DTensor<Shape, F32>(HostF32Storage(cache.copyOf()), intArrayOf(4, 1, 1, 2), F32),
        DTensor<Shape, F32>(HostF32Storage(rows.copyOf()), intArrayOf(3, 1, 2), F32),
        DTensor<Shape, I32>(HostI32Storage(slots), intArrayOf(3), I32),
    )

    @Test
    fun writesRowsAtTheirSlotsAndSkipsNegativeOnes() {
        var host = FloatArray(0)
        val fn = captureMulti(inputs(intArrayOf(3, -1, 1)), "kv_write") { (c, r, s) ->
            listOf(c.kvCacheWrite(r, s).also { host = it.entry.value.copyOf() })
        }
        assertContentEquals(want, host, "the traced value")
        val out = DxirInterpreter.evalFunction(fn, listOf(cache, rows, floatArrayOf(3f, -1f, 1f))).single()
        assertContentEquals(want, out, "the interpreter")
    }

    @Test
    fun refusesASlotWrittenTwice() {
        assertFailsWith<IllegalArgumentException> {
            captureMulti(inputs(intArrayOf(2, 2, 0)), "kv_write_twice") { (c, r, s) -> listOf(c.kvCacheWrite(r, s)) }
        }
    }
}
