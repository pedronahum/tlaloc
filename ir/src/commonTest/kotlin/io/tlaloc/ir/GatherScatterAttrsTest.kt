package io.tlaloc.ir

import io.tlaloc.core.F32
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirInterpreterF64
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * ALL_GATHER and REDUCE_SCATTER under the interpreters' single-process semantics: replica 0's
 * group of two holds two copies of the value (all-gather), or twice its own chunk
 * (reduce-scatter); the chunk follows replica 0's place in the group. Shapes that do not match
 * the group are refused by name.
 */
class GatherScatterAttrsTest {

    private fun fn(kind: OpKind, inDims: List<Int>, outDims: List<Int>, groups: List<List<Int>>, d: Int) = DxirBuilder.function("c") {
        val x = param("x", DxirType(F32, inDims))
        listOf(op(kind, listOf(x), DxirType(F32, outDims), attrs = mapOf<String, Any>("replica_groups" to groups, "dimension" to d)))
    }

    @Test
    fun aGroupOfTwoGathersTwoCopiesAndScattersTwiceTheChunk() {
        val x = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)  // [2, 3]
        val g = DxirInterpreter.evalFunction(fn(OpKind.ALL_GATHER, listOf(2, 3), listOf(2, 6), listOf(listOf(0, 1)), 1), listOf(x))[0]
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 1f, 2f, 3f, 4f, 5f, 6f, 4f, 5f, 6f), g)
        val g0 = DxirInterpreter.evalFunction(fn(OpKind.ALL_GATHER, listOf(2, 3), listOf(4, 3), listOf(listOf(0, 1)), 0), listOf(x))[0]
        assertContentEquals(x + x, g0)
        val y = floatArrayOf(1f, 2f, 3f, 4f)  // [4]
        val first = DxirInterpreter.evalFunction(fn(OpKind.REDUCE_SCATTER, listOf(4), listOf(2), listOf(listOf(0, 1)), 0), listOf(y))[0]
        assertContentEquals(floatArrayOf(2f, 4f), first)
        val second = DxirInterpreter.evalFunction(fn(OpKind.REDUCE_SCATTER, listOf(4), listOf(2), listOf(listOf(1, 0)), 0), listOf(y))[0]
        assertContentEquals(floatArrayOf(6f, 8f), second, "replica 0 second in its group takes the second chunk")
        val f64 = DxirInterpreterF64.evalFunction(fn(OpKind.REDUCE_SCATTER, listOf(4), listOf(2), listOf(listOf(1, 0)), 0), listOf(doubleArrayOf(1.0, 2.0, 3.0, 4.0)))[0]
        assertContentEquals(doubleArrayOf(6.0, 8.0), f64)
    }

    @Test
    fun shapesThatDoNotMatchTheGroupAreRefused() {
        val e = assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(fn(OpKind.ALL_GATHER, listOf(2, 3), listOf(2, 3), listOf(listOf(0, 1)), 1), listOf(FloatArray(6)))
        }
        assertTrue("ALL_GATHER" in e.message!!, e.message)
        assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(fn(OpKind.REDUCE_SCATTER, listOf(3), listOf(1), listOf(listOf(0, 1)), 0), listOf(FloatArray(3)))
        }
    }
}
