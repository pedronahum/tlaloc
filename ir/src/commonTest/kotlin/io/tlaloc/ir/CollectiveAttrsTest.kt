package io.tlaloc.ir

import io.tlaloc.core.F32
import io.tlaloc.ir.passes.DxirForwardTransform
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirInterpreterF64
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.passes.GradShardingVerify
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class CollectiveAttrsTest {

    private val groups = listOf(listOf(1, 0, 2, 3))

    private fun allToAll(split: Int, concat: Int, inDims: List<Int>, outDims: List<Int>) = DxirBuilder.function("a2a") {
        val x = param("x", DxirType(F32, inDims))
        val y = op(
            OpKind.ALL_TO_ALL, listOf(x), DxirType(F32, outDims),
            attrs = mapOf("replica_groups" to groups, "split_dimension" to split, "concat_dimension" to concat),
        )
        listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(y, y), y.type)), DxirType(F32, emptyList())))
    }

    @Test
    fun allToAllTakesFourCopiesOfReplicaZerosChunk() {
        // x [4, 2]: split rows into 4 chunks of 1, concat along columns. Replica 0 is second in
        // its group, so every sender sends it row 1.
        val fn = DxirBuilder.function("a2a") {
            val x = param("x", DxirType(F32, listOf(4, 2)))
            listOf(op(OpKind.ALL_TO_ALL, listOf(x), DxirType(F32, listOf(1, 8)), attrs = mapOf("replica_groups" to groups, "split_dimension" to 0, "concat_dimension" to 1)))
        }
        val x = FloatArray(8) { it.toFloat() }
        assertContentEquals(floatArrayOf(2f, 3f, 2f, 3f, 2f, 3f, 2f, 3f), DxirInterpreter.evalFunction(fn, listOf(x))[0])
        assertContentEquals(
            doubleArrayOf(2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0),
            DxirInterpreterF64.evalFunction(fn, listOf(DoubleArray(8) { it.toDouble() }))[0],
        )
    }

    @Test
    fun allToAllOnOneAxisRepeatsTheChunk() {
        val fn = DxirBuilder.function("a2a") {
            val x = param("x", DxirType(F32, listOf(8)))
            listOf(op(OpKind.ALL_TO_ALL, listOf(x), DxirType(F32, listOf(8)), attrs = mapOf("replica_groups" to groups, "split_dimension" to 0, "concat_dimension" to 0)))
        }
        val x = FloatArray(8) { it.toFloat() }
        assertContentEquals(floatArrayOf(2f, 3f, 2f, 3f, 2f, 3f, 2f, 3f), DxirInterpreter.evalFunction(fn, listOf(x))[0])
    }

    @Test
    fun aPermuteDeliversToTargetsAndZerosElsewhere() {
        fun permute(pairs: List<List<Int>>) = DxirBuilder.function("cp") {
            val x = param("x", DxirType(F32, listOf(3)))
            listOf(op(OpKind.COLLECTIVE_PERMUTE, listOf(x), x.type, attrs = mapOf("source_target_pairs" to pairs)))
        }
        val x = floatArrayOf(1f, 2f, 3f)
        assertContentEquals(x, DxirInterpreter.evalFunction(permute(listOf(listOf(3, 0), listOf(0, 1))), listOf(x))[0])
        assertContentEquals(FloatArray(3), DxirInterpreter.evalFunction(permute(listOf(listOf(0, 1), listOf(1, 2))), listOf(x))[0])
    }

    @Test
    fun malformedCollectivesAreRefusedByName() {
        assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(allToAll(0, 1, listOf(6, 2), listOf(1, 8)), listOf(FloatArray(12)))
        }.also { assertTrue("not a multiple of the group size" in it.message!!, it.message) }
        val twice = DxirBuilder.function("cp") {
            val x = param("x", DxirType(F32, listOf(3)))
            listOf(op(OpKind.COLLECTIVE_PERMUTE, listOf(x), x.type, attrs = mapOf("source_target_pairs" to listOf(listOf(0, 1), listOf(0, 2)))))
        }
        assertFailsWith<IllegalArgumentException> { DxirInterpreter.evalFunction(twice, listOf(FloatArray(3))) }
            .also { assertTrue("once" in it.message!!, it.message) }
    }

    @Test
    fun theAdjointsAreTheDualCollectives() {
        val a2a = DxirReverseTransform.apply(allToAll(0, 1, listOf(4, 2), listOf(1, 8)))
        // The gradient recomputes the forward all-to-all; the dual is the one back to x's shape.
        val dual = a2a.body.filterIsInstance<DxirOp>().single { it.op == OpKind.ALL_TO_ALL && it.type.dims == listOf(4, 2) }
        assertEquals(1, dual.attrs["split_dimension"])
        assertEquals(0, dual.attrs["concat_dimension"])
        assertEquals(listOf(4, 2), dual.type.dims)

        val cp = DxirBuilder.function("cp") {
            val x = param("x", DxirType(F32, listOf(3)))
            val y = op(OpKind.COLLECTIVE_PERMUTE, listOf(x), x.type, attrs = mapOf("source_target_pairs" to listOf(listOf(0, 1), listOf(1, 2), listOf(2, 0))))
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(y, y), y.type)), DxirType(F32, emptyList())))
        }
        val back = DxirReverseTransform.apply(cp).body.filterIsInstance<DxirOp>()
            .single { it.op == OpKind.COLLECTIVE_PERMUTE && it.operands[0] !is DxirParam }
        assertEquals(listOf(listOf(1, 0), listOf(2, 1), listOf(0, 2)), back.attrs["source_target_pairs"])

        for ((kind, dualKind) in listOf(OpKind.ALL_GATHER to OpKind.REDUCE_SCATTER, OpKind.REDUCE_SCATTER to OpKind.ALL_GATHER)) {
            val fn = DxirBuilder.function("gs") {
                val inDims = if (kind == OpKind.ALL_GATHER) listOf(2, 3) else listOf(8, 3)
                val outDims = if (kind == OpKind.ALL_GATHER) listOf(8, 3) else listOf(2, 3)
                val x = param("x", DxirType(F32, inDims))
                val y = op(kind, listOf(x), DxirType(F32, outDims), attrs = mapOf("replica_groups" to groups, "dimension" to 0))
                listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(y, y), y.type)), DxirType(F32, emptyList())))
            }
            val g = DxirReverseTransform.apply(fn)
            val d = g.body.filterIsInstance<DxirOp>().single { it.op == dualKind }
            assertEquals(fn.params[0].type, d.type)
            assertEquals(dualKind, GradShardingVerify.adjointOf(kind))
        }
        assertEquals(OpKind.ALL_TO_ALL, GradShardingVerify.adjointOf(OpKind.ALL_TO_ALL))
        assertEquals(OpKind.COLLECTIVE_PERMUTE, GradShardingVerify.adjointOf(OpKind.COLLECTIVE_PERMUTE))
    }

    @Test
    fun theTangentIsTheSameCollective() {
        val jvp = DxirForwardTransform.apply(allToAll(0, 1, listOf(4, 2), listOf(1, 8)))
        assertEquals(2, jvp.body.count { it is DxirOp && it.op == OpKind.ALL_TO_ALL })
    }
}
