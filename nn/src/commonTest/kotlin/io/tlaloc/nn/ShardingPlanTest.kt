package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.ir.DxirAxisRef
import io.tlaloc.ir.DxirDimSharding
import io.tlaloc.ir.DxirMesh
import io.tlaloc.ir.DxirMeshAxis
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirSharding
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * [ShardingPlan] and [CapturedStep.sharded]: where each preset puts the parameters, what the
 * sharded functions carry, and that they compute what the step computes. Running them on several
 * devices is `PjrtShardedTrainingCpuTest` in `:runtime-pjrt`.
 */
class ShardingPlanTest {

    private val config = CausalLmConfig(vocabSize = 16, dModel = 16, numLayers = 1, numHeads = 4, numKvHeads = 2, ffHidden = 32, initStd = 0.1f)
    private val start = CausalLM.llama(config, RandomKey.fromSeed(2))
    private val ids = DTensor<Shape, I32>(HostI32Storage(IntArray(8) { (5 * it) % 16 }), intArrayOf(2, 4), I32)
    private val targets = oneHot(IntArray(8) { (5 * it + 1) % 16 }, 16, intArrayOf(2, 4))
    private val step = capture(start, listOf(ids), targets = listOf(targets)) { logits, t -> crossEntropy(logits, t[0]) }

    private val data = DxirMesh("mesh", listOf(DxirMeshAxis("data", 2)))
    private val model = DxirMesh("mesh", listOf(DxirMeshAxis("model", 2)))

    private fun on(vararg axes: String?): List<DxirDimSharding> = axes.map { a -> DxirDimSharding(listOfNotNull(a).map { DxirAxisRef.Full(it) }) }

    private fun shardingOf(s: CapturedStep, key: String): DxirSharding? = s.gradient.params[s.inputCount + s.parameterKeys.indexOf(key)].sharding

    @Test
    fun dataParallelSplitsTheBatchAndKeepsTheParametersWhole() {
        val s = step.sharded(ShardingPlan.dataParallel(data))
        assertEquals(on("data", null), s.gradient.params[0].sharding!!.dimShardings)
        assertEquals(on("data", null, null), s.gradient.params[1].sharding!!.dimShardings)
        assertTrue(s.parameterKeys.all { shardingOf(s, it) == null })
        assertEquals(listOf(data), s.gradient.meshes)
        assertEquals(listOf(data), s.primal.meshes)
    }

    @Test
    fun fsdpSplitsEachParameterAlongItsLargestDividingDimension() {
        val s = step.sharded(ShardingPlan.fsdp(data))
        assertEquals(on("data", null), shardingOf(s, "embed.table")!!.dimShardings) // [16, 16]: the first of two equal
        assertEquals(on(null, "data"), shardingOf(s, "blocks.0.mlp.up.w")!!.dimShardings) // [16, 32]
        assertEquals(on("data", null), shardingOf(s, "blocks.0.mlp.down.w")!!.dimShardings) // [32, 16]
        assertEquals(on("data"), shardingOf(s, "norm.weight")!!.dimShardings)
        // Each parameter gradient is constrained to its parameter's sharding.
        for ((j, key) in s.parameterKeys.withIndex()) {
            val g = s.gradient.returns[1 + s.inputCount + j] as DxirOp
            assertEquals(OpKind.SHARD_CONSTRAINT, g.op, key)
            assertEquals(shardingOf(s, key), g.sharding, key)
        }
    }

    @Test
    fun tensorParallelSplitsTheProjectionsByColumnAndRow() {
        val s = step.sharded(ShardingPlan.tensorParallel(model))
        for (p in listOf("q", "k", "v")) assertEquals(on(null, "model"), shardingOf(s, "blocks.0.attn.$p.w")!!.dimShardings, p)
        assertEquals(on("model", null), shardingOf(s, "blocks.0.attn.o.w")!!.dimShardings)
        assertEquals(on(null, "model"), shardingOf(s, "blocks.0.mlp.gate.w")!!.dimShardings)
        assertEquals(on(null, "model"), shardingOf(s, "blocks.0.mlp.up.w")!!.dimShardings)
        assertEquals(on("model", null), shardingOf(s, "blocks.0.mlp.down.w")!!.dimShardings)
        assertEquals(on(null, "model"), shardingOf(s, "head.w")!!.dimShardings)
        assertNull(shardingOf(s, "embed.table"))
        assertNull(shardingOf(s, "norm.weight"))
        assertNull(s.gradient.params[0].sharding)
    }

    @Test
    fun theShardedStepComputesWhatTheStepComputes() {
        val inputs = listOf(FloatArray(8) { ((5 * it) % 16).toFloat() }, targets.hostF32()) + step.parameterTensors(start).map { it.hostF32() }
        val want = DxirInterpreter.evalFunction(step.gradient, inputs)
        for (plan in listOf(ShardingPlan.dataParallel(data), ShardingPlan.fsdp(data), ShardingPlan.tensorParallel(model))) {
            val got = DxirInterpreter.evalFunction(step.sharded(plan).gradient, inputs)
            for (i in want.indices) assertContentEquals(want[i], got[i])
        }
    }

    @Test
    fun deviceOptimizerStateIsShardedLikeItsParameter() {
        val s = step.sharded(ShardingPlan.fsdp(data))
        for (device in listOf(s.withDeviceAdamW(AdamW()), s.withDeviceSGD(SGD(momentum = 0.9f)))) {
            for ((k, slot) in device.stateSlots.withIndex()) {
                assertEquals(shardingOf(s, s.parameterKeys[slot.parameterIndex]), device.function.params[device.stateOffset + k].sharding, slot.name)
            }
            // The updated parameters come back sharded like the parameters.
            for (j in s.parameterKeys.indices) {
                assertEquals(shardingOf(s, s.parameterKeys[j]), (device.function.returns[1 + j] as DxirOp).sharding)
            }
        }
    }

    @Test
    fun refusesShardingsThatDoNotFit() {
        val three = DxirMesh("mesh", listOf(DxirMeshAxis("data", 3)))
        assertFailsWith<IllegalArgumentException> { step.sharded(ShardingPlan.dataParallel(three)) } // batch 2
        assertFailsWith<IllegalArgumentException> { ShardingPlan.tensorParallel(data) } // no "model" axis
        val elsewhere = ShardingPlan(data, { _, _ -> null }) { _, t -> DxirSharding("other", t.dims.map { DxirDimSharding(emptyList()) }) }
        assertFailsWith<IllegalArgumentException> { step.sharded(elsewhere) }
    }
}
