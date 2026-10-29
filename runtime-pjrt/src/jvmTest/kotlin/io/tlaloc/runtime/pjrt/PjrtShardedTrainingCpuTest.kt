package io.tlaloc.runtime.pjrt

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirMesh
import io.tlaloc.ir.DxirMeshAxis
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInline
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.nn.CapturedStep
import io.tlaloc.nn.CausalLM
import io.tlaloc.nn.CausalLmConfig
import io.tlaloc.nn.ShardingPlan
import io.tlaloc.nn.capture
import io.tlaloc.nn.crossEntropy
import io.tlaloc.nn.oneHot
import io.tlaloc.nn.sharded
import io.tlaloc.nn.withDeviceAdamW
import io.tlaloc.testing.Tolerances
import kotlin.test.Test

/**
 * A [CausalLM] training step sharded by a [ShardingPlan] and partitioned by Shardy, on four host
 * devices of XLA's CPU plugin (`scripts/pjrt-cpu-plugin/build.sh`), or on four TPU chips
 * ([MultiDeviceBackend]). Skips when neither is there.
 */
class PjrtShardedTrainingCpuTest {

    private val n = 4
    private val config = CausalLmConfig(vocabSize = 64, dModel = 64, numLayers = 2, numHeads = 8, numKvHeads = 4, ffHidden = 128, initStd = 0.05f)
    private val rows = 8
    private val seq = 16

    private val data = DxirMesh("mesh", listOf(DxirMeshAxis("data", n)))
    private val model = DxirMesh("mesh", listOf(DxirMeshAxis("model", n)))
    private val grid = DxirMesh("mesh", listOf(DxirMeshAxis("data", 2), DxirMeshAxis("model", 2)))

    private val plans = listOf(
        "data parallel" to ShardingPlan.dataParallel(data),
        "fsdp" to ShardingPlan.fsdp(data),
        "tensor parallel" to ShardingPlan.tensorParallel(model),
        "data x tensor parallel" to ShardingPlan.tensorParallel(grid, dataAxis = "data"),
    )

    private fun <T> session(block: (PjrtSession) -> T): T = MultiDeviceBackend.session(n, block = block)

    /** Token ids `[rows, seq]` (as floats) and their one-hot next tokens. */
    private fun batch(k: Int): List<FloatArray> {
        val ids = IntArray(rows * seq)
        val next = IntArray(rows * seq)
        for (r in 0 until rows) {
            var t = (r * 5 + k) % config.vocabSize
            for (s in 0 until seq) {
                ids[r * seq + s] = t
                t = (3 * t + 1 + r) % config.vocabSize
                next[r * seq + s] = t
            }
        }
        return listOf(FloatArray(ids.size) { ids[it].toFloat() }, oneHot(next, config.vocabSize, intArrayOf(rows, seq)).hostF32())
    }

    private val start = CausalLM.llama(config, RandomKey.fromSeed(3))

    private val step: CapturedStep by lazy {
        val ids = DTensor<Shape, I32>(HostI32Storage(IntArray(rows * seq)), intArrayOf(rows, seq), I32)
        val targets = oneHot(IntArray(rows * seq), config.vocabSize, intArrayOf(rows, seq))
        capture(start, listOf(ids), targets = listOf(targets)) { logits, t -> crossEntropy(logits, t[0]) }
    }

    private val parameters: List<FloatArray> by lazy { step.parameterTensors(start).map { it.hostF32() } }

    @Test
    fun eachPlansGradientEqualsTheUnshardedStep() = session { s ->
        val inputs = batch(0) + parameters
        val want = DxirInterpreter.evalFunction(step.gradient, inputs)
        for ((name, plan) in plans) {
            val got = s.runSharded(step.sharded(plan).gradient, inputs)
            Tolerances.assertClose(got[0], want[0], 1e-5, "$name loss")
            for (j in step.parameterKeys.indices) {
                val r = 1 + step.inputCount + j
                Tolerances.assertClose(got[r], want[r], 1e-4, "$name ${step.parameterKeys[j]}")
            }
        }
    }

    @Test
    fun eachPlanSplitsTheParametersAsNamed() = session { s ->
        val keys = step.parameterKeys
        for ((name, plan) in plans) {
            val (ins, outs) = s.shardDims(step.sharded(plan).gradient)
            val rowsPerDevice = when (name) {
                "data parallel", "fsdp" -> rows / n
                "data x tensor parallel" -> rows / 2
                else -> rows
            }
            kotlin.test.assertEquals(listOf(rowsPerDevice, seq), ins[0], "$name: the token ids")
            for ((j, key) in keys.withIndex()) {
                val dims = step.gradient.params[step.inputCount + j].type.dims
                val want = plan.parameter(key, step.gradient.params[step.inputCount + j].type)?.dimShardings?.mapIndexed { d, ds ->
                    dims[d] / ds.axes.fold(1) { a, ax -> a * plan.mesh.axis(ax.name)!!.size }
                } ?: dims
                kotlin.test.assertEquals(want, ins[step.inputCount + j], "$name: $key on one device")
                kotlin.test.assertEquals(want, outs[1 + step.inputCount + j], "$name: the gradient of $key on one device")
            }
            val held = keys.indices.sumOf { j -> ins[step.inputCount + j].fold(1L) { a, b -> a * b } }
            val whole = keys.indices.sumOf { j -> step.gradient.params[step.inputCount + j].type.elementCount }
            println("PjrtShardedTrainingCpuTest: $name holds ${"%.0f".format(100.0 * held / whole)}% of the parameters on each device")
        }
        // FSDP splits every matrix, the norms (64 wide) too.
        val (fsdpIns, _) = s.shardDims(step.sharded(ShardingPlan.fsdp(data)).gradient)
        kotlin.test.assertTrue(keys.indices.all { j -> fsdpIns[step.inputCount + j].fold(1) { a, b -> a * b } * n == step.gradient.params[step.inputCount + j].type.elementCount.toInt() })
    }

    @Test
    fun adamWStepsUnderEachPlanTrainLikeTheUnshardedStep() = session { s ->
        val optimizer = io.tlaloc.nn.AdamW(learningRate = 1e-2f)
        val batches = (0 until 3).map { batch(it) }
        fun loop(device: io.tlaloc.nn.DeviceOptimizerStep, run: (List<FloatArray>) -> List<FloatArray>): Pair<List<Float>, List<FloatArray>> {
            var params = parameters
            var state = device.initialState()
            val losses = (0 until 6).map { t ->
                val outs = run(batches[t % batches.size] + params + state + device.scalars(t + 1).map { floatArrayOf(it) })
                params = outs.subList(1, 1 + params.size)
                state = outs.subList(1 + params.size, outs.size)
                outs[0][0]
            }
            return losses to params
        }
        val plain = step.withDeviceAdamW(optimizer)
        val (wantLosses, wantParams) = loop(plain) { DxirInterpreter.evalFunction(plain.function, it) }
        for ((name, plan) in plans) {
            val device = step.sharded(plan).withDeviceAdamW(optimizer)
            // The moments are held like their parameters.
            val (ins, _) = s.shardDims(device.function)
            for ((k, slot) in device.stateSlots.withIndex()) {
                kotlin.test.assertEquals(ins[step.inputCount + slot.parameterIndex], ins[device.stateOffset + k], "$name: ${slot.name} of ${step.parameterKeys[slot.parameterIndex]}")
            }
            val (losses, params) = loop(device) { s.runSharded(device.function, it) }
            println("PjrtShardedTrainingCpuTest: $name AdamW losses $losses")
            Tolerances.assertClose(losses.toFloatArray(), wantLosses.toFloatArray(), 1e-4, "$name losses")
            for (j in params.indices) Tolerances.assertClose(params[j], wantParams[j], 1e-3, "$name ${step.parameterKeys[j]} after 6 steps")
        }
        kotlin.test.assertTrue(wantLosses.takeLast(3).sum() < wantLosses.take(3).sum(), "did not train: $wantLosses")
    }

    /**
     * The same six AdamW steps with the parameters and moments staged once and kept on the devices
     * ([PjrtSession.shardedProgram]), each donated to its update: a step stages its batch and four
     * scalars and reads back the loss. The losses and the parameters read at the end equal the
     * unsharded step's, and each parameter stays in the pieces its plan names.
     */
    @Test
    fun adamWStepsStayOnTheDevices() = session { s ->
        val optimizer = io.tlaloc.nn.AdamW(learningRate = 1e-2f)
        val batches = (0 until 3).map { batch(it) }
        val plain = step.withDeviceAdamW(optimizer)
        var wantParams = parameters
        var wantState = plain.initialState()
        val wantLosses = (0 until 6).map { t ->
            val outs = DxirInterpreter.evalFunction(plain.function, batches[t % 3] + wantParams + wantState + plain.scalars(t + 1).map { floatArrayOf(it) })
            wantParams = outs.subList(1, 1 + wantParams.size)
            wantState = outs.subList(1 + wantParams.size, outs.size)
            outs[0][0]
        }
        for ((name, plan) in plans) {
            val device = step.sharded(plan).withDeviceAdamW(optimizer)
            val program = s.shardedProgram(device.function, device.outputAliases)
            val nIn = step.inputCount
            var params = parameters.mapIndexed { j, v -> program.stage(nIn + j, v) }
            var state = device.initialState().mapIndexed { k, v -> program.stage(device.stateOffset + k, v) }
            val losses = (0 until 6).map { t ->
                val data = batches[t % 3].mapIndexed { i, v -> program.stage(i, v) }
                val scalars = device.scalars(t + 1).mapIndexed { i, v -> program.stage(device.scalarOffset + i, floatArrayOf(v)) }
                val outs = program.execute(data + params + state + scalars)
                (data + params + state + scalars).forEach { it.close() }   // the parameters and moments were donated
                params = outs.subList(1, 1 + params.size)
                state = outs.subList(1 + params.size, outs.size)
                outs[0].use { it.toFloatArray()[0] }
            }
            println("PjrtShardedTrainingCpuTest: $name AdamW losses on the devices $losses")
            Tolerances.assertClose(losses.toFloatArray(), wantLosses.toFloatArray(), 1e-4, "$name losses")
            for ((j, key) in step.parameterKeys.withIndex()) {
                val type = step.gradient.params[nIn + j].type
                kotlin.test.assertEquals(program.inputShardDims[nIn + j], params[j].shardDims, "$name: $key on one device after 6 steps")
                val split = plan.parameter(key, type)?.dimShardings?.any { it.axes.isNotEmpty() } == true
                kotlin.test.assertEquals(split, params[j].shardDims != type.dims, "$name: $key split as the plan says")
                Tolerances.assertClose(params[j].toFloatArray(), wantParams[j], 1e-3, "$name $key after 6 steps")
            }
            (params + state).forEach { it.close() }
        }
    }

    /**
     * One device's step for [plan] written by hand with explicit collectives, as a framework
     * without a partitioner runs it: [local] (the step at a quarter of the batch) on this device's
     * rows, each parameter that [plan] splits all-gathered before it and its gradient
     * reduce-scattered after, the others' gradients all-reduced; loss and gradients are means over
     * the devices. Takes the local inputs and each parameter's piece; returns the loss and each
     * parameter's gradient piece.
     */
    private fun explicit(local: CapturedStep, plan: ShardingPlan): DxirFunction {
        val all = listOf((0 until n).toList())
        var next = DxirInline.maxId(local.gradient) + 1
        val id = { next++ }
        val body = ArrayList<DxirNode>()
        val nIn = local.inputCount
        val dims = local.parameterKeys.mapIndexed { j, key ->
            val type = local.gradient.params[nIn + j].type
            plan.parameter(key, type)?.dimShardings?.indexOfFirst { it.axes.isNotEmpty() }?.takeIf { it >= 0 }
        }
        val inputs = (0 until nIn).map { local.gradient.params[it].let { p -> DxirParam(id(), p.name, p.type) } }
        val pieces = local.parameterKeys.indices.map { j ->
            val p = local.gradient.params[nIn + j]
            val d = dims[j]
            DxirParam(id(), p.name, if (d == null) p.type else DxirType(p.type.dtype, p.type.dims.mapIndexed { k, x -> if (k == d) x / n else x }))
        }
        fun emit(kind: OpKind, x: DxirNode, type: DxirType, attrs: Map<String, Any>) = DxirOp(id(), kind, listOf(x), attrs, type).also { body += it }
        val whole = pieces.mapIndexed { j, piece ->
            dims[j]?.let { emit(OpKind.ALL_GATHER, piece, local.gradient.params[nIn + j].type, mapOf("replica_groups" to all, "dimension" to it)) } ?: piece
        }
        val outs = DxirInline.inline(local.gradient, inputs + whole, id, body)
        fun mean(x: DxirNode): DxirNode {
            val c = DxirConst(id(), n.toFloat(), DxirType(F32, emptyList())).also { body += it }
            val cn = if (x.type.dims.isEmpty()) c else emit(OpKind.BROADCAST, c, x.type, mapOf("broadcast_dimensions" to emptyList<Int>()))
            return DxirOp(id(), OpKind.DIV, listOf(x, cn), type = x.type).also { body += it }
        }
        val loss = mean(emit(OpKind.ALL_REDUCE, outs[0], outs[0].type, mapOf("replica_groups" to all)))
        val grads = local.parameterKeys.indices.map { j ->
            val g = outs[1 + nIn + j]
            mean(dims[j]?.let { emit(OpKind.REDUCE_SCATTER, g, pieces[j].type, mapOf("replica_groups" to all, "dimension" to it)) }
                ?: emit(OpKind.ALL_REDUCE, g, g.type, mapOf("replica_groups" to all)))
        }
        return DxirFunction("${local.gradient.name}_explicit", inputs + pieces, body, listOf(loss) + grads)
    }

    /** Piece [k] of [n] of the row-major [whole] of [dims] along dimension [d]. */
    private fun piece(whole: FloatArray, dims: List<Int>, d: Int, k: Int): FloatArray {
        val outer = dims.take(d).fold(1) { a, b -> a * b }
        val inner = dims.drop(d + 1).fold(1) { a, b -> a * b }
        val len = dims[d] / n
        val out = FloatArray(outer * len * inner)
        for (o in 0 until outer) System.arraycopy(whole, (o * dims[d] + k * len) * inner, out, o * len * inner, len * inner)
        return out
    }

    @Test
    fun shardyPlacedCollectivesMatchHandWrittenOnes() = session { s ->
        val localIds = DTensor<Shape, I32>(HostI32Storage(IntArray(rows / n * seq)), intArrayOf(rows / n, seq), I32)
        val localTargets = oneHot(IntArray(rows / n * seq), config.vocabSize, intArrayOf(rows / n, seq))
        val local = capture(start, listOf(localIds), targets = listOf(localTargets)) { logits, t -> crossEntropy(logits, t[0]) }
        val whole = batch(0)
        val inputs = whole + parameters
        // Device k's rows of the ids [rows, seq] and the targets [rows, seq, vocab].
        val localInputs = listOf(listOf(rows, seq), listOf(rows, seq, config.vocabSize)).mapIndexed { i, d -> List(n) { k -> piece(whole[i], d, 0, k) } }
        for ((name, plan) in listOf("data parallel" to ShardingPlan.dataParallel(data), "fsdp" to ShardingPlan.fsdp(data))) {
            val shardy = step.sharded(plan).gradient
            val want = s.runSharded(shardy, inputs)
            val hand = explicit(local, plan)
            val paramPieces = step.parameterKeys.mapIndexed { j, key ->
                val type = step.gradient.params[step.inputCount + j].type
                val d = plan.parameter(key, type)?.dimShardings?.indexOfFirst { it.axes.isNotEmpty() }
                List(n) { k -> if (d == null) parameters[j] else piece(parameters[j], type.dims, d, k) }
            }
            val got = s.runPerDevice(hand, localInputs + paramPieces)
            for (k in 0 until n) Tolerances.assertClose(got[0][k], want[0], 1e-5, "$name loss on device $k")
            for ((j, key) in step.parameterKeys.withIndex()) {
                val type = step.gradient.params[step.inputCount + j].type
                val d = plan.parameter(key, type)?.dimShardings?.indexOfFirst { it.axes.isNotEmpty() }
                val g = want[1 + step.inputCount + j]
                for (k in 0 until n) Tolerances.assertClose(got[1 + j][k], if (d == null) g else piece(g, type.dims, d, k), 1e-4, "$name $key on device $k")
            }
            // The same results on both sides: the loss and the parameter gradients.
            val trimmed = DxirFunction(shardy.name, shardy.params, shardy.body, listOf(shardy.returns[0]) + shardy.returns.drop(1 + step.inputCount), shardy.meshes)
            // Host devices share the host's cores: medians move severalfold from round to round, so
            // print their range; the comparison that counts is on accelerators.
            val rounds = List(5) { s.medianDeviceMillisSharded(trimmed, inputs, 21) to s.medianDeviceMillisPerDevice(hand, localInputs + paramPieces, 21) }
            fun range(xs: List<Double>) = "%.1f-%.1f".format(xs.min(), xs.max())
            println("PjrtShardedTrainingCpuTest: $name step ${range(rounds.map { it.first })} ms partitioned by Shardy, ${range(rounds.map { it.second })} ms with explicit collectives (medians of 5 rounds)")
        }
    }
}
