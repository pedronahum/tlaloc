package io.tlaloc.nn

import io.tlaloc.ir.DxirAxisRef
import io.tlaloc.ir.DxirDimSharding
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirMesh
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.DxirSharding
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInline

/**
 * How a training step lays its inputs and parameters out over [mesh]: a sharding for each data
 * input (the model inputs, then the targets) and for each parameter, by key; null keeps a value
 * whole on every device. [CapturedStep.sharded] writes them on the step's parameters, and Shardy's
 * propagation shards everything else and places the collectives when the program is compiled for
 * the mesh's devices.
 */
class ShardingPlan(
    val mesh: DxirMesh,
    /** The sharding of data input [index] (inputs, then targets) of [type]. */
    val input: (index: Int, type: DxirType) -> DxirSharding?,
    /** The sharding of the parameter [key] of [type]. */
    val parameter: (key: String, type: DxirType) -> DxirSharding?,
) {
    companion object {
        /** [type]'s dimension [dim] on [axes] of [mesh], the others whole. */
        fun on(mesh: DxirMesh, type: DxirType, dim: Int, vararg axes: String): DxirSharding = DxirSharding(
            mesh.name,
            type.dims.indices.map { d -> DxirDimSharding(if (d == dim) axes.map { DxirAxisRef.Full(it) } else emptyList()) },
        )

        private fun DxirMesh.size(axis: String): Int =
            requireNotNull(axis(axis)) { "ShardingPlan: mesh '$name' has no axis '$axis' (axes ${axes.map { it.name }})" }.size

        private fun batch(mesh: DxirMesh, axis: String): (Int, DxirType) -> DxirSharding? = { _, type ->
            if (type.dims.isNotEmpty()) on(mesh, type, 0, axis) else null
        }

        /**
         * Data parallel: every data input split along its first (batch) dimension over [axis],
         * the parameters whole on every device. The parameter gradients are summed over [axis].
         */
        fun dataParallel(mesh: DxirMesh, axis: String = "data"): ShardingPlan {
            mesh.size(axis)
            return ShardingPlan(mesh, batch(mesh, axis)) { _, _ -> null }
        }

        /**
         * Fully sharded data parallel: the data inputs as in [dataParallel], and each parameter
         * split over [axis] along its largest dimension that divides by the axis size (one that
         * none divides stays whole). The parameters, their gradients and an optimizer's state
         * live in pieces; propagation gathers a parameter where it is used and reduce-scatters
         * its gradient.
         */
        fun fsdp(mesh: DxirMesh, axis: String = "data"): ShardingPlan {
            val n = mesh.size(axis)
            return ShardingPlan(mesh, batch(mesh, axis)) { _, type ->
                type.dims.withIndex().filter { (_, d) -> d % n == 0 && d >= n }.maxByOrNull { it.value }?.let { on(mesh, type, it.index, axis) }
            }
        }

        /**
         * Tensor parallel for [CausalLM] and its [TransformerBlock]s (Megatron's split): the query,
         * key, value, gate and up projections split by output column over [modelAxis], the
         * attention output and down projections by input row, the output head by vocabulary
         * column; embeddings and norms whole. The heads, and the MLP's hidden width, must divide
         * by the axis size. With [dataAxis] the data inputs are also split over it, as in
         * [dataParallel], on a two-axis mesh.
         */
        fun tensorParallel(mesh: DxirMesh, modelAxis: String = "model", dataAxis: String? = null): ShardingPlan {
            mesh.size(modelAxis)
            dataAxis?.let { mesh.size(it) }
            val columns = setOf("attn.q.w", "attn.k.w", "attn.v.w", "mlp.gate.w", "mlp.up.w", "mlp.fc1.w")
            val columnBiases = setOf("attn.q.b", "attn.k.b", "attn.v.b", "mlp.gate.b", "mlp.up.b", "mlp.fc1.b")
            val rows = setOf("attn.o.w", "mlp.down.w", "mlp.fc2.w")
            return ShardingPlan(mesh, dataAxis?.let { batch(mesh, it) } ?: { _, _ -> null }) { key, type ->
                when {
                    key == "head.w" -> on(mesh, type, 1, modelAxis)
                    columns.any { key.endsWith(".$it") } -> on(mesh, type, 1, modelAxis)
                    columnBiases.any { key.endsWith(".$it") } -> on(mesh, type, 0, modelAxis)
                    rows.any { key.endsWith(".$it") } -> on(mesh, type, 0, modelAxis)
                    else -> null
                }
            }
        }
    }
}

/**
 * This step with its data inputs and parameters sharded by [plan]: the same functions, with
 * [plan]'s mesh declared and its shardings on the parameters, and each parameter gradient
 * constrained to its parameter's sharding. Compile it for the mesh's devices
 * (`CompileOptions.spmd`) and run it with `PjrtSession.runSharded`, which splits the whole arrays
 * and puts the results back together; in the interpreter it computes what the step computes.
 * [withDeviceAdamW], [withDeviceSGD] and [withDeviceMuon] give the optimizer state its
 * parameter's sharding. Apply [withGradientAccumulation] and [withRemat] before this.
 */
fun CapturedStep.sharded(plan: ShardingPlan): CapturedStep {
    val keys = parameterKeys + frozenKeys
    require(primal.params.size == inputCount + keys.size) {
        "sharded: the step takes ${primal.params.size} values; expected $inputCount inputs and ${keys.size} parameters"
    }
    fun shardingOf(i: Int, type: DxirType): DxirSharding? = (if (i < inputCount) plan.input(i, type) else plan.parameter(keys[i - inputCount], type))?.also {
        require(it.meshName == plan.mesh.name) { "sharded: the sharding of ${describe(i)} names mesh '${it.meshName}', not '${plan.mesh.name}'" }
        require(it.dimShardings.size == type.dims.size) { "sharded: the sharding of ${describe(i)} has ${it.dimShardings.size} dimensions for $type" }
        for ((d, s) in it.dimShardings.withIndex()) {
            val pieces = s.axes.fold(1) { a, ax -> a * (plan.mesh.axis(ax.name)?.size ?: error("sharded: mesh '${plan.mesh.name}' has no axis '${ax.name}'")) }
            require(type.dims[d] % pieces == 0) { "sharded: dimension $d of ${describe(i)} ($type) does not split into $pieces pieces" }
        }
    }
    fun reparam(fn: DxirFunction): List<DxirParam> = fn.params.mapIndexed { i, p -> DxirParam(p.id, p.name, p.type, shardingOf(i, p.type)) }
    fun meshes(fn: DxirFunction) = fn.meshes.filter { it.name != plan.mesh.name } + plan.mesh

    val primalParams = reparam(primal)
    val newPrimal = DxirFunction(primal.name, primalParams, primal.body, primal.returns, meshes(primal))

    val gradParams = reparam(gradient)
    var next = DxirInline.maxId(gradient) + 1
    val body = ArrayList<DxirNode>(gradient.body)
    val returns = gradient.returns.mapIndexed { r, x ->
        val j = r - 1 - inputCount
        if (j in parameterKeys.indices) constrained(x, gradParams[inputCount + j].sharding, { next++ }, body) else x
    }
    val newGradient = DxirFunction(gradient.name, gradParams, body, returns, meshes(gradient))
    return CapturedStep(newPrimal, newGradient, parameterKeys, inputCount, frozenKeys, microbatches)
}

private fun CapturedStep.describe(i: Int): String =
    if (i < inputCount) "input $i" else "parameter '${(parameterKeys + frozenKeys)[i - inputCount]}'"

/** [x] under a [OpKind.SHARD_CONSTRAINT] to [sharding], appended to [body]; [x] itself when [sharding] is null. */
internal fun constrained(x: DxirNode, sharding: DxirSharding?, id: () -> Int, body: MutableList<DxirNode>): DxirNode =
    if (sharding == null) x
    else DxirOp(id(), OpKind.SHARD_CONSTRAINT, listOf(x), types = listOf(x.type), sharding = sharding).also { body += it }
