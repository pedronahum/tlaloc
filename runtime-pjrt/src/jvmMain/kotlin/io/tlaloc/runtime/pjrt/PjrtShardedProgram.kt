package io.tlaloc.runtime.pjrt

import io.tlaloc.core.DType
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirParam
import io.tlaloc.runtime.pjrt.ffm.OpShardingLayout
import io.tlaloc.runtime.pjrt.ffm.PjrtBuffer
import io.tlaloc.runtime.pjrt.ffm.PjrtLoadedExecutable

/**
 * An SPMD program compiled for a session's devices ([PjrtSession.shardedProgram]) that runs on
 * values held on the devices: [stage] puts an input there, cut by the sharding XLA compiled for
 * its parameter, and [execute] returns each result as it was computed, a piece per device. A
 * result fed back as the next call's input (a parameter, an optimizer's state) never leaves the
 * devices.
 */
class PjrtShardedProgram internal constructor(
    private val session: PjrtSession,
    val function: DxirFunction,
    private val exec: PjrtLoadedExecutable,
    private val inputLayouts: List<OpShardingLayout>,
    private val outputLayouts: List<OpShardingLayout>,
    private val stageOn: (device: Int, param: DxirParam, values: FloatArray, dims: List<Int>) -> PjrtBuffer,
) {
    /** The devices the program runs on. */
    val deviceCount: Int get() = inputLayouts.firstOrNull()?.deviceCount ?: outputLayouts.first().deviceCount

    /** The dims of one device's piece of each parameter. */
    val inputShardDims: List<List<Int>> get() = function.params.mapIndexed { i, p -> inputLayouts[i].shardDims(p.type.dims) }

    /** The dims of one device's piece of each result. */
    val outputShardDims: List<List<Int>> get() = function.returns.mapIndexed { i, r -> outputLayouts[i].shardDims(r.type.dims) }

    /** [value], the whole row-major array of parameter [index] (I32 values float-encoded, as
     * [PjrtSession.runOn] takes them), staged on the devices as the program expects it. */
    fun stage(index: Int, value: FloatArray): PjrtShardedBuffer = session.asInFlightCall {
        val p = function.params[index]
        require(value.size.toLong() == p.type.elementCount) {
            "PjrtShardedProgram.stage: param '${p.name}' expects ${p.type.elementCount} values, got ${value.size}"
        }
        val layout = inputLayouts[index]
        val dims = layout.shardDims(p.type.dims)
        val pieces = ArrayList<PjrtBuffer>(deviceCount)
        try {
            for (d in 0 until deviceCount) {
                val piece = if (layout.isReplicated) value else layout.slice(value, p.type.dims, d)
                pieces += stageOn(d, p, piece, dims)
            }
        } catch (t: Throwable) {
            pieces.forEach { it.close() }
            throw t
        }
        PjrtShardedBuffer(pieces, p.type.dims, p.type.dtype, layout)
    }

    /**
     * Runs the program on [inputs], one per parameter, each staged by [stage] or returned by an
     * earlier [execute] for a result laid out as that parameter is. Waits for every device.
     * An input donated to a result (the program's `outputAliases`) is consumed: close it and do
     * not pass it again. The caller closes the returned buffers.
     */
    fun execute(inputs: List<PjrtShardedBuffer>): List<PjrtShardedBuffer> = session.asInFlightCall {
        require(inputs.size == function.params.size) {
            "PjrtShardedProgram.execute: ${function.name} takes ${function.params.size} inputs, got ${inputs.size}"
        }
        for ((i, p) in function.params.withIndex()) {
            val x = inputs[i]
            require(x.dims == p.type.dims && x.dtype == p.type.dtype && x.layout.sameAs(inputLayouts[i])) {
                "PjrtShardedProgram.execute: input $i ('${p.name}') is ${x.dtype}${x.dims} laid out as ${x.layout.tileCounts} " +
                    "pieces; the program expects ${p.type} in ${inputLayouts[i].tileCounts}"
            }
        }
        val outs = exec.executeOnDevices(List(deviceCount) { d -> inputs.map { it.pieces[d] } })
        function.returns.mapIndexed { r, ret ->
            PjrtShardedBuffer(List(deviceCount) { d -> outs[d][r] }, ret.type.dims, ret.type.dtype, outputLayouts[r])
        }
    }
}

/**
 * One value of a [PjrtShardedProgram] on the devices: a buffer per device holding that device's
 * piece of the array of [dims] (the whole array where it is replicated). [toFloatArray] copies it
 * back to the host.
 */
class PjrtShardedBuffer internal constructor(
    /** One buffer per device, in the program's device order. */
    val pieces: List<PjrtBuffer>,
    val dims: List<Int>,
    val dtype: DType,
    internal val layout: OpShardingLayout,
) : AutoCloseable {

    /** The dims of one device's piece. */
    val shardDims: List<Int> get() = layout.shardDims(dims)

    /** The whole row-major array (I32 values float-encoded). A replicated value is read from the first device. */
    fun toFloatArray(): FloatArray {
        val n = shardDims.fold(1) { a, b -> a * b }
        fun read(b: PjrtBuffer) = if (dtype == I32) b.toIntArray(n).let { v -> FloatArray(n) { v[it].toFloat() } } else b.toFloatArray(n)
        return if (layout.isReplicated) read(pieces[0]) else layout.assemble(pieces.map(::read), dims)
    }

    override fun close() = pieces.forEach { it.close() }
}
