package io.tlaloc.autograd

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32

class Tracer<S : Shape> internal constructor(
    internal val tape: Tape,
    internal val entry: TapeEntry,
) {
    val id: Int get() = entry.id
    val dims: IntArray get() = entry.dims
    val rank: Int get() = entry.dims.size
    val size: Int get() = entry.size

    fun toDTensor(): DTensor<S, F32> =
        DTensor(HostF32Storage(entry.value.copyOf()), dims.copyOf(), F32)
}

internal fun <S : Shape> Tape.traceLeaf(value: DTensor<S, F32>): Tracer<S> {
    val entry = leaf(dims = value.dims.copyOf(), value = value.hostF32().copyOf())
    return Tracer(this, entry)
}

internal fun sameTape(a: Tracer<*>, b: Tracer<*>): Tape {
    require(a.tape === b.tape) { "operands come from different tapes" }
    return a.tape
}
