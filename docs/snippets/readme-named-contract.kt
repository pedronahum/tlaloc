import io.tlaloc.core.Batch
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.Hidden
import io.tlaloc.core.Named
import io.tlaloc.core.Rank2
import io.tlaloc.core.SeqLen
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.ops.contract

fun main() {
    // snippet
    check(hidden.dims.contentEquals(intArrayOf(2, 4))) { "dims ${hidden.dims.toList()}" }
}
