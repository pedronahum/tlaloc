import io.tlaloc.core.Bounded
import io.tlaloc.core.DTensor
import io.tlaloc.core.DimBound
import io.tlaloc.core.F32
import io.tlaloc.core.Hidden
import io.tlaloc.core.Named
import io.tlaloc.core.Rank2
import io.tlaloc.core.SeqLen
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors

// snippet

fun main() {
    val t: Tokens = Tensors.f32Zeros(300, 4)
    check(t.dims.contentEquals(intArrayOf(300, 4)))
}
