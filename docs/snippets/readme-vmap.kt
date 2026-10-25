import io.tlaloc.autograd.Batched
import io.tlaloc.autograd.batchAxis
import io.tlaloc.autograd.grad
import io.tlaloc.autograd.vmap2
import io.tlaloc.core.Batch
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.IndexName
import io.tlaloc.core.Named
import io.tlaloc.core.Rank2
import io.tlaloc.core.Rank3
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.ops.crossEntropyLoss
import io.tlaloc.core.ops.matmul
import io.tlaloc.core.ops.tanh
import io.tlaloc.core.ops.toFloat

object Feature : IndexName { override val name = "feature" }
object Hidden : IndexName { override val name = "hidden" }
object Class : IndexName { override val name = "class" }

typealias X = DTensor<Rank2<Sym, Named<Feature, Sym>>, F32>
typealias Y = DTensor<Rank2<Sym, Named<Class, Sym>>, F32>
typealias W1 = DTensor<Rank2<Named<Feature, Sym>, Named<Hidden, Sym>>, F32>
typealias W2 = DTensor<Rank2<Named<Hidden, Sym>, Named<Class, Sym>>, F32>

fun <S : Shape> tensor(data: FloatArray, vararg dims: Int) = DTensor<S, F32>(HostF32Storage(data), dims, F32)

fun main() {
    val w1: W1 = tensor(FloatArray(4 * 8) { (it % 7) / 7f - 0.5f }, 4, 8)
    val w2: W2 = tensor(FloatArray(8 * 3) { (it % 5) / 5f - 0.5f }, 8, 3)
    val xs = tensor<Rank3<Named<Batch, Sym>, Sym, Named<Feature, Sym>>>(FloatArray(16 * 4) { (it % 9) / 9f }, 16, 1, 4)
    val ys = tensor<Rank3<Named<Batch, Sym>, Sym, Named<Class, Sym>>>(FloatArray(16 * 3) { if (it % 3 == (it / 3) % 3) 1f else 0f }, 16, 1, 3)
    // snippet
    val typed: DTensor<Rank3<Named<Batch, Sym>, Named<Feature, Sym>, Named<Hidden, Sym>>, F32> = g1(xs, ys)
    check(typed.dims.contentEquals(intArrayOf(16, 4, 8))) { "dims ${typed.dims.toList()}" }
}
