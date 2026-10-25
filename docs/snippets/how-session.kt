// compile-only: needs a GPU and the PJRT CUDA plugin
import io.tlaloc.autograd.mean
import io.tlaloc.autograd.times
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.split
import io.tlaloc.nn.Dense
import io.tlaloc.nn.ReluLayer
import io.tlaloc.nn.Sequential
import io.tlaloc.nn.capture
import io.tlaloc.runtime.pjrt.PjrtSession
import io.tlaloc.runtime.pjrt.PjrtTarget

fun main() {
    val keys = RandomKey.fromSeed(1).split(2)
    val model = Sequential(Dense(2, 8, keys[0]), ReluLayer, Dense(8, 1, keys[1]))
    val x = Tensors.f32Matrix<Sym, Sym>(4, 2, FloatArray(8) { it / 8f })
    val step = capture(model, listOf(x)) { prediction -> (prediction * prediction).mean() }
    // snippet
}
