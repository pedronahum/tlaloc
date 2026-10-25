import io.tlaloc.autograd.constant
import io.tlaloc.autograd.mean
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.times
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.split
import io.tlaloc.nn.Adam
import io.tlaloc.nn.Dense
import io.tlaloc.nn.ReluLayer
import io.tlaloc.nn.Sequential
import io.tlaloc.nn.capture
import io.tlaloc.nn.saveCheckpoint
import io.tlaloc.nn.step
import java.nio.file.Files

fun main() {
    val n = 32
    val xs = FloatArray(2 * n) { i -> ((i * 37) % 17) / 8.5f - 1f }
    val targets = FloatArray(n) { i -> if (xs[2 * i] * xs[2 * i + 1] >= 0f) 1f else -1f }
    val x = Tensors.f32Matrix<Sym, Sym>(n, 2, xs)
    val path = Files.createTempDirectory("tlaloc-doc-").resolve("mlp.safetensors")
    // snippet
    val before = step.run(model0, listOf(x)).loss
    val after = step.run(model, listOf(x)).loss
    check(after < 0.5f * before) { "the loop must train: $before -> $after" }
    check(Files.size(path) > 0)
}
