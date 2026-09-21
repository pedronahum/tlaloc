/**
 * The task: a 2-D decision boundary with a KNOWN ground truth.
 *
 * Points live in the square [-1, 1]². A point is labelled +1 when it falls
 * inside a disc and -1 when it falls outside. The disc is deliberately
 * off-centre so the answer cannot be produced by any single linear cut — a
 * Dense-only model with no activation could not fit this, which is what makes
 * the two ReLU layers in the model worth their keep.
 *
 * Everything here is drawn from Tlaloc's threefry counter-based PRNG
 * ([io.tlaloc.core.uniformFloats]), which is bit-exact against JAX. A
 * `RandomKey` is a VALUE, not a mutable generator: the same key always yields
 * the same stream, and `split` derives independent child streams from a
 * parent. So the training data, the held-out data and the initial weights are
 * pure functions of one seed — identical on every run and every machine, with
 * no global generator state to get out of step.
 *
 * (The TRAINING TRAJECTORY reproduces bit-for-bit on the host lane, which is
 * deterministic. On the GPU lane it does not: XLA autotunes its GEMM kernels
 * at compile time, so two runs of the same program can pick different kernels
 * and land a few parts in 10⁻³ apart. The example prints both lanes' numbers
 * so you can see the size of that.)
 */
import io.tlaloc.core.RandomKey
import io.tlaloc.core.split
import io.tlaloc.core.uniformFloats
import kotlin.math.sqrt

/** The ground truth nobody tells the model about. */
object Truth {
    const val CX = 0.15f
    const val CY = -0.10f
    const val R = 0.70f

    /** +1 inside the disc, -1 outside. */
    fun label(x: Float, y: Float): Float {
        val dx = x - CX
        val dy = y - CY
        return if (sqrt(dx * dx + dy * dy) <= R) 1f else -1f
    }

    override fun toString() = "inside the disc centred (%.2f, %.2f) of radius %.2f".format(CX, CY, R)
}

/**
 * A batch of [n] uniform points in [-1, 1]² with their true labels.
 *
 * @param xs row-major [n, 2] — the model input.
 * @param ys row-major [n, 1] — the target the loss compares against.
 */
class Batch(val n: Int, val xs: FloatArray, val ys: FloatArray) {
    companion object {
        fun draw(key: RandomKey, n: Int): Batch {
            // One stream of 2n uniforms in [0,1), stretched to [-1,1).
            val u = uniformFloats(key, n * 2)
            val xs = FloatArray(u.size) { i -> 2f * u[i] - 1f }
            val ys = FloatArray(n) { i -> Truth.label(xs[2 * i], xs[2 * i + 1]) }
            return Batch(n, xs, ys)
        }
    }
}

/** The three independent streams this example uses, derived from one seed. */
class Streams(seed: Long) {
    private val children = RandomKey.fromSeed(seed).split(3)

    /** The training set. */
    val train: RandomKey = children[0]

    /** A held-out set the model never sees during training. */
    val heldOut: RandomKey = children[1]

    /** The model's weight initialisation. */
    val init: RandomKey = children[2]
}
