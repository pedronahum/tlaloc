/**
 * Tlaloc Gaussian process — fitting kernel hyperparameters by gradient descent
 * on the log marginal likelihood.
 *
 * The model: noisy observations y of an unknown function at points x, with a
 * squared-exponential covariance
 *
 *     K(θ) = σf² · exp(−(xᵢ − xⱼ)² / (2ℓ²)) + σn² · I
 *
 * and the negative log marginal likelihood
 *
 *     nll(θ) = ½ · yᵀ K⁻¹ y + ½ · log det K + (n/2) · log 2π.
 *
 * θ = (log ℓ, log σf, log σn). The whole of `nll` — building K, solving
 * K·α = y through its Cholesky factor, and the log-determinant — is written
 * inside `grad3 { }`. The Tlaloc K2 plugin differentiates it at compile time:
 * the derivative of the solve and of the log-determinant come from Murray's
 * Cholesky rule and the triangular-solve rule, not from differentiating the
 * factorization's loops.
 *
 * NOTHING HERE IS TAKEN ON TRUST. Act [1] checks the compiled gradient against
 * a central finite difference of `nllReference` below, a transcription of the
 * same formula in plain Kotlin `Double` (its own Cholesky, no Tlaloc).
 */
import io.tlaloc.autograd.grad3
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.exp
import io.tlaloc.core.hostF32
import io.tlaloc.core.ops.exp
import io.tlaloc.core.ops.get
import io.tlaloc.core.ops.identityLike
import io.tlaloc.core.ops.logDetSpd
import io.tlaloc.core.ops.plus
import io.tlaloc.core.ops.solveSpd
import io.tlaloc.core.ops.sum
import io.tlaloc.core.ops.times
import io.tlaloc.core.ops.toFloat
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

typealias Matrix = DTensor<Rank2<Sym, Sym>, F32>
typealias Vector = DTensor<Rank1<Sym>, F32>

/** Number of observations. */
const val N = 24


// ---------------------------------------------------------------------------
// The data: y = sin(x) + noise of standard deviation 0.1, at 24 points on [0, 6].
// ---------------------------------------------------------------------------

val xs = DoubleArray(N) { 6.0 * it / (N - 1) }
val ys = Random(7).let { rng -> DoubleArray(N) { sin(xs[it]) + 0.1 * rng.nextGaussianish() } }

/** A standard normal draw from two uniforms (Box–Muller), so the data needs no library. */
fun Random.nextGaussianish(): Double {
    val u1 = nextDouble().coerceAtLeast(1e-12)
    val u2 = nextDouble()
    return sqrt(-2.0 * ln(u1)) * kotlin.math.cos(2 * PI * u2)
}

// ---------------------------------------------------------------------------
// The loss, differentiated by the compiler. `d` holds the squared distances
// (xᵢ − xⱼ)² and `y` the observations as an n×1 matrix; they are parameters
// because a `grad { }` body reads tensors only through its parameters.
// ---------------------------------------------------------------------------

val nllAndGrad = grad3 { theta: Vector, d: Matrix, y: Matrix ->
    val ell = theta[0]
    val sf = theta[1]
    val sn = theta[2]
    val k = (d * (-0.5f * (-2f * ell).exp())).exp() * (2f * sf).exp() +
        d.identityLike() * (2f * sn).exp()
    val alpha = k.solveSpd(y)
    // The constant (n/2)·log 2π is left out: it has no gradient.
    0.5f * (y * alpha).sum().toFloat() + 0.5f * k.logDetSpd().toFloat()
}

// ---------------------------------------------------------------------------
// The same loss in plain Double, for the finite-difference check and for
// reporting. Its own Cholesky; nothing from Tlaloc.
// ---------------------------------------------------------------------------

fun nllReference(theta: DoubleArray): Double {
    val ell = kotlin.math.exp(theta[0])
    val sf2 = kotlin.math.exp(2 * theta[1])
    val sn2 = kotlin.math.exp(2 * theta[2])
    val k = Array(N) { i ->
        DoubleArray(N) { j ->
            val r = xs[i] - xs[j]
            sf2 * kotlin.math.exp(-r * r / (2 * ell * ell)) + if (i == j) sn2 else 0.0
        }
    }
    val l = Array(N) { DoubleArray(N) }
    for (j in 0 until N) {
        var s = k[j][j]
        for (p in 0 until j) s -= l[j][p] * l[j][p]
        l[j][j] = sqrt(s)
        for (i in j + 1 until N) {
            var t = k[i][j]
            for (p in 0 until j) t -= l[i][p] * l[j][p]
            l[i][j] = t / l[j][j]
        }
    }
    // L·z = y, then yᵀK⁻¹y = zᵀz.
    val z = DoubleArray(N)
    for (i in 0 until N) {
        var s = ys[i]
        for (p in 0 until i) s -= l[i][p] * z[p]
        z[i] = s / l[i][i]
    }
    val quad = z.sumOf { it * it }
    val logDet = 2 * (0 until N).sumOf { ln(l[it][it]) }
    return 0.5 * quad + 0.5 * logDet + 0.5 * N * ln(2 * PI)
}

fun main() {
    val d = Tensors.f32Matrix<Sym, Sym>(N, N, FloatArray(N * N) { (xs[it / N] - xs[it % N]).let { r -> (r * r).toFloat() } })
    val y = Tensors.f32Matrix<Sym, Sym>(N, 1, FloatArray(N) { ys[it].toFloat() })
    fun gradAt(theta: DoubleArray): DoubleArray {
        val t = Tensors.f32Vector<Sym>(FloatArray(3) { theta[it].toFloat() })
        return nllAndGrad(t, d, y).first.hostF32().map { it.toDouble() }.toDoubleArray()
    }

    // [1] The compiled gradient against central differences of the Double reference.
    val theta0 = doubleArrayOf(ln(3.0), ln(0.5), ln(0.5))
    val g0 = gradAt(theta0)
    val h = 1e-5
    val fd = DoubleArray(3) { i ->
        val p = theta0.copyOf().also { it[i] += h }
        val m = theta0.copyOf().also { it[i] -= h }
        (nllReference(p) - nllReference(m)) / (2 * h)
    }
    println("[1] gradient at ℓ = 3, σf = 0.5, σn = 0.5")
    println("                 d/dlog ℓ    d/dlog σf    d/dlog σn")
    println("    compiled   %10.5f   %10.5f   %10.5f".format(g0[0], g0[1], g0[2]))
    println("    finite diff%10.5f   %10.5f   %10.5f".format(fd[0], fd[1], fd[2]))
    val worst = (0 until 3).maxOf { abs(g0[it] - fd[it]) } / fd.maxOf { abs(it) }
    println("    largest difference: %.2e of the largest entry".format(worst))
    // The compiled gradient runs in F32 through a 24×24 Cholesky factor.
    check(worst < 1e-3) { "the compiled gradient disagrees with finite differences" }

    // [2] Gradient descent (Adam) on θ.
    println()
    println("[2] Adam on θ = (log ℓ, log σf, log σn)")
    println("    step        ℓ       σf       σn        nll")
    val theta = theta0.copyOf()
    val m = DoubleArray(3)
    val v = DoubleArray(3)
    val lr = 0.05
    for (step in 0..300) {
        if (step % 50 == 0) {
            println(
                "    %4d   %6.3f   %6.3f   %6.3f   %8.4f".format(
                    step, kotlin.math.exp(theta[0]), kotlin.math.exp(theta[1]), kotlin.math.exp(theta[2]),
                    nllReference(theta),
                ),
            )
        }
        if (step == 300) break
        val g = gradAt(theta)
        for (i in 0 until 3) {
            m[i] = 0.9 * m[i] + 0.1 * g[i]
            v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i]
            val mh = m[i] / (1 - Math.pow(0.9, step + 1.0))
            val vh = v[i] / (1 - Math.pow(0.999, step + 1.0))
            theta[i] -= lr * mh / (sqrt(vh) + 1e-8)
        }
    }

    // [3] At the optimum the gradient vanishes.
    val gEnd = gradAt(theta)
    println()
    println("[3] fitted: ℓ = %.3f, σf = %.3f, σn = %.3f (the data's noise is 0.1)".format(
        kotlin.math.exp(theta[0]), kotlin.math.exp(theta[1]), kotlin.math.exp(theta[2]),
    ))
    println("    |gradient| at the end: %.2e".format(sqrt(gEnd.sumOf { it * it })))
    println("    nll: %.4f → %.4f".format(nllReference(theta0), nllReference(theta)))
}
