/**
 * Tlaloc readable gradients — the derivative, printed back to you as Kotlin.
 *
 * Every autodiff framework will give you a number. Tlaloc will give you the
 * PROGRAM: the Kotlin source of the gradient it derived, which you can read,
 * diff, review — and, as this example does, compile separately and run.
 *
 * One function is differentiated here, and the answer is obtained three
 * independent ways that must agree:
 *
 *   1. `grad { }` — rewritten into synthesized bytecode by the K2 plugin at
 *      COMPILE time. No tape, no graph object, no framework at runtime.
 *   2. The PRINTED gradient — the same derivation rendered as Kotlin source by
 *      the `dumpGradSourceDir` plugin flag (see build.gradle.kts), compiled by
 *      a source set that does NOT have the Tlaloc plugin on its classpath, and
 *      invoked here by reflection.
 *   3. A central finite difference over `reference()` below — a hand-written
 *      `kotlin.math` transcription of the same formula that never touches
 *      Tlaloc, so the check is a genuine third opinion.
 *
 * (1) and (2) are the same derivation and agree to the bit. (3) is numerical
 * and agrees to finite-difference tolerance.
 *
 * Background: docs/READABLE_REVERSE.md in the Tlaloc repo.
 */
import io.tlaloc.autograd.grad
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.Tensors
import io.tlaloc.core.exp
import io.tlaloc.core.hostF32
import io.tlaloc.core.log
import io.tlaloc.core.sigmoid
import io.tlaloc.core.sqrt
import java.io.File
import kotlin.math.abs

/**
 * The function under differentiation — a "damped swish":
 *
 *     swish(x)    = x · σ(x)                  (a smooth, non-monotonic gate)
 *     softplus(x) = log(1 + eˣ)               (a smooth, positive magnitude)
 *     f(x)        = swish(x) / √(1 + softplus(x))
 *
 * It is deliberately a composite nobody differentiates in their head: a
 * quotient whose numerator needs the product rule THROUGH a sigmoid, over a
 * denominator that chains sqrt ∘ log ∘ exp. Written out by hand, f'(x) is
 *
 *     [σ(x) + x·σ(x)(1 − σ(x))] / √(1 + s)  −  x·σ(x)·σ(x) / (2·(1 + s)^1.5)
 *
 * with s = softplus(x) — four rules and two shared subexpressions. Nobody
 * writes that. The dumped gradient at build/gradients/ is the machine's
 * version of it, and you can read it line by line.
 */
private fun reference(x: Double): Double {
    val swish = x * (1.0 / (1.0 + kotlin.math.exp(-x)))
    val softplus = kotlin.math.ln(1.0 + kotlin.math.exp(x))
    return swish / kotlin.math.sqrt(1.0 + softplus)
}

/** A central difference: the numerical opinion, computed in f64 for headroom. */
private fun centralDifference(x: Double, h: Double = 1e-4): Double =
    (reference(x + h) - reference(x - h)) / (2.0 * h)

fun main() {
    // ---------------------------------------------------------------- (1) ---
    // This `grad { }` call does not exist at runtime. The Tlaloc K2 plugin
    // lowers the lambda to DXIR, runs the reverse transform over it, and
    // synthesizes the result straight into this method's bytecode. `g` is an
    // ordinary Kotlin function; call it in a hot loop and nothing allocates a
    // tape, because there is no tape.
    val g = grad { x: Float ->
        (x * x.sigmoid()) / (1.0f + (1.0f + x.exp()).log()).sqrt()
    }

    println("Tlaloc readable gradients")
    println("  f(x) = x*sigmoid(x) / sqrt(1 + log(1 + exp(x)))")
    println()

    // ---------------------------------------------------------------- (2) ---
    // The SAME derivation, rendered as Kotlin by the compiler during this
    // project's build. Print it, then call the separately-compiled version.
    val dumped = dumpedGradientSource()
    println("--- the gradient, as the compiler printed it during the build ---")
    println("--- ${dumped.first.name} (generated; not in src/) ---")
    println(dumped.second.trimEnd())
    println()

    val printedGradient = loadPrintedGradient()

    // ---------------------------------------------------------- the check ---
    println("--- three answers that must agree ---")
    println(
        "%8s  %18s  %18s  %18s".format(
            "x", "compiled grad{}", "printed source", "central difference",
        ),
    )
    var maxNumericGap = 0.0
    for (x in listOf(-3.0f, -1.0f, -0.25f, 0.0f, 0.5f, 1.75f, 4.0f)) {
        val compiled = g(x)
        val printed = printedGradient(x)
        val numeric = centralDifference(x.toDouble())

        // (1) vs (2): the printed program IS the compiled program, so this is a
        // RAW-BIT equality — no epsilon. If this ever failed, the source you
        // were shown would not be the code that ran.
        check(compiled.toRawBits() == printed.toRawBits()) {
            "printed gradient disagrees with the compiled one at x=$x: " +
                "$compiled (0x${compiled.toRawBits().toString(16)}) vs " +
                "$printed (0x${printed.toRawBits().toString(16)})"
        }
        // (1) vs (3): numerical, so a tolerance — a central difference in f64
        // on a well-scaled function is good to roughly 1e-4 relative.
        val gap = abs(compiled.toDouble() - numeric)
        maxNumericGap = maxOf(maxNumericGap, gap)

        println("%8.2f  %18.8f  %18.8f  %18.8f".format(x, compiled, printed, numeric))
    }
    println()
    println("compiled == printed: raw-bit identical at every point")
    println("compiled vs finite difference: max |gap| = %.3e".format(maxNumericGap))
    check(maxNumericGap < 1e-4) { "finite-difference gap too large: $maxNumericGap" }
    println("readable-gradients OK")

    // ------------------------------------------------- the refusals ---------
    // Tlaloc's other half is what it will NOT compile. Two gradients that no
    // framework can honestly produce sit next door in `Refusals.kt.disabled`:
    // an axis error and a differentiability error. Rename that file to
    // `Refusals.kt`, rebuild, and the compiler names both — a red squiggle in
    // the IDE, not a stack trace an hour into training. README.md quotes the
    // exact messages.
}

// --------------------------------------------------------------------------
// Plumbing: finding and calling the printed gradient.
//
// This is reflection only because the printed file does not exist yet when
// THIS file is compiled — it is written during that same compilation, and a
// second source set (`printed`, see build.gradle.kts) compiles it afterwards.
// In a real project you would commit the dump and call it by name.
// --------------------------------------------------------------------------

/** The generated `.kt` file and its text. Named `<file>_<line>_<col>_grad.kt`. */
private fun dumpedGradientSource(): Pair<File, String> {
    val dir = File(System.getProperty("tlaloc.example.gradientSourceDir") ?: "build/gradients")
    val file = dir.listFiles { f: File -> f.name.endsWith(".kt") }?.singleOrNull()
        ?: error(
            "expected exactly one dumped gradient under $dir — is the " +
                "dumpGradSourceDir plugin flag still set in build.gradle.kts?",
        )
    return file to file.readText()
}

/**
 * Loads the class the `printed` source set compiled from that file and returns
 * it as a `(Float) -> Float`. The printed signature is over `:core` host twins
 * (`DTensor<ScalarShape, F32>`), so we box on the way in and unbox on the way
 * out — the same host tensors any Tlaloc user holds.
 */
@Suppress("UNCHECKED_CAST") // reflection erases the phantom shape parameter
private fun loadPrintedGradient(): (Float) -> Float {
    val dir = File(
        System.getProperty("tlaloc.example.printedClassesDir") ?: "build/classes/kotlin/printed",
    )
    val className = dir.listFiles { f: File -> f.name.endsWith("Kt.class") }
        ?.singleOrNull()?.name?.removeSuffix(".class")
        ?: error("expected exactly one compiled printed gradient class under $dir")
    val method = Class.forName(className).declaredMethods
        .single { it.parameterCount == 1 && DTensor::class.java.isAssignableFrom(it.parameterTypes[0]) }
    return { x ->
        val out = method.invoke(null, Tensors.f32Scalar(x)) as DTensor<*, F32>
        out.hostF32()[0]
    }
}
