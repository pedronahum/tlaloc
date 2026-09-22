/**
 * Tlaloc differentiable physics — gradient descent THROUGH a simulator.
 *
 * Nothing here is a neural network. The thing being differentiated is a
 * ballistics simulation: a `for` loop that integrates Newton's laws one
 * timestep at a time, with air drag, exactly as you would write it if you had
 * never heard of automatic differentiation.
 *
 *     for (i in 0 until STEPS) {
 *         vx = vx - DRAG * vx * DT              // drag
 *         vy = vy - (G + DRAG * vy) * DT        // gravity + drag
 *         x  = x + vx * DT
 *         y  = y + vy * DT
 *     }
 *
 * The question it answers is one a person actually has: **how do I throw this
 * ball so it goes through the hoop?** Without drag you could solve that on
 * paper. With drag in the loop there is no closed form — so we differentiate
 * the loop itself, and walk downhill.
 *
 * WHY THIS IS NOT ORDINARY
 *
 * That loop lives INSIDE `grad2 { }`. At COMPILE time the Tlaloc K2 plugin
 * lowers the lambda to its own IR, applies φ-calculus coarsening to the loop
 * (the OOPSLA 2021 method: a constant trip count unrolls into straight-line
 * code that symbolic differentiation can close over), reverse-transforms the
 * result, and synthesizes the gradient directly into this method's bytecode.
 *
 * At RUNTIME there is no tape, no graph object, and nothing recording what the
 * loop did. `lossAndGrad` is an ordinary Kotlin function of two floats.
 *
 * Act [4] prints the derivative the compiler wrote — as Kotlin you can read.
 *
 * NOTHING HERE IS TAKEN ON TRUST. Act [1] checks the compiled gradient against
 * a central finite difference over `simulate` below — a hand transcription of
 * the same physics in plain `kotlin.math` that never touches Tlaloc. If the
 * compiler's derivative were wrong, the example refuses to print a result.
 */
import io.tlaloc.autograd.grad2
import io.tlaloc.autograd.valueAndGrad2
import io.tlaloc.core.cos
import io.tlaloc.core.sin
import java.io.File
import kotlin.math.abs
import kotlin.math.roundToInt
import kotlin.math.sqrt

// ---------------------------------------------------------------------------
// The world. Real units: metres, seconds.
// ---------------------------------------------------------------------------

/** Gravity, m/s². */
const val G = 9.81f

/** Linear drag coefficient, 1/s. It is what removes the closed-form answer. */
const val DRAG = 0.24f

/** Integrator timestep, s. */
const val DT = 0.025f

/** How many steps we simulate — the ball must be at the hoop at step STEPS. */
const val STEPS = 38

/** Release point: a player's hands, 2 m up. */
const val X0 = 0.0f
const val Y0 = 2.0f

/** The hoop: 4.6 m away (a free throw), 3.05 m up (a regulation rim). */
const val HOOP_X = 4.6f
const val HOOP_Y = 3.05f

/** How close counts as in: the ball must arrive within the rim's radius. */
const val RIM_RADIUS = 0.23f

/**
 * The unknowns: the launch ANGLE (radians) and SPEED (m/s). An honest first
 * guess — too flat and too soft. It falls well short, as act [3] draws.
 */
const val GUESS_ANGLE = 0.62f   // ~35.5 degrees
const val GUESS_SPEED = 6.4f

const val DESCENT_STEPS = 240

/**
 * Two learning rates, because the two unknowns have different units: a radian
 * and a metre-per-second are not comparable step sizes. This is the only
 * hand-tuning in the example.
 */
const val LR_ANGLE = 0.02f
const val LR_SPEED = 0.10f

fun main() {
    header()

    // ------------------------------------------------------------------ 1 ---
    // THE GRADIENT. This `grad2 { }` call does not exist at runtime: the
    // simulator loop inside it was differentiated during compilation.
    //
    // The lambda returns the squared miss distance — how far the ball is from
    // the hoop after 38 timesteps — so the two partial derivatives it hands
    // back are d(miss²)/d(vx₀) and d(miss²)/d(vy₀).
    //
    // NOTE: every number in this lambda is one of the `const val`s declared at
    // the top of the file — including `STEPS`, the loop's trip count. Until
    // §0.4.500 that was impossible: the body is lowered to Tlaloc IR at compile
    // time, and ANY reference out of the lambda's own scope was refused by name.
    // A capture the compiler can resolve to a constant is now folded into the
    // lowered IR as exactly the constant an inline literal would have produced,
    // so the simulator below reads like the `simulate()` transcription further
    // down instead of being a column of unnamed numbers.
    //
    // What is still refused, by name, is a captured RUNTIME value — a `var`, a
    // computed `val`, a parameter of the enclosing function. See the README
    // section "A limitation you will meet immediately".
    val dMiss = grad2 { angle: Float, speed: Float ->
        var x = X0
        var y = Y0
        var vx = speed * angle.cos()
        var vy = speed * angle.sin()
        for (i in 0 until STEPS) {
            vx = vx - DRAG * vx * DT                  // drag
            vy = vy - (G + DRAG * vy) * DT            // gravity + drag
            x = x + vx * DT
            y = y + vy * DT
        }
        (x - HOOP_X) * (x - HOOP_X) + (y - HOOP_Y) * (y - HOOP_Y)   // squared miss
    }

    // The same derivation, returning the loss as well, so the descent loop can
    // report where it is without simulating twice.
    val lossAndGrad = valueAndGrad2 { angle: Float, speed: Float ->
        var x = X0
        var y = Y0
        var vx = speed * angle.cos()
        var vy = speed * angle.sin()
        for (i in 0 until STEPS) {
            vx = vx - DRAG * vx * DT
            vy = vy - (G + DRAG * vy) * DT
            x = x + vx * DT
            y = y + vy * DT
        }
        (x - HOOP_X) * (x - HOOP_X) + (y - HOOP_Y) * (y - HOOP_Y)
    }

    // ------------------------------------------------------------------ 2 ---
    // A THIRD OPINION, before we trust it with anything.
    println("[1] is the compiler's derivative the real derivative?")
    println()
    println("    %8s %8s   %14s %14s   %14s %14s".format(
        "angle", "speed", "d/dangle", "finite diff", "d/dspeed", "finite diff"))
    var worstRelative = 0.0f
    for ((angle, speed) in listOf(
        0.62f to 6.4f, 0.85f to 8.3f, 1.05f to 6.5f, 1.20f to 9.0f,
    )) {
        val (dAngle, dSpeed) = dMiss(angle, speed)
        val fdAngle = centralDifference(angle, speed, wrtAngle = true)
        val fdSpeed = centralDifference(angle, speed, wrtAngle = false)
        worstRelative = maxOf(worstRelative, relativeGap(dAngle, fdAngle), relativeGap(dSpeed, fdSpeed))
        println("    %8.3f %8.3f   %14.6f %14.6f   %14.6f %14.6f"
            .format(angle, speed, dAngle, fdAngle, dSpeed, fdSpeed))
    }
    println()
    println("    worst relative disagreement: %.3e".format(worstRelative))
    check(worstRelative < 2e-2f) { "compiled gradient disagrees with finite differences" }
    println("    -> the compiler differentiated 38 timesteps of Newtonian motion, correctly.")
    println()

    // ------------------------------------------------------------------ 3 ---
    // THE DESCENT. An ordinary gradient-descent loop over two numbers.
    var angle = GUESS_ANGLE
    var speed = GUESS_SPEED
    val firstLoss = lossAndGrad(angle, speed).first

    println("[2] the throw, learned")
    println()
    println("    first guess: %s — misses by %.2f m".format(describe(GUESS_ANGLE, GUESS_SPEED), sqrt(firstLoss)))
    println()
    println("    %5s  %9s  %8s  %11s".format("step", "angle", "speed", "miss (m)"))

    var loss = firstLoss
    for (step in 0..DESCENT_STEPS) {
        val (l, dAngle, dSpeed) = lossAndGrad(angle, speed)
        loss = l
        if (step % (DESCENT_STEPS / 8) == 0) {
            println("    %5d  %8.2f°  %8.3f  %11.4f".format(step, degrees(angle), speed, sqrt(l)))
        }
        if (step == DESCENT_STEPS) break
        angle -= LR_ANGLE * dAngle
        speed -= LR_SPEED * dSpeed
    }
    println()

    // ------------------------------------------------------------------ 4 ---
    // DID IT GO IN? Judged by the plain-Kotlin simulator, not by the loss.
    val miss = sqrt(loss)
    val (endX, endY) = simulate(angle, speed).last()
    println("    learned throw: %s".format(describe(angle, speed)))
    println("    ball at step %d: (%.3f, %.3f)      hoop: (%.2f, %.2f)"
        .format(STEPS, endX, endY, HOOP_X, HOOP_Y))
    println("    %.4f m from the centre of the rim (rim radius %.2f m)".format(miss, RIM_RADIUS))
    println()
    check(miss < RIM_RADIUS) { "gradient descent did not sink the shot: miss = $miss m" }
    println("    SWISH.")
    println()

    // ------------------------------------------------------------------ 5 ---
    plotTrajectories(simulate(GUESS_ANGLE, GUESS_SPEED), simulate(angle, speed))

    // ------------------------------------------------------------------ 6 ---
    printDerivedGradient()

    println()
    println("differentiable-physics OK")
}

// ---------------------------------------------------------------------------
// The independent transcription. Plain kotlin.math; no Tlaloc anywhere. Used
// for the finite-difference check in act [1] and for the plot in act [3].
// ---------------------------------------------------------------------------

/** The whole trajectory as (x, y) per step, starting at the release point. */
private fun simulate(angle: Float, speed: Float): List<Pair<Float, Float>> {
    var x = X0
    var y = Y0
    var vx = speed * kotlin.math.cos(angle)
    var vy = speed * kotlin.math.sin(angle)
    val path = mutableListOf(x to y)
    for (i in 0 until STEPS) {
        vx -= DRAG * vx * DT
        vy -= (G + DRAG * vy) * DT
        x += vx * DT
        y += vy * DT
        path += x to y
    }
    return path
}

/** The same squared miss the `grad2` lambda computes, in f64 for FD headroom. */
private fun missSquared(angle: Float, speed: Float): Double {
    val (x, y) = simulate(angle, speed).last()
    val dx = (x - HOOP_X).toDouble()
    val dy = (y - HOOP_Y).toDouble()
    return dx * dx + dy * dy
}

private fun centralDifference(angle: Float, speed: Float, wrtAngle: Boolean): Float {
    val h = 1e-3f
    val up = if (wrtAngle) missSquared(angle + h, speed) else missSquared(angle, speed + h)
    val down = if (wrtAngle) missSquared(angle - h, speed) else missSquared(angle, speed - h)
    return ((up - down) / (2.0 * h)).toFloat()
}

private fun relativeGap(a: Float, b: Float): Float = abs(a - b) / maxOf(1e-3f, abs(b))

private fun degrees(radians: Float): Double = radians.toDouble() * 180.0 / Math.PI

private fun describe(angle: Float, speed: Float): String =
    "%.2f° at %.3f m/s".format(degrees(angle), speed)

// ---------------------------------------------------------------------------
// Presentation.
// ---------------------------------------------------------------------------

private fun header() {
    println("Tlaloc differentiable physics — a free throw, solved by differentiating the simulator")
    println()
    println("    release  (%.1f, %.1f) m         hoop  (%.2f, %.2f) m".format(X0, Y0, HOOP_X, HOOP_Y))
    println("    gravity  %.2f m/s²             linear drag  %.2f /s".format(G, DRAG))
    println("    %d Euler steps of %.3f s   =   %.2f s of flight".format(STEPS, DT, STEPS * DT))
    println()
    println("    The unknowns are the launch angle and speed. With drag inside the loop there")
    println("    is no formula to invert, so we differentiate the loop and walk downhill.")
    println()
}

/** Two trajectories over the court, in characters. */
private fun plotTrajectories(first: List<Pair<Float, Float>>, learned: List<Pair<Float, Float>>) {
    val cols = 58
    val rows = 15
    val xMax = 5.2f
    val yMax = (first + learned).maxOf { it.second }.coerceAtLeast(HOOP_Y) * 1.08f

    val grid = Array(rows) { CharArray(cols) { ' ' } }
    fun plot(path: List<Pair<Float, Float>>, ch: Char) {
        for ((x, y) in path) {
            val c = ((x / xMax) * (cols - 1)).roundToInt()
            val r = rows - 1 - ((y / yMax) * (rows - 1)).roundToInt()
            if (c in 0 until cols && r in 0 until rows) grid[r][c] = ch
        }
    }
    plot(first, '.')
    plot(learned, 'o')

    val rimC = ((HOOP_X / xMax) * (cols - 1)).roundToInt()
    val rimR = rows - 1 - ((HOOP_Y / yMax) * (rows - 1)).roundToInt()
    if (rimC in 0 until cols && rimR in 0 until rows) {
        grid[rimR][rimC] = '#'
        if (rimC + 1 < cols) grid[rimR][rimC + 1] = '#'
    }

    println("[3] the first guess ('.'), what the gradient found ('o'), the rim ('#')")
    println()
    for (r in grid) println("    |" + String(r))
    println("    +" + "-".repeat(cols))
    println("     0 m" + " ".repeat(cols - 10) + "%.1f m".format(xMax))
    println()
}

/**
 * Print the gradient the compiler derived — the actual file it wrote while
 * compiling this example, into `build/gradients/`.
 */
private fun printDerivedGradient() {
    println("[4] the derivative of the simulator, as the compiler wrote it")
    println()
    val dir = System.getProperty("tlaloc.example.gradientSourceDir")?.let(::File)
    if (dir == null || !dir.isDirectory) {
        println("    (no dump directory — run with `./gradlew -p examples/differentiable-physics run`)")
        return
    }
    val dumps = dir.listFiles { f -> f.name.endsWith(".kt") }.orEmpty()
    if (dumps.isEmpty()) {
        println("    (the compiler wrote no gradient source into ${dir.path})")
        return
    }
    val file = dumps.maxByOrNull { it.length() }!!
    val lines = file.readLines()
    val ops = lines.count { it.trimStart().startsWith("val v") }
    println("    ${file.name} — ${lines.size} lines, $ops operations.")
    println("    Nobody wrote this by hand: it is the chain rule carried back through")
    println("    $STEPS timesteps of the loop above, and it is ordinary Kotlin over `:core`.")
    println()
    lines.take(14).forEach { println("    $it") }
    println("    ... ${lines.size - 17} lines ...")
    lines.takeLast(3).forEach { println("    $it") }
    println()
    println("    the whole file: ${file.path}")
}
