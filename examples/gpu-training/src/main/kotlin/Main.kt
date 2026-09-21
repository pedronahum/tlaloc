/**
 * Tlaloc — training a neural network on the GPU with gradients the COMPILER
 * wrote.
 *
 * The shape of the program, and the only three ideas in it:
 *
 *  1. CAPTURE ONCE. `capture(model, inputs) { loss }` traces the model's
 *     forward through Tlaloc's Tracer into a real IR function, then applies
 *     `DxirReverseTransform` — the same reverse-mode AD pass the `grad { }`
 *     compiler plugin uses — to derive the gradient function. There is no tape
 *     at runtime, no `.backward()`, and no hand-written derivative anywhere in
 *     this example or in `:nn`. The adjoint of every operation lives in the
 *     compiler's rule registry.
 *
 *  2. EXECUTION IS A BACKEND CHOICE. The captured gradient function is a
 *     program. This example hands it to XLA through PJRT and runs it on the
 *     GPU (see Lanes.kt) — or interprets it on the JVM if there is no GPU
 *     here. Same program, same numbers, different machine.
 *
 *  3. ONE COMPILED EXECUTABLE FOR THE WHOLE RUN. The model's weights enter the
 *     graph as parameters, not constants, so all 300 training steps re-bind
 *     new values into a single compiled artifact. The example prints the
 *     session's cache size to prove it.
 *
 * The task has a ground truth you can see: the model must learn a disc in the
 * plane, and at the end we print what it learned next to the real thing.
 */
import io.tlaloc.autograd.captureN
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.mean
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.times
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.split
import io.tlaloc.ir.DxirFunction
import io.tlaloc.nn.Adam
import io.tlaloc.nn.Dense
import io.tlaloc.nn.Params
import io.tlaloc.nn.ReluLayer
import io.tlaloc.nn.Sequential
import io.tlaloc.nn.capture
import io.tlaloc.nn.step
import kotlin.math.abs

private const val SEED = 20260921L
private const val TRAIN_POINTS = 512
private const val HELD_OUT_POINTS = 1024
private const val STEPS = 600
private const val LEARNING_RATE = 0.02f
private const val PRINT_EVERY = 50

fun main() {
    println("=== Tlaloc — training on the GPU with gradients the compiler wrote ===")
    println()

    // ---- the task -------------------------------------------------------
    val streams = Streams(SEED)
    val train = Batch.draw(streams.train, TRAIN_POINTS)
    val heldOut = Batch.draw(streams.heldOut, HELD_OUT_POINTS)

    println("task     : label = +1 ${Truth}, -1 outside")
    println("data     : $TRAIN_POINTS training points in [-1,1]^2, drawn from threefry seed $SEED")
    println("           (data AND initial weights are pure functions of that seed — no global RNG)")

    // ---- the model ------------------------------------------------------
    // Two hidden ReLU layers: the disc is not linearly separable, so a model
    // without them provably cannot fit this task.
    val initKeys = streams.init.split(3)
    val model0 = Sequential(
        Dense(2, 16, initKeys[0]),
        ReluLayer,
        Dense(16, 16, initKeys[1]),
        ReluLayer,
        Dense(16, 1, initKeys[2]),
    )
    val scalars = model0.parameters.sumOf { p -> p.tensor.dims.fold(1) { a, b -> a * b } }
    println("model    : Dense(2->16) -> ReLU -> Dense(16->16) -> ReLU -> Dense(16->1)")
    println("           ${model0.parameters.size} parameter tensors, $scalars scalars, keys ${model0.parameters.map { it.key }}")
    println("loss     : mean squared error against the +/-1 labels")
    println()

    // ---- capture once ---------------------------------------------------
    // The targets ride into the graph as a constant leaf; x and the parameters
    // are re-bindable inputs. That is what makes one capture serve every step.
    val x = Tensors.f32Matrix<Sym, Sym>(train.n, 2, train.xs)
    val step = capture(model0, listOf(x), name = "disc_mlp") { prediction ->
        val target = prediction.constant<Shape>(train.ys, intArrayOf(train.n, 1))
        val residual = prediction - target
        (residual * residual).mean()
    }
    println("capture once (no tape, no .backward(), no hand-written derivatives):")
    describe("  forward ", step.primal)
    describe("  gradient", step.gradient)
    println("           the gradient graph was DERIVED from the forward graph by")
    println("           DxirReverseTransform — the compiler's own reverse-mode AD pass.")
    println()

    // ---- pick a lane ----------------------------------------------------
    // TLALOC_EXAMPLE_LANE=host forces the CPU path even on a GPU machine —
    // useful for comparing the two, and the CPU path is bit-reproducible where
    // the GPU one is not (XLA autotunes its GEMM kernels at compile time).
    val forcedHost = System.getenv("TLALOC_EXAMPLE_LANE") == "host"
    val gpuUnavailable = if (forcedHost) "TLALOC_EXAMPLE_LANE=host" else GpuLane.unavailableReason()
    if (forcedHost) {
        println("GPU lane skipped by request (TLALOC_EXAMPLE_LANE=host).")
        println()
    } else if (gpuUnavailable != null) {
        println("GPU lane unavailable: $gpuUnavailable")
        println("           -> falling back to the host interpreter. Everything below is the")
        println("              same program, just slower. This is not an error; the example")
        println("              is meant to run on a laptop too.")
        println()
    }
    val lane: Lane = if (gpuUnavailable == null) GpuLane() else HostLane()
    println("device   : ${lane.description}")

    lane.use {
        // ---- agreement between the two backends -------------------------
        // Worth doing once, because it is the whole claim: the interpreter and
        // XLA share nothing but the graph, so agreeing here means the emitted
        // StableHLO really is the program `:nn` captured.
        if (lane is GpuLane) {
            val values0 = step.bind(listOf(train.xs), model0)
            val onHost = HostLane().run(step.gradient, values0)
            val onGpu = lane.run(step.gradient, values0)
            var maxDiff = 0f
            var maxMagnitude = 0f
            for (k in onHost.indices) {
                for (i in onHost[k].indices) {
                    maxDiff = maxOf(maxDiff, abs(onGpu[k][i] - onHost[k][i]))
                    maxMagnitude = maxOf(maxMagnitude, abs(onHost[k][i]))
                }
            }
            println("check    : GPU vs host interpreter at step 0, over ${onHost.size} outputs —")
            println("           max|diff| %.3g against max|value| %.3g  (%.1e relative)"
                .format(maxDiff, maxMagnitude, maxDiff / maxMagnitude))
            println("           If that looks big for f32, it is: XLA runs f32 matmuls on NVIDIA")
            println("           tensor cores at TF32's 10 mantissa bits by default. Re-run with")
            println("             NVIDIA_TF32_OVERRIDE=0 XLA_FLAGS=--xla_gpu_enable_triton_gemm=false")
            println("           — both, since either alone changes nothing — and the relative")
            println("           difference drops to ~1e-7. A precision knob, not a correctness one.")
        }
        println()

        // ---- train ------------------------------------------------------
        println("training : Adam(lr=$LEARNING_RATE), $STEPS full-batch steps")
        val optimizer = Adam(learningRate = LEARNING_RATE)
        var model = model0
        var optState = optimizer.initialState()
        var firstLoss = Float.NaN
        var lastLoss = Float.NaN

        val startNs = System.nanoTime()
        for (s in 0 until STEPS) {
            // Bind live values, dispatch the ONE captured gradient program,
            // unpack (loss, grads), apply the optimizer. That is the loop.
            val outs = lane.run(step.gradient, step.bind(listOf(train.xs), model))
            val evaluated = step.unpack(outs)
            val (nextModel, nextState) = optimizer.step(model, evaluated.gradients, optState)
            model = nextModel
            optState = nextState

            if (s == 0) firstLoss = evaluated.loss
            lastLoss = evaluated.loss
            if (s % PRINT_EVERY == 0) println("           step %3d   loss %.6f".format(s, evaluated.loss))
        }
        val elapsedMs = (System.nanoTime() - startNs) / 1e6
        println("           step %3d   loss %.6f   (final)".format(STEPS, lastLoss))
        println()
        println("           %.3f s wall clock on %s (%.2f ms/step)".format(elapsedMs / 1000.0, lane.description, elapsedMs / STEPS))
        println("           loss %.6f -> %.6f".format(firstLoss, lastLoss))
        if (lane is GpuLane) {
            println("           compiled executables after training: ${lane.compiledExecutables}")
            println("           (one program, $STEPS dispatches — weights are graph PARAMETERS, not constants)")
        }
        println()

        // ---- what did it learn? -----------------------------------------
        // The prediction path is the same model traced WITHOUT a loss: a plain
        // function from points to scores, which runs on the same lane.
        val grid = Grid(cols = 31, rows = 15)
        val gridScores = predict(lane, model, grid.points, grid.count)
        val heldOutScores = predict(lane, model, heldOut.xs, heldOut.n)

        println("learned decision boundary vs ground truth  ('#' inside, '.' outside)")
        println()
        grid.render(gridScores).forEach { println("   $it") }
        println()

        var wrongCells = 0
        for (i in 0 until grid.count) {
            val truth = Truth.label(grid.points[2 * i], grid.points[2 * i + 1])
            if (sign(gridScores[i]) != truth) wrongCells++
        }
        println("           grid cells where the model disagrees with the truth: %d / %d (%.1f%%)"
            .format(wrongCells, grid.count, 100.0 * wrongCells / grid.count))

        var correct = 0
        for (i in 0 until heldOut.n) if (sign(heldOutScores[i]) == heldOut.ys[i]) correct++
        println("           held-out accuracy on %d fresh points never seen in training: %.1f%%"
            .format(heldOut.n, 100.0 * correct / heldOut.n))
        if (lane is GpuLane) {
            println("           compiled executables after inference: ${lane.compiledExecutables}" +
                " (training + two inference shapes)")
        }
        println()
        if (lane is GpuLane) {
            println("note     : run this twice and the losses will differ in the 3rd decimal. XLA")
            println("           autotunes its GEMM kernels at compile time, so the GPU lane is not")
            println("           bit-reproducible; TLALOC_EXAMPLE_LANE=host is, exactly, every run.")
            println()
        }
        println("done.")
    }
}

/** `+1` / `-1`, matching [Truth.label]'s convention so the two are comparable. */
private fun sign(score: Float): Float = if (score >= 0f) 1f else -1f

private fun describe(label: String, fn: DxirFunction) {
    println("$label: ${fn.body.size} IR nodes, ${fn.params.size} parameters, ${fn.returns.size} result(s)")
}

/**
 * Forward-only evaluation: trace the model with no loss attached and run the
 * resulting function. `captureN` is `:autograd`'s general tracing entry point —
 * `:nn`'s `capture` is this plus a loss plus the reverse transform.
 */
private fun predict(lane: Lane, model: Sequential, points: FloatArray, n: Int): FloatArray {
    val input = Tensors.f32Matrix<Sym, Sym>(n, 2, points)
    val keys = model.parameters.map { it.key }
    val fn = captureN(listOf(input) + model.parameters.map { it.tensor }, "predict") { leaves ->
        val byKey = keys.withIndex().associate { (j, key) -> key to leaves[1 + j] }
        model.forward(leaves[0], Params { key -> byKey.getValue(key) })
    }
    val values = listOf(points) + model.parameters.map { it.tensor.hostF32() }
    return lane.run(fn, values)[0]
}

/** A [cols] x [rows] sampling of [-1,1]^2, rendered as two side-by-side maps. */
private class Grid(val cols: Int, val rows: Int) {
    val count = cols * rows

    /** Row-major [count, 2]; row 0 is the TOP of the picture (y = +1). */
    val points = FloatArray(count * 2).also { p ->
        for (r in 0 until rows) for (c in 0 until cols) {
            val i = r * cols + c
            p[2 * i] = -1f + 2f * c / (cols - 1)
            p[2 * i + 1] = 1f - 2f * r / (rows - 1)
        }
    }

    fun render(scores: FloatArray): List<String> {
        val header = "ground truth".padCenter(cols) + "   " + "what the model learned".padCenter(cols)
        val lines = ArrayList<String>(rows + 1)
        lines += header
        for (r in 0 until rows) {
            val truth = StringBuilder(cols)
            val learned = StringBuilder(cols)
            for (c in 0 until cols) {
                val i = r * cols + c
                truth.append(if (Truth.label(points[2 * i], points[2 * i + 1]) > 0f) '#' else '.')
                learned.append(if (scores[i] >= 0f) '#' else '.')
            }
            lines += "$truth   $learned"
        }
        return lines
    }
}

private fun String.padCenter(width: Int): String {
    if (length >= width) return this
    val left = (width - length) / 2
    return " ".repeat(left) + this + " ".repeat(width - length - left)
}
