/**
 * Tlaloc on MNIST — the handwriting benchmark everybody knows, trained with a
 * gradient the compiler derived.
 *
 * The shape of the program is the same three moves as every other `:nn` model:
 *
 *  1. CAPTURE ONCE. `capture(model, inputs) { loss }` traces the forward pass
 *     and hands back the gradient function that `DxirReverseTransform` — the
 *     compiler's own reverse-mode pass — derived from it. There is no
 *     `.backward()` in this file, and no hand-written derivative anywhere.
 *
 *  2. EXECUTION IS A BACKEND CHOICE. The captured gradient is a `DxirFunction`:
 *     a program, not a GPU thing. `Lanes.kt` runs it either on the JVM
 *     interpreter or, via StableHLO and XLA, on the GPU. Nothing else in the
 *     example knows which it got.
 *
 *  3. THE LOOP IS A FOLD. Bind values, dispatch one compiled program, apply the
 *     optimizer, get a NEW model back. Nothing is mutated.
 *
 * What makes this one worth reading rather than the disc in `gpu-training/` is
 * that you can check the answer with your own eyes: act [4] prints test digits
 * as ASCII next to what the network called them.
 */
import io.tlaloc.autograd.captureN
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.mean
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.times
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.hostF32
import io.tlaloc.core.Tensors
import io.tlaloc.nn.Adam
import io.tlaloc.nn.Dense
import io.tlaloc.nn.Params
import io.tlaloc.nn.ReluLayer
import io.tlaloc.nn.Sequential
import io.tlaloc.nn.capture
import io.tlaloc.nn.step
import io.tlaloc.core.RandomKey
import io.tlaloc.core.split
import kotlin.math.abs

/** 28x28 = 784 inputs, one hidden layer, 10 outputs. */
const val PIXELS = 784
const val HIDDEN = 128
const val CLASSES = 10

/**
 * Training. The captured graph holds a FIXED training block: the images are a
 * re-bindable input, the one-hot targets ride in as a constant leaf, and the
 * weights are graph parameters. That is what lets 600 steps share ONE compiled
 * executable — and it is why this example trains full-batch on a subset rather
 * than streaming mini-batches: a baked constant cannot be re-bound per step,
 * and pretending otherwise would silently train against the first batch's
 * labels forever.
 */
const val GPU_TRAIN_IMAGES = 4096
const val HOST_TRAIN_IMAGES = 512
const val GPU_STEPS = 600
const val HOST_STEPS = 40      // the interpreter is a correctness engine, not fast
const val LEARNING_RATE = 0.001f
const val SEED = 20260922L

fun main() {
    println("Tlaloc on MNIST — 60,000 handwritten digits, one captured gradient")
    println()

    // ------------------------------------------------------------------ 0 ---
    println("[0] the data")
    val data = MnistLoader.load()
    if (data == null) {
        println()
        println("SKIP: MNIST is not cached and could not be downloaded.")
        println("      Cache directory: ${MnistLoader.cacheDir.path}")
        println("      Fetch it by hand, then re-run:")
        println()
        println("        ${MnistLoader.manualFetch()}")
        println()
        println("mnist SKIPPED (exit 0 — no network is not an error)")
        return
    }
    println("    train ${"%,d".format(data.train.n)} images    test ${"%,d".format(data.test.n)} images")
    println("    28x28 greyscale, scaled to [0,1]; no other preprocessing")
    println("    cache: ${MnistLoader.cacheDir.path}")
    println()

    // ------------------------------------------------------------------ 1 ---
    // THE MODEL. Immutable: `optimizer.step` returns a new one each time.
    val keys = RandomKey.fromSeed(SEED).split(2)
    val model0 = Sequential(
        Dense(PIXELS, HIDDEN, keys[0]),
        ReluLayer,
        Dense(HIDDEN, CLASSES, keys[1]),
    )
    val scalars = model0.parameters.sumOf { p -> p.tensor.dims.fold(1) { a, b -> a * b } }
    println("[1] the model and the captured gradient")
    println("    Dense($PIXELS -> $HIDDEN) -> ReLU -> Dense($HIDDEN -> $CLASSES)")
    println("    ${model0.parameters.size} parameter tensors, ${"%,d".format(scalars)} scalars")
    println("    weights initialised from threefry seed $SEED (bit-exact against JAX's stream)")

    // ------------------------------------------------------------------ 2 ---
    // CAPTURE ONCE. The loss is squared error against the one-hot target.
    //
    // WHY NOT CROSS-ENTROPY: `:core` has `crossEntropyLoss`, but there is no
    // traced spelling of it (nor of a `max` reduction, which a numerically
    // stable log-sum-exp needs) on `Tracer` today — see the README. Squared
    // error against one-hot needs only ops that exist, and it is honest about
    // what it costs: a percent or two of final accuracy.
    val lane = pickLane()
    val steps = if (lane is GpuLane) GPU_STEPS else HOST_STEPS
    val trainCount = if (lane is GpuLane) GPU_TRAIN_IMAGES else HOST_TRAIN_IMAGES

    val (trainX, trainY) = data.train.batch(0, trainCount)
    val x0 = Tensors.f32Matrix<Sym, Sym>(trainCount, PIXELS, trainX)
    val step = capture(model0, listOf(x0), name = "mnist_mlp") { logits ->
        val target = logits.constant<Shape>(trainY, intArrayOf(trainCount, CLASSES))
        val residual = logits - target
        (residual * residual).mean()
    }
    println("    forward : ${step.primal.body.size} IR nodes")
    println("    gradient: ${step.gradient.body.size} IR nodes, derived by DxirReverseTransform")
    println()

    // ------------------------------------------------------------------ 3 ---
    println("[2] training on ${lane.description}")
    println()
    lane.use {
        val optimizer = Adam(learningRate = LEARNING_RATE)
        var model = model0
        var optState = optimizer.initialState()
        var firstLoss = Float.NaN
        var lastLoss = Float.NaN

        println("    %6s  %12s".format("step", "loss"))
        val t0 = System.nanoTime()
        for (s in 0 until steps) {
            val outs = lane.run(step.gradient, step.bind(listOf(trainX), model))
            val evaluated = step.unpack(outs)
            val (nextModel, nextState) = optimizer.step(model, evaluated.gradients, optState)
            model = nextModel
            optState = nextState
            if (s == 0) firstLoss = evaluated.loss
            lastLoss = evaluated.loss
            if (s % (steps / 8) == 0 || s == steps - 1) {
                println("    %6d  %12.6f".format(s, evaluated.loss))
            }
        }
        val secs = (System.nanoTime() - t0) / 1e9
        println()
        println("    %d full-batch steps over %,d images in %.2f s  (%.2f ms/step)"
            .format(steps, trainCount, secs, secs * 1000 / steps))
        println("    loss %.6f -> %.6f".format(firstLoss, lastLoss))
        if (lane is GpuLane) {
            println("    compiled executables after training: ${lane.compiledExecutables}")
            println("    (one program, $steps dispatches — the weights are graph PARAMETERS)")
        }
        println()

        // -------------------------------------------------------------- 4 ---
        val evalCount = if (lane is GpuLane) 10000 else 1000
        val correct = evaluate(lane, model, data.test, evalCount)
        val accuracy = 100.0 * correct / evalCount
        println("[3] held-out accuracy")
        println()
        println("    %,d of %,d test images correct  =  %.2f %%".format(correct, evalCount, accuracy))
        println()

        // -------------------------------------------------------------- 5 ---
        showDigits(lane, model, data.test)

        check(accuracy > if (lane is GpuLane) 85.0 else 50.0) {
            "accuracy $accuracy%% is below the floor this example asserts"
        }
    }
    println()
    println("mnist OK")
}

// ---------------------------------------------------------------------------

private fun pickLane(): Lane {
    val forcedHost = System.getenv("TLALOC_EXAMPLE_LANE") == "host"
    val reason = if (forcedHost) "TLALOC_EXAMPLE_LANE=host" else GpuLane.unavailableReason()
    if (reason != null) {
        println("[!] GPU lane unavailable: $reason")
        println("    -> falling back to the JVM interpreter, with fewer steps. Same program,")
        println("       same gradient; it is just slower, and the accuracy below reflects the")
        println("       shorter run. This is not an error.")
        println()
        return HostLane()
    }
    return GpuLane()
}

/** Argmax over the model's logits for [count] test images, batch by batch. */
private fun evaluate(lane: Lane, model: Sequential, split: MnistSplit, count: Int): Int {
    var correct = 0
    var done = 0
    val chunk = 500
    while (done < count) {
        val n = minOf(chunk, count - done)
        val logits = forward(lane, model, split.images, done, n)
        for (i in 0 until n) {
            var best = 0
            for (c in 1 until CLASSES) if (logits[i * CLASSES + c] > logits[i * CLASSES + best]) best = c
            if (best == split.labels[done + i]) correct++
        }
        done += n
    }
    return correct
}

/**
 * Forward only: the model traced WITHOUT a loss is a plain function from images
 * to logits, and it runs on the same lane as training did.
 */
private fun forward(lane: Lane, model: Sequential, images: FloatArray, from: Int, n: Int): FloatArray {
    val x = FloatArray(n * PIXELS)
    System.arraycopy(images, from * PIXELS, x, 0, n * PIXELS)
    val input = Tensors.f32Matrix<Sym, Sym>(n, PIXELS, x)
    val keys = model.parameters.map { it.key }
    val fn = captureN(listOf(input) + model.parameters.map { it.tensor }, "mnist_predict") { leaves ->
        val byKey = keys.withIndex().associate { (j, key) -> key to leaves[1 + j] }
        model.forward(leaves[0], Params { key -> byKey.getValue(key) })
    }
    return lane.run(fn, listOf(x) + model.parameters.map { it.tensor.hostF32() })[0]
}

/** Print a handful of test digits as ASCII, with the model's verdict. */
private fun showDigits(lane: Lane, model: Sequential, split: MnistSplit) {
    val show = 6
    val logits = forward(lane, model, split.images, 0, show)
    println("[4] what it actually sees")
    println()
    val ramp = " .:-=+*#%@"
    val rows = Array(show) { mutableListOf<String>() }
    for (i in 0 until show) {
        for (r in 0 until 28 step 2) {
            val sb = StringBuilder()
            for (c in 0 until 28) {
                val v = split.images[i * PIXELS + r * 28 + c]
                sb.append(ramp[(v * (ramp.length - 1)).toInt().coerceIn(0, ramp.length - 1)])
            }
            rows[i].add(sb.toString())
        }
    }
    for (i in 0 until show step 3) {
        val group = (i until minOf(i + 3, show)).toList()
        for (line in 0 until 14) {
            println("    " + group.joinToString("   ") { rows[it][line] })
        }
        println("    " + group.joinToString("   ") { j ->
            var best = 0
            for (c in 1 until CLASSES) if (logits[j * CLASSES + c] > logits[j * CLASSES + best]) best = c
            val mark = if (best == split.labels[j]) "" else "  <-- WRONG"
            "predicted %d, label %d%s".format(best, split.labels[j], mark).padEnd(28)
        })
        println()
    }
}
