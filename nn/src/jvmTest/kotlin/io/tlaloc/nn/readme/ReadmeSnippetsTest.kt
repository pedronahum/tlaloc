package io.tlaloc.nn.readme

import io.tlaloc.autograd.constant
import io.tlaloc.autograd.mean
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.times
import io.tlaloc.core.Batch
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.Hidden
import io.tlaloc.core.Named
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Rank2
import io.tlaloc.core.SeqLen
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.ops.contract
import io.tlaloc.core.split
import io.tlaloc.nn.Adam
import io.tlaloc.nn.Dense
import io.tlaloc.nn.ReluLayer
import io.tlaloc.nn.Sequential
import io.tlaloc.nn.capture
import io.tlaloc.nn.loadCheckpoint
import io.tlaloc.nn.saveCheckpoint
import io.tlaloc.nn.step
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The README's "Axis names the type checker enforces" and "Train, checkpoint,
 * serve" snippets, verbatim between the SNIPPET markers, so the README cannot
 * show code that does not compile against the published API. When a snippet
 * in README.md changes, change it here too.
 */
class ReadmeSnippetsTest {

    @Test
    fun namedAxisSnippetCompilesAndContractsTheSharedAxis() {
        val activations: DTensor<Rank2<Named<Batch, Sym>, Named<SeqLen, Sym>>, F32> =
            Tensors.f32Matrix(2, 3, FloatArray(6) { it + 1f })
        val weights: DTensor<Rank2<Named<SeqLen, Sym>, Named<Hidden, Sym>>, F32> =
            Tensors.f32Matrix(3, 4, FloatArray(12) { 0.1f * (it + 1) })

        // SNIPPET
        val hidden = activations contract weights   // OK: they share SeqLen
        // END SNIPPET

        val typed: DTensor<Rank2<Named<Batch, Sym>, Named<Hidden, Sym>>, F32> = hidden
        assertContentEquals(intArrayOf(2, 4), typed.dims)
        // Row 0 of [1 2 3] . W, column 0 of W = [0.1, 0.5, 0.9]: 0.1 + 1.0 + 2.7.
        assertEquals(3.8f, typed.hostF32()[0], 1e-5f)
    }

    @Test
    fun trainingSnippetCompilesTrainsAndCheckpoints() {
        val n = 32
        val xs = FloatArray(2 * n) { i -> ((i * 37) % 17) / 8.5f - 1f }
        val targets = FloatArray(n) { i -> if (xs[2 * i] * xs[2 * i + 1] >= 0f) 1f else -1f }
        val x = Tensors.f32Matrix<Sym, Sym>(n, 2, xs)
        val dir = Files.createTempDirectory("tlaloc-readme-")
        dir.toFile().deleteOnExit()
        val path: Path = dir.resolve("mlp.safetensors")
        // SNIPPET
        val keys = RandomKey.fromSeed(7).split(2)
        val model0 = Sequential(Dense(2, 16, keys[0]), ReluLayer, Dense(16, 1, keys[1]))

        val step = capture(model0, listOf(x), name = "mlp") { prediction ->
            val residual = prediction - prediction.constant<Shape>(targets, intArrayOf(n, 1))
            (residual * residual).mean()
        }

        val optimizer = Adam(learningRate = 0.02f)
        var model = model0
        var state = optimizer.initialState()
        repeat(60) {
            val out = step.run(model, listOf(x))   // loss and gradients, on the host
            val (nextModel, nextState) = optimizer.step(model, out.gradients, state)
            model = nextModel; state = nextState
        }

        saveCheckpoint(path, model, optimizer, state)   // one safetensors file, resumable
        // END SNIPPET

        val before = step.run(model0, listOf(x)).loss
        val after = step.run(model, listOf(x)).loss
        assertTrue(after < 0.5f * before, "the loop must train: $before -> $after")
        val reloaded = loadCheckpoint(path).restore(
            Sequential(Dense(2, 16, keys[0]), ReluLayer, Dense(16, 1, keys[1])),
        )
        for (i in model.parameters.indices) {
            assertContentEquals(
                model.parameters[i].tensor.hostF32(),
                reloaded.parameters[i].tensor.hostF32(),
            )
        }
    }
}
