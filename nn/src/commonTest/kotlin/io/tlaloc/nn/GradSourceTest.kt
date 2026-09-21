package io.tlaloc.nn

import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.449 — the `:nn` user surface of the readable-reverse printer:
 * `CapturedStep.gradSource()` / `primalSource()` hand back the captured
 * functions as compilable Kotlin over the `:core` host twins. The
 * compile-and-run bit-identity certification lives in `:compiler-plugin`'s
 * PrintedGradientGoldenTest (the K2 harness module); this pins the surface
 * itself on a real model capture.
 */
class GradSourceTest {

    @Test
    fun capturedStepPrintsItsGradientAsHostTwinKotlin() {
        val model = Sequential(
            Dense(
                Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(0.5f, -1f, 1f, 0.5f)),
                Tensors.f32Vector<Sym>(floatArrayOf(0.25f, -0.5f)),
                Activation.Relu,
            ),
        )
        val step = capture(model, listOf(Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, -0.5f, 2f, 0.25f)))) { y ->
            (y * y).sum()
        }
        val grad = step.gradSource()
        assertTrue("fun " in grad, grad)
        assertTrue("matmul" in grad, "the Dense adjoint must print its matmuls:\n$grad")
        assertTrue("DTensor<Rank2<Sym, Sym>, F32>" in grad, grad)
        assertTrue("import io.tlaloc.core.ops.*" in grad, grad)

        val primal = step.primalSource()
        assertTrue(".relu()" in primal, "the primal must print its relu:\n$primal")
    }
}
