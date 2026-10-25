package io.tlaloc.plugin

import io.tlaloc.plugin.F64TestHarness.assertClose
import io.tlaloc.plugin.F64TestHarness.lit
import io.tlaloc.testing.Goldens
import kotlin.test.Test

/**
 * `vmap` through the plugin at F64 against `jax.vmap` with `jax_enable_x64`, run in a JAX
 * Python found at `TLALOC_JAX_PYTHON` or `~/.local/venvs/iree/bin/python`, or against its
 * recording when there is none (test-support `Goldens`, suite `vmap-jax`). Both sides
 * evaluate in float64; tolerance 1e-12 of the largest entry.
 *
 *  - a batched forward pass: `jax.vmap(f, in_axes=(0, None))`;
 *  - per-example gradients: `jax.vmap(jax.grad(loss), in_axes=(None, 0))`;
 *  - the gradient of a vmapped mean: `jax.grad(lambda w: jnp.mean(jax.vmap(loss, (None, 0))(w, xs)))`;
 *  - per-example gradients of a log-determinant: `jax.vmap(jax.grad(slogdet))`.
 */
class VmapJaxParityTest {

    /** JAX's result for [case]: live with the oracle venv, else its recording (test-support Goldens). */
    private fun jax(case: String, body: String): DoubleArray = Goldens.python(
        "vmap-jax", case,
        """
        |import jax
        |jax.config.update("jax_enable_x64", True)
        |jax.config.update("jax_platform_name", "cpu")
        |import jax.numpy as jnp
        |xs = jnp.array([${lit(xs)}], dtype=jnp.float64).reshape($B, 1, 3)
        |w = jnp.array([${lit(w)}], dtype=jnp.float64).reshape(3, 4)
        |def loss(w, x):
        |    h = jnp.tanh(x @ w)
        |    return jnp.sum(jax.nn.softmax(h, axis=-1) * h)
        |${body.trimMargin()}
        |print(",".join(repr(float(v)) for v in jnp.ravel(r)))
        """.trimMargin(),
    )

    private val B = 5

    // StrictMath, not kotlin.math: the inputs are part of the recording's key, and Math.sin
    // may differ in the last bit between x86 and arm64.
    private val xs = DoubleArray(B * 3) { StrictMath.sin(1.3 * it + 0.2) }
    private val w = DoubleArray(12) { StrictMath.cos(0.7 * it) * 0.8 }

    private fun tlaloc(body: String): DoubleArray {
        val src = """
            @file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)
            import io.tlaloc.autograd.*
            import io.tlaloc.core.*
            import io.tlaloc.core.ops.*
            typealias X = DTensor<Rank2<Sym, Sym>, F64>
            typealias W = DTensor<Rank2<Sym, Sym>, F64>
            fun main() {
                val xs = DTensor<Rank3<Named<Batch, Sym>, Sym, Sym>, F64>(HostF64Storage(doubleArrayOf(${lit(xs)})), intArrayOf($B, 1, 3), F64)
                val w0 = Tensors.f64Matrix<Sym, Sym>(3, 4, doubleArrayOf(${lit(w)}))
                $body
                println("r " + r.hostF64().joinToString(","))
            }
        """.trimIndent()
        return F64TestHarness.run(src).values("r")
    }

    @Test
    fun `a batched forward pass matches jax vmap`() = assertClose(
        jax("forward", "r = jax.vmap(lambda x: jnp.tanh(x @ w), in_axes=0)(xs)"),
        tlaloc("val r = vmap(batchAxis(Batch)) { x: X -> (x matmul w0).tanh() }(xs)"),
        1e-12, "vmap forward vs jax.vmap",
    )

    @Test
    fun `per-example gradients match jax vmap of jax grad`() = assertClose(
        jax("per_example_grad", "r = jax.vmap(jax.grad(loss), in_axes=(None, 0))(w, xs)"),
        tlaloc(
            """
            val r = vmap(batchAxis(Batch)) { x: X ->
                grad { w: W -> ((x matmul w).tanh().softmax() * (x matmul w).tanh()).sum().toDouble() }(w0)
            }(xs)
            """.trimIndent(),
        ),
        1e-12, "vmap(grad) vs jax.vmap(jax.grad)",
    )

    @Test
    fun `the gradient of a vmapped mean matches jax grad of jax vmap`() = assertClose(
        jax("grad_of_mean", "r = jax.grad(lambda w: jnp.mean(jax.vmap(loss, in_axes=(None, 0))(w, xs)))(w)"),
        tlaloc(
            """
            val g = grad { w: W ->
                vmap(batchAxis(Batch)) { x: X -> ((x matmul w).tanh().softmax() * (x matmul w).tanh()).sum() }(xs)
                    .mean().toDouble()
            }
            val r = g(w0)
            """.trimIndent(),
        ),
        1e-12, "grad(vmap) vs jax.grad(jax.vmap)",
    )

    @Test
    fun `per-example gradients of a log-determinant match jax`() {
        // B symmetric positive-definite 3 x 3 matrices: M Mᵀ + 3 I per example.
        val a = DoubleArray(B * 9) { k ->
            val e = k / 9
            val i = (k % 9) / 3
            val j = k % 3
            var s = 0.0
            for (q in 0 until 3) s += StrictMath.sin(e + 1.1 * i + 0.3 * q) * StrictMath.sin(e + 1.1 * j + 0.3 * q)
            s + if (i == j) 3.0 else 0.0
        }
        val want = jax(
            "logdet_grad",
            """
            a = jnp.array([${lit(a)}], dtype=jnp.float64).reshape($B, 3, 3)
            r = jax.vmap(jax.grad(lambda m: jnp.linalg.slogdet((m + m.T) / 2)[1]))(a)
            """.trimIndent(),
        )
        val got = tlaloc(
            """
            val a = DTensor<Rank3<Named<Batch, Sym>, Sym, Sym>, F64>(HostF64Storage(doubleArrayOf(${lit(a)})), intArrayOf($B, 3, 3), F64)
            val r = vmap(batchAxis(Batch)) { m: W -> grad { k: W -> k.logDetSpd().toDouble() }(m) }(a)
            """.trimIndent(),
        )
        assertClose(want, got, 1e-12, "vmap(grad(logDetSpd)) vs jax")
    }
}
