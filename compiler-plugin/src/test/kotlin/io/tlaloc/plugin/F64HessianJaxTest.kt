package io.tlaloc.plugin

import io.tlaloc.plugin.F64TestHarness.assertClose
import io.tlaloc.plugin.F64TestHarness.lit
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.test.Test

/**
 * `hessian { x -> x.logDetSpd() }` over an F64 matrix, compiled by the plugin, against
 * `jax.hessian` of `jnp.linalg.slogdet` with `jax_enable_x64`, run in a JAX Python found at
 * `TLALOC_JAX_PYTHON` or `~/.local/venvs/iree/bin/python`. Skipped by name without one.
 *
 * Both sides evaluate in float64 and are backward stable; with the matrix's condition number
 * below 2 they agree to about 1e-15. Tolerance 1e-11 of the largest entry. (The same Hessian
 * is checked against committed JAX values at IR level in `LinalgJaxParityTest`, and against
 * finite differences through the plugin in `LinalgGradientTest`.)
 */
class F64HessianJaxTest {

    private val n = 4
    private val spd = doubleArrayOf(
        6.2, 1.1, -0.7, 0.4,
        1.3, 5.1, 0.9, -1.2,
        -0.5, 0.9, 4.8, 0.6,
        0.4, -1.0, 0.6, 5.5,
    )

    private fun jaxPython(): String? {
        val candidates = listOfNotNull(
            System.getenv("TLALOC_JAX_PYTHON"),
            System.getProperty("user.home")?.let { "$it/.local/venvs/iree/bin/python" },
        )
        return candidates.firstOrNull { Files.isExecutable(Path.of(it)) && run(it, "import jax") != null }
    }

    private fun run(python: String, code: String): String? = runCatching {
        val p = ProcessBuilder(python, "-c", code).redirectErrorStream(true).start()
        val out = p.inputStream.bufferedReader().readText()
        if (p.waitFor(120, TimeUnit.SECONDS) && p.exitValue() == 0) out else null
    }.getOrNull()

    @Test
    fun `hessian of logDetSpd at F64 matches jax hessian with x64`() {
        val python = jaxPython()
        assumeTrue(python != null, "no Python with jax (TLALOC_JAX_PYTHON or ~/.local/venvs/iree) — skipping")
        val jax = run(
            python!!,
            """
            import jax
            jax.config.update("jax_enable_x64", True)
            jax.config.update("jax_platform_name", "cpu")
            import jax.numpy as jnp
            a = jnp.array([${lit(spd)}], dtype=jnp.float64).reshape($n, $n)
            h = jax.hessian(lambda x: jnp.linalg.slogdet((x + x.T) / 2)[1])(a).reshape(${n * n}, ${n * n})
            print(",".join(repr(float(v)) for v in h.ravel()))
            """.trimIndent(),
        ) ?: error("the JAX reference failed to run")
        val want = jax.trim().lines().last().split(",").map { it.toDouble() }.toDoubleArray()

        val src = """
            import io.tlaloc.autograd.hessian
            import io.tlaloc.core.*
            import io.tlaloc.core.ops.*
            fun main() {
                val a = Tensors.f64Matrix<Sym, Sym>($n, $n, doubleArrayOf(${lit(spd)}))
                val h = hessian { x: DTensor<Rank2<Sym, Sym>, F64> -> x.logDetSpd().toDouble() }
                println("h " + h(a).hostF64().joinToString(","))
            }
        """.trimIndent()
        val r = F64TestHarness.run(src)
        assertClose(want, r.values("h"), 1e-11, "hessian of log det vs JAX")
    }
}
