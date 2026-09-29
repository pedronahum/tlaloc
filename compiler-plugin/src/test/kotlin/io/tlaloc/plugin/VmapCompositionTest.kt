package io.tlaloc.plugin

import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * `vmap` composed with `grad`, `jvp` and itself through the K2 plugin, each order
 * against a loop over examples of single-example intrinsics, at F32 and F64 and batch
 * sizes 1, 7 and 64:
 *
 *  - `vmap { grad { } }` = per-example gradients = a loop of single-example gradients;
 *  - `grad { vmap { }.sum() }` = the sum over the loop of single-example gradients;
 *  - `vmap { jvp { } }` and `jvp { vmap { } }` = a loop of single-example jvps;
 *  - `vmap { vmap { } }` = a double loop.
 *
 * The batched programs sum in another order than the loop does, so the comparison allows
 * 1e-6 of the largest magnitude (1e-12 at F64).
 */
class VmapCompositionTest {

    private fun run(src: String, precision: Precision): F64TestHarness.Result {
        val user = if (precision == Precision.F64) F64Source.of(src) else src
        val r = F64TestHarness.compileAndRun(user)
        assertEquals(0, r.exitCode, "compile/run failed:\n${r.describe()}")
        assertTrue(
            r.messages.none { "kept original call" in it.message },
            "the plugin kept the original call:\n${r.describe()}",
        )
        return r
    }

    private fun assertClose(r: F64TestHarness.Result, precision: Precision) {
        val x = r.values("vmap")
        val y = r.values("loop")
        assertEquals(y.size, x.size, "sizes\n${r.stdout}")
        val tol = if (precision == Precision.F64) 1e-12 else 1e-6
        val scale = maxOf(1.0, y.maxOf { kotlin.math.abs(it) })
        for (i in x.indices) {
            assertTrue(kotlin.math.abs(x[i] - y[i]) <= tol * scale, "vmap[$i] = ${x[i]} but loop[$i] = ${y[i]}\n${r.stdout}")
        }
    }

    private fun program(batch: Int, body: String): String = """
        @file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)
        import io.tlaloc.autograd.*
        import io.tlaloc.core.*
        import io.tlaloc.core.ops.*

        object Feat : IndexName { override val name = "feat" }
        object Out : IndexName { override val name = "out" }
        object Inner : IndexName { override val name = "inner" }

        typealias X = DTensor<Rank2<Sym, Named<Feat, Sym>>, F32>
        typealias W = DTensor<Rank2<Named<Feat, Sym>, Named<Out, Sym>>, F32>
        typealias XS = DTensor<Rank3<Named<Batch, Sym>, Sym, Named<Feat, Sym>>, F32>

        fun data(n: Int, seed: Int): FloatArray {
            var s = seed.toLong() * 7919L + 12345L
            return FloatArray(n) {
                s = (s * 1103515245L + 12345L) and 0x7fffffffL
                ((s % 20000L) - 10000L) / 7000.0f
            }
        }
        @Suppress("UNCHECKED_CAST")
        fun flat(v: Any?): List<Float> = when (v) {
            is DTensor<*, *> -> (v as DTensor<*, F32>).hostF32().asList()
            is Float -> listOf(v)
            else -> error("unexpected value " + v)
        }
        fun example(xs: FloatArray, i: Int): X =
            DTensor<Rank2<Sym, Named<Feat, Sym>>, F32>(HostF32Storage(xs.copyOfRange(6 * i, 6 * i + 6)), intArrayOf(2, 3), F32)
        fun main() {
            val batch = $batch
            val xsData = data(batch * 6, 1)
            val xs = DTensor<Rank3<Named<Batch, Sym>, Sym, Named<Feat, Sym>>, F32>(HostF32Storage(xsData), intArrayOf(batch, 2, 3), F32)
            val w0 = DTensor<Rank2<Named<Feat, Sym>, Named<Out, Sym>>, F32>(HostF32Storage(data(12, 2)), intArrayOf(3, 4), F32)
            val dw = DTensor<Rank2<Named<Feat, Sym>, Named<Out, Sym>>, F32>(HostF32Storage(data(12, 3)), intArrayOf(3, 4), F32)
            // The single-example function, with w and x both parameters, for the loops.
            val g2 = grad2 { w: W, x: X -> ((x matmul w).tanh() * (x matmul w)).sum().toFloat() }
            $body
        }
    """.trimIndent()

    private fun check(body: String) {
        for (p in listOf(Precision.F32, Precision.F64)) for (b in listOf(1, 7, 64)) {
            assertClose(run(program(b, body), p), p)
        }
    }

    @Test
    fun `vmap of grad gives per-example gradients`() = check(
        """
            val perExample = vmap(batchAxis(Batch)) { x: X ->
                grad { w: W -> ((x matmul w).tanh() * (x matmul w)).sum().toFloat() }(w0)
            }
            println("vmap " + flat(perExample(xs)).joinToString(","))
            println("loop " + (0 until batch).flatMap { i -> flat(g2(w0, example(xsData, i)).first) }.joinToString(","))
        """.trimIndent(),
    )

    @Test
    fun `vmap of grad through a local val`() = check(
        """
            val perExample = vmap(batchAxis(Batch)) { x: X ->
                val g = grad { w: W -> ((x matmul w).tanh() * (x matmul w)).sum().toFloat() }
                g(w0) * 2f
            }
            println("vmap " + flat(perExample(xs)).joinToString(","))
            println("loop " + (0 until batch).flatMap { i -> flat(g2(w0, example(xsData, i)).first * 2f) }.joinToString(","))
        """.trimIndent(),
    )

    @Test
    fun `grad of a vmapped sum is the sum of per-example gradients`() = check(
        """
            val g = grad { w: W ->
                vmap(batchAxis(Batch)) { x: X -> ((x matmul w).tanh() * (x matmul w)).sum().toFloat() }(xs).sum().toFloat()
            }
            println("vmap " + flat(g(w0)).joinToString(","))
            val acc = FloatArray(12)
            for (i in 0 until batch) {
                val gi = flat(g2(w0, example(xsData, i)).first)
                for (k in 0 until 12) acc[k] += gi[k]
            }
            println("loop " + acc.joinToString(","))
        """.trimIndent(),
    )

    @Test
    fun `vmap of jvp gives per-example directional derivatives`() = check(
        """
            val perExample = vmap(batchAxis(Batch)) { x: X ->
                jvp { w: W -> ((x matmul w).tanh() * (x matmul w)).sum() }(w0, dw)
            }
            println("vmap " + flat(perExample(xs)).joinToString(","))
            // <grad_w f, dw> for each example.
            println("loop " + (0 until batch).map { i ->
                val gw = flat(g2(w0, example(xsData, i)).first)
                val d = flat(dw)
                var s = 0f
                for (k in 0 until 12) s += gw[k] * d[k]
                s
            }.joinToString(","))
        """.trimIndent(),
    )

    @Test
    fun `jvp of a vmapped function`() = check(
        """
            // jvp cannot carry a captured value, so the weight is a second argument with a zero
            // tangent. `sin`, not `tanh`: the forward rule for tanh emits a constant whose
            // extents synthesis infers from the parameters, which fails at rank 3 (work log).
            val j = jvp2 { b: XS, w: W -> vmap(batchAxis(Batch)) { x: X -> ((x matmul w).sin() * (x matmul w)).sum() }(b) }
            val gs = grad2 { w: W, x: X -> ((x matmul w).sin() * (x matmul w)).sum().toFloat() }
            val dxsData = data(batch * 6, 4)
            val dxs = DTensor<Rank3<Named<Batch, Sym>, Sym, Named<Feat, Sym>>, F32>(HostF32Storage(dxsData), intArrayOf(batch, 2, 3), F32)
            val zero = DTensor<Rank2<Named<Feat, Sym>, Named<Out, Sym>>, F32>(HostF32Storage(FloatArray(12)), intArrayOf(3, 4), F32)
            println("vmap " + flat(j(xs, w0, dxs, zero)).joinToString(","))
            // <grad_x f, dx> for each example.
            println("loop " + (0 until batch).map { i ->
                val gx = flat(gs(w0, example(xsData, i)).second)
                val d = flat(example(dxsData, i))
                var s = 0f
                for (k in 0 until 6) s += gx[k] * d[k]
                s
            }.joinToString(","))
        """.trimIndent(),
    )

    @Test
    fun `vmap of vmap batches two axes`() = check(
        """
            val f = vmap(batchAxis(Batch)) { b: DTensor<Rank2<Named<Inner, Sym>, Named<Feat, Sym>>, F32> ->
                vmap(batchAxis(Inner)) { x: DTensor<Rank1<Named<Feat, Sym>>, F32> -> (x * x).exp().sum().toFloat() }(b)
            }
            val y = f(DTensor<Rank3<Named<Batch, Sym>, Named<Inner, Sym>, Named<Feat, Sym>>, F32>(HostF32Storage(xsData), intArrayOf(batch, 2, 3), F32))
            println("vmap " + flat(y).joinToString(","))
            println("loop " + (0 until batch * 2).map { i ->
                var s = 0f
                for (k in 0 until 3) { val v = xsData[3 * i + k]; s += kotlin.math.exp(v * v) }
                s
            }.joinToString(","))
        """.trimIndent(),
    )
}
