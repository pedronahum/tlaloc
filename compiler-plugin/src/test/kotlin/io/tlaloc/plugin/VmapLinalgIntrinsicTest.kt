package io.tlaloc.plugin

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * `vmap` over the SPD linear algebra through the K2 plugin: `logDetSpd`, `solveSpd`
 * against a captured right-hand side, and per-example gradients of `logDetSpd`, each
 * against a loop over the examples, at F32 and F64 and batch sizes 1, 7 and 64.
 * The loop calls the `:core` host functions, which compute in Double and round once, so
 * the comparison allows 1e-5 of the largest magnitude at F32 (1e-12 at F64).
 */
class VmapLinalgIntrinsicTest {

    private fun run(src: String, precision: Precision): F64TestHarness.Result {
        val user = if (precision == Precision.F64) F64Source.of(src) else src
        val r = F64TestHarness.compileAndRun(user)
        assertEquals(0, r.exitCode, "compile/run failed:\n${r.describe()}")
        assertTrue(r.messages.none { "kept original call" in it.message }, "kept original call:\n${r.describe()}")
        return r
    }

    private fun program(batch: Int, body: String): String = """
        @file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)
        import io.tlaloc.autograd.*
        import io.tlaloc.core.*
        import io.tlaloc.core.ops.*

        typealias M = DTensor<Rank2<Sym, Sym>, F32>

        fun data(n: Int, seed: Int): FloatArray {
            var s = seed.toLong() * 7919L + 12345L
            return FloatArray(n) {
                s = (s * 1103515245L + 12345L) and 0x7fffffffL
                ((s % 20000L) - 10000L) / 7000.0f
            }
        }
        /** [count] symmetric positive-definite 3 x 3 matrices, row-major, back to back. */
        fun spd(count: Int): FloatArray {
            val raw = data(count * 9, 1)
            val out = FloatArray(count * 9)
            for (e in 0 until count) for (i in 0 until 3) for (j in 0 until 3) {
                var s = 0f
                for (q in 0 until 3) s += raw[e * 9 + i * 3 + q] * raw[e * 9 + j * 3 + q]
                out[e * 9 + i * 3 + j] = s + if (i == j) 3f else 0f
            }
            return out
        }
        @Suppress("UNCHECKED_CAST")
        fun flat(v: Any?): List<Float> = when (v) {
            is DTensor<*, *> -> (v as DTensor<*, F32>).hostF32().asList()
            is Float -> listOf(v)
            else -> error("unexpected value " + v)
        }
        fun example(a: FloatArray, i: Int): M =
            DTensor<Rank2<Sym, Sym>, F32>(HostF32Storage(a.copyOfRange(9 * i, 9 * i + 9)), intArrayOf(3, 3), F32)
        fun main() {
            val batch = $batch
            val aData = spd(batch)
            val a = DTensor<Rank3<Named<Batch, Sym>, Sym, Sym>, F32>(HostF32Storage(aData), intArrayOf(batch, 3, 3), F32)
            $body
        }
    """.trimIndent()

    private fun check(body: String) {
        for (p in listOf(Precision.F32, Precision.F64)) for (b in listOf(1, 7, 64)) {
            val r = run(program(b, body), p)
            val x = r.values("vmap")
            val y = r.values("loop")
            assertEquals(y.size, x.size, r.stdout)
            val tol = if (p == Precision.F64) 1e-12 else 1e-5
            val scale = maxOf(1.0, y.maxOf { kotlin.math.abs(it) })
            for (i in x.indices) {
                assertTrue(kotlin.math.abs(x[i] - y[i]) <= tol * scale, "$p batch $b [$i]: vmap ${x[i]}, loop ${y[i]}\n${r.stdout}")
            }
        }
    }

    @Test
    fun `logDetSpd per example`() = check(
        """
            val f = vmap(batchAxis(Batch)) { m: M -> m.logDetSpd() }
            println("vmap " + flat(f(a)).joinToString(","))
            println("loop " + (0 until batch).flatMap { i -> flat(example(aData, i).logDetSpd()) }.joinToString(","))
        """.trimIndent(),
    )

    @Test
    fun `solveSpd per example against a captured right-hand side`() = check(
        """
            val rhs = DTensor<Rank2<Sym, Sym>, F32>(HostF32Storage(data(6, 2)), intArrayOf(3, 2), F32)
            val f = vmap(batchAxis(Batch)) { m: M -> m.solveSpd(rhs) }
            println("vmap " + flat(f(a)).joinToString(","))
            println("loop " + (0 until batch).flatMap { i -> flat(example(aData, i).solveSpd(rhs)) }.joinToString(","))
        """.trimIndent(),
    )

    @Test
    fun `per-example gradients of logDetSpd are the inverses`() = check(
        """
            val g = vmap(batchAxis(Batch)) { m: M -> grad { k: M -> k.logDetSpd().toFloat() }(m) }
            println("vmap " + flat(g(a)).joinToString(","))
            println("loop " + (0 until batch).flatMap { i -> flat(example(aData, i).invSpd()) }.joinToString(","))
        """.trimIndent(),
    )

    @Test
    fun `solve and det per example`() = check(
        """
            val rhs = DTensor<Rank2<Sym, Sym>, F32>(HostF32Storage(data(6, 2)), intArrayOf(3, 2), F32)
            val f = vmap(batchAxis(Batch)) { m: M -> m.solve(rhs) * m.det() }
            println("vmap " + flat(f(a)).joinToString(","))
            println("loop " + (0 until batch).flatMap { i -> flat(example(aData, i).solve(rhs) * example(aData, i).det()) }.joinToString(","))
        """.trimIndent(),
    )

    @Test
    fun `per-example gradients of det`() = check(
        """
            val g = vmap(batchAxis(Batch)) { m: M -> grad { k: M -> k.det().toFloat() }(m) }
            val one = grad { k: M -> k.det().toFloat() }
            println("vmap " + flat(g(a)).joinToString(","))
            println("loop " + (0 until batch).flatMap { i -> flat(one(example(aData, i))) }.joinToString(","))
        """.trimIndent(),
    )

    @Test
    fun `qr and eigh per example`() = check(
        """
            val f = vmap(batchAxis(Batch)) { m: M -> m.qrR() + m.eighVectors() }
            val g = vmap(batchAxis(Batch)) { m: M -> m.eighValues() }
            println("vmap " + (flat(f(a)) + flat(g(a))).joinToString(","))
            println("loop " + ((0 until batch).flatMap { i -> flat(example(aData, i).qrR() + example(aData, i).eighVectors()) } +
                (0 until batch).flatMap { i -> flat(example(aData, i).eighValues()) }).joinToString(","))
        """.trimIndent(),
    )

    @Test
    fun `per-example gradients of a weighted eigenvalue sum`() = check(
        """
            val wts = Tensors.f32Vector<Sym>(floatArrayOf(1f, -2f, 0.5f))
            val g = vmap(batchAxis(Batch)) { m: M -> grad { k: M -> (k.eighValues() * wts).sum().toFloat() }(m) }
            val one = grad2 { k: M, w: DTensor<Rank1<Sym>, F32> -> (k.eighValues() * w).sum().toFloat() }
            println("vmap " + flat(g(a)).joinToString(","))
            println("loop " + (0 until batch).flatMap { i -> flat(one(example(aData, i), wts).first) }.joinToString(","))
        """.trimIndent(),
    )
}
