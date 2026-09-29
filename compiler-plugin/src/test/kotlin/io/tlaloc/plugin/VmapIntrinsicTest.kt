package io.tlaloc.plugin

import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * `vmap` through the K2 plugin: each vmapped lambda compiles, and its result equals a
 * loop over the examples calling the same body as a plain local Kotlin function (no
 * plugin involved), at F32 and F64 and at batch sizes 1, 7 and 64. Equality is exact:
 * the batched host code runs each example's arithmetic in the order the per-example
 * code does.
 */
class VmapIntrinsicTest {

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

    private fun assertSame(r: F64TestHarness.Result, a: String, b: String, tolerance: Double = 0.0) {
        val x = r.values(a)
        val y = r.values(b)
        assertEquals(y.size, x.size, "$a and $b sizes\n${r.stdout}")
        val scale = maxOf(1.0, y.maxOfOrNull { kotlin.math.abs(it) } ?: 0.0)
        for (i in x.indices) {
            assertTrue(
                x[i] == y[i] || (x[i].isNaN() && y[i].isNaN()) || kotlin.math.abs(x[i] - y[i]) <= tolerance * scale,
                "$a[$i] = ${x[i]} but $b[$i] = ${y[i]}\n${r.stdout}",
            )
        }
    }

    /**
     * One vmap case: the lambda `{ x: [xType] -> [body] }` vmapped over `Batch` and applied to
     * a batch of examples of extents [dims], against the local function with the same body
     * applied to each example. [setup] declares captured values.
     */
    private fun case(
        batch: Int,
        xType: String,
        dims: List<Int>,
        body: String,
        setup: String = "",
        axis: String = "batchAxis(Batch)",
        batchAtom: String = "Sym",
    ): String {
        val n = dims.fold(1) { a, d -> a * d }
        val atoms = Regex("""^DTensor<(ScalarShape|Rank\d<(.*)>), F32>$""").find(xType)
            ?: error("unexpected xType $xType")
        val rank = dims.size
        val inner = atoms.groupValues[2]
        val batchedShape = if (rank == 0) "Rank1<Named<Batch, $batchAtom>>"
        else "Rank${rank + 1}<Named<Batch, $batchAtom>, $inner>"
        return """
            @file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)
            import io.tlaloc.autograd.*
            import io.tlaloc.core.*
            import io.tlaloc.core.ops.*

            object Feat : IndexName { override val name = "feat" }
            object Out : IndexName { override val name = "out" }
            object MaxBatch : DimBound(64)

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
            fun main() {
                val batch = $batch
                $setup
                fun perExample(x: $xType) = run { $body }
                val f = vmap($axis) { x: $xType -> $body }
                val xs = data(batch * $n, 1)
                val y = f(DTensor<$batchedShape, F32>(HostF32Storage(xs), intArrayOf(batch, ${dims.joinToString()}), F32))
                println("vmap " + flat(y).joinToString(","))
                val loop = (0 until batch).flatMap { i ->
                    flat(perExample(DTensor<${xType.removePrefix("DTensor<").substringBeforeLast(", F32>")}, F32>(
                        HostF32Storage(xs.copyOfRange($n * i, $n * i + $n)), intArrayOf(${dims.joinToString()}), F32,
                    )))
                }
                println("loop " + loop.joinToString(","))
            }
        """.trimIndent()
    }

    private fun check(
        xType: String,
        dims: List<Int>,
        body: String,
        setup: String = "",
        axis: String = "batchAxis(Batch)",
        batchAtom: String = "Sym",
        batches: List<Int> = listOf(1, 7, 64),
        tolerance: Double = 0.0,
    ) {
        for (p in listOf(Precision.F32, Precision.F64)) for (b in batches) {
            val r = run(case(b, xType, dims, body, setup, axis, batchAtom), p)
            assertSame(r, "vmap", "loop", tolerance)
        }
    }

    private val vec = "DTensor<Rank1<Named<Feat, Sym>>, F32>"
    private val mat = "DTensor<Rank2<Sym, Named<Feat, Sym>>, F32>"

    @Test
    fun `elementwise lambda over a rank-1 example`() =
        check(vec, listOf(3), "(x * x + 1f).tanh() * x.exp()")

    @Test
    fun `a per-example sum returned as a Float`() =
        check(vec, listOf(5), "(x * x).sum().toFloat()")

    @Test
    fun `a per-example scalar tensor`() =
        check(vec, listOf(4), "x.exp().sum() * 2f")

    @Test
    fun `reductions over one axis and softmax of a rank-2 example`() =
        check(mat, listOf(3, 4), "(x.softmax() * x).sum(0).mean().toFloat()")

    @Test
    fun `mean of a rank-2 example after a transpose`() =
        check(mat, listOf(2, 3), "(x.transpose() * 3f).max().toFloat()")

    @Test
    fun `flatten`() =
        check(mat, listOf(2, 3), "(x.flatten() * 2f).sum().toFloat()")

    @Test
    fun `a matmul with a captured weight`() = check(
        mat, listOf(2, 3), "(x matmul w).tanh()",
        setup = """
            val w = DTensor<Rank2<Named<Feat, Sym>, Named<Out, Sym>>, F32>(HostF32Storage(data(12, 5)), intArrayOf(3, 4), F32)
        """.trimIndent(),
    )

    @Test
    fun `mean squared error against a captured target`() = check(
        vec, listOf(6), "((x - t) * (x - t)).mean().toFloat()",
        setup = """
            val t = DTensor<Rank1<Named<Feat, Sym>>, F32>(HostF32Storage(data(6, 9)), intArrayOf(6), F32)
        """.trimIndent(),
    )

    /** The loop calls the host `crossEntropyLoss`, which sums in another order than the ops it lowers to. */
    @Test
    fun `cross-entropy of logits against a captured one-hot row`() = check(
        mat, listOf(1, 4), "crossEntropyLoss(x, oneHot).toFloat()",
        setup = """
            val oneHot = DTensor<Rank2<Sym, Named<Feat, Sym>>, F32>(HostF32Storage(floatArrayOf(0f, 0f, 1f, 0f)), intArrayOf(1, 4), F32)
        """.trimIndent(),
        tolerance = 1e-6,
    )

    @Test
    fun `a bounded batch axis at several sizes`() = check(
        vec, listOf(3), "(x * x).sum().toFloat()",
        axis = "batchAxis(Batch, MaxBatch)",
        batchAtom = "Bounded<MaxBatch>",
        batches = listOf(1, 13, 64),
    )

    /** A `vmap2` program: [call] is the vmapped lambda, [perExample] the same body as a local fun. */
    private fun program2(batch: Int, call: String, perExample: String, apply: String, loop: String): String = """
        @file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)
        import io.tlaloc.autograd.*
        import io.tlaloc.core.*
        import io.tlaloc.core.ops.*

        object Feat : IndexName { override val name = "feat" }
        object Out : IndexName { override val name = "out" }

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
        fun main() {
            val batch = $batch
            $perExample
            val f = $call
            val y = $apply
            println("vmap " + flat(y).joinToString(","))
            println("loop " + (0 until batch).flatMap { i -> $loop }.joinToString(","))
        }
    """.trimIndent()

    private fun check2(call: String, perExample: String, apply: String, loop: String) {
        for (p in listOf(Precision.F32, Precision.F64)) for (b in listOf(1, 7, 64)) {
            assertSame(run(program2(b, call, perExample, apply, loop), p), "vmap", "loop")
        }
    }

    private val a23 = "DTensor<Rank2<Sym, Named<Feat, Sym>>, F32>"
    private val w34 = "DTensor<Rank2<Named<Feat, Sym>, Named<Out, Sym>>, F32>"

    @Test
    fun `vmap2 with both arguments batched`() = check2(
        call = "vmap2(batchAxis(Batch), Batched, Batched) { a: $a23, w: $w34 -> (a matmul w).relu() }",
        perExample = "fun g(a: $a23, w: $w34) = (a matmul w).relu()",
        apply = """
            f(
                DTensor<Rank3<Named<Batch, Sym>, Sym, Named<Feat, Sym>>, F32>(HostF32Storage(data(batch * 6, 1)), intArrayOf(batch, 2, 3), F32),
                DTensor<Rank3<Named<Batch, Sym>, Named<Feat, Sym>, Named<Out, Sym>>, F32>(HostF32Storage(data(batch * 12, 2)), intArrayOf(batch, 3, 4), F32),
            )
        """.trimIndent(),
        loop = """
            flat(g(
                DTensor<Rank2<Sym, Named<Feat, Sym>>, F32>(HostF32Storage(data(batch * 6, 1).copyOfRange(6 * i, 6 * i + 6)), intArrayOf(2, 3), F32),
                DTensor<Rank2<Named<Feat, Sym>, Named<Out, Sym>>, F32>(HostF32Storage(data(batch * 12, 2).copyOfRange(12 * i, 12 * i + 12)), intArrayOf(3, 4), F32),
            ))
        """.trimIndent(),
    )

    @Test
    fun `vmap2 with the second argument broadcast`() = check2(
        call = "vmap2(batchAxis(Batch), Batched, Broadcast) { a: $a23, w: $w34 -> (a matmul w).sum().toFloat() }",
        perExample = "fun g(a: $a23, w: $w34) = (a matmul w).sum().toFloat()",
        apply = """
            f(
                DTensor<Rank3<Named<Batch, Sym>, Sym, Named<Feat, Sym>>, F32>(HostF32Storage(data(batch * 6, 1)), intArrayOf(batch, 2, 3), F32),
                DTensor<Rank2<Named<Feat, Sym>, Named<Out, Sym>>, F32>(HostF32Storage(data(12, 2)), intArrayOf(3, 4), F32),
            )
        """.trimIndent(),
        loop = """
            flat(g(
                DTensor<Rank2<Sym, Named<Feat, Sym>>, F32>(HostF32Storage(data(batch * 6, 1).copyOfRange(6 * i, 6 * i + 6)), intArrayOf(2, 3), F32),
                DTensor<Rank2<Named<Feat, Sym>, Named<Out, Sym>>, F32>(HostF32Storage(data(12, 2)), intArrayOf(3, 4), F32),
            ))
        """.trimIndent(),
    )

    @Test
    fun `vmap2 with the first argument broadcast`() = check2(
        call = "vmap2(batchAxis(Batch), Broadcast, Batched) { a: $a23, w: $w34 -> (a matmul w) - 1f }",
        perExample = "fun g(a: $a23, w: $w34) = (a matmul w) - 1f",
        apply = """
            f(
                DTensor<Rank2<Sym, Named<Feat, Sym>>, F32>(HostF32Storage(data(6, 1)), intArrayOf(2, 3), F32),
                DTensor<Rank3<Named<Batch, Sym>, Named<Feat, Sym>, Named<Out, Sym>>, F32>(HostF32Storage(data(batch * 12, 2)), intArrayOf(batch, 3, 4), F32),
            )
        """.trimIndent(),
        loop = """
            flat(g(
                DTensor<Rank2<Sym, Named<Feat, Sym>>, F32>(HostF32Storage(data(6, 1)), intArrayOf(2, 3), F32),
                DTensor<Rank2<Named<Feat, Sym>, Named<Out, Sym>>, F32>(HostF32Storage(data(batch * 12, 2).copyOfRange(12 * i, 12 * i + 12)), intArrayOf(3, 4), F32),
            ))
        """.trimIndent(),
    )

    @Test
    fun `a Float example and a rank-erased output`() {
        for (p in listOf(Precision.F32, Precision.F64)) for (b in listOf(1, 7, 64)) {
            val src = """
                @file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)
                import io.tlaloc.autograd.*
                import io.tlaloc.core.*
                import io.tlaloc.core.ops.*
                fun main() {
                    val batch = $b
                    val xs = FloatArray(batch) { 0.37f * it - 1.1f }
                    val m = FloatArray(batch * 6) { 0.21f * it - 2.3f }
                    // A Float per example: the batch is a rank-1 tensor.
                    val f = vmap(batchAxis(Batch)) { x: Float -> x * x * 3f + x }
                    val y: DTensor<Rank1<Named<Batch, Sym>>, F32> = f(Tensors.f32Vector<Named<Batch, Sym>>(xs))
                    println("vmapA " + y.hostF32().joinToString(","))
                    println("loopA " + xs.joinToString(",") { (it * it * 3f + it).toString() })
                    // sum(0) erases the rank; so does the batched result.
                    val g = vmap(batchAxis(Batch)) { x: DTensor<Rank2<Sym, Sym>, F32> -> x.sum(0) }
                    val z: DTensor<Shape, F32> = g(DTensor<Rank3<Named<Batch, Sym>, Sym, Sym>, F32>(HostF32Storage(m), intArrayOf(batch, 2, 3), F32))
                    println("vmapB " + z.hostF32().joinToString(",") + " dims " + z.dims.toList())
                    println("loopB " + (0 until batch).flatMap { i ->
                        DTensor<Rank2<Sym, Sym>, F32>(HostF32Storage(m.copyOfRange(6 * i, 6 * i + 6)), intArrayOf(2, 3), F32).sum(0).hostF32().asList()
                    }.joinToString(","))
                }
            """.trimIndent()
            val r = run(src, p)
            assertSame(r, "vmapA", "loopA")
            val vb = r.stdout.lines().first { it.startsWith("vmapB ") }
            assertTrue(vb.endsWith("dims [$b, 3]"), vb)
            val got = vb.removePrefix("vmapB ").substringBefore(" dims").split(",").map { it.toDouble() }
            val want = r.values("loopB").toList()
            assertEquals(want, got, r.stdout)
        }
    }

    @Test
    fun `vmap2 refuses two batches of different sizes at run time`() {
        val src = """
            @file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)
            import io.tlaloc.autograd.*
            import io.tlaloc.core.*
            import io.tlaloc.core.ops.*
            fun main() {
                val f = vmap2(batchAxis(Batch), Batched, Batched) { a: DTensor<Rank1<Sym>, F32>, b: DTensor<Rank1<Sym>, F32> -> a * b }
                val one = Tensors.f32Matrix<Named<Batch, Sym>, Sym>(1, 3, floatArrayOf(1f, 2f, 3f))
                val four = Tensors.f32Matrix<Named<Batch, Sym>, Sym>(4, 3, FloatArray(12) { it.toFloat() })
                println("same " + f(four, four).hostF32().size)
                println("mixed " + f(one, four).hostF32().size)
            }
        """.trimIndent()
        val r = F64TestHarness.compileAndRun(src)
        assertEquals("same 12", r.stdout.trim().lines().first(), r.describe())
        assertTrue(r.messages.any { "batch sizes 1 and 4" in it.message }, r.describe())
    }

    @Test
    fun `a nested jvp with a captured Int`() {
        for (p in listOf(Precision.F32, Precision.F64)) {
            val src = """
                @file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)
                import io.tlaloc.autograd.*
                import io.tlaloc.core.*
                import io.tlaloc.core.ops.*
                fun main() {
                    val reps = System.getProperty("no.such.property")?.length ?: 3
                    val w0 = Tensors.f32Vector<Sym>(floatArrayOf(0.5f, -1f))
                    val dw = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f))
                    val f = vmap(batchAxis(Batch)) { x: DTensor<Rank1<Sym>, F32> ->
                        jvp { w: DTensor<Rank1<Sym>, F32> -> (x * w * w).sum() * reps.toFloat() }(w0, dw)
                    }
                    val xs = Tensors.f32Matrix<Named<Batch, Sym>, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
                    println("vmap " + f(xs).hostF32().joinToString(","))
                    // d/dw [3 Σ x w²] · dw = 3 Σ 2 x w dw
                    println("want " + listOf(0, 1).joinToString(",") { i ->
                        (3f * (2f * xs.hostF32()[2 * i] * 0.5f * 1f + 2f * xs.hostF32()[2 * i + 1] * -1f * 2f)).toString()
                    })
                }
            """.trimIndent()
            val r = run(src, p)
            assertSame(r, "vmap", "want")
        }
    }
}
