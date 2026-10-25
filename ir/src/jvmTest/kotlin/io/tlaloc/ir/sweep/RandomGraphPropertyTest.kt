package io.tlaloc.ir.sweep

import io.kotest.property.Arb
import io.kotest.property.PropTestConfig
import io.kotest.property.arbitrary.bind
import io.kotest.property.arbitrary.int
import io.kotest.property.arbitrary.list
import io.kotest.property.checkAll
import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirForwardTransform
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.passes.DxirVmapTransform
import io.tlaloc.ir.render.KotlinSourceRenderer
import io.tlaloc.testing.interpret
import io.tlaloc.testing.scaledError
import kotlinx.coroutines.runBlocking
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test

/**
 * Random DXIR graphs over smooth elementwise, reduction and shape ops (kotest property
 * tests; a failure shrinks to the shortest graph that still fails, and prints its seed).
 * Each graph `y = g(a, b)`, `a` and `b` 2×3, is checked:
 *  - F32 against F64;
 *  - the reverse-mode gradient of `Σ y ⊙ w` against central differences (F64);
 *  - the forward-mode tangent against `⟨gradient, v⟩`;
 *  - vmap against the graph applied to each example;
 *  - its gradient renders as Kotlin source.
 */
@OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)
class RandomGraphPropertyTest {

    /** One step: an op chosen by [kind] over pool values chosen by [x] and [y]. */
    data class Step(val kind: Int, val x: Int, val y: Int)

    private val steps = Arb.list(Arb.bind(Arb.int(0, KINDS - 1), Arb.int(0, 99), Arb.int(0, 99)) { k, x, y -> Step(k, x, y) }, 1..8)

    private fun build(program: List<Step>, dt: DType, loss: Boolean): DxirFunction = DxirBuilder.function("g") {
        val t = DxirType(dt, listOf(2, 3))
        val pool = mutableListOf<DxirNode>(param("a", t), param("b", t))
        fun c(v: Double) = op(OpKind.BROADCAST, listOf(const(if (dt == F64) v else v.toFloat(), DxirType(dt, emptyList()))), t)
        for (s in program) {
            val x = pool[s.x % pool.size]
            val y = pool[s.y % pool.size]
            val one = c(1.0)
            val sq = op(OpKind.MUL, listOf(x, x), t)
            pool += when (s.kind) {
                0 -> op(OpKind.ADD, listOf(x, y), t)
                1 -> op(OpKind.SUB, listOf(x, y), t)
                2 -> op(OpKind.MUL, listOf(x, y), t)
                3 -> op(OpKind.DIV, listOf(x, op(OpKind.ADD, listOf(one, op(OpKind.MUL, listOf(y, y), t)), t)), t)
                4 -> op(OpKind.TANH, listOf(x), t)
                5 -> op(OpKind.SIGMOID, listOf(x), t)
                6 -> op(OpKind.SIN, listOf(x), t)
                7 -> op(OpKind.COS, listOf(x), t)
                8 -> op(OpKind.ATAN, listOf(x), t)
                9 -> op(OpKind.SQRT, listOf(op(OpKind.ADD, listOf(sq, one), t)), t)
                10 -> op(OpKind.LOG1P, listOf(sq), t)
                11 -> op(OpKind.EXPM1, listOf(op(OpKind.TANH, listOf(x), t)), t)
                12 -> op(OpKind.SOFTMAX, listOf(x), t, attrs = mapOf("axis" to 1))
                13 -> op(OpKind.LOG_SOFTMAX, listOf(x), t, attrs = mapOf("axis" to 0))
                14 -> {
                    val r = op(OpKind.SUM, listOf(x), DxirType(dt, listOf(2, 1)), attrs = mapOf("reduction_dims" to listOf(1)))
                    op(OpKind.MUL, listOf(op(OpKind.BROADCAST_LIKE, listOf(r, x), t), y), t)
                }
                15 -> {
                    val tr = op(OpKind.TRANSPOSE, listOf(x), DxirType(dt, listOf(3, 2)), attrs = mapOf("permutation" to listOf(1, 0)))
                    op(OpKind.RESHAPE, listOf(tr), t)
                }
                16 -> op(OpKind.REVERSE, listOf(x), t, attrs = mapOf("dimensions" to listOf(1)))
                17 -> {
                    val cat = op(OpKind.CONCAT, listOf(x, y), DxirType(dt, listOf(4, 3)), attrs = mapOf("dimension" to 0))
                    op(
                        OpKind.SLICE, listOf(cat), t,
                        attrs = mapOf("start_indices" to listOf(1, 0), "limit_indices" to listOf(3, 3), "strides" to listOf(1, 1)),
                    )
                }
                18 -> {
                    val m = op(OpKind.MATMUL, listOf(x, op(OpKind.TRANSPOSE, listOf(y), DxirType(dt, listOf(3, 2)), attrs = mapOf("permutation" to listOf(1, 0)))), DxirType(dt, listOf(2, 2)))
                    op(OpKind.MATMUL, listOf(op(OpKind.TANH, listOf(m), m.type), x), t)
                }
                19 -> op(OpKind.POW, listOf(op(OpKind.ADD, listOf(sq, one), t), op(OpKind.TANH, listOf(y), t)), t)
                else -> error("kind ${s.kind}")
            }
        }
        val out = pool.last()
        if (!loss) return@function listOf(out)
        val w = param("w", t)
        listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(out, w), t)), DxirType(dt, emptyList())))
    }

    private fun inputs(seed: Int): List<DoubleArray> {
        val r = Random(seed)
        return List(3) { DoubleArray(6) { (r.nextDouble() * 2 - 1).toFloat().toDouble() } }
    }

    // Unit, not checkAll's result: JUnit runs only @Test methods that return void.
    private fun property(check: (List<Step>) -> Unit) {
        runBlocking { checkAll(PropTestConfig(iterations = 150, seed = 20261025), steps) { program -> check(program) } }
    }

    @Test
    fun f32AgreesWithF64() = property { p ->
        val x = inputs(1).take(2)
        val want = interpret(build(p, F64, false), F64, x).single()
        val got = interpret(build(p, F32, false), F32, x).single()
        val err = scaledError(got, want)
        check(err < 1e-4) { "F32 against F64: $err for $p" }
    }

    @Test
    fun reverseModeMatchesCentralDifferences() = property { p ->
        val x = inputs(2)
        val loss = build(p, F64, true)
        val grads = interpret(DxirReverseTransform.apply(loss, inputOnlyTrailingParams = 1), F64, x)
        for (ai in 0..1) {
            val fd = DoubleArray(6) { i ->
                val h = 1e-6 * maxOf(1.0, abs(x[ai][i]))
                fun at(v: Double) = interpret(loss, F64, x.toMutableList().also { it[ai] = x[ai].copyOf().also { a -> a[i] = v } }).single()[0]
                (at(x[ai][i] + h) - at(x[ai][i] - h)) / (2 * h)
            }
            val err = scaledError(grads[ai], fd)
            check(err < 1e-5) { "d input $ai: $err against central differences for $p" }
        }
    }

    @Test
    fun forwardModeMatchesReverseMode() = property { p ->
        val x = inputs(3)
        val loss = build(p, F64, true)
        val grads = interpret(DxirReverseTransform.apply(loss, inputOnlyTrailingParams = 1), F64, x)
        val v = inputs(4).take(2) + listOf(DoubleArray(6))
        val tangent = interpret(DxirForwardTransform.apply(loss), F64, x + v)[1][0]
        val want = (0..1).sumOf { ai -> (0 until 6).sumOf { grads[ai][it] * v[ai][it] } }
        check(abs(tangent - want) <= 1e-9 * maxOf(1.0, abs(want))) { "tangent $tangent against $want for $p" }
    }

    @Test
    fun vmapMatchesEachExample() = property { p ->
        val fn = build(p, F64, false)
        val ex = (5..7).map { inputs(it).take(2) }
        val batched = DxirVmapTransform.apply(fn, listOf(true, true), 3)
        val got = interpret(batched, F64, (0..1).map { i -> ex.flatMap { it[i].asIterable() }.toDoubleArray() }).single()
        val want = ex.flatMap { interpret(fn, F64, it).single().asIterable() }.toDoubleArray()
        check(scaledError(got, want) < 1e-12) { "vmap against per example for $p" }
    }

    @Test
    fun theGradientRendersAsKotlin() = property { p ->
        val loss = build(p, F32, true)
        KotlinSourceRenderer.render(loss)
        KotlinSourceRenderer.render(DxirReverseTransform.apply(loss, inputOnlyTrailingParams = 1))
    }

    private companion object {
        const val KINDS = 20
    }
}
