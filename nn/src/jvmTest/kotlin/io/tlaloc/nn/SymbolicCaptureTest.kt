package io.tlaloc.nn

import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.captureN
import io.tlaloc.core.BF16
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostBf16Storage
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.floatToBf16Bits
import io.tlaloc.core.hostF32
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.passes.DxirInterpreter
import java.nio.file.Path
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * A symbolic capture records the graph an eager one records, without computing values: the
 * committed `tiny_granite` checkpoint with LoRA adapters, captured both ways, gives the same loss
 * and gradients; a sliding-window causal attention gives the same output; bf16 parameters
 * trace as bf16 params; and a symbolic tape refuses to hand out a value.
 */
class SymbolicCaptureTest {
    private val dir: Path = Path.of("..", "ir", "src", "jvmTest", "resources", "io", "tlaloc", "ir", "inference", "tiny_granite")

    private fun i32(values: IntArray, vararg dims: Int) = DTensor<Shape, I32>(HostI32Storage(values), dims, I32)

    /** The fixture with rank-2 adapters on every linear layer, both A and B random (B = 0 would zero A's gradient). */
    private fun adapted(): CausalLM {
        val lm = HfCausalLm.load(dir)
        val m = Lora.apply(lm.model, LoraConfig(2, 4f, LoraConfig.ALL_LINEAR), RandomKey.fromSeed(5))
        val rnd = java.util.Random(7)
        return m.withParameters(
            Lora.adapterParameters(m).associate { p ->
                p.key to DTensor<Shape, F32>(HostF32Storage(FloatArray(p.tensor.size()) { (rnd.nextGaussian() * 0.1).toFloat() }), p.tensor.dims.copyOf(), F32)
            },
        )
    }

    private fun DTensor<*, *>.size() = dims.fold(1) { a, d -> a * d }

    private fun close(a: Float, b: Float, what: String) =
        assertTrue(abs(a - b) <= 1e-5f * max(1f, max(abs(a), abs(b))), "$what: eager $a, symbolic $b")

    @Test fun aLoraStepCapturedSymbolicallyMatchesTheEagerOne() {
        for (precision in listOf(Precision.F32, Precision.MIXED_BF16)) {
            val model = adapted()
            val n = 12
            val ids = i32(intArrayOf(3, 5, 9, 12, 33, 7, 40, 2, 18, 61, 4, 27), 1, n)
            val next = i32(intArrayOf(-1, -1, -1, 12, 33, 7, 40, -1, 18, 61, 4, -1), 1, n)
            val eager = captureCausalLm(model, ids, next, chunk = 4, frozen = Lora.frozen, precision = precision)
            val symbolic = captureCausalLm(model, ids, next, chunk = 4, frozen = Lora.frozen, precision = precision, symbolic = true)
            assertEquals(eager.parameterKeys, symbolic.parameterKeys)
            assertEquals(eager.frozenKeys, symbolic.frozenKeys)
            val a = eager.run(model, listOf(ids, next))
            val b = symbolic.run(model, listOf(ids, next))
            close(a.loss, b.loss, "$precision loss")
            for ((key, ga) in a.gradients) {
                val va = ga.hostF32()
                val vb = b.gradients.getValue(key).hostF32()
                assertEquals(va.size, vb.size, key)
                for (i in va.indices) close(va[i], vb[i], "$precision d$key[$i]")
            }
            assertTrue(a.gradients.values.any { g -> g.hostF32().any { it != 0f } }, "the gradients are not all zero")
        }
    }

    @Test fun theSymbolicGraphHoldsOneCopyOfEachConstantAndNoSquareMask() {
        val model = adapted()
        val n = 16
        val ids = i32(IntArray(n) { it + 1 }, 1, n)
        val symbolic = captureCausalLm(model, ids, ids, chunk = 8, frozen = Lora.frozen, symbolic = true)
        val consts = symbolic.primal.body.filterIsInstance<DxirConst>()
        assertTrue(consts.none { it.type.dims == listOf(n, n) }, "a [$n, $n] constant: ${consts.map { it.type.dims }}")
        val tables = consts.filter { it.type.dims == listOf(n, 8) }
        assertEquals(2, tables.size, "RoPE's cos and sin once each: ${consts.map { it.type.dims }}")
    }

    @Test fun aSlidingWindowMaskBuiltFromPositionsMatchesTheConstantOne() {
        val rnd = java.util.Random(11)
        fun w(r: Int, c: Int) = Dense(DTensor<Shape, F32>(HostF32Storage(FloatArray(r * c) { (rnd.nextGaussian() * 0.3).toFloat() }), intArrayOf(r, c), F32), null)
        val attn = MultiHeadAttention(w(8, 8), w(8, 8), w(8, 8), w(8, 8), numHeads = 2, numKvHeads = 2, causal = true, window = 3)
        val x = DTensor<Shape, F32>(HostF32Storage(FloatArray(2 * 7 * 8) { (rnd.nextGaussian()).toFloat() }), intArrayOf(2, 7, 8), F32)
        val params = attn.parameters
        fun captured(symbolic: Boolean) = captureN(listOf(x) + params.map { it.tensor }, "attn", symbolic) { leaves ->
            val byKey = params.withIndex().associate { (i, p) -> p.key to leaves[1 + i] }
            attn.forward(leaves[0], Params { byKey.getValue(it) })
        }
        val inputs = listOf(x.hostF32()) + params.map { it.tensor.hostF32() }
        val a = DxirInterpreter.evalFunction(captured(false), inputs)[0]
        val b = DxirInterpreter.evalFunction(captured(true), inputs)[0]
        assertEquals(a.size, b.size)
        for (i in a.indices) close(a[i], b[i], "out[$i]")
    }

    @Test fun bf16ParametersTraceAsBf16Params() {
        val model = adapted()
        @Suppress("UNCHECKED_CAST")
        val bf16 = model.withParameters(
            Lora.frozen.frozen(model).associate { p ->
                val v = p.tensor.hostF32()
                p.key to (DTensor<Shape, BF16>(HostBf16Storage(ShortArray(v.size) { floatToBf16Bits(v[it]) }), p.tensor.dims.copyOf(), BF16) as DTensor<*, F32>)
            },
        )
        val ids = i32(IntArray(8) { it + 1 }, 1, 8)
        val step = captureCausalLm(bf16, ids, ids, chunk = 4, frozen = Lora.frozen, precision = Precision.MIXED_BF16, symbolic = true)
        val frozenParams = step.primal.params.takeLast(step.frozenKeys.size)
        assertTrue(frozenParams.all { it.type.dtype == BF16 }, "frozen params: ${frozenParams.map { it.type.dtype.name }.distinct()}")
        assertTrue(step.primal.params.drop(step.inputCount).take(step.parameterKeys.size).all { it.type.dtype == F32 })
    }

    @Test fun aSymbolicTapeRefusesToHandOutAValue() {
        val x = DTensor<Shape, F32>(HostF32Storage(floatArrayOf(1f, 2f)), intArrayOf(2), F32)
        assertFailsWith<IllegalArgumentException> {
            captureN(listOf(x), "peek", symbolic = true) { leaves -> leaves[0].also { it.peek(0) } }
        }
    }
}
