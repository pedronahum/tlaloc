package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.normalFloats
import io.tlaloc.core.uniformFloats
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.408 — Phase D1: the RNG_UNIFORM / RNG_NORMAL IR arms. The interpreter
 * arms must be BIT-EXACT against the `:core/Random.kt` host kernels (they
 * call the same functions — this test is the pin that keeps it that way).
 *
 * §0.4.413 — Phase D2 v1: both differentiation transforms treat a draw as a
 * CONSTANT of its literal key/dims attrs (D1's loud refusals, flipped —
 * their spirit survives as pinned zeros). Reverse: zero gradient in the key,
 * pinned with the draw ON the differentiable path so the walk provably
 * reaches it with accumulated upstream and contributes nothing. Forward:
 * structural-zero tangent, same placement. On top of those, the
 * reparameterization certs: for loss = Σ (loc + scale ⊙ ε)² with
 * ε = normal(key, dims), d loc = 2(loc + scale⊙ε) and
 * d scale = 2(loc + scale⊙ε) ⊙ ε against host-side `normalFloats` at the
 * SAME key as the analytic oracle; the gradient evaluated twice is
 * bit-identical (the cloned RNG op re-draws deterministically from its
 * baked key — the reparameterization contract); and the JVP⇄VJP
 * cross-identity holds with ε's zero tangent in the chain.
 */
class DxirRngTest {

    private val scalar = DxirType(F32, emptyList())
    private val key = RandomKey(7, 42)

    private fun rngAttrs(dims: List<Int>): Map<String, Any> =
        mapOf("key0" to key.k0, "key1" to key.k1, "dims" to dims)

    @Test
    fun interpreterUniformBitExactVsHostKernel() {
        val r2 = DxirType(F32, listOf(2, 3))
        val fn = DxirBuilder.function("rng_uniform") {
            listOf(op(OpKind.RNG_UNIFORM, emptyList(), r2, attrs = rngAttrs(listOf(2, 3))))
        }
        val out = DxirInterpreter.evalFunction(fn, emptyList())
        assertContentEquals(uniformFloats(key, 6), out[0], "interpreter must be bit-exact vs host")
    }

    @Test
    fun interpreterNormalBitExactVsHostKernel() {
        // Odd size on purpose: exercises the end-pad counter lane through the
        // interpreter path too.
        val r1 = DxirType(F32, listOf(5))
        val fn = DxirBuilder.function("rng_normal") {
            listOf(op(OpKind.RNG_NORMAL, emptyList(), r1, attrs = rngAttrs(listOf(5))))
        }
        val out = DxirInterpreter.evalFunction(fn, emptyList())
        assertContentEquals(normalFloats(key, 5), out[0], "interpreter must be bit-exact vs host")
    }

    @Test
    fun interpreterRejectsDimsAttrDisagreeingWithType() {
        val r1 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("rng_bad_dims") {
            listOf(op(OpKind.RNG_UNIFORM, emptyList(), r1, attrs = rngAttrs(listOf(5))))
        }
        val ex = assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(fn, emptyList())
        }
        assertTrue(
            ex.message!!.contains("disagrees with result type dims"),
            "unexpected message: ${ex.message}",
        )
    }

    @Test
    fun reverseGradientInKeyIsZero() {
        // §0.4.413 (flips §0.4.408's refusal): loss = Σ (uniform ⊙ x). The
        // draw sits ON the differentiable path, so the reverse walk reaches
        // it with an accumulated upstream (= x, via MulRule) — and must
        // contribute NOTHING: a draw is piecewise-constant in its key. The
        // pin of the zero is d x = u bit-exact against the host kernel at the
        // same key: MulRule's other arm reads the draw's value, so the RNG op
        // is cloned into the gradient body with its literal attrs intact and
        // re-draws the SAME stream.
        val r1 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("rng_in_grad") {
            val x = param("x", r1)
            val u = op(OpKind.RNG_UNIFORM, emptyList(), r1, attrs = rngAttrs(listOf(4)))
            val p = op(OpKind.MUL, listOf(u, x), r1)
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val x = floatArrayOf(0.5f, -1f, 2f, 0.25f)
        val out = DxirInterpreter.evalFunction(grad, listOf(x))
        assertContentEquals(
            uniformFloats(key, 4), out[0],
            "d x must be the draw itself (bit-exact re-draw from the baked key)",
        )
    }

    @Test
    fun forwardTangentOfDrawIsStructurallyZero() {
        // §0.4.413 (flips §0.4.408's refusal): jvp of Σ (normal ⊙ x) with
        // tangent vx. The draw's tangent is the structural zero (the lazy-null
        // convention), so the product rule's surviving term is Σ ε ⊙ vx —
        // pinned against the host kernel at the same key.
        val r1 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("rng_in_jvp") {
            val x = param("x", r1)
            val z = op(OpKind.RNG_NORMAL, emptyList(), r1, attrs = rngAttrs(listOf(4)))
            val p = op(OpKind.MUL, listOf(z, x), r1)
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val jvp = DxirForwardTransform.apply(fn)
        val x = floatArrayOf(0.5f, -1f, 2f, 0.25f)
        val vx = floatArrayOf(0.11f, -0.23f, 0.37f, -0.41f)
        val out = DxirInterpreter.evalFunction(jvp, listOf(x, vx))
        val eps = normalFloats(key, 4)
        var expected = 0f
        for (i in 0 until 4) expected += eps[i] * vx[i]
        val tangent = out[1].single()
        assertTrue(
            abs(expected - tangent) < 1e-6f,
            "tangent must be Σ ε⊙vx (ε's own tangent structurally zero): $expected vs $tangent",
        )
    }

    /** loss = Σ (loc + scale ⊙ ε)², ε = normal(key, [5]) — odd length on
     * purpose (the end-pad counter lane rides through the gradient body's
     * cloned draw too). */
    private fun reparamFn(): DxirFunction {
        val r5 = DxirType(F32, listOf(5))
        return DxirBuilder.function("reparam") {
            val loc = param("loc", r5)
            val scale = param("scale", r5)
            val eps = op(OpKind.RNG_NORMAL, emptyList(), r5, attrs = rngAttrs(listOf(5)))
            val se = op(OpKind.MUL, listOf(scale, eps), r5)
            val s = op(OpKind.ADD, listOf(loc, se), r5)
            val sq = op(OpKind.MUL, listOf(s, s), r5)
            listOf(op(OpKind.SUM, listOf(sq), scalar))
        }
    }

    @Test
    fun reparameterizedGradientMatchesAnalyticOracle() {
        // d loc = 2(loc + scale⊙ε); d scale = 2(loc + scale⊙ε) ⊙ ε — the
        // oracle recomputes ε host-side with the SAME key (`normalFloats`,
        // the same kernel the interpreter's cloned RNG op calls).
        val grad = DxirReverseTransform.apply(reparamFn())
        val loc = floatArrayOf(0.3f, -1.2f, 0.9f, -0.4f, 0.7f)
        val scale = floatArrayOf(1.5f, -0.8f, 0.2f, -0.6f, 0.4f)
        val out = DxirInterpreter.evalFunction(grad, listOf(loc, scale))
        val eps = normalFloats(key, 5)
        for (i in 0 until 5) {
            val s = loc[i] + scale[i] * eps[i]
            assertTrue(
                abs(out[0][i] - 2f * s) < 1e-5f,
                "d loc[$i]: ${out[0][i]} vs analytic ${2f * s}",
            )
            assertTrue(
                abs(out[1][i] - 2f * s * eps[i]) < 1e-5f,
                "d scale[$i]: ${out[1][i]} vs analytic ${2f * s * eps[i]}",
            )
        }
    }

    @Test
    fun reparameterizedGradientIsDeterministic() {
        // Same key → same ε → bit-identical gradients across evaluations:
        // the cloned RNG op's literal attrs make the re-draw deterministic,
        // which is exactly the reparameterization contract.
        val grad = DxirReverseTransform.apply(reparamFn())
        val loc = floatArrayOf(0.3f, -1.2f, 0.9f, -0.4f, 0.7f)
        val scale = floatArrayOf(1.5f, -0.8f, 0.2f, -0.6f, 0.4f)
        val first = DxirInterpreter.evalFunction(grad, listOf(loc, scale))
        val second = DxirInterpreter.evalFunction(grad, listOf(loc, scale))
        assertContentEquals(first[0], second[0], "d loc must be bit-identical across runs")
        assertContentEquals(first[1], second[1], "d scale must be bit-identical across runs")
    }

    // --- §0.4.432 — the runtime-key operand form: RNG_UNIFORM/RNG_NORMAL
    // with two scalar-I32 key operands (dims still a literal attr). The
    // interpreter reads the key words at execution time; the pin everywhere
    // is bit-equality with the attr form / the host kernel at the SAME key. ---

    private val keyT = DxirType(I32, emptyList())

    @Test
    fun operandFormDrawsMatchAttrFormBitExact() {
        // Keys arrive as scalar I32 PARAMS — genuinely runtime values the
        // op cannot see at build time. Same key → the SAME stream as the
        // attr form, bit-for-bit (both call `:core/Random.kt`).
        val r6 = DxirType(F32, listOf(6))
        val ufn = DxirBuilder.function("rng_rtk_u") {
            val k0 = param("k0", keyT)
            val k1 = param("k1", keyT)
            listOf(op(OpKind.RNG_UNIFORM, listOf(k0, k1), r6, attrs = mapOf("dims" to listOf(6))))
        }
        val uout = DxirInterpreter.evalFunction(ufn, listOf(floatArrayOf(7f), floatArrayOf(42f)))
        assertContentEquals(uniformFloats(key, 6), uout[0], "operand-form uniform must match host/attr stream")

        // Odd length: the end-pad counter lane rides the operand form too.
        val r5 = DxirType(F32, listOf(5))
        val nfn = DxirBuilder.function("rng_rtk_n") {
            val k0 = param("k0", keyT)
            val k1 = param("k1", keyT)
            listOf(op(OpKind.RNG_NORMAL, listOf(k0, k1), r5, attrs = mapOf("dims" to listOf(5))))
        }
        val nout = DxirInterpreter.evalFunction(nfn, listOf(floatArrayOf(7f), floatArrayOf(42f)))
        assertContentEquals(normalFloats(key, 5), nout[0], "operand-form normal must match host/attr stream")
    }

    @Test
    fun operandFormConstKeysCarryHighBitsExactly() {
        // An Int-carrying DxirConst key operand is read VERBATIM — no float
        // round-trip — so high-bit key words survive the interpreter exactly
        // (the FloatArray value domain would round anything beyond 2^24).
        val hk = RandomKey(0x7FFFFFFF, -0x1235ABCD)
        val r4 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("rng_rtk_hb") {
            val k0 = const(hk.k0, keyT)
            val k1 = const(hk.k1, keyT)
            listOf(op(OpKind.RNG_UNIFORM, listOf(k0, k1), r4, attrs = mapOf("dims" to listOf(4))))
        }
        val out = DxirInterpreter.evalFunction(fn, emptyList())
        assertContentEquals(uniformFloats(hk, 4), out[0], "high-bit const key words must be exact")
    }

    @Test
    fun interpreterRefusesKeyWordsOutsideF32Domain() {
        // A non-const runtime key rides the interpreter's F32 value domain,
        // which carries integers exactly only STRICTLY below 2^24 (2^24 + 1
        // rounds INTO 2^24, so the inclusive bound would silently corrupt a
        // key). Beyond it: a loud named refusal, not a wrong stream.
        val r2 = DxirType(F32, listOf(2))
        val fn = DxirBuilder.function("rng_rtk_bad") {
            val k0 = param("k0", keyT)
            val k1 = param("k1", keyT)
            listOf(op(OpKind.RNG_UNIFORM, listOf(k0, k1), r2, attrs = mapOf("dims" to listOf(2))))
        }
        val ex = assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(fn, listOf(floatArrayOf((1 shl 24).toFloat()), floatArrayOf(1f)))
        }
        assertTrue(
            ex.message!!.contains("F32 value domain"),
            "unexpected message: ${ex.message}",
        )
    }

    @Test
    fun reverseGradientThroughOperandFormKeys() {
        // loss = Σ (u(k0, k1) ⊙ x) with the keys as scalar I32 params. The
        // reverse walk must (a) contribute NOTHING through the draw (keys
        // are integers — RngDrawRule's empty list), (b) hand the integer key
        // params their §0.4.54 typed-zero gradients, and (c) re-draw the
        // SAME stream in the gradient body: the cloned draw's key OPERANDS
        // ride the usedByAdjoint transitive walk, so d x = u bit-exact.
        val r4 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("rng_rtk_grad") {
            val x = param("x", r4)
            val k0 = param("k0", keyT)
            val k1 = param("k1", keyT)
            val u = op(OpKind.RNG_UNIFORM, listOf(k0, k1), r4, attrs = mapOf("dims" to listOf(4)))
            val p = op(OpKind.MUL, listOf(u, x), r4)
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val x = floatArrayOf(0.5f, -1f, 2f, 0.25f)
        val out = DxirInterpreter.evalFunction(grad, listOf(x, floatArrayOf(7f), floatArrayOf(42f)))
        assertContentEquals(
            uniformFloats(key, 4), out[0],
            "d x must be the draw re-drawn from the CLONED key operands (bit-exact)",
        )
        assertContentEquals(floatArrayOf(0f), out[1], "integer key param takes the typed-zero gradient")
        assertContentEquals(floatArrayOf(0f), out[2], "integer key param takes the typed-zero gradient")
    }

    @Test
    fun forwardTangentThroughOperandFormIsZero() {
        // jvp of Σ (z(k0, k1) ⊙ x): the draw's tangent stays the structural
        // zero in the operand form too, so the surviving term is Σ ε ⊙ vx.
        val r4 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("rng_rtk_jvp") {
            val x = param("x", r4)
            val k0 = const(key.k0, keyT)
            val k1 = const(key.k1, keyT)
            val z = op(OpKind.RNG_NORMAL, listOf(k0, k1), r4, attrs = mapOf("dims" to listOf(4)))
            val p = op(OpKind.MUL, listOf(z, x), r4)
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val jvp = DxirForwardTransform.apply(fn)
        val x = floatArrayOf(0.5f, -1f, 2f, 0.25f)
        val vx = floatArrayOf(0.11f, -0.23f, 0.37f, -0.41f)
        val out = DxirInterpreter.evalFunction(jvp, listOf(x, vx))
        val eps = normalFloats(key, 4)
        var expected = 0f
        for (i in 0 until 4) expected += eps[i] * vx[i]
        val tangent = out[1].single()
        assertTrue(
            abs(expected - tangent) < 1e-6f,
            "operand-form tangent must be Σ ε⊙vx: $expected vs $tangent",
        )
    }

    @Test
    fun reparameterizedJvpVjpCrossIdentity() {
        // ⟨∇f, v⟩ == forward tangent through the same loss, with ε's zero
        // tangent in the chain on both sides.
        val fn = reparamFn()
        val loc = floatArrayOf(0.3f, -1.2f, 0.9f, -0.4f, 0.7f)
        val scale = floatArrayOf(1.5f, -0.8f, 0.2f, -0.6f, 0.4f)
        val vl = floatArrayOf(0.11f, -0.23f, 0.37f, -0.41f, 0.19f)
        val vs = floatArrayOf(-0.29f, 0.31f, 0.13f, -0.17f, 0.23f)
        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(loc, scale))
        var dot = 0.0
        for (i in 0 until 5) dot += grads[0][i].toDouble() * vl[i] + grads[1][i].toDouble() * vs[i]
        val jvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(loc, scale, vl, vs),
        )
        val tangent = jvp[1].single().toDouble()
        assertTrue(
            abs(dot - tangent) < 1e-4,
            "JVP⇄VJP cross-identity broken through a draw: ⟨grad,v⟩=$dot vs tangent=$tangent",
        )
    }
}
