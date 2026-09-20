package io.tlaloc.autograd

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Rank1
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * §0.4.415 — Phase B5: the [checkCustomVjp] debug oracle (the ratified
 * "ship the JVP⇄VJP helper" decision of docs/CUSTOM_DERIVATIVES_DESIGN.md
 * §6.3) is PURE HOST MATH — no plugin, no IR — so it certifies here in
 * `:autograd`'s own suite: green for a mathematically correct user adjoint,
 * red for the deliberately NON-mathematical `3·upstream` one (which is also
 * exactly how a straight-through estimator is EXPECTED to fail the check —
 * the reason the oracle is opt-in, never a default gate). Plus the
 * [customVjp] host-stub asymmetry: unlike `grad`'s pluginMissing stub, the
 * returned function simply applies `f`.
 */
class CustomVjpOracleTest {

    private fun square(t: DTensor<Rank1<Sym>, F32>): DTensor<Rank1<Sym>, F32> {
        val d = t.hostF32()
        return DTensor(HostF32Storage(FloatArray(d.size) { d[it] * d[it] }), t.dims.copyOf(), F32)
    }

    /** The TRUE adjoint of the elementwise square: d_t = 2·t ⊙ upstream. */
    private fun squareVjp(
        upstream: DTensor<Rank1<Sym>, F32>,
        t: DTensor<Rank1<Sym>, F32>,
    ): DTensor<Rank1<Sym>, F32> {
        val u = upstream.hostF32()
        val x = t.hostF32()
        return DTensor(HostF32Storage(FloatArray(x.size) { 2f * x[it] * u[it] }), t.dims.copyOf(), F32)
    }

    /** The deliberately NON-mathematical adjoint: d_t = 3·upstream. */
    private fun threeVjp(
        upstream: DTensor<Rank1<Sym>, F32>,
        t: DTensor<Rank1<Sym>, F32>,
    ): DTensor<Rank1<Sym>, F32> {
        val u = upstream.hostF32()
        return DTensor(HostF32Storage(FloatArray(u.size) { 3f * u[it] }), t.dims.copyOf(), F32)
    }

    private val at = Tensors.f32Vector<Sym>(floatArrayOf(0.7f, -1.3f, 2.1f))

    @Test
    fun oracleIsGreenForACorrectUserAdjoint() {
        val check = checkCustomVjp(::square, ::squareVjp, at)
        assertTrue(
            check.passed,
            "correct vjpFn must satisfy the JVP⇄VJP identity; got $check",
        )
    }

    @Test
    fun oracleIsRedForTheNonMathematicalAdjoint() {
        val check = checkCustomVjp(::square, ::threeVjp, at)
        assertFalse(
            check.passed,
            "3·upstream diverges from the primal's math and MUST fail the identity; got $check",
        )
    }

    @Test
    fun hostStubAppliesThePrimalInsteadOfThrowing() {
        // The documented asymmetry vs `grad`'s pluginMissing stub: outside a
        // differentiated context the derivative attachment doesn't exist, but
        // applying the primal is the right plain-Kotlin meaning of the call.
        val f = customVjp<Float, Float>({ t -> t * t }, { u, _ -> u * 3f })
        assertEquals(25f, f(5f), "customVjp's host stub must apply f")
        val f2 = customVjp2<Float, Float, Float>({ a, b -> a * b }, { u, _, _ -> Pair(u, u) })
        assertEquals(12f, f2(3f, 4f), "customVjp2's host stub must apply f")
    }
}
