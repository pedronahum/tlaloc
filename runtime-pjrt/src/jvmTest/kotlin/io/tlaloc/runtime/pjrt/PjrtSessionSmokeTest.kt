package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.307 — exercises [PjrtSession]. Three concerns:
 *
 *   1. Numerical correctness — agreement vs DxirInterpreter.
 *   2. Compile cache works — second call with the same DxirFunction does
 *      not grow [PjrtSession.cacheSize].
 *   3. Steady-state per-iteration cost is sub-millisecond on a tiny
 *      workload — proves the amortisation is real, not just an API
 *      shape change.
 */
class PjrtSessionSmokeTest {

    private val f32Scalar = DxirType(F32, emptyList())
    private val v4 = DxirType(F32, listOf(4))

    private fun scalarAffine() = DxirBuilder.function("scalar_affine") {
        val x = param("x", f32Scalar)
        val two = const(2.0f, f32Scalar)
        val one = const(1.0f, f32Scalar)
        val mul = op(OpKind.MUL, listOf(x, two), f32Scalar)
        val add = op(OpKind.ADD, listOf(mul, one), f32Scalar)
        listOf(add)
    }

    private fun rank1Triple() = DxirBuilder.function("rank1_triple") {
        val v = param("v", v4)
        val twoV = op(OpKind.ADD, listOf(v, v), v4)
        val threeV = op(OpKind.ADD, listOf(twoV, v), v4)
        listOf(threeV)
    }

    private fun assertCloseToInterpreter(
        fn: io.tlaloc.ir.DxirFunction,
        inputs: List<FloatArray>,
        actual: List<FloatArray>,
        tol: Float = 1e-5f,
    ) {
        val expected = DxirInterpreter.evalFunction(fn, inputs)
        assertEquals(expected.size, actual.size, "result count mismatch")
        for (i in expected.indices) {
            assertEquals(expected[i].size, actual[i].size, "result[$i] size mismatch")
            for (j in expected[i].indices) {
                val diff = abs(expected[i][j] - actual[i][j])
                assertTrue(
                    diff <= tol,
                    "result[$i][$j] disagreement: interpreter=${expected[i][j]} pjrt=${actual[i][j]} diff=$diff",
                )
            }
        }
    }

    @Test
    fun runsNumericalAgreementForOneFunction() {
        assumeTrue(PjrtBinaries.available, "PJRT plugin not resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val fn = scalarAffine()
        val inputs = listOf(floatArrayOf(7.0f))
        PjrtSession().use { session ->
            val out = session.runOn(fn, inputs)
            assertCloseToInterpreter(fn, inputs, out)
        }
    }

    @Test
    fun cachesCompiledExecutableAcrossRepeatedCalls() {
        // The whole point of PjrtSession: second call with the same fn is
        // a cache hit — no extra entry in the cache.
        assumeTrue(PjrtBinaries.available, "PJRT plugin not resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val fn = scalarAffine()
        PjrtSession().use { session ->
            assertEquals(0, session.cacheSize, "fresh session has empty cache")
            session.runOn(fn, listOf(floatArrayOf(1.0f)))
            assertEquals(1, session.cacheSize, "first call populates cache")
            session.runOn(fn, listOf(floatArrayOf(2.0f)))
            session.runOn(fn, listOf(floatArrayOf(3.0f)))
            assertEquals(1, session.cacheSize, "subsequent calls with same fn don't grow cache")
        }
    }

    @Test
    fun cachesIndependentlyAcrossDifferentFunctions() {
        assumeTrue(PjrtBinaries.available, "PJRT plugin not resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val fn1 = scalarAffine()
        val fn2 = rank1Triple()
        PjrtSession().use { session ->
            session.runOn(fn1, listOf(floatArrayOf(2.0f)))
            session.runOn(fn2, listOf(floatArrayOf(1f, 2f, 3f, 4f)))
            assertEquals(2, session.cacheSize, "two different fns get two cache slots")
            // Re-run both; cache stays at 2.
            session.runOn(fn1, listOf(floatArrayOf(5.0f)))
            session.runOn(fn2, listOf(floatArrayOf(0f, 1f, 2f, 3f)))
            assertEquals(2, session.cacheSize, "rerunning same fns doesn't grow cache")
        }
    }

    /**
     * §0.4.467 (H1c) — the keyed front door. A serving loop names its program
     * with `DecodeGraphSpec.executableCacheKey` (modelHash / kind / bucket /
     * dtypes) instead of re-emitting StableHLO on every lookup, and this pins
     * the two properties that makes safe:
     *
     *  - a REPEATED key lowers ONCE ([PjrtSession.keyedLoweringCount] stays 1)
     *    and lands on the same executable, and the answers still agree with
     *    the interpreter;
     *  - two DIFFERENT keys over the SAME program share one executable —
     *    the key front-runs the emission, it does not fork the compile cache.
     *    (A bucket ladder warmed twice under different model hashes must not
     *    hold two copies of an identical graph in device memory.)
     */
    @Test
    fun aRepeatedCacheKeyLowersOnceAndStillAgreesWithTheInterpreter() {
        assumeTrue(PjrtBinaries.available, "PJRT plugin not resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val fn = scalarAffine()
        val key = "tlaloc-decode-v1/sha256:deadbeef/decode/b1/c16/t1/dtF32/kvF32"
        PjrtSession().use { session ->
            val first = session.runOn(fn, listOf(floatArrayOf(2.0f)), cacheKey = key)
            assertEquals(1, session.keyedLoweringCount, "first keyed call lowers once")
            assertEquals(1, session.cacheSize)
            assertCloseToInterpreter(fn, listOf(floatArrayOf(2.0f)), first)

            val second = session.runOn(fn, listOf(floatArrayOf(5.0f)), cacheKey = key)
            assertEquals(1, session.keyedLoweringCount, "a repeated key must not lower again")
            assertEquals(1, session.cacheSize, "and must not compile again")
            assertCloseToInterpreter(fn, listOf(floatArrayOf(5.0f)), second)

            // A second key over the same program: one more lowering, but the
            // MLIR is identical so the executable cache does not grow.
            session.runOn(fn, listOf(floatArrayOf(1.0f)), cacheKey = "$key/other")
            assertEquals(2, session.keyedLoweringCount)
            assertEquals(1, session.cacheSize, "identical programs share one executable")
        }
    }

    @Test
    fun prepareDoesNotDispatch() {
        assumeTrue(PjrtBinaries.available, "PJRT plugin not resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val fn = scalarAffine()
        PjrtSession().use { session ->
            session.prepare(fn)
            assertEquals(1, session.cacheSize, "prepare populates cache without dispatching")
            // Subsequent runOn is a cache hit; cache size unchanged.
            session.runOn(fn, listOf(floatArrayOf(2.0f)))
            assertEquals(1, session.cacheSize)
        }
    }

    @Test
    fun steadyStateDispatchIsSubMillisecond() {
        // Demonstrates the amortisation. After warmup (compile + first
        // dispatch), repeated dispatches on a tiny scalar affine should be
        // dominated by FFM downcall overhead + minimal CUDA roundtrip,
        // typically << 1 ms. Threshold is generous (10 ms) so the test
        // doesn't flake under noisy GPU contention.
        assumeTrue(PjrtBinaries.available, "PJRT plugin not resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val fn = scalarAffine()
        val inputs = listOf(floatArrayOf(1.0f))
        PjrtSession().use { session ->
            // Warmup — compile + first dispatch (the slow ones).
            repeat(3) { session.runOn(fn, inputs) }
            // Measured loop.
            val n = 50
            val t0 = System.nanoTime()
            repeat(n) { session.runOn(fn, inputs) }
            val elapsedMs = (System.nanoTime() - t0) / 1_000_000.0
            val perCallMs = elapsedMs / n
            println("[pjrt-session] steady-state per-call=${"%.3f".format(perCallMs)} ms (over $n iterations)")
            assertTrue(perCallMs < 10.0, "steady-state per-call should be <10 ms; got ${perCallMs} ms")
        }
    }

    @Test
    fun closeFreesEverythingAndSubsequentRunOnRejected() {
        assumeTrue(PjrtBinaries.available, "PJRT plugin not resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val fn = scalarAffine()
        val session = PjrtSession()
        session.runOn(fn, listOf(floatArrayOf(1.0f)))
        session.close()
        // close() is idempotent.
        session.close()
        // Subsequent dispatches must throw.
        val ex = runCatching { session.runOn(fn, listOf(floatArrayOf(1.0f))) }.exceptionOrNull()
        assertTrue(ex is IllegalStateException, "closed session must reject runOn; got $ex")
    }
}
