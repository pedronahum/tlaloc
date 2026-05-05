package io.tlaloc.ir.recognizer

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.282 — direct tests for [resolveLargestMatch], the overlap-
 * resolution helper inside [recognizeAll].
 *
 * Layer 3 v1 patterns don't overlap (FlashAttention's MATMUL/SOFTMAX/
 * MATMUL ops aren't separately recognized by RmsNorm/RoPE/CrossEntropy/
 * SwiGLU), so this resolver is a no-op for v1. It's plumbed for v2
 * patterns that contain other recognized patterns (e.g. a future
 * "TransformerBlock" that contains both FlashAttention and RmsNorm).
 * These tests pin the v2-ready overlap behavior at the unit level so
 * a regression doesn't slip in before any v2 pattern surfaces it.
 *
 * The synthetic matches use [RecognitionMatch.RmsNorm] and
 * [RecognitionMatch.Rope] as carriers — the resolver only inspects
 * `ops` and their ids, not pattern-specific metadata, so the choice
 * of subtype is incidental.
 */
class ResolveLargestMatchTest {

    private val t = DxirType(F32, listOf(4))

    /** Build a function with [n] sequential RELU ops we can slice into match groups. */
    private fun makeOps(n: Int): List<DxirOp> {
        val ops = mutableListOf<DxirOp>()
        DxirBuilder.function("f") {
            var x: io.tlaloc.ir.DxirNode = param("x", t)
            repeat(n) {
                val r = op(OpKind.RELU, listOf(x), t)
                ops += r
                x = r
            }
            listOf(x)
        }
        return ops
    }

    @Test
    fun emptyMatchListReturnsEmpty() {
        // Short-circuit: matches.size < 2.
        assertEquals(emptyList(), resolveLargestMatch(emptyList()))
    }

    @Test
    fun singleMatchPassesThroughUnchanged() {
        // Short-circuit: matches.size < 2 returns the input list as-is.
        val ops = makeOps(2)
        val m = RecognitionMatch.RmsNorm(ops = ops, input = ops[0], output = ops[1])
        assertEquals(listOf(m), resolveLargestMatch(listOf(m)))
    }

    @Test
    fun nonOverlappingMatchesAllKept() {
        // Two matches over disjoint op ids — both survive resolution.
        // Order in result follows the descending op-count sort
        // (RmsNorm has 2 ops, Rope has 2 ops; tie → input order).
        val ops = makeOps(4)
        val m1 = RecognitionMatch.RmsNorm(
            ops = listOf(ops[0], ops[1]),
            input = ops[0],
            output = ops[1],
        )
        val m2 = RecognitionMatch.Rope(
            ops = listOf(ops[2], ops[3]),
            input = ops[2],
            output = ops[3],
        )
        val result = resolveLargestMatch(listOf(m1, m2))
        assertEquals(2, result.size, "both non-overlapping matches should survive")
        assertTrue(m1 in result)
        assertTrue(m2 in result)
    }

    @Test
    fun overlappingMatchesLargerWins() {
        // Two matches sharing ops[1]. The larger (3 ops) wins; the
        // smaller (2 ops) is dropped because one of its ops was
        // already claimed by the larger.
        val ops = makeOps(4)
        val larger = RecognitionMatch.RmsNorm(
            ops = listOf(ops[0], ops[1], ops[2]),
            input = ops[0],
            output = ops[2],
        )
        val smaller = RecognitionMatch.Rope(
            ops = listOf(ops[1], ops[3]),
            input = ops[1],
            output = ops[3],
        )
        val result = resolveLargestMatch(listOf(larger, smaller))
        assertEquals(listOf(larger), result, "larger match wins; smaller dropped via overlap on ops[1]")
    }

    @Test
    fun equalSizeOverlappingMatchesPreservesInputOrder() {
        // Both matches have 2 ops and share ops[1]. Tiebreak: stable
        // sort preserves input order, so the FIRST match in the input
        // list claims its ops first; the second is dropped via overlap.
        // The recognizeAll call order (FlashAttention, RmsNorm, Rope,
        // CrossEntropy, SwiGLU) determines this for production calls.
        val ops = makeOps(3)
        val first = RecognitionMatch.RmsNorm(
            ops = listOf(ops[0], ops[1]),
            input = ops[0],
            output = ops[1],
        )
        val second = RecognitionMatch.Rope(
            ops = listOf(ops[1], ops[2]),
            input = ops[1],
            output = ops[2],
        )
        val result = resolveLargestMatch(listOf(first, second))
        assertEquals(listOf(first), result, "tie → first wins; second dropped via overlap on ops[1]")
    }
}
