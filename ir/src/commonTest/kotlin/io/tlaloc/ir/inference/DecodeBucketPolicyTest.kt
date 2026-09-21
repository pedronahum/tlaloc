package io.tlaloc.ir.inference

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * §0.4.467 — Phase H1c: the bucket-selection suite.
 *
 * Bucketing is the whole bridge between a dynamic serving loop and a static
 * compiled graph, so the interesting cases are the boundaries: a request that
 * lands EXACTLY on a ladder point must not be promoted to the next one (that
 * is a doubled executable for nothing), a request one past a point must be,
 * and a request past the cap must be REFUSED BY NAME rather than clamped.
 */
class DecodeBucketPolicyTest {

    private val policy = DecodeBucketPolicy(maxBatch = 8, maxContext = 256, blockSize = 16)

    // --- The ladders themselves. ----------------------------------------

    @Test
    fun theLaddersArePowersOfTwoEndingAtTheCaps() {
        assertEquals(listOf(1, 2, 4, 8), policy.batchLadder)
        assertEquals(listOf(16, 32, 64, 128, 256), policy.contextLadder)
        assertEquals(20, policy.executableCount)
        assertEquals(20, policy.allBuckets.size)
    }

    /** A cap that is not a power of two is still a ladder point: the
     *  scheduler's `max_num_seqs` is the steady-state shape. */
    @Test
    fun aNonPowerOfTwoCapIsAppendedToTheLadder() {
        val p = DecodeBucketPolicy(maxBatch = 5, maxContext = 100, blockSize = 16)
        assertEquals(listOf(1, 2, 4, 5), p.batchLadder)
        // 100 aligns up to 112 (7 pages of 16); the ladder stops doubling at 64.
        assertEquals(listOf(16, 32, 64, 112), p.contextLadder)
        assertEquals(112, p.maxContext)
    }

    /** Every context bucket is a whole number of pages, so `maxBlocksPerSeq`
     *  is exact rather than a rounding decision taken twice. */
    @Test
    fun everyContextBucketIsAWholeNumberOfPages() {
        for (bs in listOf(1, 8, 16, 32)) {
            val p = DecodeBucketPolicy(maxBatch = 4, maxContext = 300, blockSize = bs)
            for (c in p.contextLadder) {
                assertEquals(0, c % bs, "context bucket $c is not a multiple of blockSize $bs")
                assertEquals(c / bs, DecodeBucket(1, c).maxBlocksPerSeq(bs))
            }
        }
    }

    // --- Selection: on the boundary, and one past it. --------------------

    @Test
    fun aRequestExactlyOnABoundaryTakesThatBucketNotTheNextOne() {
        assertEquals(DecodeBucket(4, 64), policy.bucketFor(batch = 4, context = 64))
        assertEquals(DecodeBucket(1, 16), policy.bucketFor(batch = 1, context = 16))
        assertEquals(DecodeBucket(8, 256), policy.bucketFor(batch = 8, context = 256))
    }

    @Test
    fun aRequestOnePastABoundaryTakesTheNextBucket() {
        assertEquals(DecodeBucket(8, 128), policy.bucketFor(batch = 5, context = 65))
        assertEquals(DecodeBucket(2, 32), policy.bucketFor(batch = 2, context = 17))
    }

    @Test
    fun aRequestBelowTheSmallestContextBucketTakesTheSmallestOne() {
        assertEquals(DecodeBucket(1, 16), policy.bucketFor(batch = 1, context = 1))
    }

    /** Monotone and minimal: over the whole legal domain the chosen bucket
     *  covers the request, never shrinks as the request grows, and is the
     *  SMALLEST ladder point that covers it. */
    @Test
    fun selectionIsCoveringMonotoneAndMinimalOverTheWholeDomain() {
        var prevBatch = 0
        for (b in 1..policy.maxBatch) {
            val chosen = policy.bucketFor(b, 1).batch
            assertTrue(chosen >= b, "bucket $chosen does not cover batch $b")
            assertTrue(chosen >= prevBatch, "batch bucketing is not monotone at $b")
            assertTrue(
                policy.batchLadder.none { it in b until chosen },
                "bucket $chosen is not minimal for batch $b (ladder ${policy.batchLadder})",
            )
            prevBatch = chosen
        }
        var prevCtx = 0
        for (c in 1..policy.maxContext) {
            val chosen = policy.bucketFor(1, c).maxContext
            assertTrue(chosen >= c, "bucket $chosen does not cover context $c")
            assertTrue(chosen >= prevCtx, "context bucketing is not monotone at $c")
            assertTrue(
                policy.contextLadder.none { it in c until chosen },
                "bucket $chosen is not minimal for context $c",
            )
            prevCtx = chosen
        }
    }

    // --- Refusals, by name. ----------------------------------------------

    @Test
    fun aBatchOverTheCapIsRefusedByNameNotClamped() {
        val e = assertFailsWith<IllegalArgumentException> { policy.bucketFor(batch = 9, context = 16) }
        assertTrue("exceeds maxBatch 8" in (e.message ?: ""), "message was: ${e.message}")
        assertTrue("REFUSED" in (e.message ?: ""), "message was: ${e.message}")
    }

    @Test
    fun aContextOverTheCapIsRefusedByNameNotClamped() {
        val e = assertFailsWith<IllegalArgumentException> { policy.bucketFor(batch = 1, context = 257) }
        assertTrue("exceeds maxContext 256" in (e.message ?: ""), "message was: ${e.message}")
        assertTrue("truncate" in (e.message ?: ""), "message was: ${e.message}")
    }

    /** Zero context is refused with the reason it is refused: it is the one
     *  shape where PAGED_ATTENTION's interpreter and its emission disagree
     *  (zeros vs NaN), which is also why [DecodePadding.PADDING_SEQ_LEN] is 1. */
    @Test
    fun anEmptyStepOrAnEmptyContextIsRefusedByName() {
        assertTrue("scheduling bug" in refusalFor { policy.bucketFor(0, 16) })
        assertTrue("NaN" in refusalFor { policy.bucketFor(1, 0) })
    }

    @Test
    fun theNonThrowingSiblingIsNullExactlyWhereBucketForRefuses() {
        assertNull(policy.bucketForOrNull(9, 16))
        assertNull(policy.bucketForOrNull(1, 257))
        assertNull(policy.bucketForOrNull(0, 16))
        assertNull(policy.bucketForOrNull(1, 0))
        assertEquals(DecodeBucket(4, 64), policy.bucketForOrNull(3, 33))
    }

    // --- Explicit ladders get no discount on the invariants. -------------

    @Test
    fun anExplicitLadderIsCheckedExactlyLikeAGeneratedOne() {
        val p = DecodeBucketPolicy.withLadders(
            batchLadder = listOf(1, 3, 6),
            contextLadder = listOf(32, 96, 192),
            blockSize = 32,
        )
        assertEquals(DecodeBucket(3, 96), p.bucketFor(batch = 2, context = 40))
        assertEquals(9, p.executableCount)

        // Not page-aligned.
        assertTrue(
            "whole number of pages" in refusalFor {
                DecodeBucketPolicy.withLadders(listOf(1), listOf(30), blockSize = 16)
            },
        )
        // Not ascending.
        assertTrue(
            "ascending" in refusalFor {
                DecodeBucketPolicy.withLadders(listOf(4, 2), listOf(16), blockSize = 16)
            },
        )
    }

    private inline fun refusalFor(body: () -> Unit): String =
        assertFailsWith<IllegalArgumentException> { body() }.message ?: ""
}
