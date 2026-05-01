package io.tlaloc.ir.recognizer

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Layer 3 §0.4.251+ — RMS norm / RoPE / cross-entropy recognizer tests.
 *
 * Each pattern: 1 positive match + 4 adversarial cases + diagnostic
 * verification. Mirrors `FlashAttentionRecognizerTest`'s shape.
 */
class RmsNormRopeCrossEntropyTest {

    private val rank2 = DxirType(F32, listOf(8, 64))
    private val rank2Reduced = DxirType(F32, listOf(8, 1))   // mean over last axis
    private val scalar = DxirType(F32, listOf())

    // -------------------------------------------------------------------------
    // RMS norm
    // -------------------------------------------------------------------------

    @Test
    fun rmsNormPositiveMatch() {
        val fn = DxirBuilder.function("rms_norm") {
            val x = param("x", rank2)
            val sq = op(OpKind.MUL, listOf(x, x), rank2)
            val mean = op(OpKind.MEAN, listOf(sq), rank2Reduced)
            val rsq = op(OpKind.RSQRT, listOf(mean), rank2Reduced)
            val out = op(OpKind.MUL, listOf(x, rsq), rank2)
            listOf(out)
        }
        val matches = recognizeRmsNorm(fn)
        assertEquals(1, matches.size, "expected 1 RMS-norm match")
        assertEquals(4, matches.single().ops.size, "MUL→MEAN→RSQRT→MUL = 4 ops")
    }

    @Test
    fun rmsNormPositiveMatchWithEpsilon() {
        // The "with-epsilon" variant: ADD(MEAN, eps_const) before RSQRT.
        val fn = DxirBuilder.function("rms_norm_eps") {
            val x = param("x", rank2)
            val eps = const(1e-6f, rank2Reduced)
            val sq = op(OpKind.MUL, listOf(x, x), rank2)
            val mean = op(OpKind.MEAN, listOf(sq), rank2Reduced)
            val stabilised = op(OpKind.ADD, listOf(mean, eps), rank2Reduced)
            val rsq = op(OpKind.RSQRT, listOf(stabilised), rank2Reduced)
            val out = op(OpKind.MUL, listOf(x, rsq), rank2)
            listOf(out)
        }
        val matches = recognizeRmsNorm(fn)
        assertEquals(1, matches.size)
        assertEquals(5, matches.single().ops.size, "MUL→MEAN→ADD→RSQRT→MUL = 5 ops with eps")
    }

    @Test
    fun rmsNormAdversarial1RsqrtOperandNotMean() {
        // RSQRT of a generic tensor (not MEAN(x²)) — likely a different
        // norm form (e.g. inverse-variance from batch norm).
        val fn = DxirBuilder.function("rsqrt_of_param") {
            val x = param("x", rank2Reduced)
            val r = op(OpKind.RSQRT, listOf(x), rank2Reduced)
            listOf(r)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeRmsNorm(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size)
        assertTrue("expected MEAN" in diag.single().reason)
    }

    @Test
    fun rmsNormAdversarial2MeanOperandNotMul() {
        // RSQRT(MEAN(x)) where the MEAN's operand is the bare x — a
        // standard mean, not a squared-mean.
        val fn = DxirBuilder.function("rsqrt_mean_x") {
            val x = param("x", rank2)
            val mean = op(OpKind.MEAN, listOf(x), rank2Reduced)
            val r = op(OpKind.RSQRT, listOf(mean), rank2Reduced)
            listOf(r)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeRmsNorm(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size)
        assertTrue("squared step" in diag.single().reason)
    }

    @Test
    fun rmsNormAdversarial3MulOperandsDistinct() {
        // MUL(x, y) where x != y — pairwise product, not square.
        val fn = DxirBuilder.function("rsqrt_pairwise") {
            val x = param("x", rank2)
            val y = param("y", rank2)
            val pairwise = op(OpKind.MUL, listOf(x, y), rank2)
            val mean = op(OpKind.MEAN, listOf(pairwise), rank2Reduced)
            val r = op(OpKind.RSQRT, listOf(mean), rank2Reduced)
            listOf(r)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeRmsNorm(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size)
        assertTrue("aren't the same tensor" in diag.single().reason)
    }

    @Test
    fun rmsNormAdversarial4NoFinalMul() {
        // Chain ends at RSQRT — no final MUL with the original x.
        val fn = DxirBuilder.function("no_final_mul") {
            val x = param("x", rank2)
            val sq = op(OpKind.MUL, listOf(x, x), rank2)
            val mean = op(OpKind.MEAN, listOf(sq), rank2Reduced)
            val r = op(OpKind.RSQRT, listOf(mean), rank2Reduced)
            listOf(r)  // RSQRT result IS the function output; no final MUL.
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeRmsNorm(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size)
        assertTrue("no MUL consumer" in diag.single().reason)
    }

    // -------------------------------------------------------------------------
    // RoPE
    // -------------------------------------------------------------------------

    @Test
    fun ropePositiveMatch() {
        val fn = DxirBuilder.function("rope") {
            val theta = param("theta", rank2)
            val xReal = param("x_real", rank2)
            val xImag = param("x_imag", rank2)
            val cosT = op(OpKind.COS, listOf(theta), rank2)
            val sinT = op(OpKind.SIN, listOf(theta), rank2)
            val a = op(OpKind.MUL, listOf(xReal, cosT), rank2)
            val b = op(OpKind.MUL, listOf(xImag, sinT), rank2)
            val out = op(OpKind.SUB, listOf(a, b), rank2)
            listOf(out)
        }
        val matches = recognizeRope(fn)
        assertEquals(1, matches.size)
    }

    @Test
    fun ropeAdversarial1SinNoCos() {
        val fn = DxirBuilder.function("sin_alone") {
            val theta = param("theta", rank2)
            val sinT = op(OpKind.SIN, listOf(theta), rank2)
            listOf(sinT)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeRope(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size)
        assertTrue("no COS" in diag.single().reason)
    }

    @Test
    fun ropeAdversarial2SinNoMulConsumer() {
        // SIN + COS exist but neither feeds a MUL.
        val fn = DxirBuilder.function("sin_cos_no_mul") {
            val theta = param("theta", rank2)
            val cosT = op(OpKind.COS, listOf(theta), rank2)
            val sinT = op(OpKind.SIN, listOf(theta), rank2)
            val combined = op(OpKind.ADD, listOf(sinT, cosT), rank2)
            listOf(combined)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeRope(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size)
        assertTrue("no MUL consumer" in diag.single().reason)
    }

    @Test
    fun ropeAdversarial3CosFeedsSameMulAsSin() {
        // SIN and COS both feed the *same* MUL (not the paired-MULs
        // structure RoPE needs). v1 walks sinMul.id; cosMul filter
        // requires id != sinMul.id, so this rejects.
        val fn = DxirBuilder.function("shared_mul") {
            val theta = param("theta", rank2)
            val x = param("x", rank2)
            val cosT = op(OpKind.COS, listOf(theta), rank2)
            val sinT = op(OpKind.SIN, listOf(theta), rank2)
            val sumOfSc = op(OpKind.ADD, listOf(sinT, cosT), rank2)
            val combined = op(OpKind.MUL, listOf(x, sumOfSc), rank2)
            listOf(combined)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeRope(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size)
        // SIN has no direct MUL consumer; the diagnostic explains.
        assertTrue("SIN has no MUL consumer" in diag.single().reason)
    }

    @Test
    fun ropeAdversarial4NoCommonAddSubConsumer() {
        // SIN-MUL and COS-MUL exist but go to different destinations
        // (no shared ADD/SUB recombination).
        val fn = DxirBuilder.function("no_recombine") {
            val theta = param("theta", rank2)
            val xR = param("x_real", rank2)
            val xI = param("x_imag", rank2)
            val cosT = op(OpKind.COS, listOf(theta), rank2)
            val sinT = op(OpKind.SIN, listOf(theta), rank2)
            val a = op(OpKind.MUL, listOf(xR, cosT), rank2)
            val b = op(OpKind.MUL, listOf(xI, sinT), rank2)
            // No ADD/SUB recombines a and b — return both separately.
            listOf(a, b)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeRope(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size)
        assertTrue("share an ADD/SUB" in diag.single().reason)
    }

    // -------------------------------------------------------------------------
    // Cross-entropy
    // -------------------------------------------------------------------------

    @Test
    fun crossEntropyPositiveMatch() {
        val fn = DxirBuilder.function("nll") {
            val logits = param("logits", rank2)
            val labels = param("labels", rank2)
            val probs = op(OpKind.SOFTMAX, listOf(logits), rank2)
            val logp = op(OpKind.LOG, listOf(probs), rank2)
            val pointwise = op(OpKind.MUL, listOf(labels, logp), rank2)
            val loss = op(OpKind.SUM, listOf(pointwise), scalar)
            listOf(loss)
        }
        val matches = recognizeCrossEntropy(fn)
        assertEquals(1, matches.size)
        assertEquals(4, matches.single().ops.size, "SOFTMAX→LOG→MUL→SUM = 4 ops")
    }

    @Test
    fun crossEntropyAdversarial1LogOperandNotSoftmax() {
        val fn = DxirBuilder.function("log_of_relu") {
            val x = param("x", rank2)
            val activated = op(OpKind.RELU, listOf(x), rank2)
            val logp = op(OpKind.LOG, listOf(activated), rank2)
            listOf(logp)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeCrossEntropy(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size)
        assertTrue("expected SOFTMAX" in diag.single().reason)
    }

    @Test
    fun crossEntropyAdversarial2NoMulConsumer() {
        // SOFTMAX → LOG with no MUL consumer (just log-probs returned).
        val fn = DxirBuilder.function("log_probs_only") {
            val logits = param("logits", rank2)
            val probs = op(OpKind.SOFTMAX, listOf(logits), rank2)
            val logp = op(OpKind.LOG, listOf(probs), rank2)
            listOf(logp)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeCrossEntropy(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size)
        assertTrue("no MUL consumer" in diag.single().reason)
    }

    @Test
    fun crossEntropyAdversarial3NoSumConsumer() {
        // The labels · log_probs MUL exists but isn't reduced.
        val fn = DxirBuilder.function("unreduced") {
            val logits = param("logits", rank2)
            val labels = param("labels", rank2)
            val probs = op(OpKind.SOFTMAX, listOf(logits), rank2)
            val logp = op(OpKind.LOG, listOf(probs), rank2)
            val pointwise = op(OpKind.MUL, listOf(labels, logp), rank2)
            listOf(pointwise)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeCrossEntropy(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size)
        assertTrue("isn't reduced by SUM" in diag.single().reason)
    }

    @Test
    fun crossEntropyAdversarial4LogWithoutSoftmax() {
        // Bare LOG of a parameter — no SOFTMAX precursor.
        val fn = DxirBuilder.function("log_of_param") {
            val x = param("x", rank2)
            val logp = op(OpKind.LOG, listOf(x), rank2)
            listOf(logp)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeCrossEntropy(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size)
        assertTrue("non-op" in diag.single().reason || "expected SOFTMAX" in diag.single().reason)
    }

    // -------------------------------------------------------------------------
    // Aggregator + cross-recognizer
    // -------------------------------------------------------------------------

    @Test
    fun aggregatorMatchesAllPatternsInOneFunction() {
        // A function with all four patterns wired up — recognizeAll
        // should return matches for each.
        val fn = DxirBuilder.function("everything") {
            val x = param("x", rank2)
            // RMS norm
            val sq = op(OpKind.MUL, listOf(x, x), rank2)
            val mean = op(OpKind.MEAN, listOf(sq), rank2Reduced)
            val rsq = op(OpKind.RSQRT, listOf(mean), rank2Reduced)
            val xNorm = op(OpKind.MUL, listOf(x, rsq), rank2)
            // Cross entropy
            val labels = param("labels", rank2)
            val probs = op(OpKind.SOFTMAX, listOf(xNorm), rank2)
            val logp = op(OpKind.LOG, listOf(probs), rank2)
            val pointwise = op(OpKind.MUL, listOf(labels, logp), rank2)
            val loss = op(OpKind.SUM, listOf(pointwise), scalar)
            listOf(loss)
        }
        val matches = recognizeAll(fn)
        // RMS norm + Cross entropy = 2 matches. (No FlashAttention or Rope
        // primitives in this graph.)
        assertTrue(matches.size >= 2, "expected at least 2 matches; got ${matches.size}")
        val patternNames = matches.map { it.patternName }.toSet()
        assertTrue("RmsNorm" in patternNames)
        assertTrue("CrossEntropy" in patternNames)
    }
}
