package io.tlaloc.ir.recognizer

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.318 — LayerNorm recognizer tests. Mirrors the
 * [RmsNormRopeCrossEntropyTest] shape: positive matches, adversarial
 * near-misses with diagnostic verification, plus the resolver scenario
 * that demonstrates LayerNorm winning over RmsNorm on the same region.
 */
class LayerNormRecognizerTest {

    private val rank2 = DxirType(F32, listOf(8, 64))
    private val rank2Reduced = DxirType(F32, listOf(8, 1))

    @Test
    fun layerNormPositiveMatch() {
        val fn = DxirBuilder.function("layer_norm") {
            val x = param("x", rank2)
            val mean1 = op(OpKind.MEAN, listOf(x), rank2Reduced)
            val sub = op(OpKind.SUB, listOf(x, mean1), rank2)
            val sq = op(OpKind.MUL, listOf(sub, sub), rank2)
            val mean2 = op(OpKind.MEAN, listOf(sq), rank2Reduced)
            val sqrt = op(OpKind.SQRT, listOf(mean2), rank2Reduced)
            val out = op(OpKind.DIV, listOf(sub, sqrt), rank2)
            listOf(out)
        }
        val matches = recognizeLayerNorm(fn)
        assertEquals(1, matches.size, "expected 1 LayerNorm match")
        assertEquals(
            6, matches.single().ops.size,
            "MEAN→SUB→MUL→MEAN→SQRT→DIV = 6 ops without eps",
        )
    }

    @Test
    fun layerNormPositiveMatchWithEpsilon() {
        val fn = DxirBuilder.function("layer_norm_eps") {
            val x = param("x", rank2)
            val eps = const(1e-5f, rank2Reduced)
            val mean1 = op(OpKind.MEAN, listOf(x), rank2Reduced)
            val sub = op(OpKind.SUB, listOf(x, mean1), rank2)
            val sq = op(OpKind.MUL, listOf(sub, sub), rank2)
            val mean2 = op(OpKind.MEAN, listOf(sq), rank2Reduced)
            val stabilised = op(OpKind.ADD, listOf(mean2, eps), rank2Reduced)
            val sqrt = op(OpKind.SQRT, listOf(stabilised), rank2Reduced)
            val out = op(OpKind.DIV, listOf(sub, sqrt), rank2)
            listOf(out)
        }
        val matches = recognizeLayerNorm(fn)
        assertEquals(1, matches.size)
        assertEquals(
            7, matches.single().ops.size,
            "MEAN→SUB→MUL→MEAN→ADD→SQRT→DIV = 7 ops with eps",
        )
    }

    @Test
    fun layerNormAdversarialSqrtOperandNotMean() {
        // SQRT of a generic param — not a variance-reduce.
        val fn = DxirBuilder.function("sqrt_of_param") {
            val x = param("x", rank2Reduced)
            val s = op(OpKind.SQRT, listOf(x), rank2Reduced)
            listOf(s)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeLayerNorm(fn, diag)
        assertEquals(0, matches.size)
        assertTrue(diag.any { "expected MEAN" in it.reason })
    }

    @Test
    fun layerNormAdversarialSquaredTensorIsRmsNormShape() {
        // SQRT(MEAN(MUL(x, x))) where the squared tensor is the bare x —
        // this is RmsNorm shape, not LayerNorm. No SUB step.
        val fn = DxirBuilder.function("rms_norm_with_sqrt") {
            val x = param("x", rank2)
            val sq = op(OpKind.MUL, listOf(x, x), rank2)
            val mean = op(OpKind.MEAN, listOf(sq), rank2Reduced)
            val sqrt = op(OpKind.SQRT, listOf(mean), rank2Reduced)
            listOf(sqrt)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeLayerNorm(fn, diag)
        assertEquals(0, matches.size)
        assertTrue(
            diag.any { "RmsNorm" in it.reason },
            "diagnostic should call out the RmsNorm-vs-LayerNorm distinction",
        )
    }

    @Test
    fun layerNormAdversarialMean1MismatchesSub() {
        // SUB(x, mean(y)) where the centring source is a different tensor.
        val fn = DxirBuilder.function("misaligned_centre") {
            val x = param("x", rank2)
            val y = param("y", rank2)
            val meanY = op(OpKind.MEAN, listOf(y), rank2Reduced)
            val sub = op(OpKind.SUB, listOf(x, meanY), rank2)
            val sq = op(OpKind.MUL, listOf(sub, sub), rank2)
            val mean2 = op(OpKind.MEAN, listOf(sq), rank2Reduced)
            val sqrt = op(OpKind.SQRT, listOf(mean2), rank2Reduced)
            val out = op(OpKind.DIV, listOf(sub, sqrt), rank2)
            listOf(out)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeLayerNorm(fn, diag)
        assertEquals(0, matches.size)
        assertTrue(diag.any { "different mean source" in it.reason })
    }

    @Test
    fun layerNormAdversarialNoFinalDiv() {
        // Chain exists but the SQRT result is consumed by something other
        // than a DIV(centered, sqrt) — e.g. the user halted the chain.
        val fn = DxirBuilder.function("dangling_sqrt") {
            val x = param("x", rank2)
            val mean1 = op(OpKind.MEAN, listOf(x), rank2Reduced)
            val sub = op(OpKind.SUB, listOf(x, mean1), rank2)
            val sq = op(OpKind.MUL, listOf(sub, sub), rank2)
            val mean2 = op(OpKind.MEAN, listOf(sq), rank2Reduced)
            val sqrt = op(OpKind.SQRT, listOf(mean2), rank2Reduced)
            // No DIV — SQRT is the function output.
            listOf(sqrt)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeLayerNorm(fn, diag)
        assertEquals(0, matches.size)
        assertTrue(diag.any { "DIV(centered, sqrt) consumer" in it.reason })
    }

    @Test
    fun layerNormFlowsThroughRecognizeAll() {
        // End-to-end check: LayerNorm survives the [recognizeAll]
        // aggregator (no other recognizer claims its ops, no resolver
        // elimination). RmsNorm does not fire on the SQRT-form because it
        // anchors on RSQRT.
        val fn = DxirBuilder.function("layer_norm_in_aggregator") {
            val x = param("x", rank2)
            val mean1 = op(OpKind.MEAN, listOf(x), rank2Reduced)
            val sub = op(OpKind.SUB, listOf(x, mean1), rank2)
            val sq = op(OpKind.MUL, listOf(sub, sub), rank2)
            val mean2 = op(OpKind.MEAN, listOf(sq), rank2Reduced)
            val sqrt = op(OpKind.SQRT, listOf(mean2), rank2Reduced)
            val out = op(OpKind.DIV, listOf(sub, sqrt), rank2)
            listOf(out)
        }
        val matches = recognizeAll(fn)
        assertEquals(1, matches.size, "exactly one match should survive the aggregator")
        val match = matches.single()
        assertTrue(
            match is RecognitionMatch.LayerNorm,
            "expected LayerNorm; got ${match.patternName}",
        )
        assertEquals(6, match.ops.size, "no-eps LayerNorm ops")
    }
}
