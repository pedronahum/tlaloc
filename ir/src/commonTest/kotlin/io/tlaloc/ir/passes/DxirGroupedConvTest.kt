package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.429 — grouped/depthwise convolution (`feature_group_count > 1`), the last
 * C4 tail and beyond DiffKT parity (DiffKT's conv has no groups at all).
 *
 * Layout convention is StableHLO's own: the kernel's input-feature dim carries
 * `Ci / g` and its output-feature dim the full `Co` (divisible by g), so output
 * channel `o` belongs to group `o / (Co/g)` and contracts only that group's
 * `Ci / g` input channels. Depthwise is `g == Ci` with a `[C, 1, kh, kw]` kernel
 * and needs no arm of its own.
 *
 * Oracles, in cross-validating order:
 * - a HAND-COMPUTED 1×1-kernel grouped conv on a quarter-integer grid (exact in
 *   f32 — the semantics pinned by arithmetic, not by another implementation);
 * - the manual-slicing equivalence `conv(x, w, groups = g) == concat_g(conv(x_g,
 *   w_g))` over general attrs (strides, asymmetric padding, rhs_dilation), for
 *   CONV2D, depthwise CONV2D, and CONV_TRANSPOSE2D — the grouped path against g
 *   ungrouped convs the §0.4.362 certification already covers;
 * - reverse mode against directional central differences AND the JVP⇄VJP
 *   cross-identity (the forward side is the bilinear product rule over the
 *   grouped primal, never the adjoints' slicing — so agreement is real
 *   cross-validation, not a shared-bug tautology);
 * - the gradients' own slicing symmetry: grouped dX / dW equal the concatenation
 *   of the per-group ungrouped VJPs (the loss `Σ y²` decomposes across groups);
 * - the §0.4.429 named refusal: grouped CONV_TRANSPOSE2D differentiates nowhere
 *   yet, and the rule must say so at transform time in those words.
 */
class DxirGroupedConvTest {

    private val scalar = DxirType(F32, emptyList())

    private fun convFn(
        kind: OpKind,
        xDims: List<Int>,
        wDims: List<Int>,
        yDims: List<Int>,
        attrs: Map<String, Any>,
    ): DxirFunction = DxirBuilder.function("grouped_conv") {
        val x = param("x", DxirType(F32, xDims))
        val w = param("w", DxirType(F32, wDims))
        listOf(op(kind, listOf(x, w), DxirType(F32, yDims), attrs = attrs))
    }

    private fun lossFn(
        kind: OpKind,
        xDims: List<Int>,
        wDims: List<Int>,
        yDims: List<Int>,
        attrs: Map<String, Any>,
    ): DxirFunction = DxirBuilder.function("grouped_conv_loss") {
        val x = param("x", DxirType(F32, xDims))
        val w = param("w", DxirType(F32, wDims))
        val yT = DxirType(F32, yDims)
        val y = op(kind, listOf(x, w), yT, attrs = attrs)
        val y2 = op(OpKind.MUL, listOf(y, y), yT)
        listOf(op(OpKind.SUM, listOf(y2), scalar))
    }

    private fun runConv(
        kind: OpKind,
        xDims: List<Int>,
        wDims: List<Int>,
        yDims: List<Int>,
        attrs: Map<String, Any>,
        x: FloatArray,
        w: FloatArray,
    ): FloatArray =
        DxirInterpreter.evalFunction(convFn(kind, xDims, wDims, yDims, attrs), listOf(x, w))
            .single()

    /** Contiguous [count]-wide slice at [start] along [axis] — the test's own spelling. */
    private fun sliceCh(
        src: FloatArray,
        dims: List<Int>,
        axis: Int,
        start: Int,
        count: Int,
    ): FloatArray {
        var outer = 1
        for (a in 0 until axis) outer *= dims[a]
        var inner = 1
        for (a in axis + 1 until dims.size) inner *= dims[a]
        val out = FloatArray(outer * count * inner)
        for (o in 0 until outer) {
            val base = (o * dims[axis] + start) * inner
            src.copyInto(out, o * count * inner, base, base + count * inner)
        }
        return out
    }

    private fun writeCh(
        dst: FloatArray,
        dims: List<Int>,
        axis: Int,
        start: Int,
        count: Int,
        piece: FloatArray,
    ) {
        var outer = 1
        for (a in 0 until axis) outer *= dims[a]
        var inner = 1
        for (a in axis + 1 until dims.size) inner *= dims[a]
        for (o in 0 until outer) {
            val base = (o * dims[axis] + start) * inner
            piece.copyInto(dst, base, o * count * inner, (o + 1) * count * inner)
        }
    }

    private fun randomArrays(seed: Long, xN: Int, wN: Int): Pair<FloatArray, FloatArray> {
        val rng = Random(seed)
        return Pair(
            FloatArray(xN) { (rng.nextFloat() - 0.5f) * 3f },
            FloatArray(wN) { (rng.nextFloat() - 0.5f) * 3f },
        )
    }

    // ── the semantics, pinned by hand ────────────────────────────────────────

    @Test
    fun handComputedGroupedOneByOneConv() {
        // x [1, 4, 1, 1], w [2, 2, 1, 1], groups = 2. Group 0 = channels {0, 1} →
        // output 0; group 1 = channels {2, 3} → output 1. Quarter-integer values,
        // exact in f32:
        //   y0 = 1.0·0.5 + 2.0·0.25          = 1.0
        //   y1 = 0.5·1.0 + 0.25·0.75         = 0.6875
        val got = runConv(
            OpKind.CONV2D,
            listOf(1, 4, 1, 1), listOf(2, 2, 1, 1), listOf(1, 2, 1, 1),
            mapOf("window_strides" to listOf(1, 1), "feature_group_count" to 2),
            floatArrayOf(1.0f, 2.0f, 0.5f, 0.25f),
            floatArrayOf(0.5f, 0.25f, 1.0f, 0.75f),
        )
        assertEquals(1.0f, got[0], "group 0 output")
        assertEquals(0.6875f, got[1], "group 1 output")
    }

    @Test
    fun handComputedDepthwiseOneByOneConv() {
        // Depthwise = groups == Ci == Co, kernel [C, 1, 1, 1]: pure per-channel scale.
        val got = runConv(
            OpKind.CONV2D,
            listOf(1, 4, 1, 1), listOf(4, 1, 1, 1), listOf(1, 4, 1, 1),
            mapOf("window_strides" to listOf(1, 1), "feature_group_count" to 4),
            floatArrayOf(1.0f, 2.0f, 0.5f, 0.25f),
            floatArrayOf(2.0f, 0.5f, 4.0f, 8.0f),
        )
        assertTrue(
            got.toList() == listOf(2.0f, 1.0f, 2.0f, 2.0f),
            "depthwise 1×1 must be a per-channel scale; got ${got.toList()}",
        )
    }

    // ── grouped == concat of per-group ungrouped convs ───────────────────────

    @Test
    fun groupedConvMatchesManualChannelSlicing() {
        // General attrs: strides [2, 1], asymmetric padding, rhs_dilation [1, 2].
        // x [1, 4, 5, 5] ⋆ w [6, 2, 3, 3] (groups = 2) → y [1, 6, 2, 3].
        val xDims = listOf(1, 4, 5, 5)
        val wDims = listOf(6, 2, 3, 3)
        val yDims = listOf(1, 6, 2, 3)
        val attrs = mapOf<String, Any>(
            "window_strides" to listOf(2, 1),
            "padding" to listOf(listOf(1, 0), listOf(1, 1)),
            "rhs_dilation" to listOf(1, 2),
        )
        val (x, w) = randomArrays(11, 4 * 5 * 5, 6 * 2 * 3 * 3)

        val grouped = runConv(
            OpKind.CONV2D, xDims, wDims, yDims,
            attrs + ("feature_group_count" to 2), x, w,
        )

        val manual = FloatArray(1 * 6 * 2 * 3)
        for (g in 0 until 2) {
            val xg = sliceCh(x, xDims, 1, g * 2, 2)
            val wg = sliceCh(w, wDims, 0, g * 3, 3)
            val piece = runConv(
                OpKind.CONV2D, listOf(1, 2, 5, 5), listOf(3, 2, 3, 3), listOf(1, 3, 2, 3),
                attrs, xg, wg,
            )
            writeCh(manual, yDims, 1, g * 3, 3, piece)
        }
        for (i in manual.indices) {
            assertTrue(
                abs(grouped[i] - manual[i]) <= 1e-5f,
                "grouped conv ≠ manual slicing at $i: ${grouped[i]} vs ${manual[i]}",
            )
        }
    }

    @Test
    fun depthwiseConvMatchesManualChannelSlicing() {
        // groups == Ci == Co == 4 with a real 3×3 window and same-padding.
        val xDims = listOf(1, 4, 4, 4)
        val wDims = listOf(4, 1, 3, 3)
        val yDims = listOf(1, 4, 4, 4)
        val attrs = mapOf<String, Any>(
            "window_strides" to listOf(1, 1),
            "padding" to listOf(listOf(1, 1), listOf(1, 1)),
        )
        val (x, w) = randomArrays(22, 4 * 4 * 4, 4 * 3 * 3)

        val grouped = runConv(
            OpKind.CONV2D, xDims, wDims, yDims,
            attrs + ("feature_group_count" to 4), x, w,
        )

        val manual = FloatArray(4 * 4 * 4)
        for (g in 0 until 4) {
            val xg = sliceCh(x, xDims, 1, g, 1)
            val wg = sliceCh(w, wDims, 0, g, 1)
            val piece = runConv(
                OpKind.CONV2D, listOf(1, 1, 4, 4), listOf(1, 1, 3, 3), listOf(1, 1, 4, 4),
                attrs, xg, wg,
            )
            writeCh(manual, yDims, 1, g, 1, piece)
        }
        for (i in manual.indices) {
            assertTrue(
                abs(grouped[i] - manual[i]) <= 1e-5f,
                "depthwise conv ≠ per-channel convs at $i: ${grouped[i]} vs ${manual[i]}",
            )
        }
    }

    @Test
    fun groupedConvTransposeMatchesManualChannelSlicing() {
        // The transposed spelling: IOHW kernel [Ci/g, Co, kh, kw], upsampling
        // lhs_dilation [2, 2]. x [1, 4, 3, 3] ⋆ w [2, 6, 2, 2] (groups = 2) →
        // y [1, 6, 4, 4].
        val xDims = listOf(1, 4, 3, 3)
        val wDims = listOf(2, 6, 2, 2)
        val yDims = listOf(1, 6, 4, 4)
        val attrs = mapOf<String, Any>(
            "window_strides" to listOf(1, 1),
            "lhs_dilation" to listOf(2, 2),
        )
        val (x, w) = randomArrays(33, 4 * 3 * 3, 2 * 6 * 2 * 2)

        val grouped = runConv(
            OpKind.CONV_TRANSPOSE2D, xDims, wDims, yDims,
            attrs + ("feature_group_count" to 2), x, w,
        )

        val manual = FloatArray(1 * 6 * 4 * 4)
        for (g in 0 until 2) {
            val xg = sliceCh(x, xDims, 1, g * 2, 2)
            // Per-group kernel: the OUTPUT-feature axis (axis 1 of IOHW) slices.
            val wg = sliceCh(w, wDims, 1, g * 3, 3)
            val piece = runConv(
                OpKind.CONV_TRANSPOSE2D, listOf(1, 2, 3, 3), listOf(2, 3, 2, 2),
                listOf(1, 3, 4, 4), attrs, xg, wg,
            )
            writeCh(manual, yDims, 1, g * 3, 3, piece)
        }
        for (i in manual.indices) {
            assertTrue(
                abs(grouped[i] - manual[i]) <= 1e-5f,
                "grouped convT ≠ manual slicing at $i: ${grouped[i]} vs ${manual[i]}",
            )
        }
    }

    // ── reverse mode ─────────────────────────────────────────────────────────

    private val gradXDims = listOf(1, 4, 5, 4)
    private val gradWDims = listOf(6, 2, 3, 2)
    private val gradYDims = listOf(1, 6, 2, 4)
    private val gradAttrs = mapOf<String, Any>(
        "window_strides" to listOf(2, 1),
        "padding" to listOf(listOf(1, 0), listOf(1, 1)),
        "rhs_dilation" to listOf(1, 2),
        "feature_group_count" to 2,
    )

    @Test
    fun groupedConvVjpMatchesCentralDifferences() {
        val fn = lossFn(OpKind.CONV2D, gradXDims, gradWDims, gradYDims, gradAttrs)
        val (x, w) = randomArrays(44, gradXDims.reduce(Int::times), gradWDims.reduce(Int::times))
        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x, w))

        val rng = Random(45)
        val vx = FloatArray(x.size) { rng.nextFloat() - 0.5f }
        val vw = FloatArray(w.size) { rng.nextFloat() - 0.5f }
        var dot = 0.0
        for (i in x.indices) dot += grads[0][i].toDouble() * vx[i]
        for (i in w.indices) dot += grads[1][i].toDouble() * vw[i]

        val eps = 1e-2f
        fun shifted(sgn: Float): Float {
            val xS = FloatArray(x.size) { x[it] + sgn * vx[it] }
            val wS = FloatArray(w.size) { w[it] + sgn * vw[it] }
            return DxirInterpreter.evalFunction(fn, listOf(xS, wS)).single().single()
        }
        val fd = (shifted(eps) - shifted(-eps)).toDouble() / (2 * eps)
        assertTrue(
            abs(dot - fd) <= 1e-2 * maxOf(1.0, abs(fd)),
            "grouped conv VJP disagrees with central differences: ⟨∇f,v⟩=$dot vs fd=$fd",
        )
    }

    @Test
    fun groupedConvVjpAgreesWithJvp() {
        val fn = lossFn(OpKind.CONV2D, gradXDims, gradWDims, gradYDims, gradAttrs)
        val (x, w) = randomArrays(55, gradXDims.reduce(Int::times), gradWDims.reduce(Int::times))
        val rng = Random(56)
        val vx = FloatArray(x.size) { rng.nextFloat() - 0.5f }
        val vw = FloatArray(w.size) { rng.nextFloat() - 0.5f }

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x, w))
        var dot = 0.0
        for (i in x.indices) dot += grads[0][i].toDouble() * vx[i]
        for (i in w.indices) dot += grads[1][i].toDouble() * vw[i]

        // The forward side is the bilinear product rule over the grouped PRIMAL —
        // the adjoints' per-group slicing never runs on this side.
        val tangent = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(x, w, vx, vw),
        ).last().single()
        assertTrue(
            abs(dot - tangent) <= 1e-3 * maxOf(1.0, abs(tangent).toDouble()),
            "grouped conv VJP disagrees with its JVP: ⟨∇f,v⟩=$dot vs jvp=$tangent",
        )
    }

    @Test
    fun groupedGradientsGroupSliceSymmetrically() {
        // Σ y² decomposes across output-channel groups, so the grouped gradient
        // must EQUAL the concatenation of the per-group ungrouped VJPs — dX
        // slicing on input channels, dW on kernel rows, symmetrically.
        val fn = lossFn(OpKind.CONV2D, gradXDims, gradWDims, gradYDims, gradAttrs)
        val (x, w) = randomArrays(66, gradXDims.reduce(Int::times), gradWDims.reduce(Int::times))
        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x, w))

        val ungroupedAttrs = gradAttrs - "feature_group_count"
        val dxManual = FloatArray(x.size)
        val dwManual = FloatArray(w.size)
        for (g in 0 until 2) {
            val xg = sliceCh(x, gradXDims, 1, g * 2, 2)
            val wg = sliceCh(w, gradWDims, 0, g * 3, 3)
            val fnG = lossFn(
                OpKind.CONV2D, listOf(1, 2, 5, 4), listOf(3, 2, 3, 2), listOf(1, 3, 2, 4),
                ungroupedAttrs,
            )
            val gradsG = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fnG), listOf(xg, wg))
            writeCh(dxManual, gradXDims, 1, g * 2, 2, gradsG[0])
            writeCh(dwManual, gradWDims, 0, g * 3, 3, gradsG[1])
        }
        for (i in dxManual.indices) {
            assertTrue(
                abs(grads[0][i] - dxManual[i]) <= 1e-4f,
                "grouped dX ≠ per-group VJP concat at $i: ${grads[0][i]} vs ${dxManual[i]}",
            )
        }
        for (i in dwManual.indices) {
            assertTrue(
                abs(grads[1][i] - dwManual[i]) <= 1e-4f,
                "grouped dW ≠ per-group VJP concat at $i: ${grads[1][i]} vs ${dwManual[i]}",
            )
        }
    }

    // ── the named refusal ────────────────────────────────────────────────────

    @Test
    fun convTransposeVjpRefusesGroupsByName() {
        val fn = lossFn(
            OpKind.CONV_TRANSPOSE2D, listOf(1, 4, 3, 3), listOf(2, 6, 2, 2), listOf(1, 6, 4, 4),
            mapOf(
                "window_strides" to listOf(1, 1),
                "lhs_dilation" to listOf(2, 2),
                "feature_group_count" to 2,
            ),
        )
        val e = assertFailsWith<IllegalArgumentException> { DxirReverseTransform.apply(fn) }
        assertTrue(
            e.message?.contains("feature_group_count") == true,
            "the refusal must name the attr; got: ${e.message}",
        )
    }
}
