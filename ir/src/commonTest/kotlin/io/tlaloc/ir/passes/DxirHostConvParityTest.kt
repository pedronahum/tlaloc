package io.tlaloc.ir.passes

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.core.ops.avgPool2dGeneral
import io.tlaloc.core.ops.avgPool2dGrad
import io.tlaloc.core.ops.conv2d
import io.tlaloc.core.ops.conv2dDataAdjoint
import io.tlaloc.core.ops.conv2dGeneral
import io.tlaloc.core.ops.conv2dKernelAdjoint
import io.tlaloc.core.ops.convTranspose2d
import io.tlaloc.core.ops.convTranspose2dGeneral
import io.tlaloc.core.ops.transposePerm4
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.384 — Phase A3b: the `:core` host conv twins are BIT-EXACT against the dxir
 * interpreter's conv evaluation, for every attr spelling the conv/pool ADJOINTS
 * actually emit. §0.4.385 extended the walk to the fused adjoint ops
 * ([OpKind.CONV2D_DATA_ADJOINT] / [OpKind.CONV2D_KERNEL_ADJOINT]) and their twins.
 *
 * The parity is checked mechanically rather than against hand-written attrs: each
 * test runs [DxirReverseTransform] on a real loss, walks the resulting gradient
 * graph for every conv-family node, and replays each node twice — once through the
 * interpreter, once through the host twin with the attrs read off the node. So the
 * coverage follows the rules: when [VjpRegistry.Conv2dRule] or
 * [VjpRegistry.AvgPool2dRule] changes the spelling it emits, these tests start
 * exercising the new one instead of quietly pinning a stale one. Each test asserts
 * WHICH kinds it checked, so a graph walk that finds nothing (or that silently
 * stops finding a kind after a rule rewrite) fails rather than passing vacuously.
 *
 * Bit-exactness (not a tolerance) is the property under test: the K2 plugin
 * synthesises `grad {}` conv gradients into calls on these twins, and the
 * interpreted dxir is the oracle those results are certified against.
 */
class DxirHostConvParityTest {

    private val scalar = DxirType(F32, emptyList())

    // ---- attr reading, mirroring the interpreter's own defaults -------------

    private fun intPair(attrs: Map<String, Any>, key: String, def: List<Int>): List<Int> =
        (attrs[key] as? List<*>)?.map { (it as Number).toInt() } ?: def

    private fun paddingOf(attrs: Map<String, Any>): List<List<Int>> =
        (attrs["padding"] as? List<*>)
            ?.map { row -> (row as List<*>).map { (it as Number).toInt() } }
            ?: listOf(listOf(0, 0), listOf(0, 0))

    private fun reversalOf(attrs: Map<String, Any>): List<Boolean> =
        (attrs["window_reversal"] as? List<*>)?.map { it as Boolean } ?: listOf(false, false)

    private fun tensor(data: FloatArray, dims: List<Int>): DTensor<Shape, F32> =
        DTensor(HostF32Storage(data.copyOf()), dims.toIntArray(), F32)

    /** The conv/pool-family nodes in [fn]'s body — the ones the host twins must replay. */
    private fun convFamilyNodes(fn: DxirFunction): List<DxirOp> =
        fn.body.filterIsInstance<DxirOp>().filter {
            it.op == OpKind.CONV2D || it.op == OpKind.CONV_TRANSPOSE2D ||
                it.op == OpKind.CONV2D_DATA_ADJOINT || it.op == OpKind.CONV2D_KERNEL_ADJOINT ||
                it.op == OpKind.AVGPOOL2D || it.op == OpKind.AVGPOOL2D_GRAD ||
                (it.op == OpKind.TRANSPOSE && it.type.rank == 4)
        }

    /**
     * Replay [node] through the interpreter (fresh params of the operand types,
     * random data) and through the host twin with the node's own attrs, and
     * require the two to agree element-for-element.
     */
    private fun assertTwinMatchesInterpreter(node: DxirOp, seed: Long) {
        val rng = Random(seed)
        val inputs = node.operands.map { n ->
            FloatArray(n.type.elementCount.toInt()) { (rng.nextFloat() - 0.5f) * 4f }
        }
        val probe = DxirBuilder.function("probe_${node.op}_${node.id}") {
            val ps = node.operands.mapIndexed { i, n -> param("p$i", n.type) }
            listOf(op(node.op, ps, node.type, attrs = node.attrs))
        }
        val expected = DxirInterpreter.evalFunction(probe, inputs).single()

        val actual: DTensor<Shape, F32> = when (node.op) {
            OpKind.TRANSPOSE -> {
                val perm = intPair(node.attrs, "permutation", emptyList())
                assertEquals(4, perm.size, "rank-4 TRANSPOSE without a 4-permutation: ${node.attrs}")
                transposePerm4(tensor(inputs[0], node.operands[0].type.dims), perm[0], perm[1], perm[2], perm[3])
            }
            OpKind.CONV2D, OpKind.CONV_TRANSPOSE2D -> {
                val s = intPair(node.attrs, "window_strides", listOf(1, 1))
                val ld = intPair(node.attrs, "lhs_dilation", listOf(1, 1))
                val rd = intPair(node.attrs, "rhs_dilation", listOf(1, 1))
                val rev = reversalOf(node.attrs)
                val p = paddingOf(node.attrs)
                val x = tensor(inputs[0], node.operands[0].type.dims)
                val w = tensor(inputs[1], node.operands[1].type.dims)
                if (node.op == OpKind.CONV2D) {
                    conv2dGeneral(
                        x, w, s[0], s[1], ld[0], ld[1], rd[0], rd[1],
                        p[0][0], p[0][1], p[1][0], p[1][1], rev[0], rev[1],
                    )
                } else {
                    convTranspose2dGeneral(
                        x, w, s[0], s[1], ld[0], ld[1], rd[0], rd[1],
                        p[0][0], p[0][1], p[1][0], p[1][1], rev[0], rev[1],
                    )
                }
            }
            OpKind.CONV2D_DATA_ADJOINT, OpKind.CONV2D_KERNEL_ADJOINT -> {
                // §0.4.385 — the fused adjoints. Only the primal's LOW padding
                // reaches the host twin: the high side is implied by the
                // upstream's runtime shape. The template operand is shape-only,
                // but the probe still feeds it data so both engines see the same
                // three operands.
                val s = intPair(node.attrs, "window_strides", listOf(1, 1))
                val d = intPair(node.attrs, "rhs_dilation", listOf(1, 1))
                val p = paddingOf(node.attrs)
                val a = tensor(inputs[0], node.operands[0].type.dims)
                val b = tensor(inputs[1], node.operands[1].type.dims)
                val tmpl = tensor(inputs[2], node.operands[2].type.dims)
                if (node.op == OpKind.CONV2D_DATA_ADJOINT) {
                    conv2dDataAdjoint(a, b, tmpl, s[0], s[1], d[0], d[1], p[0][0], p[1][0])
                } else {
                    conv2dKernelAdjoint(a, b, tmpl, s[0], s[1], d[0], d[1], p[0][0], p[1][0])
                }
            }
            OpKind.AVGPOOL2D -> {
                // §0.4.386 — the primal twin, pinned here because count_include_pad
                // (dividing by the FULL window, padding included) is a convention
                // the two engines could easily disagree on.
                val win = intPair(node.attrs, "window", emptyList())
                val s = intPair(node.attrs, "window_strides", win)
                val p = paddingOf(node.attrs)
                assertEquals(2, win.size, "AVGPOOL2D without a `window` attr: ${node.attrs}")
                avgPool2dGeneral(
                    tensor(inputs[0], node.operands[0].type.dims),
                    win[0], win[1], s[0], s[1], p[0][0], p[0][1], p[1][0], p[1][1],
                )
            }
            OpKind.AVGPOOL2D_GRAD -> {
                // §0.4.386 — the fused avgpool adjoint: window and strides ride as
                // attrs, and only the LOW padding reaches the twin.
                val win = intPair(node.attrs, "window", emptyList())
                val s = intPair(node.attrs, "window_strides", win)
                val p = paddingOf(node.attrs)
                assertEquals(2, win.size, "AVGPOOL2D_GRAD without a `window` attr: ${node.attrs}")
                avgPool2dGrad(
                    tensor(inputs[0], node.operands[0].type.dims),
                    tensor(inputs[1], node.operands[1].type.dims),
                    win[0], win[1], s[0], s[1], p[0][0], p[1][0],
                )
            }
            else -> error("unexpected node kind ${node.op}")
        }

        assertContentEquals(
            node.type.dims.toIntArray(), actual.dims,
            "${node.op} id=${node.id}: host twin derived the wrong output extents",
        )
        assertTrue(
            expected.contentEquals(actual.hostF32()),
            "${node.op} id=${node.id} attrs=${node.attrs}: host twin disagrees with the " +
                "interpreter at ${firstDiff(expected, actual.hostF32())}",
        )
    }

    private fun firstDiff(a: FloatArray, b: FloatArray): String {
        if (a.size != b.size) return "size ${a.size} vs ${b.size}"
        for (i in a.indices) if (a[i] != b[i]) return "index $i: ${a[i]} vs ${b[i]}"
        return "no difference"
    }

    /**
     * Run the walk over [fn]'s gradient body and return the kinds actually
     * checked, so a test can assert its coverage rather than just a node count.
     */
    private fun checkAdjointNodes(primal: DxirFunction, seed: Long): List<OpKind> {
        val grads = DxirReverseTransform.apply(primal)
        val nodes = convFamilyNodes(grads)
        nodes.forEachIndexed { i, n -> assertTwinMatchesInterpreter(n, seed + i) }
        return nodes.map { it.op }
    }

    // ---- the primal losses whose adjoints are walked -----------------------

    /**
     * General-attr conv loss: N=2, Ci=3, Co=2, 5×4 input, 3×2 kernel, strides
     * [2,1], asymmetric padding [[1,0],[1,1]], rhs_dilation [1,2]. `loss = Σ y²`
     * so the upstream is non-uniform. Its adjoint is the full
     * [VjpRegistry.Conv2dRule] spelling: a lhs-dilated reversed CONV_TRANSPOSE2D
     * for `dX`, and the batch↔feature TRANSPOSE / stride-swapped CONV2D with a
     * NEGATIVE padding-high (cropping the primal's floor-division remainder) for
     * `dW`.
     */
    private fun generalConvLossFn(): DxirFunction {
        val xT = DxirType(F32, listOf(2, 3, 5, 4))
        val wT = DxirType(F32, listOf(2, 3, 3, 2))
        // hOut = (5+1+0−3)/2+1 = 2; kEffW = (2−1)·2+1 = 3; wOut = (4+1+1−3)/1+1 = 4.
        val yT = DxirType(F32, listOf(2, 2, 2, 4))
        return DxirBuilder.function("general_conv_loss") {
            val x = param("x", xT)
            val w = param("w", wT)
            val y = op(
                OpKind.CONV2D, listOf(x, w), yT,
                attrs = mapOf(
                    "window_strides" to listOf(2, 1),
                    "padding" to listOf(listOf(1, 0), listOf(1, 1)),
                    "rhs_dilation" to listOf(1, 2),
                ),
            )
            val y2 = op(OpKind.MUL, listOf(y, y), yT)
            listOf(op(OpKind.SUM, listOf(y2), scalar))
        }
    }

    /** The plain spelling: stride 1, no padding, 3×3 kernel over a 6×6 input. */
    private fun plainConvLossFn(): DxirFunction {
        val xT = DxirType(F32, listOf(1, 2, 6, 6))
        val wT = DxirType(F32, listOf(3, 2, 3, 3))
        val yT = DxirType(F32, listOf(1, 3, 4, 4))
        return DxirBuilder.function("plain_conv_loss") {
            val x = param("x", xT)
            val w = param("w", wT)
            val y = op(OpKind.CONV2D, listOf(x, w), yT)
            val y2 = op(OpKind.MUL, listOf(y, y), yT)
            listOf(op(OpKind.SUM, listOf(y2), scalar))
        }
    }

    /**
     * Avgpool loss whose adjoint ([VjpRegistry.AvgPool2dRule]) is a
     * lhs-dilated CONV_TRANSPOSE2D against a 1/(kh·kw) splat kernel — the
     * spelling §0.4.383's re-scope identified as the reason avgpool is not the
     * cheap wedge. Pinned here so slice 3 inherits a certified twin.
     */
    private fun avgPoolLossFn(): DxirFunction {
        val xT = DxirType(F32, listOf(2, 3, 6, 5))
        // Non-divisible width: wOut = (5−2)/2+1 = 2, so the adjoint's padding is
        // asymmetric ([[1,1],[1,2]]) and exercises a crop-free odd remainder.
        val yT = DxirType(F32, listOf(2, 3, 3, 2))
        return DxirBuilder.function("avgpool_loss") {
            val x = param("x", xT)
            val y = op(
                OpKind.AVGPOOL2D, listOf(x), yT,
                attrs = mapOf("window" to listOf(2, 2), "window_strides" to listOf(2, 2)),
            )
            val y2 = op(OpKind.MUL, listOf(y, y), yT)
            listOf(op(OpKind.SUM, listOf(y2), scalar))
        }
    }

    @Test
    fun hostTwinsMatchInterpreterOnGeneralConvAdjoint() {
        val kinds = checkAdjointNodes(generalConvLossFn(), seed = 100)
        // §0.4.385 — both fused adjoints, plus the primal conv the value stream
        // keeps (MUL's adjoint reads it). The general attrs are what make the
        // solved padding non-trivial: stride [2,1] means dX needs lhs_dilation
        // [2,1] and dW a NEGATIVE crop, and rhs_dilation [1,2] means the kernel's
        // effective width differs per axis.
        assertCovers(
            kinds,
            listOf(OpKind.CONV2D, OpKind.CONV2D_DATA_ADJOINT, OpKind.CONV2D_KERNEL_ADJOINT),
        )
    }

    @Test
    fun hostTwinsMatchInterpreterOnPlainConvAdjoint() {
        val kinds = checkAdjointNodes(plainConvLossFn(), seed = 200)
        assertCovers(
            kinds,
            listOf(OpKind.CONV2D_DATA_ADJOINT, OpKind.CONV2D_KERNEL_ADJOINT),
        )
    }

    @Test
    fun hostTwinsMatchInterpreterOnAvgPoolAdjoint() {
        val kinds = checkAdjointNodes(avgPoolLossFn(), seed = 300)
        // §0.4.386 — the avgpool adjoint is now the fused AVGPOOL2D_GRAD (the window
        // inverted at runtime) instead of a channel-folded lhs-dilated
        // CONV_TRANSPOSE2D against a 1/(kh·kw) splat kernel; the primal AVGPOOL2D
        // rides along in the value stream (MUL's adjoint reads it). The loss keeps a
        // NON-divisible width (5, window 2, stride 2) so the inversion has to handle
        // a floor-division remainder rather than a clean tiling.
        assertCovers(kinds, listOf(OpKind.AVGPOOL2D, OpKind.AVGPOOL2D_GRAD))
    }

    private fun assertCovers(actual: List<OpKind>, expected: List<OpKind>) {
        assertTrue(
            actual.containsAll(expected),
            "expected the gradient body to exercise $expected; found $actual",
        )
    }

    // ---- hand pins: the user-facing spellings, not just interpreter agreement

    @Test
    fun conv2dSugarDefaultsToValidConv() {
        // x = [[1..9]] (1,1,3,3), OIHW w = [[1,0],[0,2]] (1,1,2,2):
        // out[y][x] = x[y][x] + 2·x[y+1][x+1] → [[11,14],[20,23]].
        val x = DTensor<Shape, F32>(HostF32Storage(FloatArray(9) { (it + 1).toFloat() }), intArrayOf(1, 1, 3, 3), F32)
        val w = DTensor<Shape, F32>(HostF32Storage(floatArrayOf(1f, 0f, 0f, 2f)), intArrayOf(1, 1, 2, 2), F32)
        val y = x.conv2d(w)
        assertContentEquals(intArrayOf(1, 1, 2, 2), y.dims)
        assertContentEquals(floatArrayOf(11f, 14f, 20f, 23f), y.hostF32())
    }

    @Test
    fun convSugarAndGeneralDelegateAgreeOnAttrs() {
        // The sugar is defined as the general delegate with the identity dilations
        // and no reversal; pin that mapping (incl. `padTop` fanning out to all four
        // sides) rather than trusting the default arguments.
        val rng = Random(5)
        val x = DTensor<Shape, F32>(HostF32Storage(FloatArray(2 * 3 * 5 * 4) { rng.nextFloat() }), intArrayOf(2, 3, 5, 4), F32)
        val w = DTensor<Shape, F32>(HostF32Storage(FloatArray(2 * 3 * 3 * 2) { rng.nextFloat() }), intArrayOf(2, 3, 3, 2), F32)
        val viaSugar = x.conv2d(w, 2, 1, 1, 0, 1, 1)
        val viaGeneral = conv2dGeneral<Shape>(x, w, 2, 1, 1, 1, 1, 1, 1, 0, 1, 1, false, false)
        assertContentEquals(viaGeneral.hostF32(), viaSugar.hostF32())
        assertContentEquals(viaGeneral.dims, viaSugar.dims)
    }

    @Test
    fun convTranspose2dSugarMapsStrideOntoLhsDilation() {
        // x = [1, 2] (1,1,1,2), IOHW w = [3] (1,1,1,1), upsample ×2 along W:
        // lhs_dilation [1,2] spreads the input to [1, ·, 1] and the 1×1 kernel
        // scales each surviving tap → [3, 0, 6]. The interior zero IS the
        // fractionally-strided mechanism.
        val x = DTensor<Shape, F32>(HostF32Storage(floatArrayOf(1f, 2f)), intArrayOf(1, 1, 1, 2), F32)
        val w = DTensor<Shape, F32>(HostF32Storage(floatArrayOf(3f)), intArrayOf(1, 1, 1, 1), F32)
        val y = x.convTranspose2d(w, 1, 2, 0, 0, 0, 0)
        assertContentEquals(intArrayOf(1, 1, 1, 3), y.dims)
        assertContentEquals(floatArrayOf(3f, 0f, 6f), y.hostF32())
    }
}
