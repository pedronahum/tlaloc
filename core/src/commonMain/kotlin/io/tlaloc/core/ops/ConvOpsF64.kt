package io.tlaloc.core.ops

import io.tlaloc.core.DScalar
import io.tlaloc.core.DTensor
import io.tlaloc.core.F64
import io.tlaloc.core.HostF64Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.Rank3
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Shape
import io.tlaloc.core.ShapeAtom
import io.tlaloc.core.Sym
import io.tlaloc.core.digamma
import io.tlaloc.core.hostF64
import io.tlaloc.core.hostI32
import io.tlaloc.core.lgamma
import io.tlaloc.core.polygamma
import io.tlaloc.core.trigamma
import kotlin.math.pow
import kotlin.math.sqrt


// F64 twins of the convolution, pooling and batch-normalization host ops in HostOps.kt,
// generated from them at Double width. Their own file because each erases to its F64
// twin's JVM signature.

/**
 * The rank-4 substrate: the shared conv engine.
 * NCHW input; the kernel reads OIHW `[Co, Ci, kh, kw]` when [transpose] is false
 * and IOHW `[Ci, Co, kh, kw]` when true — the layouts the dxir interpreter's
 * `evalConv2d` and the StableHLO emitter fix. This is a port of that eval, kept
 * deliberately literal: same output-extent formula, same tap→input coordinate
 * mapping, same tap skipping (padding edges, lhs-dilation holes), and the same
 * `Double` accumulator walked in the same `i → ky → kx` order. Bit-exactness
 * against the interpreter is the point — the K2 plugin synthesises `grad {}` conv
 * gradients into calls on these twins, so the interpreted dxir stays a usable
 * oracle for the generated host code.
 *
 * Attrs mirror the IR's one-for-one: `window_strides` [sH, sW], `padding`
 * [[top, bottom], [left, right]] (negative values CROP, as in StableHLO),
 * `lhs_dilation` (interior-dilates the input — the transposed-conv mechanism),
 * `rhs_dilation` (à-trous kernel), `window_reversal` (spatially flips the kernel
 * taps — what [io.tlaloc.ir.passes.VjpRegistry.Conv2dRule]'s `dX` needs).
 * `feature_group_count` / `batch_group_count` are 1 here: the interpreter grew a
 * grouped arm, but the host twins (and with them the `grad {}` /
 * `jvp {}` synthesis surface) do not support grouped convolution yet — the K2
 * synthesis rejects grouped conv ops loudly rather than convolving the wrong way.
 */
private fun conv2dEngine(
    lhs: DTensor<*, F64>,
    rhs: DTensor<*, F64>,
    transpose: Boolean,
    strideH: Int,
    strideW: Int,
    lhsDilH: Int,
    lhsDilW: Int,
    rhsDilH: Int,
    rhsDilW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
    revH: Boolean,
    revW: Boolean,
): DTensor<Shape, F64> {
    val ld = lhs.dims
    val rd = rhs.dims
    val opName = if (transpose) "convTranspose2d" else "conv2d"
    require(ld.size == 4 && rd.size == 4) {
        "$opName: rank-4 NCHW lhs and rank-4 kernel required; got ${ld.toList()} / ${rd.toList()}"
    }
    require(
        strideH > 0 && strideW > 0 && lhsDilH > 0 && lhsDilW > 0 && rhsDilH > 0 && rhsDilW > 0
    ) {
        "$opName: strides and dilations must be positive; got strides [$strideH, $strideW], " +
            "lhs_dilation [$lhsDilH, $lhsDilW], rhs_dilation [$rhsDilH, $rhsDilW]"
    }
    val nB = ld[0]
    val cIn = ld[1]
    val h = ld[2]
    val w = ld[3]
    val kh = rd[2]
    val kw = rd[3]
    val cOut: Int
    val cKIn: Int
    if (transpose) {
        cKIn = rd[0]
        cOut = rd[1]
    } else {
        cOut = rd[0]
        cKIn = rd[1]
    }
    require(cKIn == cIn) { "$opName: kernel input channels $cKIn ≠ lhs channels $cIn" }

    val hDil = (h - 1) * lhsDilH + 1
    val wDil = (w - 1) * lhsDilW + 1
    val kEffH = (kh - 1) * rhsDilH + 1
    val kEffW = (kw - 1) * rhsDilW + 1
    val hOut = (hDil + padTop + padBottom - kEffH) / strideH + 1
    val wOut = (wDil + padLeft + padRight - kEffW) / strideW + 1
    require(hOut > 0 && wOut > 0) {
        "$opName: derived output extents [$hOut, $wOut] are empty — window [$kh, $kw] with " +
            "rhs_dilation [$rhsDilH, $rhsDilW] does not fit the padded, dilated input [$hDil, $wDil]"
    }

    val lv = lhs.hostF64()
    val rv = rhs.hostF64()
    val out = DoubleArray(nB * cOut * hOut * wOut)
    var outIdx = 0
    for (n in 0 until nB) {
        for (o in 0 until cOut) {
            for (y in 0 until hOut) {
                for (x in 0 until wOut) {
                    var acc = 0.0
                    for (i in 0 until cIn) {
                        for (ky in 0 until kh) {
                            // Tap position in the dilated+padded input space.
                            val yDil = y * strideH + ky * rhsDilH - padTop
                            if (yDil < 0 || yDil % lhsDilH != 0) continue
                            val inY = yDil / lhsDilH
                            if (inY >= h) continue
                            val wKy = if (revH) kh - 1 - ky else ky
                            for (kx in 0 until kw) {
                                val xDil = x * strideW + kx * rhsDilW - padLeft
                                if (xDil < 0 || xDil % lhsDilW != 0) continue
                                val inX = xDil / lhsDilW
                                if (inX >= w) continue
                                val wKx = if (revW) kw - 1 - kx else kx
                                val wIdx = if (transpose) {
                                    ((i * cOut + o) * kh + wKy) * kw + wKx
                                } else {
                                    ((o * cIn + i) * kh + wKy) * kw + wKx
                                }
                                acc += lv[((n * cIn + i) * h + inY) * w + inX].toDouble() * rv[wIdx]
                            }
                        }
                    }
                    out[outIdx++] = acc.toDouble()
                }
            }
        }
    }
    return DTensor(HostF64Storage(out), intArrayOf(nB, cOut, hOut, wOut), F64)
}

/**
 * 2-D convolution, NCHW input against an OIHW `[Co, Ci, kh, kw]`
 * kernel. The result erases to `DTensor<Shape, F64>`: its spatial extents are a
 * runtime function of the input's and of stride/padding, which no static shape
 * witness can carry — the convention `reshape`, `slice`, `concat` and
 * `broadcastTo` follow.
 *
 * Two arities, and deliberately NO default parameter values: `conv2d(w)` is the
 * plain valid conv (stride 1, no padding) and the 7-argument form takes the attrs
 * POSITIONALLY in a fixed order — `(w, strideH, strideW, padTop, padBottom,
 * padLeft, padRight)`, the same spelling as `slice(start, end, axis)`. Defaults
 * would be a silent-wrongness trap here: K2 unwraps a named argument like
 * `padTop = 1` to its bare literal BEFORE the plugin's FIR lowering sees the
 * call, without reordering it into its parameter's position, so the lowering
 * cannot tell `padTop = 1` from `strideH = 1` and would fold the padding into the
 * stride. Arity is unambiguous, so an unwritable spelling is a compile error
 * instead of a mis-folded attr.
 */
fun <S : Shape> DTensor<S, F64>.conv2d(w: DTensor<*, F64>): DTensor<Shape, F64> =
    conv2dGeneral(this, w, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, false, false)

@Suppress("LongParameterList")
fun <S : Shape> DTensor<S, F64>.conv2d(
    w: DTensor<*, F64>,
    strideH: Int,
    strideW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
): DTensor<Shape, F64> = conv2dGeneral(
    this, w, strideH, strideW, 1, 1, 1, 1, padTop, padBottom, padLeft, padRight, false, false,
)

/**
 * Transposed 2-D convolution (the "deconvolution" / fractionally-strided
 * conv), NCHW input against an IOHW `[Ci, Co, kh, kw]` kernel — note the channel
 * order is the reverse of [conv2d]'s, because the op contracts over the kernel's
 * FIRST axis and emits the second.
 *
 * [strideH]/[strideW] are the UPSAMPLING factor: they map onto the IR's
 * `lhs_dilation` with `window_strides` left at `[1, 1]`, which is exactly how
 * [io.tlaloc.ir.passes.VjpRegistry.Conv2dRule] spells `dX` and how StableHLO
 * expresses a transposed conv. `padTop`/… are the IR's `padding` attr verbatim —
 * applied to the dilated input, and NEGATIVE values crop (this is how the conv
 * adjoint lands back on the primal input's exact extents). Same two-arity,
 * no-defaults contract as [conv2d], for the same reason.
 */
fun <S : Shape> DTensor<S, F64>.convTranspose2d(w: DTensor<*, F64>): DTensor<Shape, F64> =
    convTranspose2dGeneral(this, w, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, false, false)

@Suppress("LongParameterList")
fun <S : Shape> DTensor<S, F64>.convTranspose2d(
    w: DTensor<*, F64>,
    strideH: Int,
    strideW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
): DTensor<Shape, F64> = convTranspose2dGeneral(
    this, w, 1, 1, strideH, strideW, 1, 1, padTop, padBottom, padLeft, padRight, false, false,
)

/**
 * Fixed-arity synthesis delegates: every attr the dxir CONV2D /
 * CONV_TRANSPOSE2D ops carry, as positional `Int`/`Boolean` arguments, with the
 * result's shape witness `S` supplied by the caller (synthesis passes the derived
 * IrType's shape argument). All attrs explicit and none defaulted because these
 * are the calls [io.tlaloc.plugin.DxirToIrSynthesis] builds when it replays a conv
 * gradient body — the attrs come straight off the dxir op, and a delegate that
 * omitted one (say `lhs_dilation`, or `window_reversal`) would silently evaluate a
 * different convolution. The usual IrVararg reason applies: synthesis builds
 * positional `IrCall` arguments and cannot construct an array literal, so the
 * attr lists arrive as scalars.
 */
@Suppress("LongParameterList")
fun <S : Shape> conv2dGeneral(
    x: DTensor<*, F64>,
    w: DTensor<*, F64>,
    strideH: Int,
    strideW: Int,
    lhsDilH: Int,
    lhsDilW: Int,
    rhsDilH: Int,
    rhsDilW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
    revH: Boolean,
    revW: Boolean,
): DTensor<S, F64> {
    @Suppress("UNCHECKED_CAST")
    return conv2dEngine(
        x, w, false, strideH, strideW, lhsDilH, lhsDilW, rhsDilH, rhsDilW,
        padTop, padBottom, padLeft, padRight, revH, revW,
    ) as DTensor<S, F64>
}

@Suppress("LongParameterList")
fun <S : Shape> convTranspose2dGeneral(
    x: DTensor<*, F64>,
    w: DTensor<*, F64>,
    strideH: Int,
    strideW: Int,
    lhsDilH: Int,
    lhsDilW: Int,
    rhsDilH: Int,
    rhsDilW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
    revH: Boolean,
    revW: Boolean,
): DTensor<S, F64> {
    @Suppress("UNCHECKED_CAST")
    return conv2dEngine(
        x, w, true, strideH, strideW, lhsDilH, lhsDilW, rhsDilH, rhsDilW,
        padTop, padBottom, padLeft, padRight, revH, revW,
    ) as DTensor<S, F64>
}

/**
 * The host twin of `OpKind.CONV2D_DATA_ADJOINT`: a 2-D conv's gradient
 * w.r.t. its INPUT, fused into one op.
 *
 * Mathematically it is the lhs-dilated, tap-reversed transposed convolution of
 * [upstream] against [kernel], padded so the result lands exactly on the primal
 * input's extents. That padding is why the op exists at all: it is SOLVED from
 * those extents, and under `grad {}` every extent in the graph is a -1 sentinel,
 * so padding baked at transform time is arithmetic garbage that every consumer
 * would faithfully honour. [xTemplate] therefore contributes SHAPE ONLY — its
 * values are never read — and the solve happens here, at execution, against
 * `dims` that are real:
 *
 *     kEff = (k−1)·d + 1        dilSize = (hOut−1)·s + 1
 *     low  = kEff − 1 − p_low   high    = p_low + H − dilSize
 *
 * with `k` the kernel's spatial extent, `hOut` the upstream's, `H` the target's,
 * `s` [strideH]/[strideW], `d` [rhsDilH]/[rhsDilW] and `p_low` [padTop]/[padLeft]
 * — all read off the primal conv. Note only the primal's LOW padding is needed:
 * its high side is already implicit in the upstream's runtime shape, since that
 * shape is what the primal's floor-division produced.
 *
 * The result's shape witness is [xTemplate]'s `S` — not a guess, because the
 * template IS the shape this op produces.
 */
@Suppress("LongParameterList")
fun <S : Shape> conv2dDataAdjoint(
    upstream: DTensor<*, F64>,
    kernel: DTensor<*, F64>,
    xTemplate: DTensor<S, F64>,
    strideH: Int,
    strideW: Int,
    rhsDilH: Int,
    rhsDilW: Int,
    padTop: Int,
    padLeft: Int,
): DTensor<S, F64> {
    val up = upstream.dims
    val k = kernel.dims
    val target = xTemplate.dims
    val strides = intArrayOf(strideH, strideW)
    val dil = intArrayOf(rhsDilH, rhsDilW)
    val pLow = intArrayOf(padTop, padLeft)
    fun solve(axis: Int): IntArray {
        val kEff = (k[2 + axis] - 1) * dil[axis] + 1
        val dilSize = (up[2 + axis] - 1) * strides[axis] + 1
        val low = kEff - 1 - pLow[axis]
        return intArrayOf(low, pLow[axis] + target[2 + axis] - dilSize)
    }
    val h = solve(0)
    val w = solve(1)
    val dx = conv2dEngine(
        upstream, kernel, true,
        1, 1, strideH, strideW, rhsDilH, rhsDilW,
        h[0], h[1], w[0], w[1], true, true,
    )
    require(dx.dims.contentEquals(target)) {
        "conv2dDataAdjoint: solved padding landed on ${dx.dims.toList()} but the template " +
            "is ${target.toList()} (upstream ${up.toList()}, kernel ${k.toList()}, " +
            "strides [$strideH, $strideW], rhs_dilation [$rhsDilH, $rhsDilW], " +
            "primal padding low [$padTop, $padLeft])"
    }
    @Suppress("UNCHECKED_CAST")
    return dx as DTensor<S, F64>
}

/**
 * The host twin of `OpKind.CONV2D_KERNEL_ADJOINT`: a 2-D conv's gradient
 * w.r.t. its KERNEL, fused into one op.
 *
 * This is the batch↔feature transposed trick — `dW = (Xᵀ ⋆ dYᵀ)ᵀ`, where X reads
 * as `[Ci, N, H, W]` and dY as an OIHW kernel `[Co, N, Ho, Wo]` with `o = Co`,
 * `i = N`, and the primal's stride and rhs_dilation SWAP roles — padded so the
 * inner conv's result is exactly `[Ci, Co, kh, kw]`, then transposed back to
 * OIHW. All three transposes are folded in here, so a gradient body needs no
 * rank-4 TRANSPOSE nodes and the result's shape witness is simply
 * [wTemplate]'s `S`. The padding solve, likewise read from runtime dims:
 *
 *     dilSize = (hOut−1)·s + 1
 *     low     = p_low
 *     high    = (k−1)·d + dilSize − H − p_low
 *
 * with `k` the target kernel's spatial extent and `H` the primal input's. The
 * high side is the crop that absorbs the primal's floor-division remainder (a
 * stride-2 conv over height 5 leaves an unused row), which is why it can be
 * negative. [wTemplate] contributes SHAPE ONLY.
 */
@Suppress("LongParameterList")
fun <S : Shape> conv2dKernelAdjoint(
    x: DTensor<*, F64>,
    upstream: DTensor<*, F64>,
    wTemplate: DTensor<S, F64>,
    strideH: Int,
    strideW: Int,
    rhsDilH: Int,
    rhsDilW: Int,
    padTop: Int,
    padLeft: Int,
): DTensor<S, F64> {
    val xD = x.dims
    val up = upstream.dims
    val target = wTemplate.dims
    val strides = intArrayOf(strideH, strideW)
    val dil = intArrayOf(rhsDilH, rhsDilW)
    val pLow = intArrayOf(padTop, padLeft)
    fun solve(axis: Int): IntArray {
        val dilSize = (up[2 + axis] - 1) * strides[axis] + 1
        val low = pLow[axis]
        return intArrayOf(low, (target[2 + axis] - 1) * dil[axis] + dilSize - xD[2 + axis] - low)
    }
    val h = solve(0)
    val w = solve(1)
    @Suppress("UNCHECKED_CAST")
    val xT = (x as DTensor<Shape, F64>).transpose(1, 0, 2, 3)
    @Suppress("UNCHECKED_CAST")
    val upT = (upstream as DTensor<Shape, F64>).transpose(1, 0, 2, 3)
    // [Ci, Co, kh, kw]: the target with its channel axes swapped.
    val dwt = conv2dEngine(
        xT, upT, false,
        rhsDilH, rhsDilW, 1, 1, strideH, strideW,
        h[0], h[1], w[0], w[1], false, false,
    )
    val dw = dwt.transpose(1, 0, 2, 3)
    require(dw.dims.contentEquals(target)) {
        "conv2dKernelAdjoint: solved padding landed on ${dw.dims.toList()} but the template " +
            "is ${target.toList()} (x ${xD.toList()}, upstream ${up.toList()}, " +
            "strides [$strideH, $strideW], rhs_dilation [$rhsDilH, $rhsDilW], " +
            "primal padding low [$padTop, $padLeft])"
    }
    @Suppress("UNCHECKED_CAST")
    return dw as DTensor<S, F64>
}

/**
 * The shared engine for the two transposed-conv adjoint twins. A port of
 * the interpreter's `evalConvTransposeAdjoint`, kept literal so the host result is
 * bit-exact against the interpreted dxir (same Double accumulator, same loop order,
 * same single Double conversion).
 *
 * No padding solve is involved: the primal's tap maps input↔output through
 * `yDil = yo·s + ky·d − p_low` with `yDil` a multiple of the lhs dilation, so
 * inverting that one equation per tap absorbs the padding, both dilations, the
 * strides and the kernel reversal together. Only the LOW padding is therefore
 * needed, as in [conv2dDataAdjoint].
 *
 * [up] is the upstream and [other] the kernel (IOHW `[Ci, Co, kh, kw]`) when
 * [dataAdj], or `x` when not.
 */
@Suppress("LongParameterList")
private fun convTranspose2dAdjointEngine(
    up: DoubleArray,
    other: DoubleArray,
    dataAdj: Boolean,
    nB: Int,
    cIn: Int,
    h: Int,
    w: Int,
    cOut: Int,
    hOut: Int,
    wOut: Int,
    kh: Int,
    kw: Int,
    strideH: Int,
    strideW: Int,
    lhsDilH: Int,
    lhsDilW: Int,
    rhsDilH: Int,
    rhsDilW: Int,
    padTop: Int,
    padLeft: Int,
    revH: Boolean,
    revW: Boolean,
): DoubleArray = if (dataAdj) {
    val out = DoubleArray(nB * cIn * h * w)
    var outIdx = 0
    for (n in 0 until nB) {
        for (i in 0 until cIn) {
            for (iy in 0 until h) {
                for (ix in 0 until w) {
                    var acc = 0.0
                    for (ky in 0 until kh) {
                        val num = iy * lhsDilH + padTop - ky * rhsDilH
                        if (num < 0 || num % strideH != 0) continue
                        val yo = num / strideH
                        if (yo >= hOut) continue
                        val wKy = if (revH) kh - 1 - ky else ky
                        for (kx in 0 until kw) {
                            val num2 = ix * lhsDilW + padLeft - kx * rhsDilW
                            if (num2 < 0 || num2 % strideW != 0) continue
                            val xo = num2 / strideW
                            if (xo >= wOut) continue
                            val wKx = if (revW) kw - 1 - kx else kx
                            val upBase = ((n * cOut) * hOut + yo) * wOut + xo
                            for (o in 0 until cOut) {
                                acc += up[upBase + o * hOut * wOut].toDouble() *
                                    other[((i * cOut + o) * kh + wKy) * kw + wKx]
                            }
                        }
                    }
                    out[outIdx++] = acc.toDouble()
                }
            }
        }
    }
    out
} else {
    val acc = DoubleArray(cIn * cOut * kh * kw)
    for (n in 0 until nB) {
        for (o in 0 until cOut) {
            for (yo in 0 until hOut) {
                for (xo in 0 until wOut) {
                    val upVal = up[((n * cOut + o) * hOut + yo) * wOut + xo].toDouble()
                    for (i in 0 until cIn) {
                        for (ky in 0 until kh) {
                            val yD = yo * strideH + ky * rhsDilH - padTop
                            if (yD < 0 || yD % lhsDilH != 0) continue
                            val inY = yD / lhsDilH
                            if (inY >= h) continue
                            val wKy = if (revH) kh - 1 - ky else ky
                            for (kx in 0 until kw) {
                                val xD = xo * strideW + kx * rhsDilW - padLeft
                                if (xD < 0 || xD % lhsDilW != 0) continue
                                val inX = xD / lhsDilW
                                if (inX >= w) continue
                                val wKx = if (revW) kw - 1 - kx else kx
                                acc[((i * cOut + o) * kh + wKy) * kw + wKx] +=
                                    upVal * other[((n * cIn + i) * h + inY) * w + inX].toDouble()
                            }
                        }
                    }
                }
            }
        }
    }
    DoubleArray(acc.size) { acc[it].toDouble() }
}

/**
 * The host twin of `OpKind.CONV_TRANSPOSE2D_DATA_ADJOINT`: the gradient
 * of a transposed convolution w.r.t. its INPUT. [xTemplate] contributes SHAPE ONLY.
 *
 * Attrs are the primal transposed conv's, all literals. The StableHLO emitter also
 * has an arm, reached through the conv-of-the-dilated-input identity
 * rather than this index inversion, and it rejects `window_reversal` — which this
 * twin handles, as does the interpreter.
 */
@Suppress("LongParameterList")
fun <S : Shape> convTranspose2dDataAdjoint(
    upstream: DTensor<*, F64>,
    kernel: DTensor<*, F64>,
    xTemplate: DTensor<S, F64>,
    strideH: Int,
    strideW: Int,
    lhsDilH: Int,
    lhsDilW: Int,
    rhsDilH: Int,
    rhsDilW: Int,
    padTop: Int,
    padLeft: Int,
    revH: Boolean,
    revW: Boolean,
): DTensor<S, F64> {
    val up = upstream.dims
    val k = kernel.dims
    val target = xTemplate.dims
    require(up.size == 4 && k.size == 4 && target.size == 4) {
        "convTranspose2dDataAdjoint: rank-4 NCHW upstream, IOHW kernel and template required; " +
            "got ${up.toList()} / ${k.toList()} / ${target.toList()}"
    }
    require(up[0] == target[0] && k[0] == target[1] && k[1] == up[1]) {
        "convTranspose2dDataAdjoint: channel/batch mismatch — upstream ${up.toList()}, " +
            "kernel ${k.toList()}, x ${target.toList()}"
    }
    val dx = convTranspose2dAdjointEngine(
        upstream.hostF64(), kernel.hostF64(), true,
        target[0], target[1], target[2], target[3],
        up[1], up[2], up[3], k[2], k[3],
        strideH, strideW, lhsDilH, lhsDilW, rhsDilH, rhsDilW,
        padTop, padLeft, revH, revW,
    )
    @Suppress("UNCHECKED_CAST")
    return DTensor<Shape, F64>(HostF64Storage(dx), target.copyOf(), F64) as DTensor<S, F64>
}

/**
 * The host twin of `OpKind.CONV_TRANSPOSE2D_KERNEL_ADJOINT`: the gradient
 * of a transposed convolution w.r.t. its IOHW KERNEL. [x] is a VALUE operand here (the
 * gather reads it); [wTemplate] contributes shape only.
 */
@Suppress("LongParameterList")
fun <S : Shape> convTranspose2dKernelAdjoint(
    x: DTensor<*, F64>,
    upstream: DTensor<*, F64>,
    wTemplate: DTensor<S, F64>,
    strideH: Int,
    strideW: Int,
    lhsDilH: Int,
    lhsDilW: Int,
    rhsDilH: Int,
    rhsDilW: Int,
    padTop: Int,
    padLeft: Int,
    revH: Boolean,
    revW: Boolean,
): DTensor<S, F64> {
    val xd = x.dims
    val up = upstream.dims
    val target = wTemplate.dims
    require(xd.size == 4 && up.size == 4 && target.size == 4) {
        "convTranspose2dKernelAdjoint: rank-4 NCHW x, NCHW upstream and IOHW template " +
            "required; got ${xd.toList()} / ${up.toList()} / ${target.toList()}"
    }
    require(up[0] == xd[0] && target[0] == xd[1] && target[1] == up[1]) {
        "convTranspose2dKernelAdjoint: channel/batch mismatch — x ${xd.toList()}, " +
            "upstream ${up.toList()}, kernel ${target.toList()}"
    }
    val dw = convTranspose2dAdjointEngine(
        upstream.hostF64(), x.hostF64(), false,
        xd[0], xd[1], xd[2], xd[3],
        up[1], up[2], up[3], target[2], target[3],
        strideH, strideW, lhsDilH, lhsDilW, rhsDilH, rhsDilW,
        padTop, padLeft, revH, revW,
    )
    @Suppress("UNCHECKED_CAST")
    return DTensor<Shape, F64>(HostF64Storage(dw), target.copyOf(), F64) as DTensor<S, F64>
}

/**
 * The host pooling engine: a port of the dxir interpreter's
 * `evalPool2d`, kept deliberately literal so the host result is bit-exact against
 * the interpreted dxir (same `Double` accumulator, same `n → c → y → x → ky → kx`
 * order, same single Double conversion at the end). It covers
 * both pooling kinds with an [isMax] flag, mirroring the interpreter's
 * own single-`evalPool2d`-two-kinds structure.
 *
 * count_include_pad for the average branch, the interpreter's convention: the sum
 * divides by the FULL window `kh·kw`, padding included, so a window hanging off
 * the edge is averaged over taps that were not there. That choice is what makes
 * [avgPool2dGrad]'s uniform `1/(kh·kw)` spread the exact adjoint.
 */
private fun pool2dEngine(
    x: DTensor<*, F64>,
    isMax: Boolean,
    windowH: Int,
    windowW: Int,
    strideH: Int,
    strideW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
): DTensor<Shape, F64> {
    val opName = if (isMax) "maxPool2d" else "avgPool2d"
    val d = x.dims
    require(d.size == 4) { "$opName: rank-4 NCHW input required; got ${d.toList()}" }
    require(windowH > 0 && windowW > 0 && strideH > 0 && strideW > 0) {
        "$opName: window and strides must be positive; got window [$windowH, $windowW], " +
            "strides [$strideH, $strideW]"
    }
    val nB = d[0]
    val c = d[1]
    val h = d[2]
    val w = d[3]
    val hOut = (h + padTop + padBottom - windowH) / strideH + 1
    val wOut = (w + padLeft + padRight - windowW) / strideW + 1
    require(hOut > 0 && wOut > 0) {
        "$opName: derived output extents [$hOut, $wOut] are empty — window " +
            "[$windowH, $windowW] does not fit the padded input [$h, $w]"
    }
    val v = x.hostF64()
    val windowSize = windowH * windowW
    val out = DoubleArray(nB * c * hOut * wOut)
    var outIdx = 0
    for (n in 0 until nB) {
        for (ch in 0 until c) {
            val planeBase = (n * c + ch) * h * w
            for (y in 0 until hOut) {
                for (xo in 0 until wOut) {
                    var acc = if (isMax) Double.NEGATIVE_INFINITY else 0.0
                    for (ky in 0 until windowH) {
                        val inY = y * strideH + ky - padTop
                        if (inY < 0 || inY >= h) continue
                        for (kx in 0 until windowW) {
                            val inX = xo * strideW + kx - padLeft
                            if (inX < 0 || inX >= w) continue
                            val e = v[planeBase + inY * w + inX].toDouble()
                            acc = if (isMax) maxOf(acc, e) else acc + e
                        }
                    }
                    out[outIdx++] = if (isMax) acc.toDouble() else (acc / windowSize).toDouble()
                }
            }
        }
    }
    return DTensor(HostF64Storage(out), intArrayOf(nB, c, hOut, wOut), F64)
}

/**
 * 2-D average pooling, NCHW. Two arities and no default parameter
 * values, for the reason documented on [conv2d]: K2 unwraps a named argument
 * before the plugin's FIR lowering sees it and does not reorder it, so attrs must
 * be positional to be unambiguous. `avgPool2d(windowH, windowW)` is the
 * non-overlapping pool (strides default to the window — the interpreter's own
 * default, and PyTorch's `AvgPool2d(k)` shape); the 8-argument form adds strides
 * and all four padding sides. The result erases to `DTensor<Shape, F64>` like
 * every other extent-changing host op.
 */
fun <S : Shape> DTensor<S, F64>.avgPool2d(windowH: Int, windowW: Int): DTensor<Shape, F64> =
    avgPool2dGeneral(this, windowH, windowW, windowH, windowW, 0, 0, 0, 0)

@Suppress("LongParameterList")
fun <S : Shape> DTensor<S, F64>.avgPool2d(
    windowH: Int,
    windowW: Int,
    strideH: Int,
    strideW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
): DTensor<Shape, F64> = avgPool2dGeneral(
    this, windowH, windowW, strideH, strideW, padTop, padBottom, padLeft, padRight,
)

/**
 * Fixed-arity synthesis delegate for `OpKind.AVGPOOL2D`, every attr
 * explicit and positional (the usual IrVararg reason, and the result's shape
 * witness `S` comes from the caller's derived IrType).
 */
@Suppress("LongParameterList")
fun <S : Shape> avgPool2dGeneral(
    x: DTensor<*, F64>,
    windowH: Int,
    windowW: Int,
    strideH: Int,
    strideW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
): DTensor<S, F64> {
    @Suppress("UNCHECKED_CAST")
    return pool2dEngine(
        x, false, windowH, windowW, strideH, strideW, padTop, padBottom, padLeft, padRight,
    ) as DTensor<S, F64>
}

/**
 * The host twin of `OpKind.AVGPOOL2D_GRAD`: each input element collects
 * the upstream of every output window covering it, divided by the FULL window
 * `kh·kw` (count_include_pad, mirroring [avgPool2dEngine]).
 *
 * The covering output is found by INVERTING the window rather than by solving a
 * padding: for target row `iy` and tap `ky` it is `(iy + padTop − ky) / strideH`
 * when that divides evenly and lands in range. So the only attrs are literal facts
 * off the primal, [xTemplate] contributes SHAPE ONLY (its values are never read),
 * and nothing needs an extent at compile time — which is the whole point, since
 * under `grad {}` every extent is a -1 sentinel. Only the LOW padding participates:
 * the high side is implied by the upstream's runtime shape, exactly as for
 * [conv2dDataAdjoint].
 */
@Suppress("LongParameterList")
fun <S : Shape> avgPool2dGrad(
    upstream: DTensor<*, F64>,
    xTemplate: DTensor<S, F64>,
    windowH: Int,
    windowW: Int,
    strideH: Int,
    strideW: Int,
    padTop: Int,
    padLeft: Int,
): DTensor<S, F64> {
    val up = upstream.dims
    val target = xTemplate.dims
    require(up.size == 4 && target.size == 4) {
        "avgPool2dGrad: rank-4 NCHW upstream and template required; got " +
            "${up.toList()} / ${target.toList()}"
    }
    require(windowH > 0 && windowW > 0 && strideH > 0 && strideW > 0) {
        "avgPool2dGrad: window and strides must be positive; got window [$windowH, $windowW], " +
            "strides [$strideH, $strideW]"
    }
    require(up[0] == target[0] && up[1] == target[1]) {
        "avgPool2dGrad: upstream batch/channels [${up[0]}, ${up[1]}] ≠ target " +
            "[${target[0]}, ${target[1]}]"
    }
    val nB = target[0]
    val c = target[1]
    val h = target[2]
    val w = target[3]
    val hOut = up[2]
    val wOut = up[3]
    val uv = upstream.hostF64()
    val windowSize = windowH * windowW
    val out = DoubleArray(nB * c * h * w)
    var outIdx = 0
    for (n in 0 until nB) {
        for (ch in 0 until c) {
            val upBase = (n * c + ch) * hOut * wOut
            for (iy in 0 until h) {
                for (ix in 0 until w) {
                    var acc = 0.0
                    for (ky in 0 until windowH) {
                        val dy = iy + padTop - ky
                        if (dy < 0 || dy % strideH != 0) continue
                        val y = dy / strideH
                        if (y >= hOut) continue
                        for (kx in 0 until windowW) {
                            val dx = ix + padLeft - kx
                            if (dx < 0 || dx % strideW != 0) continue
                            val x = dx / strideW
                            if (x >= wOut) continue
                            acc += uv[upBase + y * wOut + x].toDouble()
                        }
                    }
                    out[outIdx++] = (acc / windowSize).toDouble()
                }
            }
        }
    }
    @Suppress("UNCHECKED_CAST")
    return DTensor<Shape, F64>(HostF64Storage(out), target.copyOf(), F64) as DTensor<S, F64>
}

/**
 * 2-D max pooling, NCHW. Same two-arity, no-defaults contract as
 * [avgPool2d] and for the same reason (K2 unwraps named arguments before the
 * plugin's FIR lowering sees them, so attrs must be positional to be
 * unambiguous): `maxPool2d(windowH, windowW)` is the classic non-overlapping pool
 * (strides default to the window, PyTorch's `MaxPool2d(k)` shape), and the
 * 8-argument form adds strides and all four padding sides.
 */
fun <S : Shape> DTensor<S, F64>.maxPool2d(windowH: Int, windowW: Int): DTensor<Shape, F64> =
    maxPool2dGeneral(this, windowH, windowW, windowH, windowW, 0, 0, 0, 0)

@Suppress("LongParameterList")
fun <S : Shape> DTensor<S, F64>.maxPool2d(
    windowH: Int,
    windowW: Int,
    strideH: Int,
    strideW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
): DTensor<Shape, F64> = maxPool2dGeneral(
    this, windowH, windowW, strideH, strideW, padTop, padBottom, padLeft, padRight,
)

/** Fixed-arity synthesis delegate for `OpKind.MAXPOOL2D`. */
@Suppress("LongParameterList")
fun <S : Shape> maxPool2dGeneral(
    x: DTensor<*, F64>,
    windowH: Int,
    windowW: Int,
    strideH: Int,
    strideW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
): DTensor<S, F64> {
    @Suppress("UNCHECKED_CAST")
    return pool2dEngine(
        x, true, windowH, windowW, strideH, strideW, padTop, padBottom, padLeft, padRight,
    ) as DTensor<S, F64>
}

/**
 * The host twin of `OpKind.MAXPOOL2D_GRAD`: each input element receives
 * the upstream of every output window it WINS, i.e. every window whose max it
 * equals. Ties route the full upstream to every winner (the MaxRule/JAX-select
 * convention) — deliberately not XLA's `select_and_scatter`, which picks a single
 * winner and would make the GPU gradient disagree with the host one on exact ties,
 * which are common after a relu.
 *
 * Like [avgPool2dGrad] this INVERTS the window per input element rather than
 * nearest-upsampling the pooled value and the upstream back to x's shape and
 * masking — the rank-6 formulation that kept maxpool out of `grad {}`, since those
 * intermediates bake extents that are -1 sentinels there. Nothing here reads a
 * compile-time extent.
 *
 * [y] is the pooled value `maxPool2d(x)`, supplied rather than recomputed (the rule
 * materialises it in the gradient body). Unlike the conv/avgpool templates, [x] is
 * a VALUE operand — its elements are compared against [y] — though its extents
 * still come from runtime `dims`.
 */
@Suppress("LongParameterList")
fun <S : Shape> maxPool2dGrad(
    upstream: DTensor<*, F64>,
    x: DTensor<*, F64>,
    y: DTensor<*, F64>,
    windowH: Int,
    windowW: Int,
    strideH: Int,
    strideW: Int,
    padTop: Int,
    padLeft: Int,
): DTensor<S, F64> {
    val up = upstream.dims
    val target = x.dims
    require(up.size == 4 && target.size == 4 && y.dims.size == 4) {
        "maxPool2dGrad: rank-4 NCHW upstream/x/y required; got " +
            "${up.toList()} / ${target.toList()} / ${y.dims.toList()}"
    }
    require(windowH > 0 && windowW > 0 && strideH > 0 && strideW > 0) {
        "maxPool2dGrad: window and strides must be positive; got window [$windowH, $windowW], " +
            "strides [$strideH, $strideW]"
    }
    require(up[0] == target[0] && up[1] == target[1]) {
        "maxPool2dGrad: upstream batch/channels [${up[0]}, ${up[1]}] ≠ x's " +
            "[${target[0]}, ${target[1]}]"
    }
    val nB = target[0]
    val c = target[1]
    val h = target[2]
    val w = target[3]
    val hOut = up[2]
    val wOut = up[3]
    val uv = upstream.hostF64()
    val xv = x.hostF64()
    val yv = y.hostF64()
    val out = DoubleArray(nB * c * h * w)
    var outIdx = 0
    for (n in 0 until nB) {
        for (ch in 0 until c) {
            val plane = (n * c + ch) * h * w
            val upPlane = (n * c + ch) * hOut * wOut
            for (iy in 0 until h) {
                for (ix in 0 until w) {
                    val xe = xv[plane + iy * w + ix]
                    var acc = 0.0
                    for (ky in 0 until windowH) {
                        val dy = iy + padTop - ky
                        if (dy < 0 || dy % strideH != 0) continue
                        val oy = dy / strideH
                        if (oy >= hOut) continue
                        for (kx in 0 until windowW) {
                            val dx = ix + padLeft - kx
                            if (dx < 0 || dx % strideW != 0) continue
                            val ox = dx / strideW
                            if (ox >= wOut) continue
                            val wIdx = upPlane + oy * wOut + ox
                            if (xe == yv[wIdx]) acc += uv[wIdx].toDouble()
                        }
                    }
                    out[outIdx++] = acc.toDouble()
                }
            }
        }
    }
    @Suppress("UNCHECKED_CAST")
    return DTensor<Shape, F64>(HostF64Storage(out), target.copyOf(), F64) as DTensor<S, F64>
}

/**
 * TRAINING-mode batch normalisation, NCHW with the feature axis at 1:
 * per channel, subtract the mean and divide by the standard deviation computed
 * over the batch AND spatial extents of this call, then apply the per-channel
 * [scale] (γ) and [offset] (β).
 *
 * Two conventions worth stating because the AD path inherits both: the variance is
 * BIASED (divide by `N·H·W`, i.e. `mean((x−μ)²)`, matching PyTorch's training-mode
 * normalisation — the unbiased correction belongs only in the running-variance
 * update, which is not this function's job), and the statistics come from the
 * argument rather than from running estimates. That distinguishes it from
 * `OpKind.BATCHNORM`, which is the INFERENCE form (five operands: input, scale,
 * offset, mean, variance) that the Layer-3 recognizer emits for fused kernels.
 *
 * `grad {}` gets this without a VjpRule: the plugin's FIR lowering desugars the
 * call into MEAN/SUB/MUL/ADD/SQRT/DIV nodes whose adjoints already exist and are
 * sentinel-safe (the `maximum`/`clip` pattern). This host function is the
 * plain-runtime twin of that desugaring and does the same arithmetic.
 */
fun <S : Shape> DTensor<S, F64>.batchNorm(
    scale: DTensor<*, F64>,
    offset: DTensor<*, F64>,
): DTensor<Shape, F64> = batchNormGeneral(this, scale, offset, 1.0e-5)

fun <S : Shape> DTensor<S, F64>.batchNorm(
    scale: DTensor<*, F64>,
    offset: DTensor<*, F64>,
    eps: Double,
): DTensor<Shape, F64> = batchNormGeneral(this, scale, offset, eps)

/** Fixed-arity form with an explicit `eps`. Rank-4 NCHW only in v1. */
fun <S : Shape> batchNormGeneral(
    x: DTensor<*, F64>,
    scale: DTensor<*, F64>,
    offset: DTensor<*, F64>,
    eps: Double,
): DTensor<S, F64> {
    val d = x.dims
    require(d.size == 4) { "batchNorm: rank-4 NCHW input required; got ${d.toList()}" }
    val nB = d[0]
    val c = d[1]
    val h = d[2]
    val w = d[3]
    require(
        scale.dims.size == 1 && scale.dims[0] == c && offset.dims.size == 1 && offset.dims[0] == c
    ) {
        "batchNorm: scale/offset must be rank-1 [C=$c]; got " +
            "${scale.dims.toList()} / ${offset.dims.toList()}"
    }
    val spatial = h * w
    val count = (nB * spatial).toDouble()
    require(count > 0.0) { "batchNorm: empty input ${d.toList()}" }
    val v = x.hostF64()
    val g = scale.hostF64()
    val b = offset.hostF64()
    val out = DoubleArray(v.size)
    for (ch in 0 until c) {
        // μ over this channel's batch and spatial extents.
        var sum = 0.0
        for (n in 0 until nB) {
            val base = (n * c + ch) * spatial
            for (i in 0 until spatial) sum += v[base + i].toDouble()
        }
        val mean = sum / count
        // Biased variance: mean((x − μ)²).
        var sq = 0.0
        for (n in 0 until nB) {
            val base = (n * c + ch) * spatial
            for (i in 0 until spatial) {
                val e = v[base + i].toDouble() - mean
                sq += e * e
            }
        }
        val invStd = 1.0 / sqrt(sq / count + eps.toDouble())
        val gamma = g[ch].toDouble()
        val beta = b[ch].toDouble()
        for (n in 0 until nB) {
            val base = (n * c + ch) * spatial
            for (i in 0 until spatial) {
                out[base + i] = ((v[base + i].toDouble() - mean) * invStd * gamma + beta).toDouble()
            }
        }
    }
    @Suppress("UNCHECKED_CAST")
    return DTensor<Shape, F64>(HostF64Storage(out), d.copyOf(), F64) as DTensor<S, F64>
}
