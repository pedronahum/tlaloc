/**
 * The conv stack, in DiffKT's semantics, on
 * Tlaloc's native layouts: `Conv2d` (NO bias tensor — DiffKT ships none; the
 * activation composes post-op), `Conv2dWithSamePadding` (the subclass sugar),
 * `MaxPool2d` and `AvgPool2d` (window = stride, spatial divisibility required —
 * DiffKT's own contract).
 *
 * Layout: `:nn` adopts
 * NCHW input / OIHW `[Co, Ci, kh, kw]` filters — the layouts every Tlaloc
 * engine (interpreter, emitter, host twins) already speaks. DiffKT's
 * NHWC / `[Co, kh, kw, Ci]` is a layout transpose of the same maths, and the
 * PyTorch oracle the tests use is NCHW-native anyway. The fan computation is unaffected:
 * DiffKT's `fan = shape[1] · shape.drop(2).product` on ITS layout equals
 * `Ci·kh·kw`, exactly what the same formula yields on OIHW.
 *
 * All forwards are TRACE spellings (see the compiler-route contract in
 * `Training.kt`): conv2d /
 * maxPool2d / avgPool2d record onto the `:autograd` tape with the
 * interpreter's exact attrs, and `DxirReverseTransform` differentiates them
 * through the fused conv and pooling adjoints. Zero gradient math in this file.
 */
package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.split
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.avgPool2d
import io.tlaloc.autograd.conv2d
import io.tlaloc.autograd.maxPool2d
import kotlin.math.max
import kotlin.math.sqrt

/**
 * DiffKT's `PaddingStyle`, all four variants: `Valid` (no padding), `Same`
 * (TF SAME — output spatial extent `ceil(in/stride)`), `Full` (pad `k − 1` on
 * every side), `Explicit(top, bottom, left, right)`.
 */
sealed class PaddingStyle {
    object Valid : PaddingStyle()
    object Same : PaddingStyle()
    object Full : PaddingStyle()
    class Explicit(val top: Int, val bottom: Int, val left: Int, val right: Int) : PaddingStyle() {
        init {
            require(top >= 0 && bottom >= 0 && left >= 0 && right >= 0) {
                "PaddingStyle.Explicit: padding must be non-negative; got [$top, $bottom, $left, $right]"
            }
        }
    }
}

/**
 * DiffKT's TF-SAME split for one spatial axis (as in DiffKT's
 * source): total `= if (in % stride == 0) max(k − stride, 0) else
 * max(k − in % stride, 0)`, then `before = total / 2`, `after = total −
 * before` — the odd unit lands on the AFTER side (bottom/right), TF's own
 * convention. Exposed internal so the formula itself is pinned by a test,
 * independent of any conv gradient.
 */
internal fun samePadding(inSize: Int, kernel: Int, stride: Int): Pair<Int, Int> {
    val rem = inSize % stride
    val total = if (rem == 0) max(kernel - stride, 0) else max(kernel - rem, 0)
    val before = total / 2
    return before to (total - before)
}

/**
 * DiffKT's `Conv2d(filterShape, hStride, vStride, activation, paddingStyle,
 * random)` on Tlaloc layouts: one trainable `filter [Co, Ci, kh, kw]` (OIHW),
 * NO bias tensor (DiffKT ships none), forward
 * `activation(conv2d(x, filter, strides, padding))` over NCHW input.
 *
 * Stride naming follows DiffKT: [vStride] strides the ROWS (H, vertical),
 * [hStride] the COLUMNS (W, horizontal). `Same` padding is computed here at
 * the layer level from the input's spatial dims — per axis, per the
 * [samePadding] formula — and passed to the trace spelling as EXPLICIT attrs,
 * so the captured graph only ever carries literal padding.
 *
 * Groups: the layer is groups = 1 (a DiffKT-parity fact — DiffKT has no groups
 * parameter). The IR's CONV2D supports `feature_group_count` end to end, but
 * the host twin the trace forward routes through rejects grouped convolution,
 * so the layer has no groups parameter yet.
 *
 * The randomly-initialized form is the companion `invoke` — DiffKT's conv
 * default `kaimingUniform(FanIn, LeakyRelu(sqrt(5)))` (the PyTorch conv
 * default). Key discipline (the `:nn` convention): the layer splits its key once into one
 * child per parameter tensor in declaration order — one parameter here, so
 * `split(1)[0]` → filter.
 */
class Conv2d(
    val filter: DTensor<*, F32>,
    val hStride: Int = 1,
    val vStride: Int = 1,
    val activation: Activation = Activation.Identity,
    val paddingStyle: PaddingStyle = PaddingStyle.Valid,
) : TrainableLayer<Conv2d> {

    init {
        require(filter.dims.size == 4) {
            "Conv2d: filter must be rank-4 OIHW [Co, Ci, kh, kw] (got dims ${filter.dims.toList()})"
        }
        require(hStride > 0 && vStride > 0) {
            "Conv2d: strides must be positive; got hStride=$hStride vStride=$vStride"
        }
    }

    override val parameters: List<NamedParameter> = listOf(NamedParameter("filter", filter))

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): Conv2d {
        val unknown = updated.keys - setOf("filter")
        require(unknown.isEmpty()) { "Conv2d.withParameters: unknown keys $unknown" }
        return Conv2d(updated["filter"] ?: filter, hStride, vStride, activation, paddingStyle)
    }

    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> {
        require(x.rank == 4) {
            "Conv2d: input must be rank-4 NCHW (got dims ${x.dims.toList()})"
        }
        val kh = filter.dims[2]
        val kw = filter.dims[3]
        val (padTop, padBottom, padLeft, padRight) = when (val p = paddingStyle) {
            PaddingStyle.Valid -> intArrayOf(0, 0, 0, 0)
            PaddingStyle.Same -> {
                val (t, b) = samePadding(x.dims[2], kh, vStride)
                val (l, r) = samePadding(x.dims[3], kw, hStride)
                intArrayOf(t, b, l, r)
            }
            PaddingStyle.Full -> intArrayOf(kh - 1, kh - 1, kw - 1, kw - 1)
            is PaddingStyle.Explicit -> intArrayOf(p.top, p.bottom, p.left, p.right)
        }
        val z = x.conv2d<Shape>(params["filter"], vStride, hStride, padTop, padBottom, padLeft, padRight)
        return activation.apply(z)
    }

    companion object {
        /** The DiffKT constructor surface: draw the filter from [key] per the class KDoc. */
        operator fun invoke(
            filterShape: IntArray,
            key: RandomKey,
            hStride: Int = 1,
            vStride: Int = 1,
            activation: Activation = Activation.Identity,
            paddingStyle: PaddingStyle = PaddingStyle.Valid,
        ): Conv2d {
            require(filterShape.size == 4 && filterShape.all { it > 0 }) {
                "Conv2d: filterShape must be four positive dims OIHW [Co, Ci, kh, kw] " +
                    "(got ${filterShape.toList()})"
            }
            val filter = kaimingUniformInit(
                key.split(1)[0],
                filterShape.copyOf(),
                FanMode.FanIn,
                ActivationGain.LeakyRelu(sqrt(5f)),
            )
            return Conv2d(filter, hStride, vStride, activation, paddingStyle)
        }
    }
}

/** DiffKT's `Conv2dWithSamePadding`: `Conv2d` with `PaddingStyle.Same` baked in. */
class Conv2dWithSamePadding private constructor(private val inner: Conv2d) :
    TrainableLayer<Conv2dWithSamePadding> {

    constructor(
        filter: DTensor<*, F32>,
        hStride: Int = 1,
        vStride: Int = 1,
        activation: Activation = Activation.Identity,
    ) : this(Conv2d(filter, hStride, vStride, activation, PaddingStyle.Same))

    val filter: DTensor<*, F32> get() = inner.filter

    override val parameters: List<NamedParameter> get() = inner.parameters

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): Conv2dWithSamePadding =
        Conv2dWithSamePadding(inner.withParameters(updated))

    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> = inner.forward(x, params)
}

/**
 * DiffKT's `MaxPool2d(poolHeight, poolWidth)`: window = stride, zero padding,
 * NCHW rank-4, spatial dims must divide by the pool (the trace spelling keeps
 * DiffKT's own require). Not trainable. Tie note: the
 * MAXPOOL2D_GRAD adjoint routes upstream to ALL within-window ties, PyTorch to
 * the first — oracles pin on tie-free grids.
 */
class MaxPool2d(val poolHeight: Int, val poolWidth: Int) : Layer {
    init {
        require(poolHeight > 0 && poolWidth > 0) {
            "MaxPool2d: pool must be positive; got [$poolHeight, $poolWidth]"
        }
    }

    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> =
        x.maxPool2d(poolHeight, poolWidth)
}

/**
 * DiffKT's `AvgPool2d(poolHeight, poolWidth)`: the mean per non-overlapping
 * window, same shape contract as [MaxPool2d]. Not trainable.
 */
class AvgPool2d(val poolHeight: Int, val poolWidth: Int) : Layer {
    init {
        require(poolHeight > 0 && poolWidth > 0) {
            "AvgPool2d: pool must be positive; got [$poolHeight, $poolWidth]"
        }
    }

    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> =
        x.avgPool2d(poolHeight, poolWidth)
}
