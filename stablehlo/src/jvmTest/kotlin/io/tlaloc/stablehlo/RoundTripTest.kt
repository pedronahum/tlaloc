package io.tlaloc.stablehlo

import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.capture
import io.tlaloc.autograd.capture2
import io.tlaloc.autograd.matmul
import io.tlaloc.autograd.relu
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirModule
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.core.F32
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Round-trip every representative emission through `stablehlo-translate`.
 *
 * Requires the binary on PATH. All tests self-skip with a clear assumption
 * message if the tool is unavailable, so builds stay green on boxes without
 * StableHLO installed.
 */
class RoundTripTest {

    private fun requireTranslateOrSkip() {
        assumeTrue(
            StablehloTranslate.available,
            "stablehlo-translate not on PATH — skipping round-trip validation. " +
                "Install from github.com/openxla/stablehlo and symlink the binary.",
        )
    }

    private fun validate(mlir: String, label: String) {
        // --serialize parses + verifies + emits bytecode keyed to a specific StableHLO
        // target version (current stable is 1.0.0). Any invalid syntax / type / op
        // semantics fails non-zero. Successful serialization == syntactically and
        // semantically valid StableHLO.
        val result = StablehloTranslate.run(mlir, "--serialize", "--target=1.0.0")
        assertTrue(
            result.ok,
            "stablehlo-translate rejected '$label':\n  exit=${result.exitCode}\n  stderr=${result.stderr}\n--- input ---\n$mlir",
        )
        assertTrue(result.stdout.isNotEmpty(), "$label: serialized bytecode was empty")
    }

    @Test
    fun capturedAutogradFunctionRoundTrips() {
        requireTranslateOrSkip()
        val fn = capture2(
            f = { w: Tracer<Rank2<Sym, Sym>>, x: Tracer<Rank2<Sym, Sym>> ->
                (w matmul x).relu().sum()
            },
            a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f)),
            b = Tensors.f32Matrix<Sym, Sym>(2, 1, floatArrayOf(1f, 1f)),
            name = "layer",
        )
        validate(DxirModule(listOf(fn)).toStablehlo(), "layer")
    }

    @Test
    fun vectorPipelineRoundTrips() {
        requireTranslateOrSkip()
        val fn = capture(
            f = { x: Tracer<Rank1<Sym>> -> (x * x).sum() },
            input = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f, 4f)),
            name = "sumSq",
        )
        validate(DxirModule(listOf(fn)).toStablehlo(), "sumSq")
    }

    @Test
    fun scalarPipelineRoundTrips() {
        requireTranslateOrSkip()
        val fn = capture(
            f = { x: Tracer<ScalarShape> -> x * x },
            input = Tensors.f32Scalar(3f),
            name = "sq",
        )
        validate(DxirModule(listOf(fn)).toStablehlo(), "sq")
    }

    @Test
    fun elementwiseBinaryOpsRoundTrip() {
        requireTranslateOrSkip()
        for (op in listOf(OpKind.ADD, OpKind.SUB, OpKind.MUL, OpKind.DIV, OpKind.POW)) {
            val fn = DxirBuilder.function("f") {
                val a = param("a", DxirType(F32, listOf(4)))
                val b = param("b", DxirType(F32, listOf(4)))
                val c = op(op, listOf(a, b), DxirType(F32, listOf(4)))
                listOf(c)
            }
            validate(DxirModule(listOf(fn)).toStablehlo(), op.name)
        }
    }

    @Test
    fun floatArrayConstsRoundTrip() {
        // §0.4.73 — captured rank-N `FloatArray` consts (introduced by §0.4.71's
        // Capture fix) now emit as nested dense<...> literals. Round-trips
        // through `stablehlo-translate --serialize` to confirm the literal is
        // MLIR-valid at rank 1, rank 2, and rank 3.
        requireTranslateOrSkip()
        val cases = listOf(
            "rank-1 [3]" to (floatArrayOf(1f, 2f, 3f) to listOf(3)),
            "rank-1 [4]" to (floatArrayOf(1f, 2f, 3f, 4f) to listOf(4)),
            "rank-2 [2, 2]" to (floatArrayOf(1f, 2f, 3f, 4f) to listOf(2, 2)),
            "rank-2 [3, 2]" to (floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f) to listOf(3, 2)),
            "rank-3 [2, 2, 2]" to (floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f) to listOf(2, 2, 2)),
        )
        for ((label, pair) in cases) {
            val (values, dims) = pair
            val fn = DxirBuilder.function("c") {
                val c = const(values, DxirType(F32, dims))
                listOf(c)
            }
            validate(DxirModule(listOf(fn)).toStablehlo(), "FloatArray const $label")
        }
    }

    @Test
    fun booleanOpsRoundTrip() {
        // §0.4.61 — paired with §0.4.60's NOT/LAND emitter arms. The internal
        // `EmitterTest.emitsBooleanOps` only string-matches the output; this test
        // sends the emitted MLIR through `stablehlo-translate --serialize` so we
        // catch any invalid MLIR syntax / type / op semantics (e.g. a botched
        // `tensor<i1>` printing, a typo in the op name). Three shapes: rank-1
        // Bool NOT, rank-1 Bool LAND, and scalar Bool LAND — the last matches the
        // shape the §0.4.50 break-hoist cond region terminates with.
        requireTranslateOrSkip()
        val boolRank1 = DxirType(io.tlaloc.core.Bool, listOf(4))
        val boolScalar = DxirType(io.tlaloc.core.Bool, emptyList())

        val notFn = DxirBuilder.function("not_rank1") {
            val a = param("a", boolRank1)
            val b = op(OpKind.NOT, listOf(a), boolRank1)
            listOf(b)
        }
        validate(DxirModule(listOf(notFn)).toStablehlo(), "NOT rank-1 i1")

        for ((label, t) in listOf("rank-1 i1" to boolRank1, "scalar i1" to boolScalar)) {
            val fn = DxirBuilder.function("land") {
                val a = param("a", t)
                val b = param("b", t)
                val c = op(OpKind.LAND, listOf(a, b), t)
                listOf(c)
            }
            validate(DxirModule(listOf(fn)).toStablehlo(), "LAND $label")
        }
    }

    @Test
    fun elementwiseUnaryOpsRoundTrip() {
        requireTranslateOrSkip()
        val unaries = listOf(
            OpKind.NEG, OpKind.ABS, OpKind.EXP, OpKind.LOG,
            OpKind.SQRT, OpKind.RSQRT, OpKind.TANH, OpKind.SIGMOID, OpKind.RELU, OpKind.STEP,
        )
        for (op in unaries) {
            val fn = DxirBuilder.function("f") {
                val a = param("a", DxirType(F32, listOf(4)))
                val b = op(op, listOf(a), DxirType(F32, listOf(4)))
                listOf(b)
            }
            validate(DxirModule(listOf(fn)).toStablehlo(), op.name)
        }
    }

    @Test
    fun matmulShapeVariantsRoundTrip() {
        requireTranslateOrSkip()
        val shapes = listOf(
            Triple(2, 3, 4),  // 2x3 @ 3x4 -> 2x4
            Triple(1, 5, 1),  // 1x5 @ 5x1 -> 1x1
            Triple(8, 8, 8),  // 8x8 @ 8x8 -> 8x8
        )
        for ((m, k, n) in shapes) {
            val fn = DxirBuilder.function("mm_${m}_${k}_${n}") {
                val a = param("a", DxirType(F32, listOf(m, k)))
                val b = param("b", DxirType(F32, listOf(k, n)))
                val c = op(OpKind.MATMUL, listOf(a, b), DxirType(F32, listOf(m, n)))
                listOf(c)
            }
            validate(DxirModule(listOf(fn)).toStablehlo(), "matmul ${m}x${k} @ ${k}x${n}")
        }
    }

    @Test
    fun reduceOpsRoundTrip() {
        requireTranslateOrSkip()
        for (op in listOf(OpKind.SUM, OpKind.MEAN, OpKind.MAX, OpKind.MIN)) {
            val fn = DxirBuilder.function("r") {
                val a = param("a", DxirType(F32, listOf(2, 3)))
                val b = op(op, listOf(a), DxirType(F32, emptyList()))
                listOf(b)
            }
            validate(DxirModule(listOf(fn)).toStablehlo(), op.name)
        }
    }

    @Test
    fun castRoundTrips() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("c") {
            val x = param("x", DxirType(io.tlaloc.core.I32, listOf(4)))
            val y = op(OpKind.CAST, listOf(x), DxirType(F32, listOf(4)))
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "CAST i32→f32")
    }

    @Test
    fun reshapeRoundTrips() {
        requireTranslateOrSkip()
        val shapes = listOf(
            listOf(2, 3) to listOf(6),
            listOf(6) to listOf(2, 3),
            listOf(2, 2, 2) to listOf(8),
            listOf(1, 8, 1) to listOf(2, 4),
        )
        for ((from, to) in shapes) {
            val fn = DxirBuilder.function("rs") {
                val x = param("x", DxirType(F32, from))
                val y = op(OpKind.RESHAPE, listOf(x), DxirType(F32, to))
                listOf(y)
            }
            validate(DxirModule(listOf(fn)).toStablehlo(), "RESHAPE $from→$to")
        }
    }

    @Test
    fun transposeRoundTrips() {
        requireTranslateOrSkip()
        val cases = listOf(
            Triple(listOf(2, 3), listOf(1, 0), listOf(3, 2)),
            Triple(listOf(2, 3, 4), listOf(2, 0, 1), listOf(4, 2, 3)),
            Triple(listOf(5, 5), listOf(0, 1), listOf(5, 5)), // identity perm
        )
        for ((input, perm, output) in cases) {
            val fn = DxirBuilder.function("t") {
                val x = param("x", DxirType(F32, input))
                val y = op(
                    OpKind.TRANSPOSE, listOf(x), DxirType(F32, output),
                    attrs = mapOf("permutation" to perm),
                )
                listOf(y)
            }
            validate(DxirModule(listOf(fn)).toStablehlo(), "TRANSPOSE $input perm=$perm")
        }
    }

    @Test
    fun broadcastRoundTrips() {
        requireTranslateOrSkip()
        val cases = listOf(
            Triple(listOf(3), listOf(1), listOf(2, 3)),                 // rank-1 → rank-2
            Triple(listOf(2, 3), listOf(0, 2), listOf(2, 4, 3)),        // insert middle dim
            Triple(listOf(4), listOf(2), listOf(2, 3, 4)),              // scatter to last dim
        )
        for ((input, bcastDims, output) in cases) {
            val fn = DxirBuilder.function("b") {
                val x = param("x", DxirType(F32, input))
                val y = op(
                    OpKind.BROADCAST, listOf(x), DxirType(F32, output),
                    attrs = mapOf("broadcast_dimensions" to bcastDims),
                )
                listOf(y)
            }
            validate(DxirModule(listOf(fn)).toStablehlo(), "BROADCAST $input dims=$bcastDims")
        }
    }

    @Test
    fun concatRoundTrips() {
        requireTranslateOrSkip()
        // dim 0: (2,3) ++ (4,3) -> (6,3)
        val fn0 = DxirBuilder.function("c0") {
            val a = param("a", DxirType(F32, listOf(2, 3)))
            val b = param("b", DxirType(F32, listOf(4, 3)))
            val c = op(
                OpKind.CONCAT, listOf(a, b), DxirType(F32, listOf(6, 3)),
                attrs = mapOf("dimension" to 0),
            )
            listOf(c)
        }
        validate(DxirModule(listOf(fn0)).toStablehlo(), "CONCAT dim=0")

        // dim 1: (2,3) ++ (2,5) -> (2,8) with 3 operands
        val fn1 = DxirBuilder.function("c1") {
            val a = param("a", DxirType(F32, listOf(2, 3)))
            val b = param("b", DxirType(F32, listOf(2, 5)))
            val c = param("c", DxirType(F32, listOf(2, 1)))
            val d = op(
                OpKind.CONCAT, listOf(a, b, c), DxirType(F32, listOf(2, 9)),
                attrs = mapOf("dimension" to 1),
            )
            listOf(d)
        }
        validate(DxirModule(listOf(fn1)).toStablehlo(), "CONCAT dim=1 (3 operands)")
    }

    @Test
    fun sliceRoundTrips() {
        requireTranslateOrSkip()
        // 2D slice with unit stride
        val fn1 = DxirBuilder.function("s1") {
            val x = param("x", DxirType(F32, listOf(4, 4)))
            val y = op(
                OpKind.SLICE, listOf(x), DxirType(F32, listOf(2, 2)),
                attrs = mapOf(
                    "start_indices" to listOf(1, 0),
                    "limit_indices" to listOf(3, 2),
                    "strides" to listOf(1, 1),
                ),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn1)).toStablehlo(), "SLICE rank-2 unit stride")

        // 1D slice with stride 2
        val fn2 = DxirBuilder.function("s2") {
            val x = param("x", DxirType(F32, listOf(8)))
            val y = op(
                OpKind.SLICE, listOf(x), DxirType(F32, listOf(3)),
                attrs = mapOf(
                    "start_indices" to listOf(0),
                    "limit_indices" to listOf(6),
                    "strides" to listOf(2),
                ),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn2)).toStablehlo(), "SLICE rank-1 stride=2")
    }

    @Test
    fun dotRoundTrips() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("d") {
            val a = param("a", DxirType(F32, listOf(5)))
            val b = param("b", DxirType(F32, listOf(5)))
            val c = op(OpKind.DOT, listOf(a, b), DxirType(F32, emptyList()))
            listOf(c)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "DOT rank-1 inner product")
    }

    @Test
    fun siluAnyRankRoundTrips() {
        requireTranslateOrSkip()
        for (shape in listOf(listOf(4), listOf(2, 3), listOf(1, 4, 4))) {
            val fn = DxirBuilder.function("f") {
                val x = param("x", DxirType(F32, shape))
                val y = op(OpKind.SILU, listOf(x), DxirType(F32, shape))
                listOf(y)
            }
            validate(DxirModule(listOf(fn)).toStablehlo(), "SILU $shape")
        }
    }

    @Test
    fun geluAnyRankRoundTrips() {
        requireTranslateOrSkip()
        for (shape in listOf(listOf(4), listOf(2, 3), listOf(2, 2, 2))) {
            val fn = DxirBuilder.function("f") {
                val x = param("x", DxirType(F32, shape))
                val y = op(OpKind.GELU, listOf(x), DxirType(F32, shape))
                listOf(y)
            }
            validate(DxirModule(listOf(fn)).toStablehlo(), "GELU $shape")
        }
    }

    @Test
    fun softmaxRankOneRoundTrips() {
        requireTranslateOrSkip()
        for (n in listOf(1, 5, 128)) {
            val fn = DxirBuilder.function("s") {
                val x = param("x", DxirType(F32, listOf(n)))
                val y = op(OpKind.SOFTMAX, listOf(x), DxirType(F32, listOf(n)))
                listOf(y)
            }
            validate(DxirModule(listOf(fn)).toStablehlo(), "SOFTMAX rank-1 N=$n")
        }
    }

    @Test
    fun logsumexpRankOneRoundTrips() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("l") {
            val x = param("x", DxirType(F32, listOf(7)))
            val y = op(OpKind.LOGSUMEXP, listOf(x), DxirType(F32, emptyList()))
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "LOGSUMEXP rank-1")
    }

    @Test
    fun layerNormRankOneRoundTrips() {
        requireTranslateOrSkip()
        // Explicit eps
        val fn1 = DxirBuilder.function("ln1") {
            val x = param("x", DxirType(F32, listOf(8)))
            val y = op(
                OpKind.LAYERNORM, listOf(x), DxirType(F32, listOf(8)),
                attrs = mapOf("epsilon" to 1e-5f),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn1)).toStablehlo(), "LAYERNORM explicit eps")

        // Default eps
        val fn2 = DxirBuilder.function("ln2") {
            val x = param("x", DxirType(F32, listOf(16)))
            val y = op(OpKind.LAYERNORM, listOf(x), DxirType(F32, listOf(16)))
            listOf(y)
        }
        validate(DxirModule(listOf(fn2)).toStablehlo(), "LAYERNORM default eps")
    }

    @Test
    fun rmsNormRankOneRoundTrips() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("rn") {
            val x = param("x", DxirType(F32, listOf(8)))
            val y = op(
                OpKind.RMSNORM, listOf(x), DxirType(F32, listOf(8)),
                attrs = mapOf("epsilon" to 1e-6f),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "RMSNORM rank-1")
    }

    @Test
    fun conv2dRoundTripsBasicNoPadding() {
        requireTranslateOrSkip()
        // Standard 3x3 conv, NCHW, 3 input channels → 16 output channels, no pad, stride 1.
        // (1, 3, 8, 8) * (16, 3, 3, 3) -> (1, 16, 6, 6)
        val fn = DxirBuilder.function("conv_basic") {
            val x = param("x", DxirType(F32, listOf(1, 3, 8, 8)))
            val k = param("k", DxirType(F32, listOf(16, 3, 3, 3)))
            val y = op(
                OpKind.CONV2D, listOf(x, k), DxirType(F32, listOf(1, 16, 6, 6)),
                attrs = mapOf(
                    "window_strides" to listOf(1, 1),
                    "padding" to listOf(listOf(0, 0), listOf(0, 0)),
                    "lhs_dilation" to listOf(1, 1),
                    "rhs_dilation" to listOf(1, 1),
                ),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "CONV2D 3×3 no-pad")
    }

    @Test
    fun conv2dRoundTripsWithSamePadding() {
        requireTranslateOrSkip()
        // 3x3 conv with SAME-style padding (1 on each side) preserves spatial dims.
        // (2, 3, 8, 8) * (8, 3, 3, 3) -> (2, 8, 8, 8)
        val fn = DxirBuilder.function("conv_same") {
            val x = param("x", DxirType(F32, listOf(2, 3, 8, 8)))
            val k = param("k", DxirType(F32, listOf(8, 3, 3, 3)))
            val y = op(
                OpKind.CONV2D, listOf(x, k), DxirType(F32, listOf(2, 8, 8, 8)),
                attrs = mapOf(
                    "window_strides" to listOf(1, 1),
                    "padding" to listOf(listOf(1, 1), listOf(1, 1)),
                    "lhs_dilation" to listOf(1, 1),
                    "rhs_dilation" to listOf(1, 1),
                ),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "CONV2D 3×3 SAME padding")
    }

    @Test
    fun conv2dRoundTripsWithStride2() {
        requireTranslateOrSkip()
        // Strided conv halves spatial dims.
        // (1, 16, 32, 32) * (32, 16, 3, 3) stride=2 pad=[[1,1],[1,1]] -> (1, 32, 16, 16)
        val fn = DxirBuilder.function("conv_stride2") {
            val x = param("x", DxirType(F32, listOf(1, 16, 32, 32)))
            val k = param("k", DxirType(F32, listOf(32, 16, 3, 3)))
            val y = op(
                OpKind.CONV2D, listOf(x, k), DxirType(F32, listOf(1, 32, 16, 16)),
                attrs = mapOf(
                    "window_strides" to listOf(2, 2),
                    "padding" to listOf(listOf(1, 1), listOf(1, 1)),
                    "lhs_dilation" to listOf(1, 1),
                    "rhs_dilation" to listOf(1, 1),
                ),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "CONV2D stride=2 SAME")
    }

    @Test
    fun conv2dRoundTripsWithDilation() {
        requireTranslateOrSkip()
        // Dilated (atrous) 3x3 conv with dilation=2: effective receptive field 5x5.
        // (1, 4, 10, 10) * (8, 4, 3, 3) rhs_dilate=[2,2] no pad -> (1, 8, 6, 6)
        val fn = DxirBuilder.function("conv_dilated") {
            val x = param("x", DxirType(F32, listOf(1, 4, 10, 10)))
            val k = param("k", DxirType(F32, listOf(8, 4, 3, 3)))
            val y = op(
                OpKind.CONV2D, listOf(x, k), DxirType(F32, listOf(1, 8, 6, 6)),
                attrs = mapOf(
                    "window_strides" to listOf(1, 1),
                    "padding" to listOf(listOf(0, 0), listOf(0, 0)),
                    "lhs_dilation" to listOf(1, 1),
                    "rhs_dilation" to listOf(2, 2),
                ),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "CONV2D dilated 3×3")
    }

    @Test
    fun conv2dRoundTripsWithDefaultAttrs() {
        requireTranslateOrSkip()
        // Only window_strides provided; emitter fills padding/dilation defaults.
        val fn = DxirBuilder.function("conv_defaults") {
            val x = param("x", DxirType(F32, listOf(1, 3, 8, 8)))
            val k = param("k", DxirType(F32, listOf(16, 3, 3, 3)))
            val y = op(
                OpKind.CONV2D, listOf(x, k), DxirType(F32, listOf(1, 16, 6, 6)),
                attrs = mapOf("window_strides" to listOf(1, 1)),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "CONV2D defaulted attrs")
    }

    @Test
    fun axisAwareReductionsRoundTrip() {
        requireTranslateOrSkip()
        val cases = listOf(
            Triple(OpKind.SUM, listOf(1), listOf(2, 4)),
            Triple(OpKind.MEAN, listOf(1), listOf(2, 4)),
            Triple(OpKind.MAX, listOf(0), listOf(3, 4)),
            Triple(OpKind.MIN, listOf(0, 2), listOf(3)),
            Triple(OpKind.SUM, listOf(-1), listOf(2, 3)),  // negative axis
        )
        for ((opKind, dims, outShape) in cases) {
            val fn = DxirBuilder.function("r") {
                val x = param("x", DxirType(F32, listOf(2, 3, 4)))
                val y = op(opKind, listOf(x), DxirType(F32, outShape), attrs = mapOf("reduction_dims" to dims))
                listOf(y)
            }
            validate(DxirModule(listOf(fn)).toStablehlo(), "$opKind dims=$dims")
        }
    }

    @Test
    fun axisAwareSoftmaxRoundTrips() {
        requireTranslateOrSkip()
        // Default axis = -1
        val fn1 = DxirBuilder.function("s1") {
            val x = param("x", DxirType(F32, listOf(4, 10)))
            val y = op(OpKind.SOFTMAX, listOf(x), DxirType(F32, listOf(4, 10)))
            listOf(y)
        }
        validate(DxirModule(listOf(fn1)).toStablehlo(), "SOFTMAX rank-2 default axis")

        // Explicit axis = 0 on rank-2
        val fn2 = DxirBuilder.function("s2") {
            val x = param("x", DxirType(F32, listOf(4, 10)))
            val y = op(OpKind.SOFTMAX, listOf(x), DxirType(F32, listOf(4, 10)), attrs = mapOf("axis" to 0))
            listOf(y)
        }
        validate(DxirModule(listOf(fn2)).toStablehlo(), "SOFTMAX rank-2 axis=0")

        // Rank-3 softmax along middle axis
        val fn3 = DxirBuilder.function("s3") {
            val x = param("x", DxirType(F32, listOf(2, 5, 8)))
            val y = op(OpKind.SOFTMAX, listOf(x), DxirType(F32, listOf(2, 5, 8)), attrs = mapOf("axis" to 1))
            listOf(y)
        }
        validate(DxirModule(listOf(fn3)).toStablehlo(), "SOFTMAX rank-3 axis=1")
    }

    @Test
    fun axisAwareLogsumexpRoundTrips() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("lse") {
            val x = param("x", DxirType(F32, listOf(4, 6)))
            val y = op(OpKind.LOGSUMEXP, listOf(x), DxirType(F32, listOf(4)))   // reduce last axis
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "LOGSUMEXP rank-2 default axis")
    }

    @Test
    fun axisAwareNormsRoundTrip() {
        requireTranslateOrSkip()
        val fnLn = DxirBuilder.function("ln") {
            val x = param("x", DxirType(F32, listOf(2, 4, 8)))
            val y = op(
                OpKind.LAYERNORM, listOf(x), DxirType(F32, listOf(2, 4, 8)),
                attrs = mapOf("epsilon" to 1e-5f),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fnLn)).toStablehlo(), "LAYERNORM rank-3 default axis")

        val fnRn = DxirBuilder.function("rn") {
            val x = param("x", DxirType(F32, listOf(2, 4, 8)))
            val y = op(
                OpKind.RMSNORM, listOf(x), DxirType(F32, listOf(2, 4, 8)),
                attrs = mapOf("epsilon" to 1e-6f, "axis" to -1),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fnRn)).toStablehlo(), "RMSNORM rank-3 axis=-1")
    }

    @Test
    fun batchedMatmulRoundTrips() {
        requireTranslateOrSkip()
        // (2, 4, 3, 5) @ (2, 4, 5, 7) → (2, 4, 3, 7)
        val fn = DxirBuilder.function("bmm") {
            val a = param("a", DxirType(F32, listOf(2, 4, 3, 5)))
            val b = param("b", DxirType(F32, listOf(2, 4, 5, 7)))
            val c = op(
                OpKind.MATMUL, listOf(a, b), DxirType(F32, listOf(2, 4, 3, 7)),
                attrs = mapOf(
                    "lhs_batching_dims" to listOf(0, 1),
                    "rhs_batching_dims" to listOf(0, 1),
                    "lhs_contracting_dims" to listOf(3),
                    "rhs_contracting_dims" to listOf(2),
                ),
            )
            listOf(c)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "Batched MATMUL rank-4")
    }

    @Test
    fun sdpaRoundTripsRankThree() {
        requireTranslateOrSkip()
        // Rank-3 SDPA: single batch, no heads. (B=2, S_q=4, D=8) etc.
        val fn = DxirBuilder.function("sdpa") {
            val q = param("q", DxirType(F32, listOf(2, 4, 8)))
            val k = param("k", DxirType(F32, listOf(2, 6, 8)))
            val v = param("v", DxirType(F32, listOf(2, 6, 8)))
            val y = op(
                OpKind.SCALED_DOT_PRODUCT_ATTENTION,
                listOf(q, k, v),
                DxirType(F32, listOf(2, 4, 8)),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "SDPA rank-3")
    }

    @Test
    fun sdpaRoundTripsRankFourTransformerShapes() {
        requireTranslateOrSkip()
        // Realistic transformer attention block: (batch=1, heads=8, seq=64, head_dim=64)
        val fn = DxirBuilder.function("sdpa") {
            val q = param("q", DxirType(F32, listOf(1, 8, 64, 64)))
            val k = param("k", DxirType(F32, listOf(1, 8, 64, 64)))
            val v = param("v", DxirType(F32, listOf(1, 8, 64, 64)))
            val y = op(
                OpKind.SCALED_DOT_PRODUCT_ATTENTION,
                listOf(q, k, v),
                DxirType(F32, listOf(1, 8, 64, 64)),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "SDPA rank-4 transformer-scale")
    }

    @Test
    fun gatherRoundTripsBasic() {
        requireTranslateOrSkip()
        // Simple row lookup: operand (10, 4), indices (3, 1) of i32, result (3, 4)
        val fn = DxirBuilder.function("g") {
            val operand = param("o", DxirType(F32, listOf(10, 4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, listOf(3, 1)))
            val y = op(
                OpKind.GATHER, listOf(operand, idx), DxirType(F32, listOf(3, 4)),
                attrs = mapOf(
                    "offset_dims" to listOf(1),
                    "collapsed_slice_dims" to listOf(0),
                    "start_index_map" to listOf(0),
                    "index_vector_dim" to 1,
                    "slice_sizes" to listOf(1, 4),
                ),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "GATHER basic row lookup")
    }

    @Test
    fun gatherRoundTripsMultiDimIndices() {
        requireTranslateOrSkip()
        // Gather a 2x2 patch from (10, 10, 10, 10) with indices (1, 1, 4, 3).
        // Matches the canonical StableHLO testdata shape.
        val fn = DxirBuilder.function("g") {
            val operand = param("o", DxirType(F32, listOf(10, 10, 10, 10)))
            val idx = param("i", DxirType(io.tlaloc.core.I64, listOf(1, 1, 4, 3)))
            val y = op(
                OpKind.GATHER, listOf(operand, idx),
                DxirType(F32, listOf(1, 10, 2, 2, 2, 1, 4)),
                attrs = mapOf(
                    "offset_dims" to listOf(1, 2, 3, 4),
                    "collapsed_slice_dims" to emptyList<Int>(),
                    "start_index_map" to listOf(1, 2, 3),
                    "index_vector_dim" to 3,
                    "slice_sizes" to listOf(10, 2, 2, 2),
                ),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "GATHER 4D operand with multi-dim indices")
    }

    @Test
    fun embeddingRoundTripsRankOneIndices() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("e") {
            val table = param("t", DxirType(F32, listOf(10, 4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, listOf(3)))
            val y = op(OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(3, 4)))
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "EMBEDDING rank-1 indices")
    }

    @Test
    fun embeddingRoundTripsRankTwoIndicesTransformerScale() {
        requireTranslateOrSkip()
        // Realistic transformer: vocab=32000, embed=768, batch=2, seq=16.
        val fn = DxirBuilder.function("e") {
            val table = param("t", DxirType(F32, listOf(32000, 768)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, listOf(2, 16)))
            val y = op(OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(2, 16, 768)))
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "EMBEDDING transformer-scale")
    }

    @Test
    fun convTranspose2dRoundTripsUpsampling() {
        requireTranslateOrSkip()
        // 2× upsampling via transposed conv. Input (1, 16, 8, 8), kernel (16, 8, 3, 3) [i, o, Kh, Kw].
        // StableHLO math: dilated input H = 1 + (8-1)*2 = 15; padded = 15 + 2+2 = 19; output = 19 - 3 + 1 = 17.
        val fn = DxirBuilder.function("ct") {
            val x = param("x", DxirType(F32, listOf(1, 16, 8, 8)))
            val k = param("k", DxirType(F32, listOf(16, 8, 3, 3)))
            val y = op(
                OpKind.CONV_TRANSPOSE2D, listOf(x, k), DxirType(F32, listOf(1, 8, 17, 17)),
                attrs = mapOf(
                    "window_strides" to listOf(1, 1),
                    "padding" to listOf(listOf(2, 2), listOf(2, 2)),   // kernel-1 on each side
                    "lhs_dilation" to listOf(2, 2),
                    "rhs_dilation" to listOf(1, 1),
                ),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "CONV_TRANSPOSE2D 2x upsampling (17×17)")
    }

    @Test
    fun convTranspose2dRoundTripsNoPadding() {
        requireTranslateOrSkip()
        // Native StableHLO conv-transpose math with zero padding: 8 → 13.
        val fn = DxirBuilder.function("ct_nopad") {
            val x = param("x", DxirType(F32, listOf(1, 16, 8, 8)))
            val k = param("k", DxirType(F32, listOf(16, 8, 3, 3)))
            val y = op(
                OpKind.CONV_TRANSPOSE2D, listOf(x, k), DxirType(F32, listOf(1, 8, 13, 13)),
                attrs = mapOf(
                    "window_strides" to listOf(1, 1),
                    "padding" to listOf(listOf(0, 0), listOf(0, 0)),
                    "lhs_dilation" to listOf(2, 2),
                    "rhs_dilation" to listOf(1, 1),
                ),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "CONV_TRANSPOSE2D no-padding")
    }

    @Test
    fun argmaxRoundTripsRankTwo() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("am") {
            val x = param("x", DxirType(F32, listOf(2, 5)))
            val y = op(
                OpKind.ARGMAX, listOf(x), DxirType(io.tlaloc.core.I32, listOf(2)),
                attrs = mapOf("axis" to 1),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "ARGMAX rank-2 axis=1")
    }

    @Test
    fun argmaxRoundTripsRankOneI64() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("am") {
            val x = param("x", DxirType(F32, listOf(15)))
            val y = op(OpKind.ARGMAX, listOf(x), DxirType(io.tlaloc.core.I64, emptyList()))
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "ARGMAX rank-1 scalar index (i64)")
    }

    @Test
    fun crossEntropyRoundTrips() {
        requireTranslateOrSkip()
        // Standard classification head: B=4, C=10
        val fn = DxirBuilder.function("xent") {
            val logits = param("l", DxirType(F32, listOf(4, 10)))
            val labels = param("y", DxirType(io.tlaloc.core.I32, listOf(4)))
            val loss = op(OpKind.CROSS_ENTROPY, listOf(logits, labels), DxirType(F32, listOf(4)))
            listOf(loss)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "CROSS_ENTROPY rank-2 B=4 C=10")
    }

    @Test
    fun crossEntropyRoundTripsLargeVocab() {
        requireTranslateOrSkip()
        // Transformer LM head: B=2, vocab=32000, labels=i64
        val fn = DxirBuilder.function("xent") {
            val logits = param("l", DxirType(F32, listOf(2, 32000)))
            val labels = param("y", DxirType(io.tlaloc.core.I64, listOf(2)))
            val loss = op(OpKind.CROSS_ENTROPY, listOf(logits, labels), DxirType(F32, listOf(2)))
            listOf(loss)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "CROSS_ENTROPY vocab=32000 i64 labels")
    }

    @Test
    fun scatterRoundTripsMinReduction() {
        requireTranslateOrSkip()
        // From canonical StableHLO testdata: (3,f64) operand + (3,1,i64) indices + (3,f64) updates, min reduce.
        val fn = DxirBuilder.function("s") {
            val operand = param("o", DxirType(io.tlaloc.core.F64, listOf(3)))
            val idx = param("i", DxirType(io.tlaloc.core.I64, listOf(3, 1)))
            val upd = param("u", DxirType(io.tlaloc.core.F64, listOf(3)))
            val y = op(
                OpKind.SCATTER, listOf(operand, idx, upd), DxirType(io.tlaloc.core.F64, listOf(3)),
                attrs = mapOf(
                    "inserted_window_dims" to listOf(0),
                    "scatter_dims_to_operand_dims" to listOf(0),
                    "index_vector_dim" to 1,
                    "reduction" to "min",
                ),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "SCATTER 1D min")
    }

    @Test
    fun scatterRoundTripsReplaceReduction() {
        requireTranslateOrSkip()
        // 3D scatter with replace — matches a canonical testdata shape.
        val fn = DxirBuilder.function("s") {
            val operand = param("o", DxirType(F32, listOf(5, 6, 7)))
            val idx = param("i", DxirType(io.tlaloc.core.I64, listOf(2, 2, 1)))
            val upd = param("u", DxirType(F32, listOf(5, 2, 2, 7)))
            val y = op(
                OpKind.SCATTER, listOf(operand, idx, upd), DxirType(F32, listOf(5, 6, 7)),
                attrs = mapOf(
                    "update_window_dims" to listOf(0, 3),
                    "inserted_window_dims" to listOf(1),
                    "scatter_dims_to_operand_dims" to listOf(1),
                    "index_vector_dim" to 2,
                    "reduction" to "replace",
                    "unique_indices" to true,
                ),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "SCATTER 3D replace")
    }

    @Test
    fun scatterRoundTripsAddReduction() {
        requireTranslateOrSkip()
        // Common embedding-bag-style scatter: add updates into rows of a lookup table.
        val fn = DxirBuilder.function("s") {
            val operand = param("o", DxirType(F32, listOf(100, 64)))    // table: 100 rows × 64 dim
            val idx = param("i", DxirType(io.tlaloc.core.I32, listOf(8, 1)))  // 8 row-indices
            val upd = param("u", DxirType(F32, listOf(8, 64)))          // 8 row-deltas
            val y = op(
                OpKind.SCATTER, listOf(operand, idx, upd), DxirType(F32, listOf(100, 64)),
                attrs = mapOf(
                    "update_window_dims" to listOf(1),
                    "inserted_window_dims" to listOf(0),
                    "scatter_dims_to_operand_dims" to listOf(0),
                    "index_vector_dim" to 1,
                    "reduction" to "add",
                ),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "SCATTER embedding-bag add")
    }

    @Test
    fun scatterRoundTripsMaxReduction() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("s") {
            val operand = param("o", DxirType(F32, listOf(10, 10)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, listOf(3, 2)))
            val upd = param("u", DxirType(F32, listOf(3)))
            val y = op(
                OpKind.SCATTER, listOf(operand, idx, upd), DxirType(F32, listOf(10, 10)),
                attrs = mapOf(
                    "inserted_window_dims" to listOf(0, 1),
                    "scatter_dims_to_operand_dims" to listOf(0, 1),
                    "index_vector_dim" to 1,
                    "reduction" to "max",
                ),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "SCATTER 2D max (scalar updates)")
    }

    @Test
    fun splitRoundTripsEqualHalves() {
        requireTranslateOrSkip()
        // Most common case: split a transformer (B, 2D) into two (B, D) halves (gate+up).
        val fn = DxirBuilder.function("sp") {
            val x = param("x", DxirType(F32, listOf(16, 2048)))
            val split = opMulti(
                OpKind.SPLIT, listOf(x),
                types = listOf(DxirType(F32, listOf(16, 1024)), DxirType(F32, listOf(16, 1024))),
                attrs = mapOf("axis" to 1, "sizes" to listOf(1024, 1024)),
            )
            listOf(split.result(0), split.result(1))
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "SPLIT equal halves (GLU-gate)")
    }

    @Test
    fun splitRoundTripsUnequalChunks() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("sp") {
            val x = param("x", DxirType(F32, listOf(10, 4)))
            val split = opMulti(
                OpKind.SPLIT, listOf(x),
                types = listOf(
                    DxirType(F32, listOf(2, 4)),
                    DxirType(F32, listOf(3, 4)),
                    DxirType(F32, listOf(5, 4)),
                ),
                attrs = mapOf("axis" to 0, "sizes" to listOf(2, 3, 5)),
            )
            listOf(split.result(0), split.result(1), split.result(2))
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "SPLIT 3 unequal chunks")
    }

    @Test
    fun splitResultFlowsIntoNextOp() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("sp_chain") {
            val x = param("x", DxirType(F32, listOf(8, 3)))
            val split = opMulti(
                OpKind.SPLIT, listOf(x),
                types = listOf(DxirType(F32, listOf(4, 3)), DxirType(F32, listOf(4, 3))),
                attrs = mapOf("axis" to 0, "sizes" to listOf(4, 4)),
            )
            // Sum of first half.
            val r0 = split.result(0)
            val total = op(OpKind.SUM, listOf(r0), DxirType(F32, emptyList()))
            listOf(total)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "SPLIT → SUM chain")
    }

    @Test
    fun batchNormInferenceRoundTripsMatchesTestdata() {
        requireTranslateOrSkip()
        // Direct port of the stablehlo ops_stablehlo.mlir batch_norm_inference example.
        val fn = DxirBuilder.function("bn") {
            val x = param("x", DxirType(F32, listOf(4, 256)))
            val scale = param("s", DxirType(F32, listOf(256)))
            val offset = param("b", DxirType(F32, listOf(256)))
            val mean = param("m", DxirType(F32, listOf(256)))
            val variance = param("v", DxirType(F32, listOf(256)))
            val y = op(
                OpKind.BATCHNORM, listOf(x, scale, offset, mean, variance), DxirType(F32, listOf(4, 256)),
                attrs = mapOf("epsilon" to 1.001e-5f, "feature_index" to 1),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "BATCHNORM 2D canonical shape")
    }

    @Test
    fun batchNormInferenceRoundTripsNchw() {
        requireTranslateOrSkip()
        // CNN-style activation shape: (batch, channels, H, W).
        val fn = DxirBuilder.function("bn") {
            val x = param("x", DxirType(F32, listOf(2, 64, 8, 8)))
            val scale = param("s", DxirType(F32, listOf(64)))
            val offset = param("b", DxirType(F32, listOf(64)))
            val mean = param("m", DxirType(F32, listOf(64)))
            val variance = param("v", DxirType(F32, listOf(64)))
            val y = op(
                OpKind.BATCHNORM, listOf(x, scale, offset, mean, variance), DxirType(F32, listOf(2, 64, 8, 8)),
                attrs = mapOf("feature_index" to 1),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "BATCHNORM NCHW conv-output shape")
    }

    @Test
    fun batchNormInferenceRoundTripsFeatureIndexLast() {
        requireTranslateOrSkip()
        // NHWC layout — feature axis is the last dim.
        val fn = DxirBuilder.function("bn") {
            val x = param("x", DxirType(F32, listOf(2, 8, 8, 32)))
            val scale = param("s", DxirType(F32, listOf(32)))
            val offset = param("b", DxirType(F32, listOf(32)))
            val mean = param("m", DxirType(F32, listOf(32)))
            val variance = param("v", DxirType(F32, listOf(32)))
            val y = op(
                OpKind.BATCHNORM, listOf(x, scale, offset, mean, variance), DxirType(F32, listOf(2, 8, 8, 32)),
                attrs = mapOf("feature_index" to 3),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "BATCHNORM NHWC feature_index=3")
    }

    @Test
    fun embeddingRoundTripsI64Indices() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("e") {
            val table = param("t", DxirType(F32, listOf(50, 8)))
            val idx = param("i", DxirType(io.tlaloc.core.I64, listOf(4)))
            val y = op(OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(4, 8)))
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "EMBEDDING with i64 indices")
    }

    @Test
    fun sdpaRoundTripsAsymmetricSequences() {
        requireTranslateOrSkip()
        // Cross-attention: S_q != S_k, different head_dim for K/V than output dim.
        val fn = DxirBuilder.function("sdpa") {
            val q = param("q", DxirType(F32, listOf(1, 4, 3, 16)))
            val k = param("k", DxirType(F32, listOf(1, 4, 7, 16)))
            val v = param("v", DxirType(F32, listOf(1, 4, 7, 16)))
            val y = op(
                OpKind.SCALED_DOT_PRODUCT_ATTENTION,
                listOf(q, k, v),
                DxirType(F32, listOf(1, 4, 3, 16)),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "SDPA cross-attention asymmetric seq")
    }
}
