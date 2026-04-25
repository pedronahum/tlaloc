package io.tlaloc.stablehlo

import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.capture
import io.tlaloc.autograd.capture2
import io.tlaloc.autograd.matmul
import io.tlaloc.autograd.relu
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.core.F32
import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirModule
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class EmitterTest {

    @Test
    fun typeFormattingCoversScalarAndTensorShapes() {
        assertEquals("tensor<f32>", DxirType(F32, emptyList()).toMlir())
        assertEquals("tensor<4xf32>", DxirType(F32, listOf(4)).toMlir())
        assertEquals("tensor<2x3xf32>", DxirType(F32, listOf(2, 3)).toMlir())
        assertEquals("tensor<8x16x32xf32>", DxirType(F32, listOf(8, 16, 32)).toMlir())
    }

    @Test
    fun emitsFunctionSignatureWithNamedParamsAndReturnType() {
        val fn = DxirBuilder.function("id") {
            val x = param("x", DxirType(F32, listOf(4)))
            listOf(x)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("func.func @id(%0: tensor<4xf32>) -> tensor<4xf32>"), mlir)
        assertTrue(mlir.contains("return %0 : tensor<4xf32>"), mlir)
    }

    @Test
    fun emitsElementwiseBinaryOps() {
        val mlir = singleOpFunction(OpKind.ADD, DxirType(F32, listOf(4)))
        assertTrue(mlir.contains("stablehlo.add %0, %1 : tensor<4xf32>"), mlir)

        assertTrue(singleOpFunction(OpKind.SUB, DxirType(F32, listOf(4)))
            .contains("stablehlo.subtract"))
        assertTrue(singleOpFunction(OpKind.MUL, DxirType(F32, listOf(4)))
            .contains("stablehlo.multiply"))
        assertTrue(singleOpFunction(OpKind.DIV, DxirType(F32, listOf(4)))
            .contains("stablehlo.divide"))
        assertTrue(singleOpFunction(OpKind.POW, DxirType(F32, listOf(4)))
            .contains("stablehlo.power"))
    }

    @Test
    fun emitsElementwiseUnaryOps() {
        val t = DxirType(F32, listOf(4))
        assertTrue(singleUnary(OpKind.NEG, t).contains("stablehlo.negate"))
        assertTrue(singleUnary(OpKind.ABS, t).contains("stablehlo.abs"))
        assertTrue(singleUnary(OpKind.EXP, t).contains("stablehlo.exponential"))
        assertTrue(singleUnary(OpKind.LOG, t).contains("stablehlo.log"))
        assertTrue(singleUnary(OpKind.SQRT, t).contains("stablehlo.sqrt"))
        assertTrue(singleUnary(OpKind.RSQRT, t).contains("stablehlo.rsqrt"))
        assertTrue(singleUnary(OpKind.TANH, t).contains("stablehlo.tanh"))
        assertTrue(singleUnary(OpKind.SIGMOID, t).contains("stablehlo.logistic"))
    }

    @Test
    fun emitsRankNFloatArrayConstAsNestedDenseLiteral() {
        // §0.4.73 — rank-N DxirConst carrying a FloatArray (the form produced by
        // §0.4.71's Capture fix) formats as a nested dense<...> literal matching
        // the declared shape. Scalars still go through the existing Float/Double
        // arm (Capture.kt unpacks `value[0]` for rank-0 leaves, so a scalar const's
        // value is a Float, not a FloatArray).
        val rank1Fn = DxirBuilder.function("c1") {
            val c = const(floatArrayOf(1f, 2f, 3f), DxirType(F32, listOf(3)))
            listOf(c)
        }
        val mlir1 = rank1Fn.toStablehlo()
        assertTrue(mlir1.contains("dense<[1.0, 2.0, 3.0]>"), "rank-1 dense literal missing: $mlir1")
        assertTrue(mlir1.contains("tensor<3xf32>"), mlir1)

        val rank2Fn = DxirBuilder.function("c2") {
            val c = const(floatArrayOf(1f, 2f, 3f, 4f), DxirType(F32, listOf(2, 2)))
            listOf(c)
        }
        val mlir2 = rank2Fn.toStablehlo()
        assertTrue(mlir2.contains("dense<[[1.0, 2.0], [3.0, 4.0]]>"), "rank-2 nested dense missing: $mlir2")
        assertTrue(mlir2.contains("tensor<2x2xf32>"), mlir2)
    }

    @Test
    fun emitsBooleanOps() {
        // §0.4.60 — NOT and LAND lower to `stablehlo.not` / `stablehlo.and` on
        // Bool (i1) inputs. The ops were introduced by Stage B (F3 canonical-
        // isation / §0.4.50 break-hoist) and had their emission deferred until
        // benchmark bring-up demanded it; the deferral error is now a direct
        // lowering. Until D.3i PhiCalculus closure of LAND-composed WHILE lands,
        // nothing *in the compiler* produces residual NOT/LAND outside an IF or
        // WHILE region — but hand-constructed dxir can, and so can the φ-calculus
        // once D.3i unblocks, so the emitter arm is load-bearing.
        val boolT = DxirType(io.tlaloc.core.Bool, listOf(4))
        assertTrue(singleUnary(OpKind.NOT, boolT).contains("stablehlo.not"))
        assertTrue(singleOpFunction(OpKind.LAND, boolT).contains("stablehlo.and"))
        // Scalar (rank-0) Bool — the shape the break-hoist cond region terminates with.
        val boolScalar = DxirType(io.tlaloc.core.Bool, emptyList())
        val mlir = singleOpFunction(OpKind.LAND, boolScalar)
        assertTrue(mlir.contains("stablehlo.and") && mlir.contains("tensor<i1>"), mlir)
    }

    @Test
    fun reluLowersToConstantPlusMaximum() {
        val mlir = singleUnary(OpKind.RELU, DxirType(F32, listOf(4)))
        assertTrue(mlir.contains("stablehlo.constant dense<0.0> : tensor<4xf32>"), mlir)
        assertTrue(mlir.contains("stablehlo.maximum"), mlir)
    }

    @Test
    fun stepLowersToCompareGTPlusSelect() {
        val mlir = singleUnary(OpKind.STEP, DxirType(F32, listOf(4)))
        assertTrue(mlir.contains("stablehlo.constant dense<0.0> : tensor<4xf32>"), mlir)
        assertTrue(mlir.contains("stablehlo.constant dense<1.0> : tensor<4xf32>"), mlir)
        assertTrue(mlir.contains("stablehlo.compare  GT"), mlir)
        // At x == 0 the comparison is false, so STEP yields the 0 branch — matching
        // XLA's semantics.  The select body is `compare-result, one, zero`.
        assertTrue(mlir.contains("stablehlo.select"), mlir)
    }

    @Test
    fun sumLowersToReduceOverAllDims() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(2, 3)))
            val s = op(OpKind.SUM, listOf(x), DxirType(F32, emptyList()))
            listOf(s)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("stablehlo.constant dense<0.0> : tensor<f32>"), mlir)
        assertTrue(
            mlir.contains("stablehlo.reduce") &&
                mlir.contains("applies stablehlo.add across dimensions = [0, 1]"),
            mlir,
        )
        assertTrue(mlir.contains("(tensor<2x3xf32>, tensor<f32>) -> tensor<f32>"), mlir)
    }

    @Test
    fun meanLowersToReduceThenDivide() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(4)))
            val m = op(OpKind.MEAN, listOf(x), DxirType(F32, emptyList()))
            listOf(m)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("stablehlo.reduce"), mlir)
        assertTrue(mlir.contains("stablehlo.constant dense<4.0> : tensor<f32>"), "divisor missing: $mlir")
        assertTrue(mlir.contains("stablehlo.divide"), mlir)
    }

    @Test
    fun matmulLowersToDotGeneralWithContractingDims() {
        val fn = DxirBuilder.function("mm") {
            val a = param("a", DxirType(F32, listOf(2, 3)))
            val b = param("b", DxirType(F32, listOf(3, 4)))
            val c = op(OpKind.MATMUL, listOf(a, b), DxirType(F32, listOf(2, 4)))
            listOf(c)
        }
        val mlir = fn.toStablehlo()
        assertTrue(
            mlir.contains("stablehlo.dot_general %0, %1, contracting_dims = [1] x [0]"),
            mlir,
        )
        assertTrue(
            mlir.contains("(tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>"),
            mlir,
        )
    }

    @Test
    fun moduleWrapsAllFunctions() {
        val fn1 = DxirBuilder.function("a") {
            val x = param("x", DxirType(F32, listOf(2)))
            listOf(x)
        }
        val fn2 = DxirBuilder.function("b") {
            val x = param("x", DxirType(F32, listOf(2)))
            val y = op(OpKind.NEG, listOf(x), DxirType(F32, listOf(2)))
            listOf(y)
        }
        val mlir = DxirModule(listOf(fn1, fn2)).toStablehlo()
        assertTrue(mlir.startsWith("module {"), mlir)
        assertTrue(mlir.trimEnd().endsWith("}"), mlir)
        assertTrue(mlir.contains("func.func @a"), mlir)
        assertTrue(mlir.contains("func.func @b"), mlir)
    }

    @Test
    fun capturedAutogradFunctionLowersToValidStablehloShape() {
        // End-to-end: tracing → dxir → StableHLO text.
        val fn = capture2(
            f = { w: Tracer<Rank2<Sym, Sym>>, x: Tracer<Rank2<Sym, Sym>> ->
                (w matmul x).relu().sum()
            },
            a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f)),
            b = Tensors.f32Matrix<Sym, Sym>(2, 1, floatArrayOf(1f, 1f)),
            name = "layer",
        )
        val mlir = fn.toStablehlo()
        // Signature reflects capture-time dims.
        assertTrue(
            mlir.contains("func.func @layer(%0: tensor<2x2xf32>, %1: tensor<2x1xf32>) -> tensor<f32>"),
            mlir,
        )
        // Every stage shows up in order.
        val matmulIdx = mlir.indexOf("dot_general")
        val maxIdx = mlir.indexOf("stablehlo.maximum")
        val sumIdx = mlir.indexOf("stablehlo.reduce")
        assertTrue(matmulIdx in 0 until maxIdx, "matmul before relu")
        assertTrue(maxIdx in 0 until sumIdx, "relu before sum")
    }

    @Test
    fun emitsScalarIoCorrectly() {
        val fn = capture(
            f = { x: Tracer<ScalarShape> -> x * x },
            input = Tensors.f32Scalar(2f),
            name = "square",
        )
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("func.func @square(%0: tensor<f32>) -> tensor<f32>"), mlir)
        assertTrue(mlir.contains("stablehlo.multiply %0, %0 : tensor<f32>"), mlir)
    }

    @Test
    fun manualComputationEmitsSdyBlock() {
        val mesh = io.tlaloc.ir.DxirMesh("m", listOf(io.tlaloc.ir.DxirMeshAxis("a", 4)))
        val inShard = io.tlaloc.ir.DxirSharding(
            "m",
            listOf(io.tlaloc.ir.DxirDimSharding(listOf(io.tlaloc.ir.DxirAxisRef.Full("a")))),
        )
        val outShard = inShard
        val fn = DxirBuilder.function("f") {
            declareMesh(mesh)
            val x = param("x", DxirType(F32, listOf(8)))
            val mc = op(
                OpKind.MANUAL_COMPUTATION, listOf(x),
                DxirType(F32, listOf(8)),
                attrs = mapOf(
                    "in_shardings" to listOf(inShard),
                    "out_shardings" to listOf(outShard),
                    "manual_axes" to listOf("a"),
                ),
                regions = listOf(
                    region {
                        val local = arg(DxirType(F32, listOf(2)))   // 8 / 4 = 2 per-device
                        val r = op(OpKind.NEG, listOf(local), DxirType(F32, listOf(2)))
                        yields(r)
                    },
                ),
            )
            listOf(mc)
        }
        val mlir = io.tlaloc.ir.DxirModule(listOf(fn)).toStablehlo()
        assertTrue(mlir.contains("sdy.manual_computation"), mlir)
        assertTrue(mlir.contains("in_shardings=[<@m, [{\"a\"}]>]"), mlir)
        assertTrue(mlir.contains("out_shardings=[<@m, [{\"a\"}]>]"), mlir)
        assertTrue(mlir.contains("manual_axes={\"a\"}"), mlir)
        assertTrue(mlir.contains("sdy.return"), "sdy.return terminator missing: $mlir")
        assertTrue(mlir.contains("stablehlo.negate"), mlir)
        assertTrue(mlir.contains("-> tensor<8xf32>"), mlir)
    }

    @Test
    fun unsupportedOpFailsLoudly() {
        // No remaining unlowered ops — DxirCall is the one case that still errors.
        val helper = DxirBuilder.function("helper") {
            val x = param("x", DxirType(F32, listOf(4)))
            listOf(x)
        }
        val fn = DxirBuilder.function("bad") {
            val x = param("x", DxirType(F32, listOf(4)))
            listOf(call(helper, listOf(x), DxirType(F32, listOf(4))))
        }
        assertFailsWith<IllegalStateException> { fn.toStablehlo() }
    }

    @Test
    fun conv2dLowersWithExplicitAttrs() {
        val fn = DxirBuilder.function("c") {
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
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("stablehlo.convolution"), mlir)
        assertTrue(
            mlir.contains("dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]"),
            "dim_numbers missing: $mlir",
        )
        assertTrue(mlir.contains("stride = [1, 1]"), mlir)
        assertTrue(mlir.contains("pad = [[0, 0], [0, 0]]"), mlir)
        assertTrue(mlir.contains("feature_group_count = 1 : i64"), mlir)
        assertTrue(mlir.contains("batch_group_count = 1 : i64"), mlir)
        assertTrue(
            mlir.contains("(tensor<1x3x8x8xf32>, tensor<16x3x3x3xf32>) -> tensor<1x16x6x6xf32>"),
            mlir,
        )
    }

    @Test
    fun conv2dRejectsWrongInputRank() {
        val fn = DxirBuilder.function("bad") {
            val x = param("x", DxirType(F32, listOf(3, 8, 8)))
            val k = param("k", DxirType(F32, listOf(16, 3, 3, 3)))
            val y = op(
                OpKind.CONV2D, listOf(x, k), DxirType(F32, listOf(1, 16, 6, 6)),
                attrs = mapOf("window_strides" to listOf(1, 1)),
            )
            listOf(y)
        }
        assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
    }

    @Test
    fun conv2dFillsSensibleDefaultsWhenAttrsOmitted() {
        // Only window_strides required; padding/dilation default to zeros/ones.
        val fn = DxirBuilder.function("c") {
            val x = param("x", DxirType(F32, listOf(1, 3, 8, 8)))
            val k = param("k", DxirType(F32, listOf(16, 3, 3, 3)))
            val y = op(
                OpKind.CONV2D, listOf(x, k), DxirType(F32, listOf(1, 16, 6, 6)),
                attrs = mapOf("window_strides" to listOf(1, 1)),
            )
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("pad = [[0, 0], [0, 0]]"), "default padding missing: $mlir")
        assertTrue(mlir.contains("lhs_dilate = [1, 1]"), mlir)
        assertTrue(mlir.contains("rhs_dilate = [1, 1]"), mlir)
    }

    @Test
    fun sumReducesAlongSingleAxis() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(2, 3, 4)))
            val y = op(
                OpKind.SUM, listOf(x), DxirType(F32, listOf(2, 4)),
                attrs = mapOf("reduction_dims" to listOf(1)),
            )
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("across dimensions = [1]"), mlir)
        assertTrue(mlir.contains("-> tensor<2x4xf32>"), mlir)
    }

    @Test
    fun sumDefaultsToAllDimsForBackwardCompat() {
        // Existing rank-1 sum test behavior — no reduction_dims attr.
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(4)))
            val y = op(OpKind.SUM, listOf(x), DxirType(F32, emptyList()))
            listOf(y)
        }
        assertTrue(fn.toStablehlo().contains("across dimensions = [0]"))
    }

    @Test
    fun meanWithAxisEmitsCorrectDivisor() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(2, 5)))
            val y = op(
                OpKind.MEAN, listOf(x), DxirType(F32, listOf(2)),
                attrs = mapOf("reduction_dims" to listOf(1)),
            )
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        // Divisor = size of reduced dim (5), emitted at output shape.
        assertTrue(mlir.contains("dense<5.0> : tensor<2xf32>"), "expected per-output divisor 5: $mlir")
    }

    @Test
    fun sumNormalizesNegativeAxis() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(3, 5)))
            val y = op(
                OpKind.SUM, listOf(x), DxirType(F32, listOf(3)),
                attrs = mapOf("reduction_dims" to listOf(-1)),
            )
            listOf(y)
        }
        assertTrue(fn.toStablehlo().contains("across dimensions = [1]"))
    }

    @Test
    fun sumRejectsMismatchedOutputShape() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(2, 3)))
            val y = op(
                OpKind.SUM, listOf(x), DxirType(F32, listOf(5)),   // wrong
                attrs = mapOf("reduction_dims" to listOf(1)),
            )
            listOf(y)
        }
        assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
    }

    @Test
    fun softmaxRankTwoAlongLastAxis() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(4, 10)))
            val y = op(OpKind.SOFTMAX, listOf(x), DxirType(F32, listOf(4, 10)))
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        // Reduce along dim 1; broadcast back with dims=[0].
        assertTrue(mlir.contains("across dimensions = [1]"), mlir)
        assertTrue(mlir.contains("broadcast_in_dim") && mlir.contains("dims = [0]"), mlir)
        assertTrue(mlir.contains("(tensor<4xf32>) -> tensor<4x10xf32>"), mlir)
    }

    @Test
    fun softmaxExplicitAxisOverridesDefault() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(4, 10)))
            val y = op(
                OpKind.SOFTMAX, listOf(x), DxirType(F32, listOf(4, 10)),
                attrs = mapOf("axis" to 0),
            )
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("across dimensions = [0]"), mlir)
        // Broadcast back uses dim 1 since dim 0 was reduced.
        assertTrue(mlir.contains("dims = [1]"), mlir)
    }

    @Test
    fun layerNormRankThreeAlongLastAxis() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(2, 4, 8)))
            val y = op(
                OpKind.LAYERNORM, listOf(x), DxirType(F32, listOf(2, 4, 8)),
                attrs = mapOf("epsilon" to 1e-5f),
            )
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("across dimensions = [2]"), mlir)
        // Divisor = 8 (last-axis size), emitted at reduced shape tensor<2x4xf32>.
        assertTrue(mlir.contains("dense<8.0> : tensor<2x4xf32>"), "per-axis divisor missing: $mlir")
    }

    @Test
    fun matmulBatchedRankFour() {
        // (B, H, M, K) @ (B, H, K, N) -> (B, H, M, N)
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
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("batching_dims = [0, 1] x [0, 1]"), mlir)
        assertTrue(mlir.contains("contracting_dims = [3] x [2]"), mlir)
    }

    @Test
    fun matmulDefaultRankTwoUnchanged() {
        // No attrs → existing rank-2 path.
        val fn = DxirBuilder.function("m") {
            val a = param("a", DxirType(F32, listOf(2, 3)))
            val b = param("b", DxirType(F32, listOf(3, 4)))
            val c = op(OpKind.MATMUL, listOf(a, b), DxirType(F32, listOf(2, 4)))
            listOf(c)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("contracting_dims = [1] x [0]"), mlir)
        assertTrue(!mlir.contains("batching_dims"), "rank-2 default must not emit batching")
    }

    @Test
    fun sdpaLowersRankFour() {
        // Standard transformer SDPA shapes: (B=1, H=2, S_q=4, D=8) / (B=1, H=2, S_k=6, D=8) / (B=1, H=2, S_k=6, D=8)
        val fn = DxirBuilder.function("sdpa") {
            val q = param("q", DxirType(F32, listOf(1, 2, 4, 8)))
            val k = param("k", DxirType(F32, listOf(1, 2, 6, 8)))
            val v = param("v", DxirType(F32, listOf(1, 2, 6, 8)))
            val y = op(
                OpKind.SCALED_DOT_PRODUCT_ATTENTION,
                listOf(q, k, v),
                DxirType(F32, listOf(1, 2, 4, 8)),
            )
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        // Two dot_general calls: Q@K^T and attn@V
        val dotCount = Regex("stablehlo\\.dot_general").findAll(mlir).count()
        assertEquals(2, dotCount, "SDPA should emit exactly two dot_general ops")
        // Scale: 1/sqrt(8) ≈ 0.3535...
        assertTrue(mlir.contains("0.353"), "scale constant missing: $mlir")
        // Softmax over S_k (axis 3)
        assertTrue(mlir.contains("across dimensions = [3]"), mlir)
        // Final output shape
        assertTrue(mlir.contains("-> tensor<1x2x4x8xf32>"), mlir)
    }

    @Test
    fun gatherEmitsDimensionNumbers() {
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
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("\"stablehlo.gather\""), mlir)
        assertTrue(mlir.contains("offset_dims = [1]"), mlir)
        assertTrue(mlir.contains("collapsed_slice_dims = [0]"), mlir)
        assertTrue(mlir.contains("start_index_map = [0]"), mlir)
        assertTrue(mlir.contains("index_vector_dim = 1"), mlir)
        assertTrue(mlir.contains("slice_sizes = array<i64: 1, 4>"), mlir)
    }

    @Test
    fun gatherEmitsIndicesAreSortedWhenTrue() {
        val fn = DxirBuilder.function("g") {
            val operand = param("o", DxirType(F32, listOf(10)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, listOf(5, 1)))
            val y = op(
                OpKind.GATHER, listOf(operand, idx), DxirType(F32, listOf(5)),
                attrs = mapOf(
                    "offset_dims" to emptyList<Int>(),
                    "collapsed_slice_dims" to listOf(0),
                    "start_index_map" to listOf(0),
                    "index_vector_dim" to 1,
                    "slice_sizes" to listOf(1),
                    "indices_are_sorted" to true,
                ),
            )
            listOf(y)
        }
        assertTrue(fn.toStablehlo().contains("indices_are_sorted = true"))
    }

    @Test
    fun gatherRejectsSliceSizeRankMismatch() {
        val fn = DxirBuilder.function("bad") {
            val operand = param("o", DxirType(F32, listOf(10, 4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, listOf(3, 1)))
            val y = op(
                OpKind.GATHER, listOf(operand, idx), DxirType(F32, listOf(3, 4)),
                attrs = mapOf(
                    "offset_dims" to listOf(1),
                    "start_index_map" to listOf(0),
                    "index_vector_dim" to 1,
                    "slice_sizes" to listOf(1),  // wrong: operand rank is 2
                ),
            )
            listOf(y)
        }
        assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
    }

    @Test
    fun embeddingLookupRankOneIndices() {
        // table (10, 4) + indices (3,) -> output (3, 4)
        val fn = DxirBuilder.function("e") {
            val table = param("t", DxirType(F32, listOf(10, 4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, listOf(3)))
            val y = op(OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(3, 4)))
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("\"stablehlo.gather\""), mlir)
        assertTrue(mlir.contains("offset_dims = [1]"), mlir)
        assertTrue(mlir.contains("collapsed_slice_dims = [0]"), mlir)
        assertTrue(mlir.contains("index_vector_dim = 1"), mlir)
        assertTrue(mlir.contains("slice_sizes = array<i64: 1, 4>"), mlir)
        assertTrue(mlir.contains("(tensor<10x4xf32>, tensor<3xi32>) -> tensor<3x4xf32>"), mlir)
    }

    @Test
    fun embeddingLookupRankTwoIndices() {
        // table (1000, 64) + indices (2, 8) -> output (2, 8, 64)
        val fn = DxirBuilder.function("e") {
            val table = param("t", DxirType(F32, listOf(1000, 64)))
            val idx = param("i", DxirType(io.tlaloc.core.I64, listOf(2, 8)))
            val y = op(OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(2, 8, 64)))
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("offset_dims = [2]"), "offset_dims must equal indices rank: $mlir")
        assertTrue(mlir.contains("index_vector_dim = 2"), mlir)
        assertTrue(mlir.contains("slice_sizes = array<i64: 1, 64>"), mlir)
    }

    @Test
    fun embeddingRejectsRankThreeTable() {
        val fn = DxirBuilder.function("bad") {
            val table = param("t", DxirType(F32, listOf(10, 4, 2)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, listOf(3)))
            val y = op(OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(3, 4, 2)))
            listOf(y)
        }
        assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
    }

    @Test
    fun embeddingRejectsOutputShapeMismatch() {
        val fn = DxirBuilder.function("bad") {
            val table = param("t", DxirType(F32, listOf(10, 4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, listOf(3)))
            val y = op(OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(3, 5)))
            listOf(y)
        }
        assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
    }

    @Test
    fun convTranspose2dUsesFlippedKernelLayout() {
        val fn = DxirBuilder.function("ct") {
            val x = param("x", DxirType(F32, listOf(1, 16, 16, 16)))
            val k = param("k", DxirType(F32, listOf(16, 8, 3, 3)))  // [i, o, Kh, Kw]
            val y = op(
                OpKind.CONV_TRANSPOSE2D, listOf(x, k), DxirType(F32, listOf(1, 8, 32, 32)),
                attrs = mapOf(
                    "window_strides" to listOf(1, 1),
                    "padding" to listOf(listOf(1, 1), listOf(1, 1)),
                    "lhs_dilation" to listOf(2, 2),
                    "rhs_dilation" to listOf(1, 1),
                ),
            )
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        // Key difference from CONV2D: kernel layout `[i, o, 0, 1]` (input/output swapped).
        assertTrue(mlir.contains("dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1]"), mlir)
        assertTrue(mlir.contains("lhs_dilate = [2, 2]"), "upsampling via lhs_dilate missing: $mlir")
    }

    @Test
    fun argmaxEmitsIotaAndReduceWithBody() {
        val fn = DxirBuilder.function("am") {
            val x = param("x", DxirType(F32, listOf(2, 5)))
            val y = op(
                OpKind.ARGMAX, listOf(x), DxirType(io.tlaloc.core.I32, listOf(2)),
                attrs = mapOf("axis" to 1),
            )
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("stablehlo.iota dim = 1"), mlir)
        assertTrue(mlir.contains("stablehlo.reduce"), mlir)
        assertTrue(mlir.contains("reducer"), mlir)
        assertTrue(mlir.contains("stablehlo.compare  GT"), mlir)
        assertTrue(mlir.contains("stablehlo.select"), mlir)
        // Returns the second result of the pair.
        assertTrue(mlir.contains("#1 : tensor<2xi32>"), "must reference #1 of reduce pair: $mlir")
    }

    @Test
    fun argmaxDefaultsToLastAxis() {
        val fn = DxirBuilder.function("am") {
            val x = param("x", DxirType(F32, listOf(2, 5)))
            val y = op(OpKind.ARGMAX, listOf(x), DxirType(io.tlaloc.core.I32, listOf(2)))
            listOf(y)
        }
        // No axis attr → reduces along dim 1 (rank-1 = 1).
        assertTrue(fn.toStablehlo().contains("across dimensions = [1]"))
    }

    @Test
    fun argmaxRejectsFloatOutput() {
        val fn = DxirBuilder.function("bad") {
            val x = param("x", DxirType(F32, listOf(4)))
            val y = op(OpKind.ARGMAX, listOf(x), DxirType(F32, emptyList()))
            listOf(y)
        }
        assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
    }

    @Test
    fun crossEntropyEmitsLogSoftmaxAndOneHot() {
        val fn = DxirBuilder.function("xent") {
            val logits = param("l", DxirType(F32, listOf(4, 10)))
            val labels = param("y", DxirType(io.tlaloc.core.I32, listOf(4)))
            val loss = op(OpKind.CROSS_ENTROPY, listOf(logits, labels), DxirType(F32, listOf(4)))
            listOf(loss)
        }
        val mlir = fn.toStablehlo()
        // logsumexp pieces
        assertTrue(mlir.contains("applies stablehlo.maximum"), mlir)
        assertTrue(mlir.contains("stablehlo.exponential"), mlir)
        assertTrue(mlir.contains("stablehlo.log "), mlir)
        // one_hot pieces
        assertTrue(mlir.contains("stablehlo.iota dim = 0 : tensor<10xi32>"), mlir)
        assertTrue(mlir.contains("stablehlo.compare  EQ"), mlir)
        assertTrue(mlir.contains("stablehlo.convert"), mlir)
        // final negate of per-sample sum
        assertTrue(mlir.contains("stablehlo.negate") && mlir.contains("-> tensor<4xf32>"), mlir)
    }

    @Test
    fun scatterEmitsDimensionNumbersAndReductionBody() {
        // 1D scatter with min reduction — based on a canonical StableHLO testdata shape.
        val fn = DxirBuilder.function("s") {
            val operand = param("o", DxirType(F32, listOf(3)))
            val idx = param("i", DxirType(io.tlaloc.core.I64, listOf(3, 1)))
            val upd = param("u", DxirType(F32, listOf(3)))
            val y = op(
                OpKind.SCATTER, listOf(operand, idx, upd), DxirType(F32, listOf(3)),
                attrs = mapOf(
                    "inserted_window_dims" to listOf(0),
                    "scatter_dims_to_operand_dims" to listOf(0),
                    "index_vector_dim" to 1,
                    "reduction" to "min",
                ),
            )
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("\"stablehlo.scatter\""), mlir)
        assertTrue(mlir.contains("#stablehlo.scatter<"), mlir)
        assertTrue(mlir.contains("inserted_window_dims = [0]"), mlir)
        assertTrue(mlir.contains("stablehlo.minimum"), "min reduction body missing: $mlir")
        assertTrue(mlir.contains("stablehlo.return"), mlir)
    }

    @Test
    fun scatterReplaceReductionOmitsBinaryOp() {
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
        val mlir = fn.toStablehlo()
        // Replace: no binary op, just `stablehlo.return` of the update value.
        assertTrue(mlir.contains("update_window_dims = [0, 3]"), mlir)
        assertTrue(mlir.contains("unique_indices = true"), mlir)
        // Body should only have the return — no add/multiply/maximum/minimum.
        assertTrue(!mlir.contains("stablehlo.add") || mlir.indexOf("stablehlo.return") > mlir.indexOf("^bb0"))
    }

    @Test
    fun scatterAddReductionEmitsAddBody() {
        val fn = DxirBuilder.function("s") {
            val operand = param("o", DxirType(F32, listOf(4, 4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, listOf(2, 1)))
            val upd = param("u", DxirType(F32, listOf(2, 4)))
            val y = op(
                OpKind.SCATTER, listOf(operand, idx, upd), DxirType(F32, listOf(4, 4)),
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
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("stablehlo.add"), mlir)
    }

    // §0.4.114 — substrate-shape SCATTER lowering (no attrs, scalar I32 idx).

    @Test
    fun substrateScatterRank1EmitsReplaceBody() {
        // `base: tensor<4xf32>, idx: tensor<i32>, v: tensor<f32>` → tensor<4xf32>.
        // Replace semantics: body returns `upd` directly (no add).
        val fn = DxirBuilder.function("s") {
            val base = param("b", DxirType(F32, listOf(4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, emptyList()))
            val v = param("v", DxirType(F32, emptyList()))
            val y = op(OpKind.SCATTER, listOf(base, idx, v), DxirType(F32, listOf(4)))
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("\"stablehlo.scatter\""), mlir)
        assertTrue(mlir.contains("inserted_window_dims = [0]"), mlir)
        assertTrue(mlir.contains("scatter_dims_to_operand_dims = [0]"), mlir)
        assertTrue(mlir.contains("index_vector_dim = 0"), mlir)
        assertTrue(!mlir.contains("update_window_dims"), "rank-1 should have no update_window_dims: $mlir")
        // Replace body has no `stablehlo.add` — just `stablehlo.return`.
        assertTrue(!mlir.contains("stablehlo.add"), "replace body must not contain add: $mlir")
        assertTrue(mlir.contains("stablehlo.return"), mlir)
    }

    @Test
    fun substrateScatterRank2EmitsRowReplaceShape() {
        // `base: tensor<3x4xf32>, idx: tensor<i32>, v: tensor<4xf32>` → tensor<3x4xf32>.
        val fn = DxirBuilder.function("s") {
            val base = param("b", DxirType(F32, listOf(3, 4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, emptyList()))
            val v = param("v", DxirType(F32, listOf(4)))
            val y = op(OpKind.SCATTER, listOf(base, idx, v), DxirType(F32, listOf(3, 4)))
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("update_window_dims = [0]"), mlir)
        assertTrue(mlir.contains("inserted_window_dims = [0]"), mlir)
        assertTrue(!mlir.contains("stablehlo.add"), "replace body must not contain add: $mlir")
        assertTrue(
            mlir.contains("(tensor<3x4xf32>, tensor<i32>, tensor<4xf32>) -> tensor<3x4xf32>"),
            "expected scatter type signature for rank-2 substrate: $mlir",
        )
    }

    @Test
    fun generalScatterStillRoutesThroughAttrPath() {
        // The substrate-shape detection only fires when there's no `scatter_dims_to_operand_dims`
        // attr. A SCATTER with explicit attrs (the canonical multi-dim form) still routes
        // through the original attr-driven path.
        val fn = DxirBuilder.function("s") {
            val operand = param("o", DxirType(F32, listOf(3)))
            val idx = param("i", DxirType(io.tlaloc.core.I64, listOf(3, 1)))
            val upd = param("u", DxirType(F32, listOf(3)))
            val y = op(
                OpKind.SCATTER, listOf(operand, idx, upd), DxirType(F32, listOf(3)),
                attrs = mapOf(
                    "inserted_window_dims" to listOf(0),
                    "scatter_dims_to_operand_dims" to listOf(0),
                    "index_vector_dim" to 1,
                    "reduction" to "min",
                ),
            )
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        // The `min` reduction body fires (substrate would have been `replace`).
        assertTrue(mlir.contains("stablehlo.minimum"), mlir)
    }

    // §0.4.113 — substrate-shape GATHER lowering (no attrs, scalar I32 idx).

    @Test
    fun substrateGatherRank1EmitsCanonicalAttrs() {
        // `arr: tensor<4xf32>, idx: tensor<i32>` → tensor<f32>. Substrate shape:
        // no attrs on the DxirOp; the emitter synthesizes the canonical stablehlo.gather
        // attrs.
        val fn = DxirBuilder.function("g") {
            val arr = param("a", DxirType(F32, listOf(4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, emptyList()))
            val y = op(OpKind.GATHER, listOf(arr, idx), DxirType(F32, emptyList()))
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("\"stablehlo.gather\""), mlir)
        // No offset_dims for rank-1 substrate (scalar result).
        assertTrue(mlir.contains("offset_dims = []"), mlir)
        assertTrue(mlir.contains("collapsed_slice_dims = [0]"), mlir)
        assertTrue(mlir.contains("start_index_map = [0]"), mlir)
        assertTrue(mlir.contains("index_vector_dim = 0"), mlir)
        assertTrue(mlir.contains("slice_sizes = array<i64: 1>"), mlir)
        // Function signature should reference the original tensor types unchanged.
        assertTrue(
            mlir.contains("(tensor<4xf32>, tensor<i32>) -> tensor<f32>"),
            "expected gather type signature for rank-1 substrate: $mlir",
        )
    }

    @Test
    fun substrateGatherRank2EmitsRowSliceAttrs() {
        // `arr: tensor<3x4xf32>, idx: tensor<i32>` → tensor<4xf32>. The single offset
        // axis is axis 0 of the rank-1 result; slice_sizes = [1, N].
        val fn = DxirBuilder.function("g") {
            val arr = param("a", DxirType(F32, listOf(3, 4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, emptyList()))
            val y = op(OpKind.GATHER, listOf(arr, idx), DxirType(F32, listOf(4)))
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("offset_dims = [0]"), mlir)
        assertTrue(mlir.contains("collapsed_slice_dims = [0]"), mlir)
        assertTrue(mlir.contains("start_index_map = [0]"), mlir)
        assertTrue(mlir.contains("index_vector_dim = 0"), mlir)
        assertTrue(mlir.contains("slice_sizes = array<i64: 1, 4>"), mlir)
        assertTrue(
            mlir.contains("(tensor<3x4xf32>, tensor<i32>) -> tensor<4xf32>"),
            "expected gather type signature for rank-2 substrate: $mlir",
        )
    }

    @Test
    fun generalGatherStillRoutesThroughAttrPath() {
        // The substrate-shape detection only fires when there are NO `offset_dims`
        // attr. A GATHER with explicit attrs (the canonical multi-dim stablehlo
        // form) still routes through the original attr-driven path, even when its
        // indices happen to be a scalar.
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
        val mlir = fn.toStablehlo()
        // The user-supplied index_vector_dim = 1 must survive (substrate would have set 0).
        assertTrue(mlir.contains("index_vector_dim = 1"), mlir)
        assertTrue(mlir.contains("slice_sizes = array<i64: 1, 4>"), mlir)
    }

    // §0.4.112 — SCATTER_ADD substrate-shape lowering.

    @Test
    fun scatterAddRank1EmitsScatterWithAddBody() {
        // Substrate shape: rank-1 base + scalar idx + scalar value (no attrs).
        // Expect: stablehlo.scatter with `add` reduction body and no reshape ops
        // (rank-0 scatter_indices + index_vector_dim=0 triggers implicit expansion).
        val fn = DxirBuilder.function("g") {
            val base = param("b", DxirType(F32, listOf(4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, emptyList()))
            val v = param("v", DxirType(F32, emptyList()))
            val y = op(OpKind.SCATTER_ADD, listOf(base, idx, v), DxirType(F32, listOf(4)))
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("\"stablehlo.scatter\""), mlir)
        assertTrue(mlir.contains("inserted_window_dims = [0]"), mlir)
        assertTrue(mlir.contains("scatter_dims_to_operand_dims = [0]"), mlir)
        assertTrue(mlir.contains("index_vector_dim = 0"), mlir)
        // No update_window_dims for rank-1.
        assertTrue(!mlir.contains("update_window_dims"), "rank-1 should have no update_window_dims: $mlir")
        // Body computation is `add`.
        assertTrue(mlir.contains("stablehlo.add"), "expected add body: $mlir")
        assertTrue(mlir.contains("stablehlo.return"), mlir)
        // No reshape — operands flow directly.
        assertTrue(
            !mlir.contains("stablehlo.reshape"),
            "rank-1 scatter_add should not emit a reshape: $mlir",
        )
    }

    @Test
    fun scatterAddRank2EmitsRowUpdateShape() {
        // Substrate shape: rank-2 base + scalar idx + rank-1 [N] value (no attrs).
        // update_window_dims includes the inner axis (the value's rank-1 maps to it).
        val fn = DxirBuilder.function("g") {
            val base = param("b", DxirType(F32, listOf(3, 4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, emptyList()))
            val v = param("v", DxirType(F32, listOf(4)))
            val y = op(OpKind.SCATTER_ADD, listOf(base, idx, v), DxirType(F32, listOf(3, 4)))
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        // update_window_dims indexes UPDATES' axes; for rank-1 updates (the row),
        // the single window dim is axis 0 of updates → axis 1 of the operand.
        assertTrue(mlir.contains("update_window_dims = [0]"), mlir)
        assertTrue(mlir.contains("inserted_window_dims = [0]"), mlir)
        assertTrue(mlir.contains("stablehlo.add"), mlir)
        // The function signature line should reference the original tensor types.
        assertTrue(
            mlir.contains("(tensor<3x4xf32>, tensor<i32>, tensor<4xf32>) -> tensor<3x4xf32>"),
            "expected scatter type signature for rank-2 substrate: $mlir",
        )
    }

    @Test
    fun scatterAddIgnoresInPlaceAttr() {
        // §0.4.46's `in_place` attr is a synthesis-side hint; it must NOT leak into
        // the emitted MLIR (stablehlo.scatter is functional, not in-place).
        val fn = DxirBuilder.function("g") {
            val b = param("b", DxirType(F32, listOf(4)))
            val i = param("i", DxirType(io.tlaloc.core.I32, emptyList()))
            val v = param("v", DxirType(F32, emptyList()))
            listOf(
                op(
                    OpKind.SCATTER_ADD,
                    listOf(b, i, v),
                    DxirType(F32, listOf(4)),
                    attrs = mapOf("in_place" to true),
                ),
            )
        }
        val mlir = fn.toStablehlo()
        assertTrue(
            !mlir.contains("in_place"),
            "in_place hint must not appear in emitted MLIR: $mlir",
        )
    }

    @Test
    fun batchNorm2dEmitsInferenceOp() {
        val fn = DxirBuilder.function("bn") {
            val x = param("x", DxirType(F32, listOf(4, 256)))
            val scale = param("s", DxirType(F32, listOf(256)))
            val offset = param("b", DxirType(F32, listOf(256)))
            val mean = param("m", DxirType(F32, listOf(256)))
            val variance = param("v", DxirType(F32, listOf(256)))
            val y = op(
                OpKind.BATCHNORM, listOf(x, scale, offset, mean, variance), DxirType(F32, listOf(4, 256)),
                attrs = mapOf("epsilon" to 1e-5f, "feature_index" to 1),
            )
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("\"stablehlo.batch_norm_inference\""), mlir)
        assertTrue(mlir.contains("epsilon = 1.0E-5 : f32"), "eps: $mlir")
        assertTrue(mlir.contains("feature_index = 1 : i64"), "feature_index: $mlir")
        assertTrue(
            mlir.contains("(tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256xf32>"),
            mlir,
        )
    }

    @Test
    fun batchNormNchwWithExplicitFeatureIndex() {
        // Conv-output shape: (batch=2, channels=64, H=8, W=8), per-channel stats.
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
        assertTrue(fn.toStablehlo().contains("feature_index = 1 : i64"))
    }

    @Test
    fun batchNormDefaultsEpsilonAndFeatureIndex() {
        val fn = DxirBuilder.function("bn") {
            val x = param("x", DxirType(F32, listOf(4, 16)))
            val scale = param("s", DxirType(F32, listOf(16)))
            val offset = param("b", DxirType(F32, listOf(16)))
            val mean = param("m", DxirType(F32, listOf(16)))
            val variance = param("v", DxirType(F32, listOf(16)))
            val y = op(
                OpKind.BATCHNORM, listOf(x, scale, offset, mean, variance), DxirType(F32, listOf(4, 16)),
            )
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        // Default eps = 1e-5, default feature_index = 1.
        assertTrue(mlir.contains("epsilon = 1.0E-5"), "default eps missing: $mlir")
        assertTrue(mlir.contains("feature_index = 1 : i64"), "default feature_index missing: $mlir")
    }

    @Test
    fun batchNormRejectsPerChannelSizeMismatch() {
        val fn = DxirBuilder.function("bad") {
            val x = param("x", DxirType(F32, listOf(4, 256)))
            val scale = param("s", DxirType(F32, listOf(100)))   // wrong size
            val offset = param("b", DxirType(F32, listOf(256)))
            val mean = param("m", DxirType(F32, listOf(256)))
            val variance = param("v", DxirType(F32, listOf(256)))
            val y = op(
                OpKind.BATCHNORM, listOf(x, scale, offset, mean, variance), DxirType(F32, listOf(4, 256)),
            )
            listOf(y)
        }
        assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
    }

    @Test
    fun batchNormRejectsBadFeatureIndex() {
        val fn = DxirBuilder.function("bad") {
            val x = param("x", DxirType(F32, listOf(4, 256)))
            val scale = param("s", DxirType(F32, listOf(256)))
            val offset = param("b", DxirType(F32, listOf(256)))
            val mean = param("m", DxirType(F32, listOf(256)))
            val variance = param("v", DxirType(F32, listOf(256)))
            val y = op(
                OpKind.BATCHNORM, listOf(x, scale, offset, mean, variance), DxirType(F32, listOf(4, 256)),
                attrs = mapOf("feature_index" to 9),
            )
            listOf(y)
        }
        assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
    }

    @Test
    fun splitEmitsNSliceOpsAndReturnsViaResultIndex() {
        val fn = DxirBuilder.function("sp") {
            val x = param("x", DxirType(F32, listOf(6, 4)))
            val split = opMulti(
                OpKind.SPLIT, listOf(x),
                types = listOf(DxirType(F32, listOf(3, 4)), DxirType(F32, listOf(3, 4))),
                attrs = mapOf("axis" to 0, "sizes" to listOf(3, 3)),
            )
            listOf(split.result(0), split.result(1))
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("stablehlo.slice %0 [0:3, 0:4]"), "first slice: $mlir")
        assertTrue(mlir.contains("stablehlo.slice %0 [3:6, 0:4]"), "second slice: $mlir")
        // Two return values since there are 2 results.
        assertTrue(
            mlir.contains("return") && mlir.contains("tensor<3x4xf32>, tensor<3x4xf32>"),
            "multi-return types: $mlir",
        )
    }

    @Test
    fun splitUnequalSizes() {
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
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("stablehlo.slice %0 [0:2"), mlir)
        assertTrue(mlir.contains("stablehlo.slice %0 [2:5"), mlir)
        assertTrue(mlir.contains("stablehlo.slice %0 [5:10"), mlir)
    }

    @Test
    fun splitAlongNonZeroAxis() {
        val fn = DxirBuilder.function("sp") {
            val x = param("x", DxirType(F32, listOf(2, 3, 8)))
            val split = opMulti(
                OpKind.SPLIT, listOf(x),
                types = listOf(DxirType(F32, listOf(2, 3, 4)), DxirType(F32, listOf(2, 3, 4))),
                attrs = mapOf("axis" to -1, "sizes" to listOf(4, 4)),
            )
            listOf(split.result(0), split.result(1))
        }
        val mlir = fn.toStablehlo()
        // Axis -1 normalizes to 2; first slice is [full, full, 0:4], second [full, full, 4:8].
        assertTrue(mlir.contains("[0:2, 0:3, 0:4]"), mlir)
        assertTrue(mlir.contains("[0:2, 0:3, 4:8]"), mlir)
    }

    @Test
    fun splitRejectsSizesSumMismatch() {
        val fn = DxirBuilder.function("bad") {
            val x = param("x", DxirType(F32, listOf(6, 4)))
            val split = opMulti(
                OpKind.SPLIT, listOf(x),
                types = listOf(DxirType(F32, listOf(3, 4)), DxirType(F32, listOf(2, 4))),
                attrs = mapOf("axis" to 0, "sizes" to listOf(3, 2)),   // sum=5 ≠ 6
            )
            listOf(split.result(0), split.result(1))
        }
        assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
    }

    @Test
    fun splitResultTypeMismatchIsCaught() {
        val fn = DxirBuilder.function("bad") {
            val x = param("x", DxirType(F32, listOf(6, 4)))
            val split = opMulti(
                OpKind.SPLIT, listOf(x),
                // Declared shape for result 1 is wrong.
                types = listOf(DxirType(F32, listOf(3, 4)), DxirType(F32, listOf(5, 4))),
                attrs = mapOf("axis" to 0, "sizes" to listOf(3, 3)),
            )
            listOf(split.result(0), split.result(1))
        }
        assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
    }

    @Test
    fun splitResultUsableAsOperandInSubsequentOp() {
        // One half of a split fed into a relu — exercises DxirOpResult as operand.
        val fn = DxirBuilder.function("sp_then_relu") {
            val x = param("x", DxirType(F32, listOf(4, 2)))
            val split = opMulti(
                OpKind.SPLIT, listOf(x),
                types = listOf(DxirType(F32, listOf(2, 2)), DxirType(F32, listOf(2, 2))),
                attrs = mapOf("axis" to 0, "sizes" to listOf(2, 2)),
            )
            val firstHalf = split.result(0)
            val r = op(OpKind.RELU, listOf(firstHalf), DxirType(F32, listOf(2, 2)))
            listOf(r)
        }
        val mlir = fn.toStablehlo()
        // The relu must consume the first slice's SSA name — not a named dxir id.
        assertTrue(mlir.contains("stablehlo.maximum") && mlir.contains("stablehlo.slice"), mlir)
    }

    @Test
    fun scatterRejectsUnknownReduction() {
        val fn = DxirBuilder.function("bad") {
            val operand = param("o", DxirType(F32, listOf(3)))
            val idx = param("i", DxirType(io.tlaloc.core.I64, listOf(3, 1)))
            val upd = param("u", DxirType(F32, listOf(3)))
            val y = op(
                OpKind.SCATTER, listOf(operand, idx, upd), DxirType(F32, listOf(3)),
                attrs = mapOf(
                    "scatter_dims_to_operand_dims" to listOf(0),
                    "index_vector_dim" to 1,
                    "reduction" to "bogus",
                ),
            )
            listOf(y)
        }
        assertFailsWith<IllegalStateException> { fn.toStablehlo() }
    }

    @Test
    fun crossEntropyRejectsRankThreeLogits() {
        val fn = DxirBuilder.function("bad") {
            val logits = param("l", DxirType(F32, listOf(4, 10, 2)))
            val labels = param("y", DxirType(io.tlaloc.core.I32, listOf(4)))
            val loss = op(OpKind.CROSS_ENTROPY, listOf(logits, labels), DxirType(F32, listOf(4)))
            listOf(loss)
        }
        assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
    }

    @Test
    fun meshDeclarationEmittedAtModuleScope() {
        val mesh = io.tlaloc.ir.DxirMesh(
            "m",
            listOf(io.tlaloc.ir.DxirMeshAxis("data", 8), io.tlaloc.ir.DxirMeshAxis("model", 4)),
        )
        val fn = DxirBuilder.function("f") {
            declareMesh(mesh)
            val x = param("x", DxirType(F32, listOf(32, 16)))
            listOf(x)
        }
        val mlir = io.tlaloc.ir.DxirModule(listOf(fn)).toStablehlo()
        assertTrue(mlir.contains("sdy.mesh @m = <[\"data\"=8, \"model\"=4]>"), mlir)
    }

    @Test
    fun shardConstraintLowersToSdyOp() {
        val mesh = io.tlaloc.ir.DxirMesh("m", listOf(io.tlaloc.ir.DxirMeshAxis("data", 8)))
        val sharding = io.tlaloc.ir.DxirSharding(
            "m",
            listOf(
                io.tlaloc.ir.DxirDimSharding(listOf(io.tlaloc.ir.DxirAxisRef.Full("data"))),
                io.tlaloc.ir.DxirDimSharding(emptyList()),
            ),
        )
        val fn = DxirBuilder.function("f") {
            declareMesh(mesh)
            val x = param("x", DxirType(F32, listOf(8, 16)))
            val y = op(
                OpKind.SHARD_CONSTRAINT, listOf(x), DxirType(F32, listOf(8, 16)),
                sharding = sharding,
            )
            listOf(y)
        }
        val mlir = io.tlaloc.ir.DxirModule(listOf(fn)).toStablehlo()
        assertTrue(mlir.contains("sdy.sharding_constraint"), mlir)
        assertTrue(mlir.contains("<@m, [{\"data\"}, {}]>"), mlir)
    }

    @Test
    fun shardingAttrEmitsOpenAxisAndPriority() {
        val mesh = io.tlaloc.ir.DxirMesh("m", listOf(io.tlaloc.ir.DxirMeshAxis("a", 2)))
        val sharding = io.tlaloc.ir.DxirSharding(
            "m",
            listOf(
                io.tlaloc.ir.DxirDimSharding(
                    axes = listOf(io.tlaloc.ir.DxirAxisRef.Full("a")),
                    closed = false,
                    priority = 1,
                ),
            ),
        )
        val fn = DxirBuilder.function("f") {
            declareMesh(mesh)
            val x = param("x", DxirType(F32, listOf(4)))
            val y = op(
                OpKind.SHARD_CONSTRAINT, listOf(x), DxirType(F32, listOf(4)),
                sharding = sharding,
            )
            listOf(y)
        }
        val mlir = io.tlaloc.ir.DxirModule(listOf(fn)).toStablehlo()
        // Open-axis marker `?` and priority suffix `p1`.
        assertTrue(mlir.contains("<@m, [{\"a\", ?}p1]>"), mlir)
    }

    @Test
    fun shardingWithReplicatedAxes() {
        val mesh = io.tlaloc.ir.DxirMesh(
            "m",
            listOf(
                io.tlaloc.ir.DxirMeshAxis("data", 4),
                io.tlaloc.ir.DxirMeshAxis("model", 2),
            ),
        )
        val sharding = io.tlaloc.ir.DxirSharding(
            "m",
            dimShardings = listOf(
                io.tlaloc.ir.DxirDimSharding(listOf(io.tlaloc.ir.DxirAxisRef.Full("data"))),
                io.tlaloc.ir.DxirDimSharding(emptyList()),
            ),
            replicated = listOf(io.tlaloc.ir.DxirAxisRef.Full("model")),
        )
        val fn = DxirBuilder.function("f") {
            declareMesh(mesh)
            val x = param("x", DxirType(F32, listOf(8, 4)))
            val y = op(
                OpKind.SHARD_CONSTRAINT, listOf(x), DxirType(F32, listOf(8, 4)),
                sharding = sharding,
            )
            listOf(y)
        }
        val mlir = io.tlaloc.ir.DxirModule(listOf(fn)).toStablehlo()
        assertTrue(mlir.contains("replicated={\"model\"}"), mlir)
    }

    @Test
    fun shardConstraintRequiresShardingAttr() {
        val fn = DxirBuilder.function("bad") {
            declareMesh(io.tlaloc.ir.DxirMesh("m", listOf(io.tlaloc.ir.DxirMeshAxis("a", 2))))
            val x = param("x", DxirType(F32, listOf(4)))
            // Note: sharding = null
            val y = op(OpKind.SHARD_CONSTRAINT, listOf(x), DxirType(F32, listOf(4)))
            listOf(y)
        }
        assertFailsWith<IllegalStateException> { io.tlaloc.ir.DxirModule(listOf(fn)).toStablehlo() }
    }

    @Test
    fun embeddingRejectsFloatIndices() {
        val fn = DxirBuilder.function("bad") {
            val table = param("t", DxirType(F32, listOf(10, 4)))
            val idx = param("i", DxirType(F32, listOf(3)))  // wrong dtype
            val y = op(OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(3, 4)))
            listOf(y)
        }
        assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
    }

    @Test
    fun sdpaRejectsShapeMismatch() {
        val fn = DxirBuilder.function("bad") {
            val q = param("q", DxirType(F32, listOf(1, 2, 4, 8)))
            val k = param("k", DxirType(F32, listOf(1, 2, 6, 16)))  // different head_dim
            val v = param("v", DxirType(F32, listOf(1, 2, 6, 8)))
            val y = op(
                OpKind.SCALED_DOT_PRODUCT_ATTENTION,
                listOf(q, k, v),
                DxirType(F32, listOf(1, 2, 4, 8)),
            )
            listOf(y)
        }
        assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
    }

    @Test
    fun castLowersToConvert() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(io.tlaloc.core.I32, listOf(4)))
            val y = op(OpKind.CAST, listOf(x), DxirType(F32, listOf(4)))
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("stablehlo.convert %0 : (tensor<4xi32>) -> tensor<4xf32>"), mlir)
    }

    @Test
    fun reshapeLowersCleanly() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(2, 3)))
            val y = op(OpKind.RESHAPE, listOf(x), DxirType(F32, listOf(6)))
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("stablehlo.reshape %0 : (tensor<2x3xf32>) -> tensor<6xf32>"), mlir)
    }

    @Test
    fun transposeEmitsPermutation() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(2, 3)))
            val y = op(
                OpKind.TRANSPOSE, listOf(x), DxirType(F32, listOf(3, 2)),
                attrs = mapOf("permutation" to listOf(1, 0)),
            )
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("stablehlo.transpose %0, dims = [1, 0]"), mlir)
        assertTrue(mlir.contains("(tensor<2x3xf32>) -> tensor<3x2xf32>"), mlir)
    }

    @Test
    fun broadcastEmitsDims() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(3)))
            val y = op(
                OpKind.BROADCAST, listOf(x), DxirType(F32, listOf(2, 3)),
                attrs = mapOf("broadcast_dimensions" to listOf(1)),
            )
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("stablehlo.broadcast_in_dim %0, dims = [1]"), mlir)
        assertTrue(mlir.contains("(tensor<3xf32>) -> tensor<2x3xf32>"), mlir)
    }

    @Test
    fun concatLowersVariadicOperands() {
        val fn = DxirBuilder.function("f") {
            val a = param("a", DxirType(F32, listOf(2, 3)))
            val b = param("b", DxirType(F32, listOf(4, 3)))
            val c = op(
                OpKind.CONCAT, listOf(a, b), DxirType(F32, listOf(6, 3)),
                attrs = mapOf("dimension" to 0),
            )
            listOf(c)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("stablehlo.concatenate %0, %1, dim = 0"), mlir)
        assertTrue(mlir.contains("(tensor<2x3xf32>, tensor<4x3xf32>) -> tensor<6x3xf32>"), mlir)
    }

    @Test
    fun maxAndMinReducersEmitInfinityInits() {
        val fMax = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(4)))
            val y = op(OpKind.MAX, listOf(x), DxirType(F32, emptyList()))
            listOf(y)
        }.toStablehlo()
        assertTrue(fMax.contains("stablehlo.constant dense<0xFF800000> : tensor<f32>"), fMax)
        assertTrue(fMax.contains("applies stablehlo.maximum"), fMax)

        val fMin = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(4)))
            val y = op(OpKind.MIN, listOf(x), DxirType(F32, emptyList()))
            listOf(y)
        }.toStablehlo()
        assertTrue(fMin.contains("stablehlo.constant dense<0x7F800000> : tensor<f32>"), fMin)
        assertTrue(fMin.contains("applies stablehlo.minimum"), fMin)
    }

    @Test
    fun sliceEmitsRangeSyntax() {
        val fn = DxirBuilder.function("f") {
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
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("stablehlo.slice %0 [1:3, 0:2]"), "unexpected slice form: $mlir")
    }

    @Test
    fun sliceEmitsStrideWhenNonUnit() {
        val fn = DxirBuilder.function("f") {
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
        assertTrue(fn.toStablehlo().contains("stablehlo.slice %0 [0:6:2]"))
    }

    @Test
    fun dotLowersToDotGeneralWithScalarContracting() {
        val fn = DxirBuilder.function("f") {
            val a = param("a", DxirType(F32, listOf(5)))
            val b = param("b", DxirType(F32, listOf(5)))
            val c = op(OpKind.DOT, listOf(a, b), DxirType(F32, emptyList()))
            listOf(c)
        }
        val mlir = fn.toStablehlo()
        assertTrue(
            mlir.contains("stablehlo.dot_general %0, %1, contracting_dims = [0] x [0]"),
            mlir,
        )
        assertTrue(mlir.contains("(tensor<5xf32>, tensor<5xf32>) -> tensor<f32>"), mlir)
    }

    @Test
    fun transposeRejectsBadPermutation() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(2, 3)))
            val y = op(
                OpKind.TRANSPOSE, listOf(x), DxirType(F32, listOf(3, 2)),
                attrs = mapOf("permutation" to listOf(0, 0)),
            )
            listOf(y)
        }
        assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
    }

    @Test
    fun reshapeRejectsSizeMismatch() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(2, 3)))
            val y = op(OpKind.RESHAPE, listOf(x), DxirType(F32, listOf(5)))
            listOf(y)
        }
        assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
    }

    @Test
    fun siluLowersToSigmoidTimesInput() {
        val mlir = singleUnary(OpKind.SILU, DxirType(F32, listOf(4)))
        assertTrue(mlir.contains("stablehlo.logistic"), mlir)
        // Result is x * sigmoid(x): the final multiply takes the input %0 twice-adjacent is wrong;
        // it's %0 (input) × %sN (sigmoid output).
        assertTrue(Regex("stablehlo\\.multiply %0, %s\\d+").containsMatchIn(mlir), mlir)
    }

    @Test
    fun geluEmitsConstantsAndTanh() {
        val mlir = singleUnary(OpKind.GELU, DxirType(F32, listOf(4)))
        assertTrue(mlir.contains("dense<0.5>"), "missing 0.5 constant: $mlir")
        assertTrue(mlir.contains("dense<0.7978845608>"), "missing √(2/π) constant: $mlir")
        assertTrue(mlir.contains("dense<0.044715>"), "missing gelu coefficient: $mlir")
        assertTrue(mlir.contains("stablehlo.tanh"), mlir)
    }

    @Test
    fun softmaxRankOneLoweringSequence() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(5)))
            val y = op(OpKind.SOFTMAX, listOf(x), DxirType(F32, listOf(5)))
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        // Stable form: max → broadcast → subtract → exp → sum → broadcast → divide
        assertTrue(mlir.contains("dense<0xFF800000> : tensor<f32>"), "max init missing: $mlir")
        assertTrue(mlir.contains("applies stablehlo.maximum"), mlir)
        assertTrue(mlir.contains("stablehlo.broadcast_in_dim"), mlir)
        assertTrue(mlir.contains("stablehlo.subtract"), mlir)
        assertTrue(mlir.contains("stablehlo.exponential"), mlir)
        assertTrue(mlir.contains("applies stablehlo.add"), mlir)
        assertTrue(mlir.contains("stablehlo.divide"), mlir)
    }

    @Test
    fun softmaxLowersForRankTwoWithDefaultAxis() {
        // Default axis = -1 → last dim. Output shape equals input shape.
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(2, 3)))
            val y = op(OpKind.SOFTMAX, listOf(x), DxirType(F32, listOf(2, 3)))
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("across dimensions = [1]"), "should reduce along last dim: $mlir")
        assertTrue(
            mlir.contains("dims = [0]") && mlir.contains("tensor<2xf32>"),
            "broadcast back should keep dim 0: $mlir",
        )
    }

    @Test
    fun logsumexpLowersToScalarResult() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(5)))
            val y = op(OpKind.LOGSUMEXP, listOf(x), DxirType(F32, emptyList()))
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("dense<0xFF800000>"), mlir)
        assertTrue(mlir.contains("stablehlo.exponential"), mlir)
        assertTrue(mlir.contains("stablehlo.log"), mlir)
        // Final add is log(sum(exp)) + max — both scalars.
        assertTrue(mlir.contains("-> tensor<f32>"), "final type should be scalar: $mlir")
    }

    @Test
    fun layerNormEmitsMeanVarSqrt() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(8)))
            val y = op(
                OpKind.LAYERNORM, listOf(x), DxirType(F32, listOf(8)),
                attrs = mapOf("epsilon" to 1e-5f),
            )
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("applies stablehlo.add"), "reduce_add missing: $mlir")
        assertTrue(mlir.contains("dense<8.0>"), "divisor N missing: $mlir")
        assertTrue(mlir.contains("stablehlo.sqrt"), mlir)
        // eps appears — float toString yields scientific notation for 1e-5f
        assertTrue(mlir.contains("1.0E-5") || mlir.contains("0.00001"), "eps missing: $mlir")
    }

    @Test
    fun layerNormDefaultsEpsWhenAttrMissing() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(4)))
            val y = op(OpKind.LAYERNORM, listOf(x), DxirType(F32, listOf(4)))
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("stablehlo.sqrt"), mlir)
    }

    @Test
    fun rmsNormEmitsMeanSquareAndSqrt() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(4)))
            val y = op(
                OpKind.RMSNORM, listOf(x), DxirType(F32, listOf(4)),
                attrs = mapOf("epsilon" to 1e-5f),
            )
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("stablehlo.multiply %0, %0"), "x*x missing: $mlir")
        assertTrue(mlir.contains("stablehlo.sqrt"), mlir)
        assertTrue(mlir.contains("dense<4.0>"), "divisor N missing: $mlir")
    }

    @Test
    fun vectorPipelineEmitsExpectedLineSequence() {
        val fn = capture(
            f = { x: Tracer<Rank1<Sym>> -> (x * x).sum() },
            input = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f, 4f)),
            name = "sumSq",
        )
        val mlir = fn.toStablehlo()
        // Multiply elementwise, then reduce over single dim.
        assertTrue(
            mlir.contains("stablehlo.multiply %0, %0 : tensor<4xf32>"),
            mlir,
        )
        assertTrue(
            mlir.contains("applies stablehlo.add across dimensions = [0]"),
            mlir,
        )
        assertTrue(
            mlir.contains("(tensor<4xf32>, tensor<f32>) -> tensor<f32>"),
            mlir,
        )
    }

    // --- helpers ---

    private fun singleOpFunction(op: OpKind, type: DxirType): String {
        val fn = DxirBuilder.function("f") {
            val a = param("a", type)
            val b = param("b", type)
            val c = op(op, listOf(a, b), type)
            listOf(c)
        }
        return fn.toStablehlo()
    }

    private fun singleUnary(op: OpKind, type: DxirType): String {
        val fn = DxirBuilder.function("f") {
            val a = param("a", type)
            val b = op(op, listOf(a), type)
            listOf(b)
        }
        return fn.toStablehlo()
    }
}
