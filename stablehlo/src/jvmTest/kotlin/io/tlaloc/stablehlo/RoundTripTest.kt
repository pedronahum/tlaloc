package io.tlaloc.stablehlo

import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.capture
import io.tlaloc.autograd.capture2
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.matmul
import io.tlaloc.autograd.plus
import io.tlaloc.autograd.relu
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.Rank3
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
    fun capturedScalarBroadcastRoundTripsThroughStablehloTranslate() {
        // §0.4.80 — §0.4.77's scalar-broadcast Tracer operator records
        // OpKind.BROADCAST on the tape. Capture carries that op through to a
        // DxirFunction; the StableHLO emitter formats it as `stablehlo.
        // broadcast_in_dim ... dims = [...]`. This test verifies the captured
        // function is valid MLIR — catches any mismatch between the tape-side
        // BROADCAST recording (no attrs) and the emitter's attr requirements.
        requireTranslateOrSkip()
        val rank1 = capture2(
            f = { x: Tracer<Rank1<Sym>>, c: Tracer<ScalarShape> -> (x * c).sum() },
            a = Tensors.f32Vector<Sym>(floatArrayOf(2f, 4f, 8f)),
            b = Tensors.f32Scalar(3f),
            name = "cap_scalar_bcast",
        )
        validate(DxirModule(listOf(rank1)).toStablehlo(), "capture rank-1 with scalar broadcast")
    }

    @Test
    fun capturedRank3ScalarBroadcastRoundTripsThroughStablehloTranslate() {
        // §0.4.99 — exercises §0.4.97's f32Tensor3 + rank-3 scalar-broadcast
        // through the full Capture → emitter → stablehlo-translate pipeline.
        // BroadcastRule's reverse for rank-3 emits SUM with `reduction_dims =
        // [0, 1, 2]` (axis-aware path, §0.4.84); the emitter's
        // `readReductionDims` defaults missing attrs to all-axes for the
        // simple all-axis SUM that .sum() produces.
        requireTranslateOrSkip()
        val rank3 = capture2(
            f = { x: Tracer<Rank3<Sym, Sym, Sym>>, c: Tracer<ScalarShape> -> (x * c).sum() },
            a = Tensors.f32Tensor3<Sym, Sym, Sym>(2, 2, 2, FloatArray(8) { (it + 1).toFloat() }),
            b = Tensors.f32Scalar(0.5f),
            name = "cap_rank3_scalar_bcast",
        )
        validate(DxirModule(listOf(rank3)).toStablehlo(), "capture rank-3 with scalar broadcast")
    }

    @Test
    fun capturedRank2ScalarBroadcastRoundTripsThroughStablehloTranslate() {
        // §0.4.83 — rank-2 extension of §0.4.80's round-trip. §0.4.78's rank-2
        // scalar-broadcast overload records the SAME OpKind.BROADCAST shape on
        // the tape (with rank-2 target dims), so the same emitter arm handles
        // it. This test pins that the rank axis doesn't drift — if the
        // generalised `broadcastScalar` helper ever regresses a rank-specific
        // attr, the stablehlo-translate validator catches it here.
        requireTranslateOrSkip()
        val rank2 = capture2(
            f = { x: Tracer<Rank2<Sym, Sym>>, c: Tracer<ScalarShape> -> (x * c).sum() },
            a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f)),
            b = Tensors.f32Scalar(0.5f),
            name = "cap_rank2_scalar_bcast",
        )
        validate(DxirModule(listOf(rank2)).toStablehlo(), "capture rank-2 with scalar broadcast")
    }

    @Test
    fun capturedLambdaWithRankNConstantRoundTripsThroughStablehloTranslate() {
        // §0.4.74 — full pipeline integration: a user lambda that creates a
        // non-param leaf via `Tracer.constant(FloatArray)`, gets captured into
        // a DxirFunction (§0.4.71 Capture fix stores it as a FloatArray-valued
        // DxirConst), lowered to StableHLO (§0.4.73 emitter's nested dense
        // literal arm), and validated by `stablehlo-translate --serialize`.
        //
        // Previously each piece had its own test: CaptureTest for the DxirConst
        // value shape, DxirInterpreter for local eval (§0.4.72), EmitterTest /
        // RoundTripTest for the emission. Composing them into one test ensures
        // a future refactor that breaks the chain at any intermediate point
        // fails a specific test with a clear breadcrumb.
        requireTranslateOrSkip()

        // Rank-1: `f(x) = x + [10, 20, 30]`.
        val rank1Fn = capture(
            f = { x: Tracer<Rank1<Sym>> -> x + x.constant(floatArrayOf(10f, 20f, 30f)) },
            input = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f)),
            name = "cap_rank1",
        )
        validate(DxirModule(listOf(rank1Fn)).toStablehlo(), "capture rank-1 + const")

        // Rank-2: `f(x) = x + [[10, 20], [30, 40]]`.
        val rank2Fn = capture(
            f = { x: Tracer<Rank2<Sym, Sym>> ->
                val m: Tracer<Rank2<Sym, Sym>> =
                    x.constant(floatArrayOf(10f, 20f, 30f, 40f), intArrayOf(2, 2))
                x + m
            },
            input = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f)),
            name = "cap_rank2",
        )
        validate(DxirModule(listOf(rank2Fn)).toStablehlo(), "capture rank-2 + const")
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
            // §0.4.395 — TAN round-trips as `stablehlo.tan`; ATAN as the
            // constant + `stablehlo.atan2` pair.
            OpKind.TAN, OpKind.ATAN,
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
    fun reverseRoundTrips() {
        // §0.4.396 — REVERSE (flip): single axis, leading axis, both axes.
        requireTranslateOrSkip()
        val cases = listOf(
            listOf(1),
            listOf(0),
            listOf(0, 1),
        )
        for (axes in cases) {
            val fn = DxirBuilder.function("rev") {
                val x = param("x", DxirType(F32, listOf(2, 3)))
                val y = op(
                    OpKind.REVERSE, listOf(x), DxirType(F32, listOf(2, 3)),
                    attrs = mapOf("dimensions" to axes),
                )
                listOf(y)
            }
            validate(DxirModule(listOf(fn)).toStablehlo(), "REVERSE axes=$axes")
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
    fun namedMatmulExplicitAttrsRoundTrips() {
        // Layer 1 §0.4.241+ — N.3 round-trip pin. The K2 plugin's `contract`
        // lowering emits MATMUL ops with `axisNames` populated on operand +
        // result types AND explicit `*_contracting_dims` attrs (so the
        // existing emitter explicit-mode path fires). Confirm the resulting
        // MLIR survives `stablehlo-translate --serialize` end-to-end.
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("contract_named") {
            val a = param("a", DxirType(F32, listOf(8, 16), listOf("Batch", "SeqLen")))
            val b = param("b", DxirType(F32, listOf(16, 32), listOf("SeqLen", "Hidden")))
            val c = op(
                OpKind.MATMUL,
                listOf(a, b),
                DxirType(F32, listOf(8, 32), listOf("Batch", "Hidden")),
                attrs = mapOf(
                    "lhs_contracting_dims" to listOf(1),
                    "rhs_contracting_dims" to listOf(0),
                    "lhs_batching_dims" to emptyList<Int>(),
                    "rhs_batching_dims" to emptyList<Int>(),
                    "contracted_names" to setOf("SeqLen"),
                    "preserved_names" to listOf("Batch", "Hidden"),
                ),
            )
            listOf(c)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "named MATMUL with explicit attrs")
    }

    @Test
    fun namedMatmulInferredAttrsRoundTrips() {
        // Layer 1 §0.4.241+ — N.3 defensive named-inference path. When a
        // hand-built DXIR carries `axisNames` on both operands but omits the
        // explicit `*_contracting_dims` attrs, the emitter's third path
        // derives the contracting dim positions from the shared axis name.
        // Round-trip pin: the inferred output is dialect-valid MLIR.
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("contract_named_inferred") {
            val a = param("a", DxirType(F32, listOf(8, 16), listOf("Batch", "SeqLen")))
            val b = param("b", DxirType(F32, listOf(16, 32), listOf("SeqLen", "Hidden")))
            val c = op(
                OpKind.MATMUL,
                listOf(a, b),
                DxirType(F32, listOf(8, 32), listOf("Batch", "Hidden")),
                // No explicit *_contracting_dims; the emitter must infer from
                // the shared "SeqLen" axis name.
            )
            listOf(c)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "named MATMUL with inferred attrs")
    }

    @Test
    fun namedAttentionRank4BatchedRoundTrips() {
        // Layer 1.5 §0.4.242+ — rank-4 attention QK^T pattern with two
        // batching dims (Batch, Heads) and one contracting dim (Dim).
        // This is the use case the user flagged as load-bearing for Layer 2:
        // a hole here would force the typed-step-boundary work to walk
        // around a missing rank-4 named-contract surface.
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("attention_qkT") {
            val q = param(
                "q",
                DxirType(F32, listOf(2, 8, 64, 64), listOf("Batch", "Heads", "SeqLen", "Dim")),
            )
            val kT = param(
                "kT",
                DxirType(F32, listOf(2, 8, 64, 64), listOf("Batch", "Heads", "Dim", "Vocab")),
            )
            val scores = op(
                OpKind.MATMUL,
                listOf(q, kT),
                DxirType(F32, listOf(2, 8, 64, 64), listOf("Batch", "Heads", "SeqLen", "Vocab")),
                attrs = mapOf(
                    "lhs_contracting_dims" to listOf(3),
                    "rhs_contracting_dims" to listOf(2),
                    "lhs_batching_dims" to listOf(0, 1),
                    "rhs_batching_dims" to listOf(0, 1),
                    "contracted_names" to setOf("Dim"),
                    "preserved_names" to listOf("SeqLen", "Vocab"),
                    "batching_names" to listOf("Batch", "Heads"),
                ),
            )
            listOf(scores)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "named attention QK^T (rank-4, 2 batching axes)")
    }

    @Test
    fun namedDotRank1RoundTrips() {
        // Layer 1 §0.4.241+ — rank-1 × rank-1 named contraction emits DOT.
        // Round-trip pin: the resulting `stablehlo.dot_general %a, %b,
        // contracting_dims = [0] x [0]` survives serialization with
        // axis-named operand types.
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("dot_named") {
            val a = param("a", DxirType(F32, listOf(64), listOf("SeqLen")))
            val b = param("b", DxirType(F32, listOf(64), listOf("SeqLen")))
            val c = op(
                OpKind.DOT,
                listOf(a, b),
                DxirType(F32, emptyList()),  // scalar result
                attrs = mapOf(
                    "contracted_names" to setOf("SeqLen"),
                    "preserved_names" to emptyList<String>(),
                ),
            )
            listOf(c)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "named DOT rank-1")
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

    // §0.4.114 — substrate-shape SCATTER must round-trip through stablehlo-translate.

    @Test
    fun substrateScatterRank1SubstrateRoundTrips() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("s") {
            val base = param("b", DxirType(F32, listOf(4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, emptyList()))
            val v = param("v", DxirType(F32, emptyList()))
            val y = op(OpKind.SCATTER, listOf(base, idx, v), DxirType(F32, listOf(4)))
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "SCATTER rank-1 substrate")
    }

    @Test
    fun substrateScatterRank2SubstrateRoundTrips() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("s") {
            val base = param("b", DxirType(F32, listOf(3, 4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, emptyList()))
            val v = param("v", DxirType(F32, listOf(4)))
            val y = op(OpKind.SCATTER, listOf(base, idx, v), DxirType(F32, listOf(3, 4)))
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "SCATTER rank-2 substrate")
    }

    // §0.4.113 — substrate-shape GATHER must round-trip through stablehlo-translate.
    // Both rank-1 (`arr[idx]` → scalar) and rank-2 (`arr[idx, :]` → row) shapes
    // are covered.

    @Test
    fun substrateGatherRank1SubstrateRoundTrips() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("g") {
            val arr = param("a", DxirType(F32, listOf(4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, emptyList()))
            val y = op(OpKind.GATHER, listOf(arr, idx), DxirType(F32, emptyList()))
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "GATHER rank-1 substrate")
    }

    @Test
    fun substrateGatherRank2SubstrateRoundTrips() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("g") {
            val arr = param("a", DxirType(F32, listOf(3, 4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, emptyList()))
            val y = op(OpKind.GATHER, listOf(arr, idx), DxirType(F32, listOf(4)))
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "GATHER rank-2 substrate")
    }

    // §0.4.112 — SCATTER_ADD substrate-shape lowering must round-trip through
    // stablehlo-translate. Both rank-1 (scalar value) and rank-2 (rank-1 row value)
    // shapes are exercised; pre-§0.4.112 the emitter would `error("not yet implemented")`
    // on either.

    @Test
    fun scatterAddRank1SubstrateRoundTrips() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("g") {
            val base = param("b", DxirType(F32, listOf(4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, emptyList()))
            val v = param("v", DxirType(F32, emptyList()))
            val y = op(OpKind.SCATTER_ADD, listOf(base, idx, v), DxirType(F32, listOf(4)))
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "SCATTER_ADD rank-1 substrate")
    }

    @Test
    fun scatterAddRank2SubstrateRoundTrips() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("g") {
            val base = param("b", DxirType(F32, listOf(3, 4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, emptyList()))
            val v = param("v", DxirType(F32, listOf(4)))
            val y = op(OpKind.SCATTER_ADD, listOf(base, idx, v), DxirType(F32, listOf(3, 4)))
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "SCATTER_ADD rank-2 substrate")
    }

    // §0.4.132 — substrate-shape GATHER / SCATTER / SCATTER_ADD must round-trip
    // for any rank ≥ 1. These pin the rank-3 case (the next-step generalisation
    // exercised by AD against rank-3 GATHER primals).

    @Test
    fun substrateGatherRank3SubstrateRoundTrips() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("g") {
            val arr = param("a", DxirType(F32, listOf(2, 3, 4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, emptyList()))
            val y = op(OpKind.GATHER, listOf(arr, idx), DxirType(F32, listOf(3, 4)))
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "GATHER rank-3 substrate")
    }

    @Test
    fun substrateScatterRank3SubstrateRoundTrips() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("g") {
            val base = param("b", DxirType(F32, listOf(2, 3, 4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, emptyList()))
            val v = param("v", DxirType(F32, listOf(3, 4)))
            val y = op(OpKind.SCATTER, listOf(base, idx, v), DxirType(F32, listOf(2, 3, 4)))
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "SCATTER rank-3 substrate")
    }

    @Test
    fun scatterAddRank3SubstrateRoundTrips() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("g") {
            val base = param("b", DxirType(F32, listOf(2, 3, 4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, emptyList()))
            val v = param("v", DxirType(F32, listOf(3, 4)))
            val y = op(OpKind.SCATTER_ADD, listOf(base, idx, v), DxirType(F32, listOf(2, 3, 4)))
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "SCATTER_ADD rank-3 substrate")
    }

    // §0.4.135 — no-attrs MATMUL of rank ≥ 3 emits canonical batched
    // dot_general; verify the resulting MLIR still parses through stablehlo-translate.

    @Test
    fun matmulRank3BatchedNoAttrsRoundTrips() {
        requireTranslateOrSkip()
        val fn = DxirBuilder.function("bmm") {
            val a = param("a", DxirType(F32, listOf(2, 2, 3)))
            val b = param("b", DxirType(F32, listOf(2, 3, 4)))
            val c = op(OpKind.MATMUL, listOf(a, b), DxirType(F32, listOf(2, 2, 4)))
            listOf(c)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "MATMUL rank-3 batched no-attrs")
    }

    // §0.4.133 — scatter-into-zeros peephole's emitted MLIR must still round-trip.

    @Test
    fun scatterAddIntoZeroBroadcastRoundTrips() {
        requireTranslateOrSkip()
        // The canonical [GatherRule] adjoint shape: SCATTER_ADD into BROADCAST(0).
        // After §0.4.133's peephole, the inner body becomes `return upd` (no add).
        // Confirm the optimised MLIR still parses + lowers through stablehlo-translate.
        val fn = DxirBuilder.function("g") {
            val idx = param("i", DxirType(io.tlaloc.core.I32, emptyList()))
            val v = param("v", DxirType(F32, listOf(4)))
            val zeroScalar = const(0f, DxirType(F32, emptyList()))
            val zeroBase = op(
                OpKind.BROADCAST,
                listOf(zeroScalar),
                DxirType(F32, listOf(3, 4)),
                attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
            )
            val y = op(OpKind.SCATTER_ADD, listOf(zeroBase, idx, v), DxirType(F32, listOf(3, 4)))
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "SCATTER_ADD into BROADCAST(0) peephole")
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
    fun embeddingGradRoundTripsScatterAddRegion() {
        requireTranslateOrSkip()
        // §0.4.400 — EmbeddingRule's fused adjoint: the scatter+add-region
        // emission over a splat-zero base, with an I32 index CONST so the
        // integer dense-literal spelling round-trips too.
        val fn = DxirBuilder.function("eg") {
            val up = param("u", DxirType(F32, listOf(4, 8)))
            val template = param("t", DxirType(F32, listOf(10, 8)))
            val idx = const(floatArrayOf(0f, 2f, 0f, 7f), DxirType(io.tlaloc.core.I32, listOf(4)))
            val y = op(
                OpKind.EMBEDDING_GRAD,
                listOf(idx, up, template),
                DxirType(F32, listOf(10, 8)),
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "EMBEDDING_GRAD scatter+add region")
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
