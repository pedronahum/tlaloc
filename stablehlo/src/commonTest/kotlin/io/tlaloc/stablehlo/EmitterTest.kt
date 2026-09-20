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
    fun binaryWithBroadcastableOperandInjectsBroadcastInDim() {
        // Result type [4] but operand a is [1] — broadcast injection picks
        // up `a` and re-emits it as broadcast_in_dim before the add. `b`
        // matches the result and is left alone. Surfaces in LlamaDecoder
        // when MEAN's keep-dims [tokens, 1] result combines with [tokens, d]
        // tensors via the elementwise tail of RmsNorm.
        val r = DxirType(F32, listOf(4))
        val s = DxirType(F32, listOf(1))
        val fn = DxirBuilder.function("f") {
            val a = param("a", s)
            val b = param("b", r)
            val c = op(OpKind.ADD, listOf(a, b), r)
            listOf(c)
        }
        val mlir = fn.toStablehlo()
        assertTrue(
            mlir.contains("stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1xf32>) -> tensor<4xf32>"),
            "expected broadcast_in_dim re-emitting %0 from [1] to [4]; got: $mlir",
        )
        // The add must consume the broadcast result, not the raw %0.
        assertTrue(
            mlir.contains("stablehlo.add %") && !mlir.contains("stablehlo.add %0,"),
            "add must consume the broadcasted operand, not the raw [1]-shaped %0; got: $mlir",
        )
        assertTrue(mlir.contains(", %1 : tensor<4xf32>"), mlir)
    }

    @Test
    fun binaryWithBothOperandsBroadcastableInjectsBothBroadcasts() {
        // Result type [3, 4]; both operands need a broadcast: a is [1, 4]
        // (row repeated across the 3 axis) and b is [3, 1] (col repeated
        // across the 4 axis). Two distinct broadcast_in_dim ops must
        // appear before the multiply.
        val r = DxirType(F32, listOf(3, 4))
        val sa = DxirType(F32, listOf(1, 4))
        val sb = DxirType(F32, listOf(3, 1))
        val fn = DxirBuilder.function("f") {
            val a = param("a", sa)
            val b = param("b", sb)
            val c = op(OpKind.MUL, listOf(a, b), r)
            listOf(c)
        }
        val mlir = fn.toStablehlo()
        assertTrue(
            mlir.contains("stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<3x4xf32>"),
            "expected broadcast_in_dim widening %0 from [1,4] to [3,4]; got: $mlir",
        )
        assertTrue(
            mlir.contains("stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<3x1xf32>) -> tensor<3x4xf32>"),
            "expected broadcast_in_dim widening %1 from [3,1] to [3,4]; got: $mlir",
        )
        assertTrue(mlir.contains("stablehlo.multiply %") && mlir.contains(": tensor<3x4xf32>"), mlir)
    }

    @Test
    fun binaryWithScalarOperandSplatsWithAnEmptyAxisMap() {
        // Phase A5c — the rank-0 end of implicit broadcasting: a scalar operand
        // against a [3,4] result splats through `broadcast_in_dim` with an EMPTY
        // axis map (every result axis is a new replicated one). The rank-1
        // right-alignment is pinned by
        // [binaryRankDeficientOperandRightAlignsInsteadOfRejecting].
        val r = DxirType(F32, listOf(3, 4))
        val scalarFn = DxirBuilder.function("g") {
            val s = param("s", DxirType(F32, emptyList()))
            val b = param("b", r)
            listOf(op(OpKind.MUL, listOf(s, b), r))
        }
        val scalarMlir = scalarFn.toStablehlo()
        assertTrue(
            scalarMlir.contains("stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<3x4xf32>"),
            "expected an empty-axis-map splat for the scalar operand; got: $scalarMlir",
        )
    }

    @Test
    fun binaryWithHigherRankOperandIsRefused() {
        // Broadcasting never DROPS axes: a [2,3,4] operand against a [3,4] result
        // must fail loudly rather than emit an invalid broadcast_in_dim.
        val r = DxirType(F32, listOf(3, 4))
        val fn = DxirBuilder.function("f") {
            val a = param("a", DxirType(F32, listOf(2, 3, 4)))
            val b = param("b", r)
            listOf(op(OpKind.ADD, listOf(a, b), r))
        }
        assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
    }

    @Test
    fun sliceLikeEmitsAStaticWindowSlice() {
        // Phase A2b — SLICE_LIKE's bounds come from its templates' RUNTIME extents,
        // but at emit time every dim is concrete, so they fold to literals and the
        // op lowers to the same static `stablehlo.slice` the SLICE arm emits:
        // a [2,5] value, a [2,3] window starting after a [2,2] prior on axis 1 →
        // columns 2..4. The template params go unreferenced in the MLIR, which is
        // legal (and DCE'd downstream).
        val fn = DxirBuilder.function("f") {
            val v = param("v", DxirType(F32, listOf(2, 5)))
            val t = param("t", DxirType(F32, listOf(2, 3)))
            val p = param("p", DxirType(F32, listOf(2, 2)))
            listOf(
                op(
                    OpKind.SLICE_LIKE, listOf(v, t, p), DxirType(F32, listOf(2, 3)),
                    attrs = mapOf("axis" to 1),
                ),
            )
        }
        val mlir = fn.toStablehlo()
        assertTrue(
            mlir.contains("stablehlo.slice %0 [0:2, 2:5] : (tensor<2x5xf32>) -> tensor<2x3xf32>"),
            mlir,
        )
    }

    @Test
    fun padLikeEmitsAStaticPad() {
        // §0.4.404 — PAD_LIKE is SLICE_LIKE's transpose: place the value into
        // the outTemplate's shape at the window after the prior templates. At
        // emit time every dim is concrete, so the offset (Σ priors' axis
        // extents) and the trailing pad (outTemplate − offset − value) fold to
        // literals and the op lowers to the same static `stablehlo.pad` the
        // PAD_TO arm emits: a [2,3] value into a [2,7] outTemplate after a
        // [2,2] prior on axis 1 → low = [0,2], high = [0,2]. The template
        // params go unreferenced in the MLIR (legal, DCE'd downstream).
        val fn = DxirBuilder.function("f") {
            val v = param("v", DxirType(F32, listOf(2, 3)))
            val t = param("t", DxirType(F32, listOf(2, 7)))
            val p = param("p", DxirType(F32, listOf(2, 2)))
            listOf(
                op(
                    OpKind.PAD_LIKE, listOf(v, t, p), DxirType(F32, listOf(2, 7)),
                    attrs = mapOf("axis" to 1),
                ),
            )
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("stablehlo.pad %0, %"), mlir)
        assertTrue(
            mlir.contains(
                "low = [0, 2], high = [0, 2], interior = [0, 0] : " +
                    "(tensor<2x3xf32>, tensor<f32>) -> tensor<2x7xf32>",
            ),
            mlir,
        )
    }

    @Test
    fun broadcastLikeEmitsAStaticBroadcastInDim() {
        // §0.4.399 — BROADCAST_LIKE's target extents come from its template's
        // RUNTIME shape, but at emit time every dim is concrete, so it folds to
        // a static `broadcast_in_dim` with the identity right-aligned axis map:
        // a [3] value against a [2,3] template maps its axis 0 to output axis 1.
        // The template param goes unreferenced in the MLIR (legal, DCE'd).
        val fn = DxirBuilder.function("f") {
            val v = param("v", DxirType(F32, listOf(3)))
            val t = param("t", DxirType(F32, listOf(2, 3)))
            listOf(op(OpKind.BROADCAST_LIKE, listOf(v, t), DxirType(F32, listOf(2, 3))))
        }
        val mlir = fn.toStablehlo()
        assertTrue(
            mlir.contains("stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>"),
            mlir,
        )
    }

    @Test
    fun sliceAtEmitsAStaticWindowSlice() {
        // §0.4.399 — SLICE_AT's window extents come from its template's RUNTIME
        // shape and its offset from the literal `low` attr; at emit time both
        // fold to the same static `stablehlo.slice` the SLICE arm emits: a [3,4]
        // value, a [2,2] template at low=[1,1] → rows 1..2, cols 1..2.
        val fn = DxirBuilder.function("f") {
            val v = param("v", DxirType(F32, listOf(3, 4)))
            val t = param("t", DxirType(F32, listOf(2, 2)))
            listOf(
                op(
                    OpKind.SLICE_AT, listOf(v, t), DxirType(F32, listOf(2, 2)),
                    attrs = mapOf("low" to listOf(1, 1)),
                ),
            )
        }
        val mlir = fn.toStablehlo()
        assertTrue(
            mlir.contains("stablehlo.slice %0 [1:3, 1:3] : (tensor<3x4xf32>) -> tensor<2x2xf32>"),
            mlir,
        )
    }

    @Test
    fun binaryWithMatchingOperandsEmitsNoBroadcastInjection() {
        // Same-shape operands hit the no-op path; emit must not mention
        // broadcast_in_dim around the add. Pins the regression that
        // §0.4.277 introduced the broadcast helper without breaking the
        // existing same-shape lowering.
        val mlir = singleOpFunction(OpKind.ADD, DxirType(F32, listOf(8, 16)))
        // Existing keep-dims reduce paths emit broadcast_in_dim too, but
        // singleOpFunction is just `add(param_a, param_b)`; no reduce.
        assertTrue(!mlir.contains("broadcast_in_dim"), "no broadcast injection expected for same-shape add; got: $mlir")
        assertTrue(mlir.contains("stablehlo.add %0, %1 : tensor<8x16xf32>"), mlir)
    }

    @Test
    fun binaryRankDeficientOperandRightAlignsInsteadOfRejecting() {
        // Was `binaryRejectsRankMismatchedOperand`: §0.4.277's v1 was same-rank
        // only and refused this case loudly "so a future generaliser can't
        // silently miscompile". Phase A5c IS that generaliser — implicit
        // broadcasting on the elementwise binaries — so the refusal is replaced by
        // an explicit pin of the emitted right-alignment: the rank-1 [4] operand
        // maps its axis 0 to result axis 1 (`dims = [1]`), gaining a replicated
        // leading axis. The surviving refusal is the OTHER direction (an operand
        // of higher rank than the result — broadcasting never drops axes), pinned
        // by [binaryWithHigherRankOperandIsRefused].
        val r = DxirType(F32, listOf(3, 4))
        val s = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("f") {
            val a = param("a", s)
            val b = param("b", r)
            val c = op(OpKind.ADD, listOf(a, b), r)
            listOf(c)
        }
        val mlir = fn.toStablehlo()
        assertTrue(
            mlir.contains("stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<4xf32>) -> tensor<3x4xf32>"),
            "expected the rank-1 operand right-aligned onto result axis 1; got: $mlir",
        )
        assertTrue(
            !mlir.contains("rank mismatch"),
            "the v1 rank-mismatch refusal must be gone; got: $mlir",
        )
    }

    @Test
    fun binaryRejectsIncompatibleDimOperand() {
        // Same-rank but operand dim is neither equal to result nor 1 →
        // not broadcast-compatible. Fails loudly with the offending dim
        // index in the message so callers can localise the bug.
        val r = DxirType(F32, listOf(4, 4))
        val s = DxirType(F32, listOf(2, 4))
        val fn = DxirBuilder.function("f") {
            val a = param("a", s)
            val b = param("b", r)
            val c = op(OpKind.ADD, listOf(a, b), r)
            listOf(c)
        }
        val ex = assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
        assertTrue(
            ex.message!!.contains("dim 0") && ex.message!!.contains("not broadcast-compatible"),
            "expected incompatible-dim require-message naming dim 0; got: ${ex.message}",
        )
    }

    @Test
    fun binaryRejectsDtypeMismatchedOperand() {
        // §0.4.279 — `stablehlo.broadcast_in_dim` is shape-only; emitting
        // `(tensor<NxF32>) -> tensor<NxF64>` is invalid MLIR. The
        // broadcastIfNeeded helper guards against this with a
        // load-bearing error rather than silently miscompiling. DXIR's
        // elementwise-op contract should already enforce same-dtype on
        // operand vs result, but a hand-built DxirOp via the public
        // builder API can still violate it — this guard catches that.
        val r = DxirType(io.tlaloc.core.F64, listOf(4))
        val s = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("f") {
            val a = param("a", s)
            val b = param("b", r)
            val c = op(OpKind.ADD, listOf(a, b), r)
            listOf(c)
        }
        val ex = assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
        assertTrue(
            ex.message!!.contains("dtype mismatch") && ex.message!!.contains("CAST"),
            "expected dtype-mismatch require-message naming CAST as the fix; got: ${ex.message}",
        )
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
    fun tanAndAtanEmitPinnedSpellings() {
        // §0.4.395 — TAN is a first-class `stablehlo.tan` (the trailing-space
        // match keeps a hypothetical tanh mis-emission from passing); ATAN has
        // no unary StableHLO op and pins to `atan2(x, splat 1.0)`.
        val t = DxirType(F32, listOf(4))
        assertTrue(singleUnary(OpKind.TAN, t).contains("stablehlo.tan %"))
        val atan = singleUnary(OpKind.ATAN, t)
        assertTrue(atan.contains("dense<1.0>"), "missing the atan2 splat-one operand: $atan")
        assertTrue(
            Regex("stablehlo\\.atan2 %0, %s\\d+").containsMatchIn(atan),
            "expected atan(x) to lower as atan2(x, 1.0): $atan",
        )
    }

    @Test
    fun lgammaDigammaTrigammaEmitPinnedChloSpellings() {
        // §0.4.402 — Phase C1 special functions lower through the CHLO dialect
        // (StableHLO has none of them; XLA's PJRT compile path parses +
        // legalizes CHLO — smoke-certified on the GB10). LGAMMA/DIGAMMA pin the
        // unary-elementwise pretty form `chlo.op %x : t -> t`; TRIGAMMA pins
        // `chlo.polygamma(splat 1.0, x)` in generic MLIR form, the spelling the
        // smoke test certified.
        val t = DxirType(F32, listOf(4))
        val lg = singleUnary(OpKind.LGAMMA, t)
        assertTrue(
            lg.contains("chlo.lgamma %0 : tensor<4xf32> -> tensor<4xf32>"),
            "expected the CHLO lgamma spelling: $lg",
        )
        val dg = singleUnary(OpKind.DIGAMMA, t)
        assertTrue(
            dg.contains("chlo.digamma %0 : tensor<4xf32> -> tensor<4xf32>"),
            "expected the CHLO digamma spelling: $dg",
        )
        val tg = singleUnary(OpKind.TRIGAMMA, t)
        assertTrue(tg.contains("dense<1.0>"), "missing the polygamma order-1 splat: $tg")
        assertTrue(
            Regex("\"chlo\\.polygamma\"\\(%s\\d+, %0\\) : \\(tensor<4xf32>, tensor<4xf32>\\) -> tensor<4xf32>")
                .containsMatchIn(tg),
            "expected trigamma to lower as chlo.polygamma(1, x): $tg",
        )
    }

    @Test
    fun polygammaEmitsOrderSplatChloSpelling() {
        // §0.4.405 — general POLYGAMMA carries its literal `order` attr as the
        // float splat order operand: `"chlo.polygamma"(splat n.0, x)` in the
        // generic MLIR form the §0.4.402 spike certified against the GB10's
        // XLA (identical to TRIGAMMA's spelling but for the splat value).
        val t = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("f") {
            val a = param("a", t)
            val b = op(OpKind.POLYGAMMA, listOf(a), t, attrs = mapOf("order" to 3))
            listOf(b)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("dense<3.0>"), "missing the order-3 splat: $mlir")
        assertTrue(
            Regex("\"chlo\\.polygamma\"\\(%s\\d+, %0\\) : \\(tensor<4xf32>, tensor<4xf32>\\) -> tensor<4xf32>")
                .containsMatchIn(mlir),
            "expected polygamma(3) to lower as chlo.polygamma(3, x): $mlir",
        )
    }

    @Test
    fun reverseEmitsPinnedSpelling() {
        // §0.4.396 — REVERSE (flip) pins to `stablehlo.reverse` with the
        // literal `dims` list; shape-preserving, so operand and result types
        // coincide. A missing/empty/duplicate/out-of-range axis list fails
        // loudly at emit rather than shipping a malformed op.
        val t = DxirType(F32, listOf(2, 3))
        fun rev(axes: List<Int>): String {
            val fn = DxirBuilder.function("f") {
                val a = param("a", t)
                listOf(op(OpKind.REVERSE, listOf(a), t, attrs = mapOf("dimensions" to axes)))
            }
            return fn.toStablehlo()
        }
        assertTrue(
            rev(listOf(0)).contains(
                "stablehlo.reverse %0, dims = [0] : (tensor<2x3xf32>) -> tensor<2x3xf32>",
            ),
            "single-axis reverse spelling drifted:\n${rev(listOf(0))}",
        )
        assertTrue(
            rev(listOf(0, 1)).contains("stablehlo.reverse %0, dims = [0, 1]"),
            "two-axis reverse spelling drifted:\n${rev(listOf(0, 1))}",
        )
        assertFailsWith<IllegalArgumentException> { rev(emptyList()) }
        assertFailsWith<IllegalArgumentException> { rev(listOf(0, 0)) }
        assertFailsWith<IllegalArgumentException> { rev(listOf(2)) }
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

    // §0.4.274 — keep-dims (size-1 reduced axes preserved in the output)
    // is detected by output-rank == input-rank. Real Llama / Mistral
    // RMS-norm chains emit MEAN with keepdims; without this, the
    // LlamaDecoder primal can't lower.

    @Test
    fun sumKeepDimsLowersToReduceThenBroadcast() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(8, 64)))
            val s = op(
                OpKind.SUM, listOf(x),
                DxirType(F32, listOf(8, 1)),  // keep-dims: rank preserved, reduced axis = 1
                attrs = mapOf("reduction_dims" to listOf(1)),
            )
            listOf(s)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("stablehlo.reduce"), "expected reduce; got: $mlir")
        assertTrue(
            mlir.contains("(tensor<8x64xf32>, tensor<f32>) -> tensor<8xf32>"),
            "expected reduce intermediate at rank-1 (drop-dims); got: $mlir",
        )
        assertTrue(
            mlir.contains("stablehlo.broadcast_in_dim") &&
                mlir.contains(": (tensor<8xf32>) -> tensor<8x1xf32>"),
            "expected broadcast_in_dim re-inflating to keep-dims [8,1]; got: $mlir",
        )
    }

    @Test
    fun meanKeepDimsLowersToReduceDivideThenBroadcast() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(8, 64)))
            val m = op(
                OpKind.MEAN, listOf(x),
                DxirType(F32, listOf(8, 1)),  // keep-dims
                attrs = mapOf("reduction_dims" to listOf(1)),
            )
            listOf(m)
        }
        val mlir = fn.toStablehlo()
        // reduce + divide both happen on the dropped-dims intermediate
        assertTrue(mlir.contains("stablehlo.reduce"), mlir)
        assertTrue(
            mlir.contains("stablehlo.constant dense<64.0> : tensor<8xf32>"),
            "divisor at intermediate type missing; got: $mlir",
        )
        assertTrue(
            mlir.contains("stablehlo.divide") && mlir.contains(": tensor<8xf32>"),
            "divide should run at intermediate type; got: $mlir",
        )
        // then broadcast_in_dim re-inflates to keep-dims output
        assertTrue(
            mlir.contains("stablehlo.broadcast_in_dim") &&
                mlir.contains(": (tensor<8xf32>) -> tensor<8x1xf32>"),
            "expected broadcast_in_dim re-inflating to keep-dims [8,1]; got: $mlir",
        )
    }

    @Test
    fun reduceRejectsAmbiguousOutputRank() {
        // Output rank that's neither drop-dims (input.rank - dims.size) nor
        // input.rank (keep-dims) is a malformed reduce — surface a clear error
        // rather than silent miscompile.
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(8, 64, 16)))
            val s = op(
                OpKind.SUM, listOf(x),
                DxirType(F32, listOf(8, 64)),  // rank 2: neither drop-dims (rank 1) nor keep-dims (rank 3)
                attrs = mapOf("reduction_dims" to listOf(0, 2)),  // would drop to rank 1
            )
            listOf(s)
        }
        val ex = assertFailsWith<IllegalStateException> { fn.toStablehlo() }
        assertTrue(
            "reduce output rank" in ex.message.orEmpty(),
            "expected rank-mismatch error; got: ${ex.message}",
        )
    }

    @Test
    fun rngOpsRefuseEmissionLoudlyByName() {
        // §0.4.408 — Phase D1: the RNG ops deliberately have no StableHLO
        // arm. `stablehlo.rng_bit_generator`'s counter layout would not
        // reproduce the host/interpreter threefry stream bit-for-bit, and a
        // silent stream fork between engines is exactly what the stateless
        // PRNG design exists to prevent. Explicit-threefry emission (JAX's
        // own approach) is the recorded Phase D tail.
        for (kind in listOf(OpKind.RNG_UNIFORM, OpKind.RNG_NORMAL)) {
            val fn = DxirBuilder.function("rng") {
                listOf(
                    op(
                        kind, emptyList(), DxirType(F32, listOf(2, 3)),
                        attrs = mapOf("key0" to 7, "key1" to 42, "dims" to listOf(2, 3)),
                    ),
                )
            }
            val ex = assertFailsWith<IllegalStateException> { fn.toStablehlo() }
            assertTrue(
                "$kind has no StableHLO emission" in ex.message.orEmpty(),
                "expected a named RNG refusal; got: ${ex.message}",
            )
        }
    }

    @Test
    fun checkShapeLikeRefusesEmissionLoudlyByName() {
        // §0.4.415 — Phase B5 (customVjp): CHECK_SHAPE_LIKE is the runtime
        // assert wrapped around a USER gradient_body's returns. StableHLO has
        // no assert primitive, and emitting the op as a silent value alias
        // would DROP the check on GPU while the host/interpreter enforce it —
        // the RNG-refusal reasoning verbatim. customVjp gradient bodies are
        // host/interpreter-certified in v1; the honest GPU story is a recorded
        // Phase B5 tail.
        val fn = DxirBuilder.function("chk") {
            val v = param("v", DxirType(F32, listOf(3)))
            val t = param("t", DxirType(F32, listOf(3)))
            listOf(op(OpKind.CHECK_SHAPE_LIKE, listOf(v, t), DxirType(F32, listOf(3))))
        }
        val ex = assertFailsWith<IllegalStateException> { fn.toStablehlo() }
        assertTrue(
            "CHECK_SHAPE_LIKE has no StableHLO emission" in ex.message.orEmpty(),
            "expected the named customVjp shape-assert refusal; got: ${ex.message}",
        )
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

    // §0.4.135 — no-attrs MATMUL with rank ≥ 3 follows the canonical batched
    // matmul convention: leading axes are batching dims, last two are M/K vs K/N.

    @Test
    fun matmulRank3WithoutAttrsLowersAsBatchedDotGeneral() {
        // (2, 2, 3) × (2, 3, 4) → (2, 2, 4): single batch axis [0], contracting [2] x [1].
        val fn = DxirBuilder.function("bmm") {
            val a = param("a", DxirType(F32, listOf(2, 2, 3)))
            val b = param("b", DxirType(F32, listOf(2, 3, 4)))
            val c = op(OpKind.MATMUL, listOf(a, b), DxirType(F32, listOf(2, 2, 4)))
            listOf(c)
        }
        val mlir = fn.toStablehlo()
        assertTrue(
            mlir.contains(
                "stablehlo.dot_general %0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1]",
            ),
            "expected canonical rank-3 batched dot_general; got: $mlir",
        )
        assertTrue(
            mlir.contains("(tensor<2x2x3xf32>, tensor<2x3x4xf32>) -> tensor<2x2x4xf32>"),
            mlir,
        )
    }

    @Test
    fun matmulRank4WithoutAttrsLowersAsTwoBatchAxesDotGeneral() {
        // (3, 2, 2, 5) × (3, 2, 5, 4) → (3, 2, 2, 4): two batch axes [0, 1],
        // contracting [3] x [2].
        val fn = DxirBuilder.function("bmm4") {
            val a = param("a", DxirType(F32, listOf(3, 2, 2, 5)))
            val b = param("b", DxirType(F32, listOf(3, 2, 5, 4)))
            val c = op(OpKind.MATMUL, listOf(a, b), DxirType(F32, listOf(3, 2, 2, 4)))
            listOf(c)
        }
        val mlir = fn.toStablehlo()
        assertTrue(
            mlir.contains(
                "stablehlo.dot_general %0, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2]",
            ),
            "expected canonical rank-4 batched dot_general; got: $mlir",
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
    fun embeddingGradEmitsScatterWithAddRegionOverZeros() {
        // §0.4.400 — EmbeddingRule's fused adjoint: indices (3,) + upstream (3, 4)
        // + shape template (10, 4) -> dTable (10, 4), as a scatter with a REAL add
        // region over a splat-zero base. Collisions must accumulate, so neither
        // `unique_indices` nor the return-upd peephole may appear.
        val fn = DxirBuilder.function("eg") {
            val idx = param("i", DxirType(io.tlaloc.core.I32, listOf(3)))
            val up = param("u", DxirType(F32, listOf(3, 4)))
            val template = param("t", DxirType(F32, listOf(10, 4)))
            val y = op(
                OpKind.EMBEDDING_GRAD,
                listOf(idx, up, template),
                DxirType(F32, listOf(10, 4)),
            )
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("\"stablehlo.scatter\""), mlir)
        assertTrue(mlir.contains("stablehlo.constant dense<0.0> : tensor<10x4xf32>"), mlir)
        assertTrue(mlir.contains("update_window_dims = [1]"), mlir)
        assertTrue(mlir.contains("inserted_window_dims = [0]"), mlir)
        assertTrue(mlir.contains("scatter_dims_to_operand_dims = [0]"), mlir)
        assertTrue(mlir.contains("index_vector_dim = 1"), mlir)
        assertTrue(mlir.contains("stablehlo.add"), mlir)
        assertTrue(!mlir.contains("unique_indices"), "collisions must accumulate: $mlir")
        assertTrue(mlir.contains("(tensor<10x4xf32>, tensor<3xi32>, tensor<3x4xf32>) -> tensor<10x4xf32>"), mlir)
    }

    @Test
    fun embeddingGradRankTwoIndicesEmitScatterWithBatchWindow() {
        // §0.4.409 — the v1 rank-1 restriction fell with the host surface's
        // [B, N] batch spelling: every index axis is a scatter dim, the
        // trailing upstream axis the window, so r = 2 shifts both dim numbers.
        val fn = DxirBuilder.function("eg2") {
            val idx = param("i", DxirType(io.tlaloc.core.I32, listOf(2, 3)))
            val up = param("u", DxirType(F32, listOf(2, 3, 4)))
            val template = param("t", DxirType(F32, listOf(10, 4)))
            val y = op(
                OpKind.EMBEDDING_GRAD,
                listOf(idx, up, template),
                DxirType(F32, listOf(10, 4)),
            )
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("\"stablehlo.scatter\""), mlir)
        assertTrue(mlir.contains("update_window_dims = [2]"), mlir)
        assertTrue(mlir.contains("inserted_window_dims = [0]"), mlir)
        assertTrue(mlir.contains("index_vector_dim = 2"), mlir)
        assertTrue(mlir.contains("stablehlo.add"), mlir)
        assertTrue(!mlir.contains("unique_indices"), "collisions must accumulate: $mlir")
        assertTrue(mlir.contains("(tensor<10x4xf32>, tensor<2x3xi32>, tensor<2x3x4xf32>) -> tensor<10x4xf32>"), mlir)
    }

    @Test
    fun embeddingWithPaddingIndexMasksGatheredRows() {
        // §0.4.409 — `padding_index` attr: gather, then compare + select zeroes
        // the padded positions' rows (XLA clamps out-of-bounds gathers, so the
        // select is what makes the row zero — the gather alone cannot).
        val fn = DxirBuilder.function("ep") {
            val table = param("t", DxirType(F32, listOf(10, 4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, listOf(3)))
            val y = op(
                OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(3, 4)),
                attrs = mapOf("padding_index" to 7),
            )
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("\"stablehlo.gather\""), mlir)
        assertTrue(mlir.contains("stablehlo.constant dense<7> : tensor<3xi32>"), mlir)
        assertTrue(
            mlir.contains("stablehlo.compare  EQ,") && mlir.contains("SIGNED : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>"),
            mlir,
        )
        assertTrue(mlir.contains("stablehlo.broadcast_in_dim") && mlir.contains("(tensor<3xi1>) -> tensor<3x4xi1>"), mlir)
        assertTrue(mlir.contains("stablehlo.select"), mlir)
        assertTrue(mlir.contains("stablehlo.constant dense<0.0> : tensor<3x4xf32>"), mlir)
    }

    @Test
    fun embeddingWithoutPaddingIndexEmitsNoMask() {
        // The unpadded spelling must stay the §0.4.400 bare gather — no compare,
        // no select.
        val fn = DxirBuilder.function("e") {
            val table = param("t", DxirType(F32, listOf(10, 4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, listOf(3)))
            val y = op(OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(3, 4)))
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(!mlir.contains("stablehlo.compare"), mlir)
        assertTrue(!mlir.contains("stablehlo.select"), mlir)
    }

    @Test
    fun embeddingGradWithPaddingIndexMasksUpstreamBeforeScatter() {
        // §0.4.409 — the padded adjoint zeroes the padded positions' upstream
        // rows BEFORE the scatter: adding a zero row is a numeric no-op, so the
        // padded vocab row stays exactly zero over the splat-zero base.
        val fn = DxirBuilder.function("egp") {
            val idx = param("i", DxirType(io.tlaloc.core.I32, listOf(3)))
            val up = param("u", DxirType(F32, listOf(3, 4)))
            val template = param("t", DxirType(F32, listOf(10, 4)))
            val y = op(
                OpKind.EMBEDDING_GRAD,
                listOf(idx, up, template),
                DxirType(F32, listOf(10, 4)),
                attrs = mapOf("padding_index" to 2),
            )
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("stablehlo.constant dense<2> : tensor<3xi32>"), mlir)
        assertTrue(mlir.contains("stablehlo.compare  EQ,"), mlir)
        assertTrue(mlir.contains("(tensor<3xi1>) -> tensor<3x4xi1>"), mlir)
        assertTrue(mlir.contains("stablehlo.select"), mlir)
        assertTrue(mlir.contains("stablehlo.constant dense<0.0> : tensor<3x4xf32>"), mlir)
        // The scatter itself is unchanged — and the select must land BEFORE it.
        assertTrue(mlir.contains("\"stablehlo.scatter\""), mlir)
        assertTrue(
            mlir.indexOf("stablehlo.select") < mlir.indexOf("\"stablehlo.scatter\""),
            "the padding mask must be applied to the updates, before the scatter: $mlir",
        )
        assertTrue(mlir.contains("update_window_dims = [1]"), mlir)
        assertTrue(!mlir.contains("unique_indices"), mlir)
    }

    @Test
    fun embeddingGradRejectsUpstreamShapeMismatch() {
        val fn = DxirBuilder.function("bad") {
            val idx = param("i", DxirType(io.tlaloc.core.I32, listOf(3)))
            val up = param("u", DxirType(F32, listOf(3, 5)))  // wrong: D is 4
            val template = param("t", DxirType(F32, listOf(10, 4)))
            val y = op(
                OpKind.EMBEDDING_GRAD,
                listOf(idx, up, template),
                DxirType(F32, listOf(10, 4)),
            )
            listOf(y)
        }
        assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
    }

    @Test
    fun integerConstsEmitIntegerDenseLiterals() {
        // §0.4.400 — an I32 index const (a gradient graph's cloned indices) must
        // print integer literals against its i32 tensor type, not the float
        // spelling of the dxir const carrier.
        val fn = DxirBuilder.function("ic") {
            val table = param("t", DxirType(F32, listOf(10, 4)))
            val idx = const(floatArrayOf(0f, 2f, 0f), DxirType(io.tlaloc.core.I32, listOf(3)))
            val y = op(OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(3, 4)))
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("stablehlo.constant dense<[0, 2, 0]> : tensor<3xi32>"), mlir)
        assertTrue(!mlir.contains("0.0, 2.0"), mlir)
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

    /**
     * §0.4.385 — the fused conv adjoints solve their padding at EMIT time, so the
     * MLIR must carry the same numbers Conv2dRule used to bake (and that
     * `PjrtConvSmokeTest` certifies against real XLA). Primal here is the
     * general-attr conv: strides [2,1], padding [[1,0],[1,1]], rhs_dilation [1,2],
     * x [2,2,5,4] ⋆ w [3,2,3,2] → y [2,3,2,4].
     *
     * dX: kEff = [3,3], dilSize = [(2−1)·2+1, (4−1)·1+1] = [3,4], so
     *     low = kEff−1−p_low = [1,1] and high = p_low + H − dilSize = [1+5−3, 1+4−4] = [3,1].
     */
    @Test
    fun convDataAdjointSolvesPaddingAtEmitTime() {
        val fn = DxirBuilder.function("dx") {
            val x = param("x", DxirType(F32, listOf(2, 2, 5, 4)))
            val w = param("w", DxirType(F32, listOf(3, 2, 3, 2)))
            val up = param("up", DxirType(F32, listOf(2, 3, 2, 4)))
            val dx = op(
                OpKind.CONV2D_DATA_ADJOINT, listOf(up, w, x), DxirType(F32, listOf(2, 2, 5, 4)),
                attrs = mapOf(
                    "window_strides" to listOf(2, 1),
                    "padding" to listOf(listOf(1, 0), listOf(1, 1)),
                    "rhs_dilation" to listOf(1, 2),
                ),
            )
            listOf(dx)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("pad = [[1, 3], [1, 1]]"), "solved dX padding wrong: $mlir")
        assertTrue(mlir.contains("lhs_dilate = [2, 1]"), "stride must land on lhs_dilate: $mlir")
        assertTrue(mlir.contains("rhs_dilate = [1, 2]"), "primal rhs_dilation must carry over: $mlir")
        assertTrue(mlir.contains("reverse = [true, true]"), "dX needs the tap flip: $mlir")
        assertTrue(
            mlir.contains("dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1]"),
            "dX is a transposed conv (IOHW kernel): $mlir",
        )
        // One convolution, no transposes: the data adjoint is a single op.
        assertEquals(1, mlir.split("stablehlo.convolution").size - 1, mlir)
    }

    /**
     * §0.4.385 — the kernel adjoint: `low = p_low`, `high = (k−1)·d + dilSize − H − p_low`,
     * which for this primal is [[1, (3−1)·1+3−5−1], [1, (2−1)·2+4−4−1]] = [[1,−1],[1,1]] —
     * the negative high side being the crop that absorbs the stride-2 floor-division
     * remainder over height 5. Stride and rhs_dilation SWAP roles, and the batch↔feature
     * transposes are emitted explicitly around the convolution.
     */
    @Test
    fun convKernelAdjointEmitsSwapConvSwapWithSolvedCrop() {
        val fn = DxirBuilder.function("dw") {
            val x = param("x", DxirType(F32, listOf(2, 2, 5, 4)))
            val w = param("w", DxirType(F32, listOf(3, 2, 3, 2)))
            val up = param("up", DxirType(F32, listOf(2, 3, 2, 4)))
            val dw = op(
                OpKind.CONV2D_KERNEL_ADJOINT, listOf(x, up, w), DxirType(F32, listOf(3, 2, 3, 2)),
                attrs = mapOf(
                    "window_strides" to listOf(2, 1),
                    "padding" to listOf(listOf(1, 0), listOf(1, 1)),
                    "rhs_dilation" to listOf(1, 2),
                ),
            )
            listOf(dw)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("pad = [[1, -1], [1, 1]]"), "solved dW padding wrong: $mlir")
        assertTrue(mlir.contains("stride = [1, 2]"), "primal rhs_dilation becomes the stride: $mlir")
        assertTrue(mlir.contains("rhs_dilate = [2, 1]"), "primal stride becomes rhs_dilate: $mlir")
        assertTrue(
            mlir.contains("dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]"),
            "dW's inner conv reads an OIHW kernel: $mlir",
        )
        assertEquals(3, mlir.split("stablehlo.transpose").size - 1, "expected Xᵀ, dYᵀ and the swap back: $mlir")
        assertEquals(1, mlir.split("stablehlo.convolution").size - 1, mlir)
    }

    /**
     * §0.4.386 — the fused avgpool adjoint expands into the MLIR the pre-fusion
     * rule produced: fold channels into the batch dim, one lhs-dilated convolution
     * against a `1/(kh·kw)` splat kernel (IOHW `[1,1,kh,kw]`, so a single-channel
     * kernel applies depthwise without grouped-conv support), fold back. Primal:
     * x [2,3,6,5], window [2,2], strides [2,2] → y [2,3,3,2] — a NON-divisible
     * width, so the solved padding is asymmetric.
     *
     * dilSize = [(3−1)·2+1, (2−1)·2+1] = [5,3]; low = kh−1−p_low = [1,1];
     * high = p_low + H − dilSize = [0+6−5, 0+5−3] = [1,2].
     */
    @Test
    fun avgPoolGradExpandsToFoldedTransposedConv() {
        val fn = DxirBuilder.function("ap") {
            val x = param("x", DxirType(F32, listOf(2, 3, 6, 5)))
            val up = param("up", DxirType(F32, listOf(2, 3, 3, 2)))
            val dx = op(
                OpKind.AVGPOOL2D_GRAD, listOf(up, x), DxirType(F32, listOf(2, 3, 6, 5)),
                attrs = mapOf(
                    "window" to listOf(2, 2),
                    "window_strides" to listOf(2, 2),
                    "padding" to listOf(listOf(0, 0), listOf(0, 0)),
                ),
            )
            listOf(dx)
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("pad = [[1, 1], [1, 2]]"), "solved avgpool padding wrong: $mlir")
        assertTrue(mlir.contains("lhs_dilate = [2, 2]"), "the stride must land on lhs_dilate: $mlir")
        assertTrue(mlir.contains("dense<0.25>"), "splat kernel must carry 1/(kh·kw): $mlir")
        assertTrue(
            mlir.contains("dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1]"),
            "the folded conv is a transposed conv against an IOHW splat: $mlir",
        )
        assertEquals(2, mlir.split("stablehlo.reshape").size - 1, "fold and unfold: $mlir")
        assertEquals(1, mlir.split("stablehlo.convolution").size - 1, mlir)
    }

    /**
     * §0.4.389 — the fused maxpool adjoint expands into the upsample-and-mask MLIR
     * the pre-fusion rule produced (and §0.4.363 certified on the GB10): both the
     * pooled value and the upstream go `[N,C,Ho,Wo] → [N,C,Ho,1,Wo,1] →
     * broadcast → [N,C,Ho,kh,Wo,kw] → [N,C,H,W]`, then `select(x == U(y), U(dY), 0)`.
     *
     * The rank-6 intermediates are fine in MLIR — it was the *dxir types* baking
     * `n`/`c`/`Ho`/`Wo` under `grad {}`'s sentinels that made the old spelling
     * unusable there, and emit time has concrete dims. Note the compare runs at
     * x's shape, not the pooled one: `U(y)` is the upsampled value.
     */
    @Test
    fun maxPoolGradExpandsToUpsampleAndMask() {
        val fn = DxirBuilder.function("mp") {
            val x = param("x", DxirType(F32, listOf(1, 2, 4, 4)))
            val up = param("up", DxirType(F32, listOf(1, 2, 2, 2)))
            val y = param("y", DxirType(F32, listOf(1, 2, 2, 2)))
            val dx = op(
                OpKind.MAXPOOL2D_GRAD, listOf(up, x, y), DxirType(F32, listOf(1, 2, 4, 4)),
                attrs = mapOf(
                    "window" to listOf(2, 2),
                    "window_strides" to listOf(2, 2),
                    "padding" to listOf(listOf(0, 0), listOf(0, 0)),
                ),
            )
            listOf(dx)
        }
        val mlir = fn.toStablehlo()
        assertEquals(4, mlir.split("stablehlo.reshape").size - 1, "two upsamples × (fold, unfold): $mlir")
        assertEquals(2, mlir.split("stablehlo.broadcast_in_dim").size - 1, mlir)
        assertTrue(mlir.contains("tensor<1x2x2x1x2x1xf32>"), "narrow rank-6 type missing: $mlir")
        assertTrue(mlir.contains("tensor<1x2x2x2x2x2xf32>"), "wide rank-6 type missing: $mlir")
        assertTrue(mlir.contains("stablehlo.compare  EQ"), mlir)
        assertTrue(mlir.contains("stablehlo.select"), mlir)
        assertTrue(
            mlir.contains("(tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xi1>"),
            "the mask must compare at x's shape and produce i1: $mlir",
        )
    }

    /**
     * §0.4.392 — the maxpool adjoint's expansion needs EXACT tiling, so an
     * overlapping pool must fail loudly at emit with a message naming the
     * restriction. The host and interpreter paths handle overlapping windows
     * correctly (they invert the window per input element); this is an EMISSION
     * limitation, not a semantic one, and the difference has to be obvious to
     * whoever hits it rather than surfacing as a mask that silently routes gradient
     * to the wrong input positions.
     *
     * The restriction is inherent, not an oversight: the all-ties convention (every
     * within-window winner receives the full upstream, matching MaxRule and the
     * interpreter) is not expressible over overlapping windows in StableHLO.
     * `select_and_scatter` is the primitive built for this and it picks ONE winner
     * per window; a transposed-conv spread of `dy` sums over covering windows but
     * cannot apply the per-window `x == max(w)` mask, because with overlapping
     * windows an input position belongs to several windows with different maxima.
     * So the choice is "general strides on the GPU with backend-dependent ties" or
     * "non-overlapping on the GPU with identical semantics everywhere" — and a
     * gradient that differs between host and device is worse than a loud refusal,
     * especially since exact ties are common after a relu.
     */
    @Test
    fun maxPoolGradRefusesOverlappingWindowsAtEmitTime() {
        val fn = DxirBuilder.function("mp_overlap") {
            val x = param("x", DxirType(F32, listOf(1, 1, 4, 4)))
            val up = param("up", DxirType(F32, listOf(1, 1, 3, 3)))
            val y = param("y", DxirType(F32, listOf(1, 1, 3, 3)))
            listOf(
                op(
                    OpKind.MAXPOOL2D_GRAD, listOf(up, x, y), DxirType(F32, listOf(1, 1, 4, 4)),
                    attrs = mapOf(
                        "window" to listOf(2, 2),
                        "window_strides" to listOf(1, 1),
                        "padding" to listOf(listOf(0, 0), listOf(0, 0)),
                    ),
                ),
            )
        }
        val e = assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
        val msg = e.message ?: ""
        assertTrue("non-overlapping" in msg, "error must name the restriction: $msg")
        assertTrue("MAXPOOL2D_GRAD" in msg, "error must name the op: $msg")
    }

    /** §0.4.392 — same for a window that does not tile the input exactly. */
    @Test
    fun maxPoolGradRefusesNonDivisibleDimsAtEmitTime() {
        val fn = DxirBuilder.function("mp_remainder") {
            val x = param("x", DxirType(F32, listOf(1, 1, 5, 5)))
            val up = param("up", DxirType(F32, listOf(1, 1, 2, 2)))
            val y = param("y", DxirType(F32, listOf(1, 1, 2, 2)))
            listOf(
                op(
                    OpKind.MAXPOOL2D_GRAD, listOf(up, x, y), DxirType(F32, listOf(1, 1, 5, 5)),
                    attrs = mapOf(
                        "window" to listOf(2, 2),
                        "window_strides" to listOf(2, 2),
                        "padding" to listOf(listOf(0, 0), listOf(0, 0)),
                    ),
                ),
            )
        }
        val e = assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
        val msg = e.message ?: ""
        assertTrue("window-divisible" in msg, "error must name the divisibility need: $msg")
        assertTrue("remainder" in msg, "error should say the other engines cope: $msg")
    }

    /**
     * §0.4.393 — the transposed-conv DATA adjoint emits through the
     * `convT(x,w) ≡ conv(dilate(x,L), swap01(w))` identity: swap the kernel's channel
     * axes, run the §0.4.385 data-adjoint convolution against the DILATED shape, then
     * take every L-th element to undo the dilation.
     *
     * Primal: x [1,2,3,3] ⋆ IOHW w [2,3,2,2], lhs_dilation [2,2], padding 1 → y
     * [1,3,6,6]. So hDil = (3−1)·2+1 = 5, and the convolution must land on 5×5
     * before the stride-2 slice brings it back to 3×3. Its solved padding is
     * `dilSize = (6−1)·1+1 = 6`, `kEff = 2` → `low = 2−1−1 = 0`,
     * `high = 1+5−6 = 0`.
     */
    @Test
    fun convTransposeDataAdjointEmitsSwapConvStridedSlice() {
        val fn = DxirBuilder.function("ct_dx") {
            val x = param("x", DxirType(F32, listOf(1, 2, 3, 3)))
            val w = param("w", DxirType(F32, listOf(2, 3, 2, 2)))
            val up = param("up", DxirType(F32, listOf(1, 3, 6, 6)))
            val dx = op(
                OpKind.CONV_TRANSPOSE2D_DATA_ADJOINT, listOf(up, w, x),
                DxirType(F32, listOf(1, 2, 3, 3)),
                attrs = mapOf(
                    "window_strides" to listOf(1, 1),
                    "lhs_dilation" to listOf(2, 2),
                    "padding" to listOf(listOf(1, 1), listOf(1, 1)),
                ),
            )
            listOf(dx)
        }
        val mlir = fn.toStablehlo()
        // Kernel swap IOHW [2,3,2,2] → OIHW [3,2,2,2].
        assertTrue(
            mlir.contains("tensor<2x3x2x2xf32>) -> tensor<3x2x2x2xf32>"),
            "kernel channel swap missing: $mlir",
        )
        // `lhs_dilate` is [1,1] here, NOT the primal's [2,2]: the identity dilates x
        // explicitly, so the inner convolution is an ordinary one over the dilated
        // tensor and the dilation is undone by the strided slice below. (Worth
        // pinning, because "the primal's lhs_dilation should ride along" is the
        // intuitive and wrong expectation.)
        assertTrue(mlir.contains("lhs_dilate = [1, 1]"), "inner conv must not re-dilate: $mlir")
        assertTrue(mlir.contains("pad = [[0, 0], [0, 0]]"), "solved padding wrong: $mlir")
        assertTrue(mlir.contains("reverse = [true, true]"), "a data adjoint flips the taps: $mlir")
        // The undilating strided slice: 5 → 3 at stride 2 on both spatial axes.
        assertTrue(mlir.contains("0:5:2, 0:5:2"), "strided undilate missing: $mlir")
        assertTrue(
            mlir.contains("(tensor<1x2x5x5xf32>) -> tensor<1x2x3x3xf32>"),
            "slice must go dilated 5×5 → 3×3: $mlir",
        )
    }

    /**
     * §0.4.393 — the transposed-conv KERNEL adjoint: interior-pad `x` by
     * `lhs_dilation` (this is the first use of a non-zero `interior` in the emitter —
     * `emitPad` hardcoded zeros), run the §0.4.385 kernel-adjoint expansion over it,
     * then swap the channel axes back to IOHW.
     *
     * Same primal as above: x dilates 3×3 → 5×5, and the kernel adjoint's solve is
     * `dilSize = 6`, `low = p_low = 1`, `high = (2−1)·1 + 6 − 5 − 1 = 1`, which lands
     * the inner conv on exactly kh=kw=2.
     */
    @Test
    fun convTransposeKernelAdjointEmitsInteriorPadConvSwap() {
        val fn = DxirBuilder.function("ct_dw") {
            val x = param("x", DxirType(F32, listOf(1, 2, 3, 3)))
            val w = param("w", DxirType(F32, listOf(2, 3, 2, 2)))
            val up = param("up", DxirType(F32, listOf(1, 3, 6, 6)))
            val dw = op(
                OpKind.CONV_TRANSPOSE2D_KERNEL_ADJOINT, listOf(x, up, w),
                DxirType(F32, listOf(2, 3, 2, 2)),
                attrs = mapOf(
                    "window_strides" to listOf(1, 1),
                    "lhs_dilation" to listOf(2, 2),
                    "padding" to listOf(listOf(1, 1), listOf(1, 1)),
                ),
            )
            listOf(dw)
        }
        val mlir = fn.toStablehlo()
        assertTrue(
            mlir.contains("interior = [0, 0, 1, 1]"),
            "x must be interior-dilated by lhs_dilation−1: $mlir",
        )
        assertTrue(
            mlir.contains("(tensor<1x2x3x3xf32>, tensor<f32>) -> tensor<1x2x5x5xf32>"),
            "the interior pad must go 3×3 → 5×5: $mlir",
        )
        assertTrue(mlir.contains("pad = [[1, 1], [1, 1]]"), "solved dW padding wrong: $mlir")
        // Four transposes: the kernel-adjoint expansion's two operand swaps and its
        // result swap, plus this arm's final OIHW → IOHW swap back.
        assertEquals(4, mlir.split("stablehlo.transpose").size - 1, mlir)
        assertEquals(1, mlir.split("stablehlo.convolution").size - 1, mlir)
    }

    /**
     * §0.4.393 — `stablehlo.sign`. SIGN was the one user-reachable kind with no
     * emitter arm: the MAX/MIN reduction rule builds its extremum indicator as
     * `1 − sign(y − x)`, so `grad { x.max(1).sum() }` could be lowered and
     * synthesised but never emitted.
     */
    @Test
    fun signEmitsStablehloSign() {
        val fn = DxirBuilder.function("sgn") {
            val x = param("x", DxirType(F32, listOf(2, 3)))
            listOf(op(OpKind.SIGN, listOf(x), DxirType(F32, listOf(2, 3))))
        }
        val mlir = fn.toStablehlo()
        assertTrue(mlir.contains("stablehlo.sign"), mlir)
    }

    /**
     * §0.4.393 — the EMPTY `broadcast_dimensions` form with an equal-rank,
     * non-scalar input is the reduction adjoints' "un-reduce stretch"
     * (`[N,1] → [N,K]`, emitted by the Max/Min/Tanh/Softmax rules). §0.4.359
     * documented the interpreter's polymorphic handling and claimed the emitter
     * matched; it did not, and rejected the form outright, so none of those
     * gradients were emittable. Identity dims is the correct spelling.
     */
    @Test
    fun broadcastWithEmptyDimsAndEqualRankStretchesByShape() {
        val fn = DxirBuilder.function("stretch") {
            val v = param("v", DxirType(F32, listOf(2, 1)))
            val t = param("t", DxirType(F32, listOf(2, 3)))
            listOf(
                op(
                    OpKind.BROADCAST, listOf(v, t), DxirType(F32, listOf(2, 3)),
                    attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
                ),
            )
        }
        val mlir = fn.toStablehlo()
        assertTrue(
            mlir.contains("dims = [0, 1]") && mlir.contains("(tensor<2x1xf32>) -> tensor<2x3xf32>"),
            "equal-rank empty-dims broadcast must stretch by identity: $mlir",
        )
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

    // §0.4.133 — scatter-into-zeros peephole tests.

    @Test
    fun scatterAddIntoZeroBroadcastEmitsReplaceBody() {
        // The canonical [GatherRule] adjoint shape: SCATTER_ADD whose base is
        // BROADCAST(const(0), arr.type). 0 + x = x, so the emitter elides the
        // add op and emits a `return upd` body — same shape as plain SCATTER.
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
        val mlir = fn.toStablehlo()
        // Body: return upd directly, no `stablehlo.add` inside the scatter block.
        // Find the scatter block and check its body lines.
        val scatterStart = mlir.indexOf("\"stablehlo.scatter\"")
        assertTrue(scatterStart >= 0, "expected stablehlo.scatter in: $mlir")
        val scatterEnd = mlir.indexOf("})", scatterStart)
        val scatterBlock = mlir.substring(scatterStart, scatterEnd)
        assertTrue(
            !scatterBlock.contains("stablehlo.add"),
            "peephole should elide add inside the scatter body: $scatterBlock",
        )
        assertTrue(
            scatterBlock.contains("stablehlo.return"),
            "scatter body must still terminate with return: $scatterBlock",
        )
    }

    @Test
    fun scatterAddWithNonZeroBaseStillEmitsAddBody() {
        // Counter-test: when the base is NOT a zero broadcast (e.g., a function
        // param), the peephole must NOT fire — the add body is required for
        // correctness.
        val fn = DxirBuilder.function("g") {
            val base = param("b", DxirType(F32, listOf(3, 4)))
            val idx = param("i", DxirType(io.tlaloc.core.I32, emptyList()))
            val v = param("v", DxirType(F32, listOf(4)))
            val y = op(OpKind.SCATTER_ADD, listOf(base, idx, v), DxirType(F32, listOf(3, 4)))
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        val scatterStart = mlir.indexOf("\"stablehlo.scatter\"")
        val scatterEnd = mlir.indexOf("})", scatterStart)
        val scatterBlock = mlir.substring(scatterStart, scatterEnd)
        assertTrue(
            scatterBlock.contains("stablehlo.add"),
            "non-zero base must keep the add body: $scatterBlock",
        )
    }

    @Test
    fun scatterAddIntoNonZeroBroadcastDoesNotTriggerPeephole() {
        // BROADCAST of a non-zero const (e.g., 1.0) must NOT trigger the peephole —
        // 1 + x ≠ x, so the add body is semantically required.
        val fn = DxirBuilder.function("g") {
            val idx = param("i", DxirType(io.tlaloc.core.I32, emptyList()))
            val v = param("v", DxirType(F32, listOf(4)))
            val oneScalar = const(1f, DxirType(F32, emptyList()))
            val oneBase = op(
                OpKind.BROADCAST,
                listOf(oneScalar),
                DxirType(F32, listOf(3, 4)),
                attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
            )
            val y = op(OpKind.SCATTER_ADD, listOf(oneBase, idx, v), DxirType(F32, listOf(3, 4)))
            listOf(y)
        }
        val mlir = fn.toStablehlo()
        val scatterStart = mlir.indexOf("\"stablehlo.scatter\"")
        val scatterEnd = mlir.indexOf("})", scatterStart)
        val scatterBlock = mlir.substring(scatterStart, scatterEnd)
        assertTrue(
            scatterBlock.contains("stablehlo.add"),
            "non-zero broadcast must keep the add body: $scatterBlock",
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
    fun broadcastRejectsLengthMismatchedDims() {
        // The existing length-vs-input-rank require: input is rank 2 but
        // broadcast_dimensions has length 1, so the attr can't possibly
        // map every input axis to an output axis.
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(2, 3)))
            val y = op(
                OpKind.BROADCAST, listOf(x), DxirType(F32, listOf(4, 2, 3)),
                attrs = mapOf("broadcast_dimensions" to listOf(1)),
            )
            listOf(y)
        }
        val ex = assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
        assertTrue(
            ex.message!!.contains("must equal input rank"),
            "expected length-mismatch require-message; got: ${ex.message}",
        )
    }

    @Test
    fun broadcastRejectsOutOfRangeDim() {
        // §0.4.283 — output is rank 2 (valid dims 0, 1) but
        // broadcast_dimensions points to dim 3. Must fail loudly rather
        // than emit invalid MLIR.
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(3)))
            val y = op(
                OpKind.BROADCAST, listOf(x), DxirType(F32, listOf(2, 3)),
                attrs = mapOf("broadcast_dimensions" to listOf(3)),
            )
            listOf(y)
        }
        val ex = assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
        assertTrue(
            ex.message!!.contains("out of range"),
            "expected out-of-range require-message; got: ${ex.message}",
        )
    }

    @Test
    fun broadcastRejectsDuplicateDims() {
        // §0.4.283 — input is rank 2; broadcast_dimensions has length 2
        // (length OK) but both entries point to output dim 0. StableHLO
        // requires unique target dims; emitter must fail loudly.
        val fn = DxirBuilder.function("f") {
            val x = param("x", DxirType(F32, listOf(2, 3)))
            val y = op(
                OpKind.BROADCAST, listOf(x), DxirType(F32, listOf(4, 5)),
                attrs = mapOf("broadcast_dimensions" to listOf(0, 0)),
            )
            listOf(y)
        }
        val ex = assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
        assertTrue(
            ex.message!!.contains("must be unique"),
            "expected duplicate-dims require-message; got: ${ex.message}",
        )
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
