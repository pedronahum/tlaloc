package io.tlaloc.testing

import io.tlaloc.core.BF16
import io.tlaloc.core.Bool
import io.tlaloc.core.DType
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.OpKind.*

/**
 * One entry per [OpKind]: samples and the surfaces each op is not expected to work on,
 * with reasons. [OpCatalogueTest] fails when an OpKind has no entry, so a new op is
 * swept from its first commit. The sweep runs excluded surfaces too and reports an
 * exclusion that no longer holds.
 */
object OpCatalogue {

    private fun t(dt: DType, vararg dims: Int) = DxirType(dt, dims.toList())

    private fun arg(name: String, vararg dims: Int, gen: Gen = Gen.Uniform()) = Arg(name, dims.toList(), gen)

    private fun ints(name: String, vararg dims: Int, lo: Int, until: Int) =
        Arg(name, dims.toList(), Gen.Ints(lo, until), integer = true)

    private val unitInterval = Gen.Uniform(-1.0, 1.0)
    private val positive = Gen.Uniform(0.5, 2.0)

    /** Values in `±[0.2, 1]`: away from the kink of ABS, RELU, SIGN and STEP. */
    private val signed = Gen.Fixed(doubleArrayOf(0.7, -0.3, 0.9, -0.8, 0.25, -0.55))

    private fun unary(kind: OpKind, gen: Gen = unitInterval, attrs: Map<String, Any> = emptyMap(), tol: Double = 1e-6) =
        OpSample(kind.name.lowercase(), listOf(arg("x", 2, 3, gen = gen)), gradTolerance = tol) { dt, ps ->
            op(kind, ps, t(dt, 2, 3), attrs = attrs)
        }

    private fun binary(kind: OpKind, a: Gen = unitInterval, b: Gen = unitInterval) =
        OpSample(kind.name.lowercase(), listOf(arg("a", 2, 3, gen = a), arg("b", 2, 3, gen = b))) { dt, ps ->
            op(kind, ps, t(dt, 2, 3))
        }

    /** `WHERE(mask(a, b), a, b)`: a Bool-valued op seen through a select. */
    private fun selectBy(name: String, mask: DxirBuilder.(List<DxirNode>) -> DxirNode) =
        OpSample(name, listOf(arg("a", 2, 3), arg("b", 2, 3))) { dt, ps ->
            op(WHERE, listOf(mask(ps), ps[0], ps[1]), t(dt, 2, 3))
        }

    private fun DxirBuilder.compare(a: DxirNode, b: DxirNode, direction: String) =
        op(COMPARE, listOf(a, b), DxirType(Bool, a.type.dims), attrs = mapOf("direction" to direction))

    private fun reduce(kind: OpKind) =
        OpSample(kind.name.lowercase(), listOf(arg("x", 2, 3))) { dt, ps ->
            op(kind, ps, t(dt, 2, 1), attrs = mapOf("reduction_dims" to listOf(1)))
        }

    private fun entry(kind: OpKind, vararg samples: OpSample, excluded: Map<Surface, String> = emptyMap()) =
        OpEntry(kind, samples.toList(), exclusions[kind].orEmpty() + excluded)

    private fun none(kind: OpKind, where: String) = OpEntry(kind, emptyList(), noSample = where)

    // Reasons shared by several entries.
    private const val ADJOINT_OP = "an adjoint op, emitted by a VJP rule; it is differentiated only through its own primal"
    private const val DEMOTED = "demoted to coarsener and emitter only (passes/DemotedOpKinds.kt)"

    private val demoted = mapOf(
        Surface.INTERPRETER to DEMOTED, Surface.INTERPRETER_F64 to DEMOTED, Surface.VJP to DEMOTED,
        Surface.JVP to DEMOTED, Surface.VMAP to DEMOTED, Surface.RENDER to DEMOTED, Surface.DEVICE to "$DEMOTED; the device check compares with the interpreter",
    )


    private fun on(reason: String, vararg surfaces: Surface) = surfaces.associateWith { reason }

    /** The surfaces that need the interpreter (every comparison does). */
    private fun noInterpreter(reason: String) = on(
        reason, Surface.INTERPRETER, Surface.INTERPRETER_F64, Surface.VJP, Surface.JVP, Surface.VMAP, Surface.DEVICE,
    )

    /** No VJP rule: the gradient cannot be built, so its rendering cannot either. */
    private fun noGradient(reason: String) = on(reason, Surface.VJP, Surface.JVP, Surface.RENDER)

    private const val STRETCH_RANK = "the gradient broadcasts at a rank the stretch twins (ranks 1 to 3) do not cover"
    private const val NO_BATCH_RULE = "no vmap batching rule"
    private const val CAST_TANGENT =
        "the JVP rounds the tangent to the narrower dtype and the VJP leaves the cotangent unrounded (a gap)"
    private const val FUSED = "a fused kind the coarsener emits; no interpreter arm or VJP rule, it is spelled through its composition"

    /**
     * The surfaces each op is not expected to work on, and why. docs/work-log/torch-tpu-
     * testing-assessment.md lists the ones that are gaps rather than decisions.
     */
    private val exclusions: Map<OpKind, Map<Surface, String>> = mapOf(
        ABS to on(":core has no tensor abs to render to", Surface.RENDER),
        RSQRT to on(":core has no tensor rsqrt to render to", Surface.RENDER),
        GELU to noInterpreter("below the user surface: emitter and coarseners only, no host op, interpreter arm or VJP rule") +
            on("no host op to render to", Surface.RENDER),
        SILU to noGradient("below the user surface: emitter and coarseners only, no host op or VJP rule"),
        NOT to on("boolean algebra renders only inside control flow", Surface.RENDER),
        LAND to on("boolean algebra renders only inside control flow", Surface.RENDER),
        ARGMAX to on("no :core host spelling", Surface.RENDER),
        LOGSUMEXP to noInterpreter("no interpreter arm (a gap)") + on("no VJP rule and no :core host spelling (a gap)", Surface.RENDER),
        DOT to noInterpreter("no interpreter arm; the plugin lowers dot products to MATMUL or SUM(MUL)") +
            on("no :core host spelling", Surface.RENDER),
        CONV2D to on(STRETCH_RANK, Surface.RENDER) + on(NO_BATCH_RULE, Surface.VMAP),
        CONV_TRANSPOSE2D to on(STRETCH_RANK, Surface.RENDER) + on(NO_BATCH_RULE, Surface.VMAP),
        DET to on(STRETCH_RANK, Surface.RENDER),
        MAXPOOL2D to on(STRETCH_RANK, Surface.RENDER) + on(NO_BATCH_RULE, Surface.VMAP),
        AVGPOOL2D to on(STRETCH_RANK, Surface.RENDER) + on(NO_BATCH_RULE, Surface.VMAP),
        MAXPOOL2D_GRAD to noGradient(ADJOINT_OP) + on(NO_BATCH_RULE, Surface.VMAP),
        AVGPOOL2D_GRAD to noGradient(ADJOINT_OP) + on(NO_BATCH_RULE, Surface.VMAP),
        CONV2D_DATA_ADJOINT to noGradient(ADJOINT_OP) + on(NO_BATCH_RULE, Surface.VMAP),
        CONV2D_KERNEL_ADJOINT to noGradient(ADJOINT_OP) + on(NO_BATCH_RULE, Surface.VMAP),
        CONV_TRANSPOSE2D_DATA_ADJOINT to noGradient(ADJOINT_OP) + on(NO_BATCH_RULE, Surface.VMAP),
        CONV_TRANSPOSE2D_KERNEL_ADJOINT to noGradient(ADJOINT_OP) + on(NO_BATCH_RULE, Surface.VMAP),
        GATHER to on("only a constant position is batched", Surface.VMAP) +
            on("the scalar-index gather family reads a runtime index the Int-taking host twins cannot express", Surface.RENDER),
        SCATTER to noGradient("no VJP rule; GATHER's adjoint is the only producer") + on(NO_BATCH_RULE, Surface.VMAP),
        SCATTER_ADD to noGradient("no VJP rule; GATHER's adjoint is the only producer") +
            on("only a constant position is batched", Surface.VMAP),
        TOP_K to on("TAKE_ALONG has no batching rule (a gap, FOLLOWUPS section 14)", Surface.VMAP),
        TAKE_ALONG to on("no batching rule (a gap, FOLLOWUPS section 14)", Surface.VMAP),
        TAKE_ALONG_GRAD to on("no batching rule (a gap, FOLLOWUPS section 14)", Surface.VMAP),
        PAD to on("standalone PAD has no host twin; the transform emits PAD_TO", Surface.RENDER),
        RMSNORM to noInterpreter(FUSED) + on(FUSED, Surface.RENDER),
        BATCHNORM to noInterpreter(FUSED) + on(FUSED, Surface.RENDER),
        CROSS_ENTROPY to noInterpreter("no interpreter arm; the plugin spells cross entropy through LOG_SOFTMAX") +
            on("no :core host spelling", Surface.RENDER),
        EMBEDDING to on("a batched table is not batched (only the indices may be)", Surface.VMAP),
        EMBEDDING_GRAD to noGradient(ADJOINT_OP) + on(NO_BATCH_RULE, Surface.VMAP),
        SPARSE_MATMUL to on("StableHLO has no sparse types; refused by name", Surface.EMIT, Surface.EMIT_F64, Surface.DEVICE) +
            on("the CSR family is not wired into the printer", Surface.RENDER) + on(NO_BATCH_RULE, Surface.VMAP),
        SPARSE_MATMUL_VALUES_ADJOINT to noGradient(ADJOINT_OP) + on(NO_BATCH_RULE, Surface.VMAP) +
            on("StableHLO has no sparse types; refused by name", Surface.EMIT, Surface.EMIT_F64, Surface.DEVICE),
        RNG_UNIFORM to on("emission is F32 only (draw at F32 and CAST)", Surface.EMIT_F64),
        RNG_NORMAL to on("emission is F32 only (draw at F32 and CAST)", Surface.EMIT_F64),
        CHECK_SHAPE_LIKE to on("a host assert StableHLO cannot express; refused by name", Surface.EMIT, Surface.EMIT_F64, Surface.DEVICE),
    )

    private val conv = mapOf("window_strides" to listOf(1, 1), "padding" to listOf(listOf(1, 1), listOf(1, 1)))
    private val convT = mapOf(
        "window_strides" to listOf(1, 1),
        "lhs_dilation" to listOf(2, 2),
        "padding" to listOf(listOf(1, 1), listOf(1, 1)),
    )
    private val pool = mapOf("window" to listOf(2, 2), "window_strides" to listOf(2, 2))

    val entries: Map<OpKind, OpEntry> = listOf(
        // Elementwise unary.
        entry(NEG, unary(NEG)),
        entry(ABS, unary(ABS, signed)),
        entry(EXP, unary(EXP)),
        entry(LOG, unary(LOG, positive)),
        entry(SQRT, unary(SQRT, positive)),
        entry(RSQRT, unary(RSQRT, positive)),
        entry(TANH, unary(TANH)),
        entry(SIGMOID, unary(SIGMOID)),
        entry(RELU, unary(RELU, signed)),
        entry(GELU, unary(GELU)),
        entry(SILU, unary(SILU)),
        entry(STEP, unary(STEP, signed)),
        entry(SIN, unary(SIN)),
        entry(COS, unary(COS)),
        entry(TAN, unary(TAN)),
        entry(ATAN, unary(ATAN)),
        entry(EXPM1, unary(EXPM1)),
        entry(LOG1P, unary(LOG1P, Gen.Uniform(-0.5, 1.0))),
        entry(LGAMMA, unary(LGAMMA, positive)),
        entry(DIGAMMA, unary(DIGAMMA, positive)),
        entry(TRIGAMMA, unary(TRIGAMMA, positive, tol = 1e-5)),
        entry(POLYGAMMA, unary(POLYGAMMA, positive, mapOf("order" to 3), tol = 1e-5)),
        entry(SIGN, unary(SIGN, signed)),
        entry(NOT, selectBy("not") { ps -> op(NOT, listOf(compare(ps[0], ps[1], "GT")), DxirType(Bool, ps[0].type.dims)) }),
        entry(
            LAND,
            selectBy("land") { ps ->
                val zero = op(ZEROS_LIKE, listOf(ps[0]), ps[0].type)
                op(LAND, listOf(compare(ps[0], ps[1], "GT"), compare(ps[0], zero, "GT")), DxirType(Bool, ps[0].type.dims))
            },
        ),
        // Elementwise binary.
        entry(ADD, binary(ADD)),
        entry(SUB, binary(SUB)),
        entry(MUL, binary(MUL)),
        entry(DIV, binary(DIV, b = positive)),
        entry(POW, binary(POW, a = positive)),
        // Reductions.
        entry(SUM, reduce(SUM)),
        entry(MEAN, reduce(MEAN)),
        entry(MAX, reduce(MAX)),
        entry(MIN, reduce(MIN)),
        entry(
            ARGMAX,
            OpSample("argmax", listOf(arg("x", 2, 3))) { dt, ps ->
                op(CAST, listOf(op(ARGMAX, ps, DxirType(I32, listOf(2)), attrs = mapOf("axis" to 1))), t(dt, 2))
            },
        ),
        entry(SOFTMAX, OpSample("softmax", listOf(arg("x", 2, 3))) { dt, ps -> op(SOFTMAX, ps, t(dt, 2, 3), attrs = mapOf("axis" to 1)) }),
        entry(LOGSUMEXP, OpSample("logsumexp", listOf(arg("x", 2, 3))) { dt, ps -> op(LOGSUMEXP, ps, t(dt, 2), attrs = mapOf("axis" to 1)) }),
        entry(LOG_SOFTMAX, OpSample("log_softmax", listOf(arg("x", 2, 3))) { dt, ps -> op(LOG_SOFTMAX, ps, t(dt, 2, 3), attrs = mapOf("axis" to 1)) }),
        entry(
            SOFT_CROSS_ENTROPY,
            OpSample("soft_cross_entropy", listOf(arg("s", 2, 3), arg("t", 2, 3))) { dt, ps ->
                op(SOFT_CROSS_ENTROPY, ps, t(dt, 2, 1), attrs = mapOf("axis" to 1))
            },
        ),
        // Linear algebra.
        entry(MATMUL, OpSample("matmul", listOf(arg("a", 2, 3), arg("b", 3, 4)), usesDot = true) { dt, ps -> op(MATMUL, ps, t(dt, 2, 4)) }),
        entry(DOT, OpSample("dot", listOf(arg("a", 3), arg("b", 3)), usesDot = true) { dt, ps -> op(DOT, ps, t(dt)) }),
        entry(
            CONV2D,
            OpSample("conv2d", listOf(arg("x", 1, 2, 4, 4), arg("w", 3, 2, 3, 3)), usesDot = true) { dt, ps ->
                op(CONV2D, ps, t(dt, 1, 3, 4, 4), attrs = conv)
            },
        ),
        entry(
            CONV_TRANSPOSE2D,
            OpSample("conv_transpose2d", listOf(arg("x", 1, 2, 3, 3), arg("w", 2, 3, 2, 2)), usesDot = true) { dt, ps ->
                op(CONV_TRANSPOSE2D, ps, t(dt, 1, 3, 6, 6), attrs = convT)
            },
        ),
        entry(CHOLESKY, OpSample("cholesky", listOf(arg("a", 3, 3, gen = Gen.Spd(3))), usesDot = true) { dt, ps -> op(CHOLESKY, ps, t(dt, 3, 3)) }),
        entry(
            TRIANGULAR_SOLVE,
            *listOf(true, false).flatMap { lower ->
                listOf(true, false).map { tr ->
                    OpSample("triangular_solve_l${lower}_t$tr", listOf(arg("a", 3, 3, gen = Gen.Spd(3)), arg("b", 3, 2)), usesDot = true) { dt, ps ->
                        op(TRIANGULAR_SOLVE, ps, t(dt, 3, 2), attrs = mapOf("lower" to lower, "transpose_a" to tr, "unit_diagonal" to false))
                    }
                }
            }.toTypedArray(),
        ),
        entry(
            TRIANGLE,
            OpSample("triangle", listOf(arg("a", 3, 3))) { dt, ps ->
                op(TRIANGLE, ps, t(dt, 3, 3), attrs = mapOf("lower" to 1.0, "diagonal" to 0.5, "upper" to 0.0))
            },
        ),
        entry(
            SOLVE,
            *listOf(false, true).map { tr ->
                OpSample("solve_t$tr", listOf(arg("a", 3, 3, gen = Gen.Spd(3)), arg("b", 3, 2)), usesDot = true) { dt, ps ->
                    op(SOLVE, ps, t(dt, 3, 2), attrs = mapOf("transpose_a" to tr))
                }
            }.toTypedArray(),
        ),
        entry(DET, OpSample("det", listOf(arg("a", 3, 3, gen = Gen.Spd(3))), usesDot = true) { dt, ps -> op(DET, ps, t(dt)) }),
        entry(QR_Q, OpSample("qr_q", listOf(arg("a", 4, 3)), usesDot = true) { dt, ps -> op(QR_Q, ps, t(dt, 4, 3)) }),
        entry(QR_R, OpSample("qr_r", listOf(arg("a", 4, 3)), usesDot = true) { dt, ps -> op(QR_R, ps, t(dt, 3, 3)) }),
        entry(EIGH_W, OpSample("eigh_w", listOf(arg("a", 3, 3)), usesDot = true) { dt, ps -> op(EIGH_W, ps, t(dt, 3)) }),
        entry(EIGH_V, OpSample("eigh_v", listOf(arg("a", 3, 3)), usesDot = true, gradTolerance = 1e-5) { dt, ps -> op(EIGH_V, ps, t(dt, 3, 3)) }),
        // Pooling and the fused conv and pool adjoints.
        entry(MAXPOOL2D, OpSample("maxpool2d", listOf(arg("x", 1, 2, 4, 4))) { dt, ps -> op(MAXPOOL2D, ps, t(dt, 1, 2, 2, 2), attrs = pool) }),
        entry(AVGPOOL2D, OpSample("avgpool2d", listOf(arg("x", 1, 2, 4, 4))) { dt, ps -> op(AVGPOOL2D, ps, t(dt, 1, 2, 2, 2), attrs = pool) }),
        entry(
            MAXPOOL2D_GRAD,
            OpSample("maxpool2d_grad", listOf(arg("u", 1, 2, 2, 2), arg("x", 1, 2, 4, 4))) { dt, ps ->
                val y = op(MAXPOOL2D, listOf(ps[1]), t(dt, 1, 2, 2, 2), attrs = pool)
                op(MAXPOOL2D_GRAD, listOf(ps[0], ps[1], y), t(dt, 1, 2, 4, 4), attrs = pool)
            },
        ),
        entry(
            AVGPOOL2D_GRAD,
            OpSample("avgpool2d_grad", listOf(arg("u", 1, 2, 2, 2), arg("x", 1, 2, 4, 4))) { dt, ps ->
                op(AVGPOOL2D_GRAD, ps, t(dt, 1, 2, 4, 4), attrs = pool)
            },
        ),
        entry(
            CONV2D_DATA_ADJOINT,
            OpSample("conv2d_data_adjoint", listOf(arg("u", 1, 3, 4, 4), arg("w", 3, 2, 3, 3), arg("x", 1, 2, 4, 4)), usesDot = true) { dt, ps ->
                op(CONV2D_DATA_ADJOINT, ps, t(dt, 1, 2, 4, 4), attrs = conv)
            },
        ),
        entry(
            CONV2D_KERNEL_ADJOINT,
            OpSample("conv2d_kernel_adjoint", listOf(arg("x", 1, 2, 4, 4), arg("u", 1, 3, 4, 4), arg("w", 3, 2, 3, 3)), usesDot = true) { dt, ps ->
                op(CONV2D_KERNEL_ADJOINT, ps, t(dt, 3, 2, 3, 3), attrs = conv)
            },
        ),
        entry(
            CONV_TRANSPOSE2D_DATA_ADJOINT,
            OpSample("conv_transpose2d_data_adjoint", listOf(arg("u", 1, 3, 6, 6), arg("w", 2, 3, 2, 2), arg("x", 1, 2, 3, 3)), usesDot = true) { dt, ps ->
                op(CONV_TRANSPOSE2D_DATA_ADJOINT, ps, t(dt, 1, 2, 3, 3), attrs = convT)
            },
        ),
        entry(
            CONV_TRANSPOSE2D_KERNEL_ADJOINT,
            OpSample("conv_transpose2d_kernel_adjoint", listOf(arg("x", 1, 2, 3, 3), arg("u", 1, 3, 6, 6), arg("w", 2, 3, 2, 2)), usesDot = true) { dt, ps ->
                op(CONV_TRANSPOSE2D_KERNEL_ADJOINT, ps, t(dt, 2, 3, 2, 2), attrs = convT)
            },
        ),
        // Shape.
        entry(RESHAPE, OpSample("reshape", listOf(arg("x", 2, 3))) { dt, ps -> op(RESHAPE, ps, t(dt, 3, 2)) }),
        entry(TRANSPOSE, OpSample("transpose", listOf(arg("x", 2, 3))) { dt, ps -> op(TRANSPOSE, ps, t(dt, 3, 2), attrs = mapOf("permutation" to listOf(1, 0))) }),
        entry(
            BROADCAST,
            OpSample("broadcast", listOf(arg("x", 3))) { dt, ps -> op(BROADCAST, ps, t(dt, 2, 3), attrs = mapOf("broadcast_dimensions" to listOf(1))) },
        ),
        entry(CONCAT, OpSample("concat", listOf(arg("a", 2, 3), arg("b", 1, 3))) { dt, ps -> op(CONCAT, ps, t(dt, 3, 3), attrs = mapOf("dimension" to 0)) }),
        entry(
            SLICE,
            OpSample("slice", listOf(arg("x", 3, 4))) { dt, ps ->
                op(SLICE, ps, t(dt, 2, 3), attrs = mapOf("start_indices" to listOf(1, 0), "limit_indices" to listOf(3, 3), "strides" to listOf(1, 1)))
            },
            OpSample(
                "slice_strided", listOf(arg("x", 3, 4)),
                excluded = noGradient("strided slices have no VJP rule (needs interior padding; a gap)") +
                    on("a strided slice has no host twin", Surface.RENDER),
            ) { dt, ps ->
                op(SLICE, ps, t(dt, 2, 2), attrs = mapOf("start_indices" to listOf(1, 0), "limit_indices" to listOf(3, 4), "strides" to listOf(1, 2)))
            },
        ),
        entry(GATHER, OpSample("gather", listOf(arg("x", 4, 3), ints("i", lo = 0, until = 4))) { dt, ps -> op(GATHER, ps, t(dt, 3)) }),
        entry(
            SCATTER,
            OpSample("scatter", listOf(arg("x", 4, 3), ints("i", lo = 0, until = 4), arg("v", 3))) { dt, ps -> op(SCATTER, ps, t(dt, 4, 3)) },
        ),
        entry(
            TOP_K,
            OpSample("top_k", listOf(arg("x", 1, 5, 1))) { dt, ps ->
                val s = op(RESHAPE, ps, t(dt, 1, 5))
                val idx = op(TOP_K, listOf(s), DxirType(I32, listOf(1, 2)), attrs = mapOf("k" to 2))
                op(TAKE_ALONG, listOf(ps[0], idx), t(dt, 1, 2, 1))
            },
        ),
        entry(TAKE_ALONG, OpSample("take_along", listOf(arg("x", 2, 3, 2), ints("i", 2, 4, lo = 0, until = 3))) { dt, ps -> op(TAKE_ALONG, ps, t(dt, 2, 4, 2)) }),
        entry(
            TAKE_ALONG_GRAD,
            OpSample("take_along_grad", listOf(arg("u", 2, 4, 2), ints("i", 2, 4, lo = 0, until = 3), arg("x", 2, 3, 2))) { dt, ps ->
                op(TAKE_ALONG_GRAD, ps, t(dt, 2, 3, 2))
            },
        ),
        entry(REVERSE, OpSample("reverse", listOf(arg("x", 2, 3))) { dt, ps -> op(REVERSE, ps, t(dt, 2, 3), attrs = mapOf("dimensions" to listOf(1))) }),
        entry(SUM_TO, OpSample("sum_to", listOf(arg("x", 2, 3), arg("t", 1, 3))) { dt, ps -> op(SUM_TO, ps, t(dt, 1, 3)) }),
        entry(BROADCAST_LIKE, OpSample("broadcast_like", listOf(arg("x", 1, 3), arg("t", 2, 3))) { dt, ps -> op(BROADCAST_LIKE, ps, t(dt, 2, 3)) }),
        entry(PAD_TO, OpSample("pad_to", listOf(arg("x", 2, 2), arg("t", 3, 4))) { dt, ps -> op(PAD_TO, ps, t(dt, 3, 4), attrs = mapOf("low" to listOf(1, 1))) }),
        entry(SLICE_AT, OpSample("slice_at", listOf(arg("x", 3, 4), arg("t", 2, 2))) { dt, ps -> op(SLICE_AT, ps, t(dt, 2, 2), attrs = mapOf("low" to listOf(1, 1))) }),
        entry(
            SLICE_LIKE,
            OpSample("slice_like", listOf(arg("x", 5, 3), arg("t", 2, 3), arg("p", 3, 3))) { dt, ps -> op(SLICE_LIKE, ps, t(dt, 2, 3), attrs = mapOf("axis" to 0)) },
        ),
        entry(
            PAD_LIKE,
            OpSample("pad_like", listOf(arg("x", 2, 3), arg("t", 5, 3), arg("p", 3, 3))) { dt, ps -> op(PAD_LIKE, ps, t(dt, 5, 3), attrs = mapOf("axis" to 0)) },
        ),
        entry(
            ZEROS_LIKE,
            OpSample("zeros_like", listOf(arg("x", 2, 3))) { dt, ps -> op(ADD, listOf(ps[0], op(ZEROS_LIKE, ps, t(dt, 2, 3))), t(dt, 2, 3)) },
        ),
        entry(WHERE, selectBy("where") { ps -> compare(ps[0], ps[1], "GT") }),
        entry(
            COMPARE,
            *listOf("EQ", "NE", "LT", "LE", "GT", "GE").map { d -> selectBy("compare_${d.lowercase()}") { ps -> compare(ps[0], ps[1], d) } }.toTypedArray(),
        ),
        entry(PAD, OpSample("pad", listOf(arg("x", 2, 3))) { dt, ps -> op(PAD, ps, t(dt, 3, 5), attrs = mapOf("low" to listOf(1, 0), "high" to listOf(0, 2))) }),
        entry(
            SCATTER_ADD,
            OpSample("scatter_add", listOf(arg("x", 4), ints("i", lo = 0, until = 4), arg("v"))) { dt, ps -> op(SCATTER_ADD, ps, t(dt, 4)) },
        ),
        // Normalization and attention.
        entry(LAYERNORM, OpSample("layernorm", listOf(arg("x", 2, 4))) { dt, ps -> op(LAYERNORM, ps, t(dt, 2, 4)) }, excluded = demoted),
        entry(RMSNORM, OpSample("rmsnorm", listOf(arg("x", 2, 4))) { dt, ps -> op(RMSNORM, ps, t(dt, 2, 4), attrs = mapOf("epsilon" to 1e-6)) }),
        entry(
            BATCHNORM,
            OpSample("batchnorm", listOf(arg("x", 4, 3), arg("scale", 3), arg("offset", 3), arg("mean", 3), arg("var", 3, gen = positive))) { dt, ps ->
                op(BATCHNORM, ps, t(dt, 4, 3), attrs = mapOf("epsilon" to 1e-5, "feature_index" to 1))
            },
        ),
        entry(
            SCALED_DOT_PRODUCT_ATTENTION,
            OpSample("sdpa", listOf(arg("q", 1, 2, 3), arg("k", 1, 2, 3), arg("v", 1, 2, 3)), usesDot = true) { dt, ps ->
                op(SCALED_DOT_PRODUCT_ATTENTION, ps, t(dt, 1, 2, 3))
            },
            excluded = demoted,
        ),
        // Inference only: dedicated suites cover them.
        none(PAGED_ATTENTION, "PagedAttentionEmitTest, DxirPagedAttentionTest, PjrtPagedAttentionTest; inference only"),
        none(KV_CACHE_WRITE, "DxirKvCacheWriteTest, PjrtKvCacheWriteTest; inference only"),
        none(DEQUANTIZE_KV, "DxirDequantizeKvTest, PjrtDequantizeKvTest; inference only"),
        none(CAUSAL_CONV1D, "the Qwen3.5 parity tests (HfQwen35*, PjrtQwen35*); inference only"),
        none(GATED_DELTA_RULE, "the Qwen3.5 parity tests (HfQwen35*, PjrtQwen35*); inference only"),
        none(MOE_EXPERTS, "HfQwen35MoeParityTest, PjrtQwen35MoeParityTest, PjrtMoeBenchTest; inference only"),
        none(NVFP4_MATMUL, "PjrtQwen4ExpNvfp4TableTest; inference only"),
        none(NVFP4_EMBEDDING, "PjrtQwen4ExpNvfp4TableTest; inference only"),
        none(NGRAM_HASH, "the qwen4_exp tests (HfQwen4Exp*, PjrtQwen4Exp*); inference only"),
        none(INDEXER_TOPK, "PjrtFusedSparseAttentionTest and the qwen4_exp tests; inference only"),
        none(SAMPLE_TOKENS, "the sampler tests in :runtime-pjrt; inference only"),
        none(MOSAIC_KERNEL, "MosaicCustomCallEmitTest, TpuCustomCallParseCheckTest; a TPU custom call"),
        // Embedding, cross entropy, casts.
        entry(EMBEDDING, OpSample("embedding", listOf(arg("table", 5, 3), ints("i", 4, lo = 0, until = 5))) { dt, ps -> op(EMBEDDING, ps, t(dt, 4, 3)) }),
        entry(
            EMBEDDING_GRAD,
            OpSample("embedding_grad", listOf(ints("i", 4, lo = 0, until = 5), arg("u", 4, 3), arg("table", 5, 3))) { dt, ps ->
                op(EMBEDDING_GRAD, ps, t(dt, 5, 3))
            },
        ),
        entry(
            CROSS_ENTROPY,
            OpSample("cross_entropy", listOf(arg("logits", 2, 4), ints("labels", 2, lo = 0, until = 4))) { dt, ps -> op(CROSS_ENTROPY, ps, t(dt, 2)) },
        ),
        entry(
            CAST,
            OpSample(
                "cast_f32_f64", listOf(arg("x", 2, 3)),
                // The step must clear F32's rounding; the error is then F32 noise.
                fdStep = 1e-2, gradTolerance = 2e-5,
                excluded = on(CAST_TANGENT, Surface.JVP) + on("only the Bool and F32 mask identity renders", Surface.RENDER),
            ) { dt, ps ->
                val other = if (dt == io.tlaloc.core.F64) io.tlaloc.core.F32 else io.tlaloc.core.F64
                op(CAST, listOf(op(CAST, ps, t(other, 2, 3))), t(dt, 2, 3))
            },
            OpSample(
                "cast_bf16", listOf(arg("x", 2, 3)),
                excluded = on("rounding to bf16 is a step function, so central differences do not see the derivative 1", Surface.VJP) +
                    on(CAST_TANGENT, Surface.JVP) +
                    on("only the Bool and F32 mask identity renders", Surface.RENDER) +
                    on("XLA folds the F32 to bf16 to F32 convert pair (TestBackend.foldsConvertPairs)", Surface.DEVICE),
            ) { dt, ps -> op(CAST, listOf(op(CAST, ps, t(BF16, 2, 3))), t(dt, 2, 3)) },
        ),
        // Sparse (CSR, three rows of a 3×4 matrix with 5 stored values).
        entry(
            SPARSE_MATMUL,
            OpSample(
                "sparse_matmul",
                listOf(
                    arg("values", 5),
                    Arg("col", listOf(5), Gen.Fixed(doubleArrayOf(0.0, 2.0, 1.0, 0.0, 3.0)), integer = true),
                    Arg("row", listOf(4), Gen.Fixed(doubleArrayOf(0.0, 2.0, 3.0, 5.0)), integer = true),
                    arg("b", 4, 2),
                ),
                usesDot = true,
            ) { dt, ps -> op(SPARSE_MATMUL, ps, t(dt, 3, 2)) },
        ),
        entry(
            SPARSE_MATMUL_VALUES_ADJOINT,
            OpSample(
                "sparse_matmul_values_adjoint",
                listOf(
                    arg("u", 3, 2),
                    arg("b", 4, 2),
                    Arg("col", listOf(5), Gen.Fixed(doubleArrayOf(0.0, 2.0, 1.0, 0.0, 3.0)), integer = true),
                    Arg("row", listOf(4), Gen.Fixed(doubleArrayOf(0.0, 2.0, 3.0, 5.0)), integer = true),
                ),
                usesDot = true,
            ) { dt, ps -> op(SPARSE_MATMUL_VALUES_ADJOINT, ps, t(dt, 5)) },
        ),
        // Random numbers: the draw times x, so the gradient is the draw.
        *listOf(RNG_UNIFORM, RNG_NORMAL).map { kind ->
            entry(
                kind,
                OpSample(kind.name.lowercase(), listOf(arg("x", 4))) { dt, ps ->
                    val e = op(kind, emptyList(), t(dt, 4), attrs = mapOf("key0" to 42, "key1" to 7, "dims" to listOf(4)))
                    op(MUL, listOf(ps[0], e), t(dt, 4))
                },
            )
        }.toTypedArray(),
        entry(CHECK_SHAPE_LIKE, OpSample("check_shape_like", listOf(arg("x", 2, 3))) { dt, ps -> op(CHECK_SHAPE_LIKE, listOf(ps[0], ps[0]), t(dt, 2, 3)) }),
        // Regions, splices and distribution: built by their own passes.
        none(IF, "DxirIfTest, the IF arms of DxirForwardTransformTest and the K2 plugin's branch tests"),
        none(WHILE, "PhiCalculusTest and the plugin's loop tests (coarsened before AD)"),
        none(COARSENED, "PhiCalculusTest, CoarsenedCustomCallTest and the coarsener suites"),
        none(SHARD_CONSTRAINT, "ShardingTest, GradShardingVerifyTest, SdyParseTest"),
        none(MANUAL_COMPUTATION, "EmitterTest, SdyRoundTripTest"),
        none(ALL_REDUCE, "AllReduceTest, EmitterTest, PjrtCollectiveCompileTest, MultiDeviceCpuTest"),
        none(ALL_GATHER, "AllReduceTest, EmitterTest, PjrtCollectiveCompileTest, MultiDeviceCpuTest"),
        none(REDUCE_SCATTER, "AllReduceTest, EmitterTest, PjrtCollectiveCompileTest, MultiDeviceCpuTest"),
    ).associateBy { it.kind }

    private val kindBySample: Map<OpSample, OpKind> by lazy {
        entries.values.flatMap { e -> e.samples.map { it to e.kind } }.toMap()
    }

    fun kindOf(sample: OpSample): OpKind? = kindBySample[sample]

    /** Whether [sample] is expected to work on [surface]. */
    fun claims(sample: OpSample, surface: Surface): Boolean {
        val e = kindOf(sample)?.let { entries[it] } ?: return true
        return surface !in e.excluded && surface !in sample.excluded
    }

    /** Every OpKind without an entry. */
    fun missing(): List<OpKind> = OpKind.entries.filter { it !in entries }
}
