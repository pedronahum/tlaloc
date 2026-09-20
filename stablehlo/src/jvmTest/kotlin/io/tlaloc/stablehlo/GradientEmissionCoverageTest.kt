package io.tlaloc.stablehlo

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.393 — an emission COVERAGE SWEEP over the differentiable surface: for every
 * primal here, reverse-transform it and require the gradient graph to emit.
 *
 * This exists because of what §0.4.393 found. An OpKind-level audit ("which kinds
 * have no emitter arm?") turned up `SIGN` and nothing else, but running one real
 * reduction gradient through the emitter immediately hit a second, worse gap:
 * `BROADCAST` with empty `broadcast_dimensions` at equal rank — the reduction
 * adjoints' un-reduce stretch — which §0.4.359 had made polymorphic in the
 * INTERPRETER while the emitter still rejected it. Every Max/Min/Tanh/Softmax
 * gradient was unemittable, and no test caught it because no test emitted one. An
 * audit of kinds is not an audit of configurations; only exercising the graphs is.
 *
 * Each case is self-validating: the interpreter runs the primal AND the gradient
 * before emission, so a malformed hand-built graph fails as an interpreter error
 * rather than masquerading as an emitter gap.
 *
 * Deliberately NOT in the catalogue, with reasons:
 * - `EMBEDDING` — its adjoint `EMBEDDING_GRAD` has no emitter arm by design
 *   (scatter+add region, deferred), and `embedding` has no FIR arm either, so no
 *   `grad {}` body can contain one. Unreachable, not a gap.
 * - overlapping or padded `MAXPOOL2D` — the all-ties convention is not expressible
 *   over overlapping windows in StableHLO (§0.4.392); `EmitterTest` pins that this
 *   fails loudly. The non-overlapping case IS covered below.
 * - `window_reversal` on the deconv adjoints — rejected at emit, pinned in
 *   `EmitterTest`, and unreachable from user code.
 * - `SILU` / `GELU` — not part of the differentiable surface at all: they have an
 *   emitter arm and appear in the recognizers/coarseners (SwiGLU, TransformerMLP)
 *   but have no host op, no FIR arm and no VjpRule, so no `grad {}` body can
 *   contain one. Listed because the sweep's first run reported them as "no VJP
 *   rule" and that reads like a gap rather than what it is: an op that only exists
 *   below the user surface.
 */
class GradientEmissionCoverageTest {

    private val scalar = DxirType(F32, emptyList())

    private data class Case(val name: String, val build: () -> DxirFunction)

    /** `Σ y²` over `y = op(x…)`: the standard non-uniform-upstream loss shape. */
    private fun squaredSumLoss(
        name: String,
        params: List<Pair<String, DxirType>>,
        yType: DxirType,
        buildY: DxirBuilder.(List<io.tlaloc.ir.DxirNode>) -> io.tlaloc.ir.DxirNode,
    ) = Case(name) {
        DxirBuilder.function(name) {
            val ps = params.map { (n, t) -> param(n, t) }
            val y = buildY(ps)
            val y2 = op(OpKind.MUL, listOf(y, y), yType)
            listOf(op(OpKind.SUM, listOf(y2), scalar))
        }
    }

    private fun unaryLoss(name: String, kind: OpKind, dims: List<Int>, attrs: Map<String, Any> = emptyMap()) =
        squaredSumLoss(
            name,
            listOf("x" to DxirType(F32, dims)),
            DxirType(F32, dims),
        ) { ps -> op(kind, listOf(ps[0]), DxirType(F32, dims), attrs = attrs) }

    private val r2 = DxirType(F32, listOf(2, 3))

    private val cases = listOf(
        // Elementwise unaries whose adjoints are elementwise.
        unaryLoss("tanh", OpKind.TANH, listOf(2, 3)),
        unaryLoss("sigmoid", OpKind.SIGMOID, listOf(2, 3)),
        // §0.4.395 — the Phase C2 trig tails: TanRule's adjoint recomputes TAN
        // (the `stablehlo.tan` arm), AtanRule's divides by 1 + x² (the atan2
        // spelling never appears in the ADJOINT — only the primal emits it).
        unaryLoss("tan", OpKind.TAN, listOf(2, 3)),
        unaryLoss("atan", OpKind.ATAN, listOf(2, 3)),
        unaryLoss("sqrt", OpKind.SQRT, listOf(2, 3)),
        unaryLoss("sign", OpKind.SIGN, listOf(2, 3)),
        // RELU is in the catalogue even though the INTERPRETER has no arm for it
        // (it is a composite: the emitter expands it and synthesis calls the host
        // `relu`, while the interpreter's "bridge supported set" stops at STEP).
        // The sweep treats that specific rejection as "interpreter does not model
        // this op" and still asserts emission — which is the point of the test.
        unaryLoss("relu", OpKind.RELU, listOf(2, 3)),
        // Reductions — the un-reduce stretch is what §0.4.393 fixed.
        squaredSumLoss(
            "max_reduction", listOf("x" to r2), DxirType(F32, listOf(2, 1)),
        ) { ps -> op(OpKind.MAX, listOf(ps[0]), DxirType(F32, listOf(2, 1)), attrs = mapOf("reduction_dims" to listOf(1))) },
        squaredSumLoss(
            "min_reduction", listOf("x" to r2), DxirType(F32, listOf(2, 1)),
        ) { ps -> op(OpKind.MIN, listOf(ps[0]), DxirType(F32, listOf(2, 1)), attrs = mapOf("reduction_dims" to listOf(1))) },
        squaredSumLoss(
            "mean_reduction", listOf("x" to r2), DxirType(F32, listOf(2, 1)),
        ) { ps -> op(OpKind.MEAN, listOf(ps[0]), DxirType(F32, listOf(2, 1)), attrs = mapOf("reduction_dims" to listOf(1))) },
        squaredSumLoss(
            "softmax_axis1", listOf("x" to r2), r2,
        ) { ps -> op(OpKind.SOFTMAX, listOf(ps[0]), r2, attrs = mapOf("axis" to 1)) },
        // Linear algebra.
        squaredSumLoss(
            "matmul", listOf("a" to DxirType(F32, listOf(2, 3)), "b" to DxirType(F32, listOf(3, 4))),
            DxirType(F32, listOf(2, 4)),
        ) { ps -> op(OpKind.MATMUL, listOf(ps[0], ps[1]), DxirType(F32, listOf(2, 4))) },
        // Shape plumbing.
        squaredSumLoss(
            "concat_axis0",
            listOf("a" to DxirType(F32, listOf(2, 3)), "b" to DxirType(F32, listOf(1, 3))),
            DxirType(F32, listOf(3, 3)),
        ) { ps ->
            op(
                OpKind.CONCAT, listOf(ps[0], ps[1]), DxirType(F32, listOf(3, 3)),
                attrs = mapOf("dimension" to 0),
            )
        },
        squaredSumLoss(
            "slice", listOf("x" to r2), DxirType(F32, listOf(2, 2)),
        ) { ps ->
            op(
                OpKind.SLICE, listOf(ps[0]), DxirType(F32, listOf(2, 2)),
                attrs = mapOf(
                    "start_indices" to listOf(0, 0),
                    "limit_indices" to listOf(2, 2),
                    "strides" to listOf(1, 1),
                ),
            )
        },
        squaredSumLoss(
            "pad", listOf("x" to r2), DxirType(F32, listOf(3, 4)),
        ) { ps ->
            op(
                OpKind.PAD, listOf(ps[0]), DxirType(F32, listOf(3, 4)),
                attrs = mapOf("low" to listOf(1, 0), "high" to listOf(0, 1)),
            )
        },
        squaredSumLoss(
            "reshape", listOf("x" to r2), DxirType(F32, listOf(6)),
        ) { ps -> op(OpKind.RESHAPE, listOf(ps[0]), DxirType(F32, listOf(6))) },
        squaredSumLoss(
            "transpose", listOf("x" to r2), DxirType(F32, listOf(3, 2)),
        ) { ps ->
            op(OpKind.TRANSPOSE, listOf(ps[0]), DxirType(F32, listOf(3, 2)), attrs = mapOf("permutation" to listOf(1, 0)))
        },
        // Masking.
        squaredSumLoss(
            "where_compare", listOf("x" to r2), r2,
        ) { ps ->
            val mask = op(
                OpKind.COMPARE, listOf(ps[0], ps[0]), DxirType(io.tlaloc.core.Bool, listOf(2, 3)),
                attrs = mapOf("direction" to "GE"),
            )
            op(OpKind.WHERE, listOf(mask, ps[0], ps[0]), r2)
        },
        // The conv/pool arc (NCHW), all four kinds and both directions.
        squaredSumLoss(
            "conv2d",
            listOf("x" to DxirType(F32, listOf(1, 2, 4, 4)), "w" to DxirType(F32, listOf(3, 2, 3, 3))),
            DxirType(F32, listOf(1, 3, 4, 4)),
        ) { ps ->
            op(
                OpKind.CONV2D, listOf(ps[0], ps[1]), DxirType(F32, listOf(1, 3, 4, 4)),
                attrs = mapOf("window_strides" to listOf(1, 1), "padding" to listOf(listOf(1, 1), listOf(1, 1))),
            )
        },
        squaredSumLoss(
            "conv_transpose2d",
            listOf("x" to DxirType(F32, listOf(1, 2, 3, 3)), "w" to DxirType(F32, listOf(2, 3, 2, 2))),
            DxirType(F32, listOf(1, 3, 6, 6)),
        ) { ps ->
            op(
                OpKind.CONV_TRANSPOSE2D, listOf(ps[0], ps[1]), DxirType(F32, listOf(1, 3, 6, 6)),
                attrs = mapOf(
                    "window_strides" to listOf(1, 1),
                    "lhs_dilation" to listOf(2, 2),
                    "padding" to listOf(listOf(1, 1), listOf(1, 1)),
                ),
            )
        },
        squaredSumLoss(
            "avgpool2d", listOf("x" to DxirType(F32, listOf(1, 2, 4, 4))), DxirType(F32, listOf(1, 2, 2, 2)),
        ) { ps ->
            op(
                OpKind.AVGPOOL2D, listOf(ps[0]), DxirType(F32, listOf(1, 2, 2, 2)),
                attrs = mapOf("window" to listOf(2, 2), "window_strides" to listOf(2, 2)),
            )
        },
        squaredSumLoss(
            "maxpool2d", listOf("x" to DxirType(F32, listOf(1, 2, 4, 4))), DxirType(F32, listOf(1, 2, 2, 2)),
        ) { ps ->
            op(
                OpKind.MAXPOOL2D, listOf(ps[0]), DxirType(F32, listOf(1, 2, 2, 2)),
                attrs = mapOf("window" to listOf(2, 2), "window_strides" to listOf(2, 2)),
            )
        },
    )

    @Test
    fun everyDifferentiableSurfaceEmitsItsGradientGraph() {
        val failures = mutableListOf<String>()
        for (case in cases) {
            val fn = try {
                case.build()
            } catch (t: Throwable) {
                failures += "${case.name}: test graph is malformed — ${t.message}"
                continue
            }
            val grads = try {
                DxirReverseTransform.apply(fn)
            } catch (t: Throwable) {
                failures += "${case.name}: reverse transform — ${t.message}"
                continue
            }

            // Self-validation: if the interpreter cannot run the graph, the graph is
            // wrong, not the emitter — report it distinctly so a broken catalogue
            // entry cannot masquerade as an emitter gap.
            val inputs = fn.params.map { p ->
                val n = if (p.type.dims.isEmpty()) 1 else p.type.dims.reduce(Int::times)
                FloatArray(n) { i -> ((i * 37) % 23 - 11) / 24.0f }
            }
            try {
                DxirInterpreter.evalFunction(fn, inputs)
                DxirInterpreter.evalFunction(grads, inputs)
            } catch (t: Throwable) {
                // The interpreter's "bridge supported set" is deliberately narrower
                // than the emitter's — composites like RELU are expanded by the
                // emitter and served by a host op under synthesis, but the
                // interpreter has no arm for them. That is not an emitter gap, so
                // note it and still assert emission; anything else means the
                // catalogue entry itself is malformed.
                val msg = t.message ?: ""
                if ("not in the bridge's supported set" !in msg) {
                    failures += "${case.name}: interpreter rejected the graph — $msg"
                    continue
                }
            }

            try {
                grads.toStablehlo()
            } catch (t: Throwable) {
                failures += "${case.name}: GRADIENT GRAPH DOES NOT EMIT — ${t.message}"
            }
        }
        assertTrue(
            failures.isEmpty(),
            "${failures.size} of ${cases.size} gradient graphs failed:\n" + failures.joinToString("\n"),
        )
    }
}
