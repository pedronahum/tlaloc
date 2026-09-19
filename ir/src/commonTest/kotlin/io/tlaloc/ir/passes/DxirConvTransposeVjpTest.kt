package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.391 — CONV_TRANSPOSE2D's own adjoint ([VjpRegistry.ConvTranspose2dRule],
 * the fused `CONV_TRANSPOSE2D_DATA_ADJOINT` / `_KERNEL_ADJOINT` pair).
 *
 * Differentiating THROUGH a transposed conv — a deconvolution, or the
 * fractionally-strided upsample that `convTranspose2d(w, stride)` spells — was a
 * loud "no VJP rule registered" before this, even though the primal's FIR arm and
 * host twin shipped in §0.4.384.
 *
 * Two independent oracles, as `DxirConvTest` does for CONV2D:
 * - directional central differences of the interpreted primal (so the adjoint is
 *   checked against the op's own definition, not against a re-derivation of it);
 * - the JVP⇄VJP cross-identity `⟨∇f, v⟩ == jvp_f(x, v)`, whose forward side uses
 *   only the bilinear product rule (the primal op twice) and therefore never
 *   touches the new rule's index inversion.
 *
 * The configurations are the ones the index-inversion formulas were verified
 * against in a standalone model before any Kotlin was written: lhs_dilation 1 and
 * 2, window_strides 1 and 2, rhs_dilation, window_reversal, asymmetric padding,
 * and a case combining all of them. Each is load-bearing — the inversion's
 * divisibility test is what makes lhs_dilation and window_strides interact, and a
 * formula that only worked at L=1, s=1 would pass the first case alone.
 */
class DxirConvTransposeVjpTest {

    private data class Config(
        val name: String,
        val xDims: List<Int>,
        val wDims: List<Int>,   // IOHW [Ci, Co, kh, kw]
        val yDims: List<Int>,
        val attrs: Map<String, Any>,
        val seed: Long,
    )

    private val configs = listOf(
        Config(
            "plain stride-1 pad-1", listOf(1, 2, 4, 4), listOf(2, 3, 3, 3), listOf(1, 3, 4, 4),
            mapOf(
                "window_strides" to listOf(1, 1),
                "padding" to listOf(listOf(1, 1), listOf(1, 1)),
            ),
            seed = 101,
        ),
        Config(
            "upsampling lhs_dilation 2", listOf(1, 2, 3, 3), listOf(2, 3, 2, 2), listOf(1, 3, 4, 4),
            mapOf(
                "window_strides" to listOf(1, 1),
                "lhs_dilation" to listOf(2, 2),
            ),
            seed = 202,
        ),
        Config(
            "upsampling lhs_dilation 2 with pad 1", listOf(1, 2, 3, 3), listOf(2, 3, 2, 2),
            listOf(1, 3, 6, 6),
            mapOf(
                "window_strides" to listOf(1, 1),
                "lhs_dilation" to listOf(2, 2),
                "padding" to listOf(listOf(1, 1), listOf(1, 1)),
            ),
            seed = 303,
        ),
        Config(
            "reversed and rhs_dilated, asymmetric padding",
            listOf(1, 2, 5, 4), listOf(2, 3, 3, 2), listOf(1, 3, 4, 4),
            mapOf(
                "window_strides" to listOf(1, 1),
                "padding" to listOf(listOf(1, 0), listOf(1, 1)),
                "rhs_dilation" to listOf(1, 2),
                "window_reversal" to listOf(true, true),
            ),
            seed = 404,
        ),
        Config(
            "window_strides 2", listOf(1, 2, 5, 5), listOf(2, 3, 2, 2), listOf(1, 3, 2, 2),
            mapOf("window_strides" to listOf(2, 2)),
            seed = 505,
        ),
        Config(
            "everything at once: strides [2,1], lhs_dilation 2, reversal [true,false], " +
                "asymmetric padding",
            listOf(1, 1, 4, 4), listOf(1, 1, 3, 3), listOf(1, 1, 4, 6),
            mapOf(
                "window_strides" to listOf(2, 1),
                "lhs_dilation" to listOf(2, 2),
                "padding" to listOf(listOf(1, 1), listOf(0, 1)),
                "window_reversal" to listOf(true, false),
            ),
            seed = 606,
        ),
    )

    private fun lossFn(cfg: Config): DxirFunction {
        val xT = DxirType(F32, cfg.xDims)
        val wT = DxirType(F32, cfg.wDims)
        val yT = DxirType(F32, cfg.yDims)
        return DxirBuilder.function("convT_loss_${cfg.seed}") {
            val x = param("x", xT)
            val w = param("w", wT)
            val y = op(OpKind.CONV_TRANSPOSE2D, listOf(x, w), yT, attrs = cfg.attrs)
            val y2 = op(OpKind.MUL, listOf(y, y), yT)
            listOf(op(OpKind.SUM, listOf(y2), DxirType(F32, emptyList())))
        }
    }

    private fun randomInputs(cfg: Config): Pair<FloatArray, FloatArray> {
        val rng = Random(cfg.seed)
        val n = cfg.xDims.reduce(Int::times)
        val m = cfg.wDims.reduce(Int::times)
        return Pair(
            FloatArray(n) { (rng.nextFloat() - 0.5f) * 3f },
            FloatArray(m) { (rng.nextFloat() - 0.5f) * 3f },
        )
    }

    @Test
    fun convTransposeVjpMatchesCentralDifferences() {
        for (cfg in configs) {
            val fn = lossFn(cfg)
            val (x, w) = randomInputs(cfg)
            val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x, w))

            // Directional central differences along a fixed direction v: ⟨∇f, v⟩ vs
            // (f(x+εv) − f(x−εv)) / 2ε, aggregated into one scalar so the f32 oracle
            // stays well-conditioned.
            val rng = Random(cfg.seed + 1)
            val vx = FloatArray(x.size) { rng.nextFloat() - 0.5f }
            val vw = FloatArray(w.size) { rng.nextFloat() - 0.5f }
            var dot = 0.0
            for (i in x.indices) dot += grads[0][i].toDouble() * vx[i]
            for (i in w.indices) dot += grads[1][i].toDouble() * vw[i]

            val eps = 1e-2f
            fun shifted(s: Float): Float {
                val xS = FloatArray(x.size) { x[it] + s * vx[it] }
                val wS = FloatArray(w.size) { w[it] + s * vw[it] }
                return DxirInterpreter.evalFunction(fn, listOf(xS, wS)).single().single()
            }
            val fd = (shifted(eps) - shifted(-eps)).toDouble() / (2 * eps)
            assertTrue(
                abs(dot - fd) <= 1e-2 * maxOf(1.0, abs(fd)),
                "[${cfg.name}] convT VJP disagrees with central differences: " +
                    "⟨∇f,v⟩=$dot vs fd=$fd (attrs=${cfg.attrs})",
            )
        }
    }

    @Test
    fun convTransposeVjpAgreesWithJvp() {
        for (cfg in configs) {
            val fn = lossFn(cfg)
            val (x, w) = randomInputs(cfg)
            val rng = Random(cfg.seed + 2)
            val vx = FloatArray(x.size) { rng.nextFloat() - 0.5f }
            val vw = FloatArray(w.size) { rng.nextFloat() - 0.5f }

            val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x, w))
            var dot = 0.0
            for (i in x.indices) dot += grads[0][i].toDouble() * vx[i]
            for (i in w.indices) dot += grads[1][i].toDouble() * vw[i]

            // The forward side uses only the bilinear product rule over the primal
            // op, never the new rule's index inversion — so agreement is real
            // cross-validation rather than a shared-bug tautology.
            val tangent = DxirInterpreter.evalFunction(
                DxirForwardTransform.apply(fn), listOf(x, w, vx, vw),
            ).last().single()
            assertTrue(
                abs(dot - tangent) <= 1e-3 * maxOf(1.0, abs(tangent).toDouble()),
                "[${cfg.name}] convT VJP disagrees with its JVP: ⟨∇f,v⟩=$dot vs jvp=$tangent " +
                    "(attrs=${cfg.attrs})",
            )
        }
    }

    /**
     * The adjoint nodes must carry nothing but the primal's literal attrs — the
     * sentinel-safety property, checked here on CONCRETE dims so a regression shows
     * up as a changed attr set rather than only under `grad {}`.
     */
    @Test
    fun convTransposeAdjointCarriesOnlyPrimalLiteralAttrs() {
        for (cfg in configs) {
            val grads = DxirReverseTransform.apply(lossFn(cfg))
            val adjoints = grads.body.filterIsInstance<DxirOp>().filter {
                it.op == OpKind.CONV_TRANSPOSE2D_DATA_ADJOINT ||
                    it.op == OpKind.CONV_TRANSPOSE2D_KERNEL_ADJOINT
            }
            assertTrue(
                adjoints.size == 2,
                "[${cfg.name}] expected both fused adjoints, got ${adjoints.map { it.op }}",
            )
            for (adj in adjoints) {
                // Defaults filled in are fine (the interpreter supplies the same
                // ones); what must NOT appear is any extent-derived number, i.e. a
                // `padding` value that is not the primal's.
                val pad = adj.attrs["padding"]
                val primalPad = cfg.attrs["padding"] ?: listOf(listOf(0, 0), listOf(0, 0))
                assertTrue(
                    pad == primalPad,
                    "[${cfg.name}] ${adj.op} carries padding $pad, not the primal's $primalPad " +
                        "— something solved an extent at transform time",
                )
                assertTrue(
                    adj.operands.size == 3,
                    "[${cfg.name}] ${adj.op} must carry its shape template",
                )
            }
        }
    }
}
