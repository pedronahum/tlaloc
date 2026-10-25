package io.tlaloc.ir.recognizer.coarsener

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.recognizeCrossEntropy
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Layer 4 §0.4.266 — cross-entropy analytical-backward coarsener tests.
 *
 * Structural verification (op count, body signatures, readsPrimalIndices),
 * and the gradient body's values against exact ones for a confident row and
 * for labels that do not sum to one. Mirrors `RmsNormCoarsenerTest` /
 * `RopeCoarsenerTest`.
 */
class CrossEntropyCoarsenerTest {

    private val logitsType = DxirType(F32, listOf(8, 64))
    private val labelsType = DxirType(F32, listOf(8, 64))
    private val scalarType = DxirType(F32, listOf())

    /**
     * Canonical NLL form: `loss = SUM(MUL(labels, LOG(SOFTMAX(logits))))`.
     * Mirrors `RmsNormRopeCrossEntropyTest.crossEntropyPositiveMatch`.
     */
    private fun buildCrossEntropyFn(): DxirFunction = DxirBuilder.function("nll") {
        val logits = param("logits", logitsType)
        val labels = param("labels", labelsType)
        val probs = op(OpKind.SOFTMAX, listOf(logits), logitsType)
        val logp = op(OpKind.LOG, listOf(probs), logitsType)
        val pw = op(OpKind.MUL, listOf(labels, logp), logitsType)
        val loss = op(OpKind.SUM, listOf(pw), scalarType)
        listOf(loss)
    }

    @Test
    fun coarsensCanonicalCrossEntropyIntoOneCoarsenedOp() {
        val fn = buildCrossEntropyFn()
        val matches = recognizeCrossEntropy(fn)
        assertEquals(1, matches.size, "recognizer prerequisite")

        val coarsened = coarsenRecognizedPatterns(fn, matches)

        val ops = coarsened.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size, "expected one body op (the COARSENED); got ${ops.map { it.op }}")
        val co = ops.single()
        assertEquals(OpKind.COARSENED, co.op)
        assertEquals(2, co.operands.size, "COARSENED takes (logits, labels)")
        assertEquals(scalarType, co.types.single())
        assertEquals(co.id, coarsened.returns.single().id)
    }

    @Test
    fun coarsenedOpHasCrossEntropyPrimalAndGradientBodies() {
        val fn = buildCrossEntropyFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeCrossEntropy(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()

        val primal = co.attrs["primal_body"]
        assertNotNull(primal)
        primal as DxirFunction
        assertEquals("cross_entropy_primal", primal.name)
        assertEquals(2, primal.params.size, "primal takes (logits, labels)")
        assertEquals(logitsType, primal.params[0].type)
        assertEquals(labelsType, primal.params[1].type)
        assertEquals(1, primal.returns.size)
        assertEquals(scalarType, primal.returns.single().type)

        val grad = co.attrs["gradient_body"]
        assertNotNull(grad)
        grad as DxirFunction
        assertEquals("cross_entropy_grad", grad.name)
        // K=1 (single-result COARSENED) + N=2 (logits, labels) = 3 params.
        assertEquals(3, grad.params.size)
        assertEquals(scalarType, grad.params[0].type, "param[0] is upstream d_loss (scalar)")
        assertEquals(logitsType, grad.params[1].type, "param[1] is logits")
        assertEquals(labelsType, grad.params[2].type, "param[2] is labels")
        assertEquals(2, grad.returns.size, "returns (d_logits, d_labels)")
        assertEquals(logitsType, grad.returns[0].type)
        assertEquals(labelsType, grad.returns[1].type)
    }

    @Test
    fun gradientBodyForLabelsIsDLossTimesLogSoftmax() {
        // §0.4.292 — the prior shortcut "d_labels = const(0)" produced bit-exact
        // disagreement with PyTorch's torch.autograd.grad. Coarsener now emits
        // the analytical d_labels = d_loss · log(softmax(logits)).
        //
        // Structurally that's a MUL whose operands are (dLossBroadcast, logp),
        // where logp = LOG(SOFTMAX(logits)). The probs/SOFTMAX are already
        // recomputed for the d_logits arm and reused here.
        val fn = buildCrossEntropyFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeCrossEntropy(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()
        val grad = co.attrs["gradient_body"] as DxirFunction

        val dLabelsRet = grad.returns[1]
        assertTrue(
            dLabelsRet is DxirOp && dLabelsRet.op == OpKind.MUL,
            "d_labels should be a MUL, not a const-zero shortcut; got $dLabelsRet",
        )
        assertEquals(labelsType, dLabelsRet.type)

        // Pin the analytical structure: d_labels = MUL(dLossBroadcast, LOG_SOFTMAX(logits)),
        // and a SOFTMAX recompute for the d_logits arm.
        val logs = grad.body.filterIsInstance<DxirOp>().count { it.op == OpKind.LOG_SOFTMAX }
        val softmaxes = grad.body.filterIsInstance<DxirOp>().count { it.op == OpKind.SOFTMAX }
        assertTrue(logs >= 1, "expected a LOG_SOFTMAX in the gradient body for d_labels = logSoftmax")
        assertTrue(softmaxes >= 1, "expected at least one SOFTMAX recompute in the gradient body")
    }

    @Test
    fun gradientBodyBroadcastsScalarUpstream() {
        // Scalar d_loss must be broadcast to logits' shape before the
        // MUL with (labels − softmax(logits)). Pinning the BROADCAST op +
        // its empty broadcast_dimensions catches regressions where the
        // scalar→rank-N broadcast is skipped.
        val fn = buildCrossEntropyFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeCrossEntropy(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()
        val grad = co.attrs["gradient_body"] as DxirFunction

        val broadcasts = grad.body.filterIsInstance<DxirOp>()
            .filter { it.op == OpKind.BROADCAST && it.operands.single().id == grad.params[0].id }
        assertEquals(1, broadcasts.size, "exactly one BROADCAST of the scalar d_loss → logits shape")
        val bcast = broadcasts.single()
        assertEquals(logitsType, bcast.type)
        @Suppress("UNCHECKED_CAST")
        val dims = bcast.attrs["broadcast_dimensions"] as List<Int>
        assertTrue(dims.isEmpty(), "scalar broadcast uses empty broadcast_dimensions; got $dims")
    }

    @Test
    fun readsPrimalIndicesIncludesLogitsAndLabels() {
        val fn = buildCrossEntropyFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeCrossEntropy(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()

        @Suppress("UNCHECKED_CAST")
        val reads = co.attrs["reads_primal_indices"] as Set<Int>
        // Gradient body recomputes softmax(logits) (param idx 0) and
        // computes (labels − softmax) (param idx 1). Both are dereferenced.
        assertEquals(setOf(0, 1), reads, "gradient references both logits and labels")
    }

    private fun ceGradient(rows: Int, classes: Int, fused: Boolean): DxirFunction {
        val t = DxirType(F32, listOf(rows, classes))
        val fn = DxirBuilder.function("nll") {
            val logits = param("logits", t)
            val labels = param("labels", t)
            val logp = if (fused) {
                op(OpKind.LOG_SOFTMAX, listOf(logits), t, attrs = mapOf("axis" to 1))
            } else {
                op(OpKind.LOG, listOf(op(OpKind.SOFTMAX, listOf(logits), t, attrs = mapOf("axis" to 1))), t)
            }
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(labels, logp), t)), scalarType))
        }
        val co = coarsenRecognizedPatterns(fn, recognizeCrossEntropy(fn)).body.filterIsInstance<DxirOp>().single()
        assertEquals(OpKind.COARSENED, co.op)
        return co.attrs["gradient_body"] as DxirFunction
    }

    @Test
    fun theLogSoftmaxFormCoarsensToo() {
        val t = DxirType(F32, listOf(2, 3))
        val fn = DxirBuilder.function("nll") {
            val logits = param("logits", t)
            val labels = param("labels", t)
            val logp = op(OpKind.LOG_SOFTMAX, listOf(logits), t, attrs = mapOf("axis" to 1))
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(labels, logp), t)), scalarType))
        }
        val match = recognizeCrossEntropy(fn).single()
        assertEquals(listOf(OpKind.LOG_SOFTMAX, OpKind.MUL, OpKind.SUM), match.ops.map { it.op })
        val primal = coarsenRecognizedPatterns(fn, listOf(match)).body.filterIsInstance<DxirOp>().single()
            .attrs["primal_body"] as DxirFunction
        assertTrue(primal.body.filterIsInstance<DxirOp>().any { it.op == OpKind.LOG_SOFTMAX })
    }

    @Test
    fun gradientOfAConfidentRowKeepsTheTargetEntry() {
        // d/dlogits Σ labels ⊙ logSoftmax = labels − softmax: at the target of
        // [9, −9, −9], 2e/(1 + 2e) with e = exp(−18), which `labels − softmax` rounds to 0.
        for (fused in listOf(true, false)) {
            val out = io.tlaloc.ir.passes.DxirInterpreter.evalFunction(
                ceGradient(1, 3, fused),
                listOf(floatArrayOf(1f), floatArrayOf(9f, -9f, -9f), floatArrayOf(1f, 0f, 0f)),
            )
            val e = kotlin.math.exp(-18.0)
            val want = doubleArrayOf(2 * e / (1 + 2 * e), -e / (1 + 2 * e), -e / (1 + 2 * e))
            for (k in 0..2) {
                val err = kotlin.math.abs(out[0][k] - want[k]) / kotlin.math.abs(want[k])
                assertTrue(err < 1e-5, "fused=$fused d_logits[$k] = ${out[0][k]} against ${want[k]}")
            }
            assertTrue(kotlin.math.abs(out[1][0] + 3.046e-8f) < 1e-11f, "d_labels[0] = logSoftmax = ${out[1][0]}")
        }
    }

    @Test
    fun gradientDoesNotAssumeLabelsSumToOne() {
        // Labels of row mass 2: d_logits = labels − softmax · 2.
        val x = floatArrayOf(0.5f, -0.3f, 0.1f)
        val labels = floatArrayOf(1.5f, 0.5f, 0f)
        val out = io.tlaloc.ir.passes.DxirInterpreter.evalFunction(
            ceGradient(1, 3, fused = true), listOf(floatArrayOf(1f), x, labels),
        )
        val m = x.maxOf { it.toDouble() }
        val z = x.sumOf { kotlin.math.exp(it - m) }
        for (k in 0..2) {
            val want = labels[k] - 2 * kotlin.math.exp(x[k] - m) / z
            assertTrue(kotlin.math.abs(out[0][k] - want) < 1e-6, "d_logits[$k] = ${out[0][k]} against $want")
        }
    }
}
