package io.tlaloc.runtime.pjrt

import io.tlaloc.core.BF16
import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * MOE_EXPERTS's StableHLO form (top-k by a sort, pairs sorted by expert,
 * a while over tiles of one expert's rows) on the device against the
 * interpreter, which MoeExpertsFixtureTest checks against transformers: a
 * decode-sized block and a prefill-sized one, with f32 weights and with bf16.
 */
class PjrtMoeExpertsTest {

    private fun case(rows: Int, h: Int, e: Int, inter: Int, k: Int, wdt: DType, tol: Float, sigmoid: Boolean = false) {
        // bf16 weights take x in bf16: the graph casts first, as a projection does.
        val fn = DxirBuilder.function("moe") {
            val x = param("x", DxirType(F32, listOf(rows, h)))
            val l = param("logits", DxirType(F32, listOf(rows, e)))
            val gu = param("gateUp", DxirType(F32, listOf(e, 2 * inter, h)))
            val dn = param("down", DxirType(F32, listOf(e, h, inter)))
            fun c(n: io.tlaloc.ir.DxirNode) = if (wdt == F32) n else op(OpKind.CAST, listOf(n), DxirType(wdt, n.type.dims))
            if (sigmoid) {
                // DeepSeek's routing: top k of sigmoid + bias, weights normalized and scaled.
                val bias = param("bias", DxirType(F32, listOf(e)))
                listOf(
                    op(
                        OpKind.MOE_EXPERTS, listOf(c(x), l, c(gu), c(dn), bias), DxirType(F32, listOf(rows, h)),
                        mapOf("top_k" to k, "routing" to io.tlaloc.ir.MoeExpertsAttrs.SIGMOID_BIAS, "routed_scale" to 2.5),
                    ),
                )
            } else {
                listOf(op(OpKind.MOE_EXPERTS, listOf(c(x), l, c(gu), c(dn)), DxirType(F32, listOf(rows, h)), mapOf("top_k" to k)))
            }
        }
        val rnd = Random(rows * 31 + e)
        val ins = listOf(
            FloatArray(rows * h) { rnd.nextFloat() - 0.5f },
            FloatArray(rows * e) { 3f * (rnd.nextFloat() - 0.5f) },
            FloatArray(e * 2 * inter * h) { (rnd.nextFloat() - 0.5f) * 0.4f },
            FloatArray(e * h * inter) { (rnd.nextFloat() - 0.5f) * 0.4f },
        ) + if (sigmoid) listOf(FloatArray(e) { (rnd.nextFloat() - 0.5f) * 0.5f }) else emptyList()
        val want = DxirInterpreter.evalFunction(fn, ins)[0]
        TestBackend.session().use { s ->
            val got = s.runOn(fn, ins)[0]
            var worst = 0f
            for (j in want.indices) worst = maxOf(worst, abs(got[j] - want[j]) / maxOf(0.05f, abs(want[j])))
            println("[pjrt-moe] R=$rows E=$e k=$k H=$h I=$inter ${wdt.name}${if (sigmoid) " sigmoid+bias" else ""}: worst relative error $worst")
            assertTrue(worst <= tol, "worst relative error $worst > $tol")
        }
    }

    @Test
    fun decodeSizedRowsMatchTheInterpreterOnTheDevice() {
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        case(rows = 4, h = 64, e = 16, inter = 32, k = 4, wdt = F32, tol = 1e-4f)
        case(rows = 4, h = 64, e = 16, inter = 32, k = 4, wdt = BF16, tol = 1e-2f)
    }

    @Test
    fun prefillSizedRowsMatchTheInterpreterOnTheDevice() {
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        case(rows = 300, h = 64, e = 16, inter = 32, k = 4, wdt = F32, tol = 1e-4f)
        case(rows = 300, h = 64, e = 16, inter = 32, k = 4, wdt = BF16, tol = 1e-2f)
    }

    @Test
    fun sigmoidRoutingWithABiasMatchesTheInterpreter() {
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        case(rows = 4, h = 64, e = 16, inter = 32, k = 4, wdt = F32, tol = 1e-4f, sigmoid = true)
        case(rows = 300, h = 64, e = 16, inter = 32, k = 4, wdt = F32, tol = 1e-4f, sigmoid = true)
    }
}
