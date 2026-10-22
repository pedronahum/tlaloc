package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.MoeExpertsAttrs
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.runtime.pjrt.ffm.PjrtFfiRegistry
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * MOE_EXPERTS routed by the CUDA kernel `tlaloc_moe_route` (`route_kernel = true`), against the
 * same op routed by the sort and against the interpreter: softmax and sigmoid + bias routing, the
 * gathered form (decode rows) and the tiled loop (prefill rows), 64 and 256 experts. Logits drawn
 * from a few values make ties, which both forms break toward the lower expert.
 */
class PjrtMoeRouteTest {

    private val library = Path.of("..", "triton", "backends", "tlaloc", "libtlaloc_kernels.so").toAbsolutePath().normalize()

    private fun registered(): Boolean {
        if (!TestBackend.deviceAvailable || TestBackend.isTpu || TestBackend.isMetal || !Files.isRegularFile(library)) return false
        PjrtFfiRegistry.registerNativeLibrary(PjrtBinaries.requireCudaPlugin("PjrtMoeRouteTest"), library)
        return true
    }

    private fun fn(rows: Int, h: Int, e: Int, inter: Int, k: Int, sigmoid: Boolean, kernel: Boolean): DxirFunction =
        DxirBuilder.function("moe_route") {
            val x = param("x", DxirType(F32, listOf(rows, h)))
            val l = param("logits", DxirType(F32, listOf(rows, e)))
            val gu = param("gateUp", DxirType(F32, listOf(e, 2 * inter, h)))
            val dn = param("down", DxirType(F32, listOf(e, h, inter)))
            val route = if (kernel) mapOf<String, Any>(MoeExpertsAttrs.ROUTE_KERNEL to true) else emptyMap()
            if (sigmoid) {
                val bias = param("bias", DxirType(F32, listOf(e)))
                listOf(
                    op(
                        OpKind.MOE_EXPERTS, listOf(x, l, gu, dn, bias), DxirType(F32, listOf(rows, h)),
                        mapOf("top_k" to k, "routing" to MoeExpertsAttrs.SIGMOID_BIAS, "routed_scale" to 2.5) + route,
                    ),
                )
            } else {
                listOf(op(OpKind.MOE_EXPERTS, listOf(x, l, gu, dn), DxirType(F32, listOf(rows, h)), mapOf("top_k" to k) + route))
            }
        }

    private fun case(rows: Int, e: Int, k: Int, sigmoid: Boolean, ties: Boolean) {
        val h = 64
        val inter = 32
        val kernel = fn(rows, h, e, inter, k, sigmoid, kernel = true)
        val sorted = fn(rows, h, e, inter, k, sigmoid, kernel = false)
        val text = kernel.toStablehlo()
        assertTrue("tlaloc_moe_route" in text && "stablehlo.sort" !in text.substringBefore("tlaloc_moe_route"), "the kernel routes")
        val rnd = Random(rows * 7 + e + if (sigmoid) 1 else 0)
        val ins = listOf(
            FloatArray(rows * h) { rnd.nextFloat() - 0.5f },
            FloatArray(rows * e) { if (ties) rnd.nextInt(4) * 0.5f else 3f * (rnd.nextFloat() - 0.5f) },
            FloatArray(e * 2 * inter * h) { (rnd.nextFloat() - 0.5f) * 0.4f },
            FloatArray(e * h * inter) { (rnd.nextFloat() - 0.5f) * 0.4f },
        ) + if (sigmoid) listOf(FloatArray(e) { if (ties) 0f else (rnd.nextFloat() - 0.5f) * 0.5f }) else emptyList()
        val want = DxirInterpreter.evalFunction(kernel, ins)[0]
        TestBackend.session().use { s ->
            val got = s.runOn(kernel, ins)[0]
            val bySort = s.runOn(sorted, ins)[0]
            var worst = 0f
            var apart = 0f
            for (j in want.indices) {
                worst = maxOf(worst, abs(got[j] - want[j]) / maxOf(0.05f, abs(want[j])))
                apart = maxOf(apart, abs(got[j] - bySort[j]) / maxOf(0.05f, abs(bySort[j])))
            }
            println(
                "[moe-route] R=$rows E=$e k=$k ${if (sigmoid) "sigmoid+bias" else "softmax"}${if (ties) " ties" else ""}: " +
                    "against the interpreter $worst, against the sort $apart",
            )
            assertTrue(worst <= 1e-4f, "against the interpreter: $worst")
            assertTrue(apart <= 1e-5f, "against the sort: $apart")
        }
    }

    @Test
    fun theKernelRoutesAsTheSort() {
        assumeTrue(registered(), "no CUDA device or no libtlaloc_kernels.so (triton/build_backend.sh)")
        for (sigmoid in listOf(false, true)) {
            for (ties in listOf(false, true)) {
                case(rows = 4, e = 64, k = 4, sigmoid = sigmoid, ties = ties)
                case(rows = 300, e = 64, k = 8, sigmoid = sigmoid, ties = ties)
            }
        }
        case(rows = 40, e = 256, k = 8, sigmoid = false, ties = false)
    }
}
