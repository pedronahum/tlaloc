package io.tlaloc.runtime.pjrt.kptx

import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.ffm.PjrtFfiRegistry
import io.tlaloc.runtime.pjrt.ffm.PjrtFfm
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.lang.foreign.Arena
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * KPTX v1.4 (§0.4.330) — the first **computing** KPTX dispatch: a
 * hand-written PTX kernel registered via [KptxKernelRegistry], launched
 * by XLA inside a compiled PJRT executable, writing into XLA's output
 * buffer, with results copied back to host and asserted exactly
 * (`x + 1.0f` is representable in f32 — bit-exact expectation is safe).
 *
 * Composes every §0.4.326–330 piece: FFI registration (spike) →
 * driver-JIT + launch (CudaDriverFfm) → call-frame decode (XlaFfi) →
 * launch registry marshalling inputs-then-outputs on XLA's stream.
 */
class KptxKernelRegistryTest {

    private val addOnePtx = """
        .version 7.0
        .target sm_75
        .address_size 64

        .visible .entry kptx_add_one(
            .param .u64 in_ptr,
            .param .u64 out_ptr,
            .param .u32 n
        )
        {
            .reg .pred %p<2>;
            .reg .b32 %r<6>;
            .reg .f32 %f<3>;
            .reg .b64 %rd<8>;

            ld.param.u64 %rd1, [in_ptr];
            ld.param.u64 %rd2, [out_ptr];
            ld.param.u32 %r1, [n];
            cvta.to.global.u64 %rd3, %rd1;
            cvta.to.global.u64 %rd4, %rd2;
            mov.u32 %r2, %ctaid.x;
            mov.u32 %r3, %ntid.x;
            mov.u32 %r4, %tid.x;
            mad.lo.s32 %r5, %r2, %r3, %r4;
            setp.ge.s32 %p1, %r5, %r1;
            @%p1 bra DONE;
            mul.wide.s32 %rd5, %r5, 4;
            add.s64 %rd6, %rd3, %rd5;
            ld.global.f32 %f1, [%rd6];
            add.f32 %f2, %f1, 0f3F800000;
            add.s64 %rd7, %rd4, %rd5;
            st.global.f32 [%rd7], %f2;
        DONE:
            ret;
        }
    """.trimIndent()

    @Test
    fun kptxKernelComputesInsideXlaExecutable() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val pluginPath = PjrtBinaries.pluginPath!!
        assumeTrue(PjrtFfiRegistry.isGpuCustomCallSupported(pluginPath), "no GPU custom-call extension — skipping.")

        val block = 256
        KptxKernelRegistry.registerKernel(
            pluginPath,
            "kptx_add_one",
            KptxKernelRegistry.LaunchConfig(
                ptx = addOnePtx,
                entryName = "kptx_add_one",
                grid = { args ->
                    val n = args[0].elementCount.toInt()
                    KptxKernelRegistry.Dim3((n + block - 1) / block)
                },
                block = KptxKernelRegistry.Dim3(block),
                trailingI32Params = { args -> intArrayOf(args[0].elementCount.toInt()) },
            ),
        )

        // Duplicate registration fails fast.
        assertFailsWith<IllegalStateException> {
            KptxKernelRegistry.registerKernel(
                pluginPath, "kptx_add_one",
                KptxKernelRegistry.LaunchConfig(addOnePtx, "kptx_add_one", { KptxKernelRegistry.Dim3(1) }, KptxKernelRegistry.Dim3(1)),
            )
        }

        val n = 1024
        val mlir = """
            func.func @main(%arg0: tensor<${n}xf32>) -> tensor<${n}xf32> {
              %0 = stablehlo.custom_call @kptx_add_one(%arg0) {api_version = 4 : i32} : (tensor<${n}xf32>) -> tensor<${n}xf32>
              return %0 : tensor<${n}xf32>
            }
        """.trimIndent()

        val input = FloatArray(n) { it * 0.5f }
        Arena.ofShared().use { arena ->
            PjrtFfm.load(pluginPath, arena).createClient().use { client ->
                val device = client.addressableDevices().first()
                client.compile(mlir).use { exec ->
                    client.bufferFromHostF32(device, input, listOf(n)).use { inBuf ->
                        val outputs = exec.execute(listOf(inBuf), device)
                        try {
                            val result = outputs.single().toFloatArray(n)
                            for (i in 0 until n) {
                                assertEquals(input[i] + 1f, result[i], "element $i")
                            }
                        } finally {
                            outputs.forEach { it.close() }
                        }
                    }
                }
            }
        }
        println("[kptx-registry] kptx_add_one computed $n f32 elements inside an XLA executable — values exact")
    }
}
