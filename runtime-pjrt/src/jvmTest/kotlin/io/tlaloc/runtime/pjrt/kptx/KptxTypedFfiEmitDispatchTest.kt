package io.tlaloc.runtime.pjrt.kptx

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.kernel.KernelDescriptor
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.KernelTemplate
import io.tlaloc.ir.recognizer.kernel.lowerKernelChoice
import io.tlaloc.ir.recognizer.recognizeFlashAttention
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.ffm.PjrtFfiRegistry
import io.tlaloc.runtime.pjrt.ffm.PjrtFfm
import io.tlaloc.runtime.pjrt.ffm.XlaFfi
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.util.concurrent.atomic.AtomicReference
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * KPTX v1.6 (§0.4.332) — the loop the §0.4.325 negative pin promised:
 * **Tlaloc dispatches the same MLIR shape JAX rejects.** A DXIR
 * function is coarsened (FlashAttention recognizer), annotated with a
 * `typedFfi = true` [KernelDescriptor], emitted by [toStablehlo] as a
 * typed-FFI `stablehlo.custom_call` — and that *emitter-produced* MLIR
 * compiles and dispatches into a registered Kotlin handler through the
 * stock JAX CUDA plugin. `JaxRejectsTlalocCustomCallMlirTest` pins that
 * JAX cannot compile custom-call-bearing Tlaloc MLIR; this test pins
 * that Tlaloc-PJRT can, closing L4's "same source, our runtime wins"
 * argument at the dispatch level (kernel *math* is pinned separately by
 * [KptxRmsNormKernelTest]).
 *
 * The probe handler asserts the typed-FFI attrs from the
 * `backend_config` dictionary arrive decoded on the call frame — proving
 * the whole attr pipeline: Kotlin map → MLIR dict emit → XLA compile →
 * XLA_FFI_Attrs → [XlaFfi] decode. (The JAX-lowering spelling
 * `mhlo.backend_config` compiles but delivers empty attrs through this
 * plugin's StableHLO import — hence the op-native dict form.)
 */
class KptxTypedFfiEmitDispatchTest {

    private val qType = DxirType(F32, listOf(8, 4))
    private val kType = DxirType(F32, listOf(4, 8))
    private val vType = DxirType(F32, listOf(8, 4))
    private val sType = DxirType(F32, listOf(8, 8))
    private val oType = DxirType(F32, listOf(8, 4))

    private fun emitTypedFfiMlir(): String {
        val raw = DxirBuilder.function("attn") {
            val q = param("Q", qType)
            val k = param("K", kType)
            val v = param("V", vType)
            val qk = op(OpKind.MATMUL, listOf(q, k), sType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), sType)
            val out = op(OpKind.MATMUL, listOf(sm, v), oType)
            listOf(out)
        }
        val coarsened = coarsenRecognizedPatterns(raw, recognizeFlashAttention(raw))
        val registry = mapOf<String, KernelTemplate>(
            "FlashAttention" to KernelTemplate { _, _ ->
                KernelDescriptor(
                    "kptx_e2e_probe", "tlaloc", "gb10", typedFfi = true,
                    customCallAttrs = mapOf("head_dim" to 4, "kernel_tag" to "probe"),
                )
            },
        )
        return lowerKernelChoice(coarsened, KernelTarget.NVIDIA_GB10, registry).toStablehlo()
    }

    private data class Seen(
        val stage: Int,
        val argDims: List<List<Long>>,
        val attrs: Map<String, XlaFfi.AttrValue>,
    )

    @Test
    fun emitterProducedTypedFfiMlirDispatchesThroughStockJaxPlugin() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val pluginPath = PjrtBinaries.pluginPath!!
        assumeTrue(PjrtFfiRegistry.isGpuCustomCallSupported(pluginPath), "no GPU custom-call extension — skipping.")

        val mlir = emitTypedFfiMlir()
        assertTrue(mlir.contains("stablehlo.custom_call @kptx_e2e_probe"), mlir)
        assertTrue(mlir.contains("api_version = 4 : i32"), mlir)
        assertTrue(
            mlir.contains("backend_config = {head_dim = 4 : i64, kernel_tag = \"probe\"}"),
            mlir,
        )

        val seen = AtomicReference<Seen>()
        PjrtFfiRegistry.registerExecuteHandler(pluginPath, "kptx_e2e_probe") { framePtr ->
            val f = XlaFfi.decode(framePtr)
            seen.set(Seen(f.stage, f.args.map { it.dims.toList() }, f.attrs))
            MemorySegment.NULL
        }

        Arena.ofShared().use { arena ->
            PjrtFfm.load(pluginPath, arena).createClient().use { client ->
                val device = client.addressableDevices().first()
                client.compile(mlir).use { exec ->
                    client.bufferFromHostF32(device, FloatArray(32) { it.toFloat() }, listOf(8, 4)).use { q ->
                        client.bufferFromHostF32(device, FloatArray(32) { it * 0.5f }, listOf(4, 8)).use { k ->
                            client.bufferFromHostF32(device, FloatArray(32) { it * 0.25f }, listOf(8, 4)).use { v ->
                                exec.execute(listOf(q, k, v), device).forEach { it.close() }
                            }
                        }
                    }
                }
            }
        }

        val s = seen.get()
        checkNotNull(s) { "probe handler was never invoked" }
        assertEquals(XlaFfi.STAGE_EXECUTE, s.stage)
        assertEquals(listOf(listOf(8L, 4L), listOf(4L, 8L), listOf(8L, 4L)), s.argDims)
        assertEquals(XlaFfi.AttrValue.I64(4L), s.attrs["head_dim"], "attrs=${s.attrs}")
        assertEquals(XlaFfi.AttrValue.Str("probe"), s.attrs["kernel_tag"], "attrs=${s.attrs}")
        println(
            "[kptx-e2e-emit] emitter-produced typed-FFI custom_call dispatched: " +
                "args=${s.argDims}, attrs=${s.attrs}",
        )
    }
}
