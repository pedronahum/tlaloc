package io.tlaloc.runtime.pjrt.ffm

import io.tlaloc.runtime.pjrt.PjrtBinaries
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.ValueLayout.JAVA_LONG
import java.util.concurrent.atomic.AtomicReference
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * KPTX v1.3 (§0.4.329) — [XlaFfi] call-frame decoder tests.
 *
 * 1. [decodesSyntheticCallFrame] — a hand-built `XLA_FFI_CallFrame` in an
 *    Arena, byte-for-byte per the jaxlib-0.10.0 layouts documented in
 *    [XlaFfi]. Runs everywhere (no GPU): pins the offset arithmetic.
 * 2. [decodesLiveCallFrameFromXlaExecution] — registers a probe handler,
 *    runs a real custom_call through the CUDA plugin, and asserts the
 *    decoded args/rets shapes + dtype + non-null device pointers + a
 *    non-null CUstream from `XLA_FFI_Stream_Get`. GPU-gated.
 */
class XlaFfiCallFrameTest {

    // ---------------------------------------------------------------------
    // 1. Synthetic frame.
    // ---------------------------------------------------------------------

    private fun byteSpan(arena: Arena, text: String): MemorySegment {
        val bytes = text.toByteArray()
        val data = arena.allocate(bytes.size.toLong().coerceAtLeast(1))
        for ((i, b) in bytes.withIndex()) data.set(java.lang.foreign.ValueLayout.JAVA_BYTE, i.toLong(), b)
        val span = arena.allocate(16)
        span.set(ADDRESS, 0L, data)
        span.set(JAVA_LONG, 8L, bytes.size.toLong())
        return span
    }

    private fun buffer(arena: Arena, dtype: Int, dataAddress: Long, dims: LongArray): MemorySegment {
        val dimsSeg = arena.allocate((dims.size * 8L).coerceAtLeast(8))
        for ((i, d) in dims.withIndex()) dimsSeg.set(JAVA_LONG, i * 8L, d)
        val buf = arena.allocate(48)
        buf.set(JAVA_LONG, 0L, 48L)                       // struct_size
        buf.set(ADDRESS, 8L, MemorySegment.NULL)          // extension_start
        buf.set(JAVA_INT, XlaFfi.OFF_BUF_DTYPE, dtype)
        buf.set(ADDRESS, XlaFfi.OFF_BUF_DATA, MemorySegment.ofAddress(dataAddress))
        buf.set(JAVA_LONG, XlaFfi.OFF_BUF_RANK, dims.size.toLong())
        buf.set(ADDRESS, XlaFfi.OFF_BUF_DIMS, dimsSeg)
        return buf
    }

    @Test
    fun decodesSyntheticCallFrame() {
        Arena.ofConfined().use { arena ->
            val frame = arena.allocate(XlaFfi.FRAME_MIN_SIZE)
            frame.set(JAVA_LONG, 0L, XlaFfi.FRAME_MIN_SIZE)
            frame.set(JAVA_INT, XlaFfi.OFF_FRAME_STAGE, XlaFfi.STAGE_EXECUTE)

            // Two f32 args (2x3 and scalar), one f32 ret (2x3).
            val arg0 = buffer(arena, XlaFfi.DTYPE_F32, 0xA000L, longArrayOf(2, 3))
            val arg1 = buffer(arena, XlaFfi.DTYPE_F32, 0xB000L, LongArray(0))
            val ret0 = buffer(arena, XlaFfi.DTYPE_F32, 0xC000L, longArrayOf(2, 3))

            val argTypes = arena.allocate(8).also { it.set(JAVA_INT, 0, XlaFfi.ARG_TYPE_BUFFER); it.set(JAVA_INT, 4, XlaFfi.ARG_TYPE_BUFFER) }
            val argValues = arena.allocate(16).also { it.set(ADDRESS, 0, arg0); it.set(ADDRESS, 8, arg1) }
            frame.set(JAVA_LONG, XlaFfi.OFF_FRAME_ARGS + XlaFfi.OFF_SEQ_SIZE, 2L)
            frame.set(ADDRESS, XlaFfi.OFF_FRAME_ARGS + XlaFfi.OFF_SEQ_TYPES, argTypes)
            frame.set(ADDRESS, XlaFfi.OFF_FRAME_ARGS + XlaFfi.OFF_SEQ_VALUES, argValues)

            val retTypes = arena.allocate(4).also { it.set(JAVA_INT, 0, XlaFfi.ARG_TYPE_BUFFER) }
            val retValues = arena.allocate(8).also { it.set(ADDRESS, 0, ret0) }
            frame.set(JAVA_LONG, XlaFfi.OFF_FRAME_RETS + XlaFfi.OFF_SEQ_SIZE, 1L)
            frame.set(ADDRESS, XlaFfi.OFF_FRAME_RETS + XlaFfi.OFF_SEQ_TYPES, retTypes)
            frame.set(ADDRESS, XlaFfi.OFF_FRAME_RETS + XlaFfi.OFF_SEQ_VALUES, retValues)

            // Attrs: eps = i64 7, kernel = "rms_norm".
            val epsValue = arena.allocate(8).also { it.set(JAVA_LONG, 0, 7L) }
            val epsScalar = arena.allocate(16).also {
                it.set(JAVA_INT, 0, XlaFfi.DTYPE_S64)
                it.set(ADDRESS, 8, epsValue)
            }
            val attrTypes = arena.allocate(8).also { it.set(JAVA_INT, 0, XlaFfi.ATTR_TYPE_SCALAR); it.set(JAVA_INT, 4, XlaFfi.ATTR_TYPE_STRING) }
            val attrNames = arena.allocate(16).also { it.set(ADDRESS, 0, byteSpan(arena, "eps")); it.set(ADDRESS, 8, byteSpan(arena, "kernel")) }
            val attrValues = arena.allocate(16).also { it.set(ADDRESS, 0, epsScalar); it.set(ADDRESS, 8, byteSpan(arena, "rms_norm")) }
            frame.set(JAVA_LONG, XlaFfi.OFF_FRAME_ATTRS + XlaFfi.OFF_SEQ_SIZE, 2L)
            frame.set(ADDRESS, XlaFfi.OFF_FRAME_ATTRS + XlaFfi.OFF_SEQ_TYPES, attrTypes)
            frame.set(ADDRESS, XlaFfi.OFF_FRAME_ATTRS + XlaFfi.OFF_ATTRS_NAMES, attrNames)
            frame.set(ADDRESS, XlaFfi.OFF_FRAME_ATTRS + XlaFfi.OFF_ATTRS_VALUES, attrValues)

            val decoded = XlaFfi.decode(frame)
            assertEquals(XlaFfi.STAGE_EXECUTE, decoded.stage)
            assertEquals(
                listOf(
                    XlaFfi.Buffer(XlaFfi.DTYPE_F32, 0xA000L, longArrayOf(2, 3)),
                    XlaFfi.Buffer(XlaFfi.DTYPE_F32, 0xB000L, LongArray(0)),
                ),
                decoded.args,
            )
            assertEquals(listOf(XlaFfi.Buffer(XlaFfi.DTYPE_F32, 0xC000L, longArrayOf(2, 3))), decoded.rets)
            assertEquals(6L, decoded.args[0].elementCount)
            assertEquals(1L, decoded.args[1].elementCount)
            assertEquals(
                mapOf<String, XlaFfi.AttrValue>(
                    "eps" to XlaFfi.AttrValue.I64(7L),
                    "kernel" to XlaFfi.AttrValue.Str("rms_norm"),
                ),
                decoded.attrs,
            )
        }
    }

    // ---------------------------------------------------------------------
    // 2. Live frame from a real XLA execution.
    // ---------------------------------------------------------------------

    private data class Probe(
        val stage: Int,
        val args: List<XlaFfi.Buffer>,
        val rets: List<XlaFfi.Buffer>,
        val streamAddress: Long,
    )

    @Test
    fun decodesLiveCallFrameFromXlaExecution() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val pluginPath = PjrtBinaries.pluginPath!!
        assumeTrue(PjrtFfiRegistry.isGpuCustomCallSupported(pluginPath), "no GPU custom-call extension — skipping.")

        val probe = AtomicReference<Probe>()
        PjrtFfiRegistry.registerExecuteHandler(pluginPath, "tlaloc_kptx_frame_probe") { framePtr ->
            val f = XlaFfi.decode(framePtr)
            probe.set(Probe(f.stage, f.args, f.rets, f.streamGet().address()))
            MemorySegment.NULL
        }

        val mlir = """
            func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
              %0 = stablehlo.custom_call @tlaloc_kptx_frame_probe(%arg0, %arg1) {api_version = 4 : i32} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
              return %0 : tensor<2x3xf32>
            }
        """.trimIndent()

        Arena.ofShared().use { arena ->
            PjrtFfm.load(pluginPath, arena).createClient().use { client ->
                val device = client.addressableDevices().first()
                client.compile(mlir).use { exec ->
                    client.bufferFromHostF32(device, FloatArray(6) { it.toFloat() }, listOf(2, 3)).use { a ->
                        client.bufferFromHostF32(device, FloatArray(6) { it * 2f }, listOf(2, 3)).use { b ->
                            exec.execute(listOf(a, b), device).forEach { it.close() }
                        }
                    }
                }
            }
        }

        val seen = probe.get()
        checkNotNull(seen) { "probe handler was never invoked" }
        assertEquals(XlaFfi.STAGE_EXECUTE, seen.stage)
        assertEquals(2, seen.args.size, "two input buffers")
        assertEquals(1, seen.rets.size, "one output buffer")
        seen.args.forEach { buf ->
            assertEquals(XlaFfi.DTYPE_F32, buf.dtype)
            assertTrue(buf.dims.contentEquals(longArrayOf(2, 3)), "arg dims ${buf.dims.toList()}")
            assertTrue(buf.dataAddress != 0L, "device pointer must be non-null")
        }
        assertTrue(seen.rets[0].dims.contentEquals(longArrayOf(2, 3)), "ret dims ${seen.rets[0].dims.toList()}")
        assertTrue(seen.rets[0].dataAddress != 0L, "output device pointer must be non-null")
        assertTrue(seen.streamAddress != 0L, "XLA_FFI_Stream_Get must yield a CUstream")
        println(
            "[kptx-frame-probe] live frame decoded: args=${seen.args.map { it.dims.toList() }}, " +
                "rets=${seen.rets.map { it.dims.toList() }}, stream=0x${seen.streamAddress.toString(16)}",
        )
    }
}
