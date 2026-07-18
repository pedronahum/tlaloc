package io.tlaloc.runtime.pjrt.ffm

import io.tlaloc.runtime.pjrt.PjrtBinaries
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.ValueLayout.JAVA_LONG
import java.lang.invoke.MethodHandles
import java.lang.invoke.MethodType
import java.nio.file.Path
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * KPTX Phase-2 feasibility spike — can an out-of-tree process register a
 * **typed-FFI custom-call handler** with the JAX-shipped `xla_cuda_plugin.so`
 * through the PJRT C API's GPU extension, and have an XLA-compiled program
 * dispatch into a **Kotlin upcall stub** at execute time?
 *
 * This is the load-bearing uncertainty for the KPTX kernel tier (the
 * pyptx-style "hand-written PTX kernels inside PJRT executables" plan):
 * everything else — `cuLaunchKernel` via FFM, `stablehlo.custom_call` emit
 * (§0.4.261), MLIR plumbing — is proven. What is *not* proven is that the
 * **stock JAX plugin** accepts external handler registration at all.
 *
 * # ABI facts this spike is built on
 *
 * - `PJRT_Api.extension_start` at offset 8 (pjrt_c_api.h:2930 prefix layout,
 *   same as [PjrtFfm]'s offset table). Extension chain nodes are
 *   `PJRT_Extension_Base { struct_size(0), type(int @8), next(@16) }`.
 * - `PJRT_Extension_Type_Gpu_Custom_Call = 0` (pjrt_c_api.h:62).
 * - `PJRT_Gpu_Custom_Call` extension struct: base(24 bytes) + fn ptr
 *   `custom_call` @24 (pjrt_c_api_gpu_extension.h, extension version 2).
 * - `PJRT_Gpu_Register_Custom_Call_Args` (64 bytes): struct_size(0),
 *   function_name(8), function_name_size(16), api_version(int @24 — 1 =
 *   typed FFI), handler_instantiate(32), handler_prepare(40),
 *   handler_initialize(48), handler_execute(56).
 * - XLA FFI handler: `XLA_FFI_Error* (*)(XLA_FFI_CallFrame*)`, NULL return
 *   = success. Call frame: struct_size(0), extension_start(8), api(16),
 *   ctx(24), stage(int @32; 3 = EXECUTE). Verified against the jaxlib-0.10.0
 *   **shipped** header `jaxlib/include/xla/ffi/api/c_api.h` (API 0.3).
 * - Metadata protocol: XLA may invoke the handler with an
 *   `XLA_FFI_Metadata_Extension` (type 1) in the frame's extension chain;
 *   the handler must fill `metadata->api_version` (major 0 / minor 3 for
 *   this jaxlib) and return before treating the frame as an execution.
 */
class PjrtCustomCallRegistrationSpikeTest {

    companion object {
        /** Execution stages observed by the handler (3 = EXECUTE). Written
         * from XLA's dispatch thread; read from the test thread. */
        private val observedStages = ConcurrentLinkedQueue<Int>()
        private val metadataQueries = AtomicInteger()

        /** Process-lifetime arena: the plugin's FFI registry holds the upcall
         * stub forever, so the stub (and the name bytes) must never be freed. */
        private val handlerArena: Arena = Arena.ofShared()

        // XLA_FFI_CallFrame field offsets (jaxlib 0.10.0 c_api.h).
        private const val OFF_FRAME_EXTENSION_START = 8L
        private const val OFF_FRAME_STAGE = 32L

        // XLA_FFI_Extension_Base: struct_size(0), type(int @8), next(@16).
        private const val OFF_EXT_TYPE = 8L
        private const val OFF_EXT_NEXT = 16L
        private const val XLA_FFI_EXTENSION_METADATA = 1

        // XLA_FFI_Metadata_Extension: extension_base(24 bytes) + metadata* @24.
        private const val OFF_METADATA_EXT_METADATA_PTR = 24L

        // XLA_FFI_Metadata: struct_size(0), api_version @8 (an inline
        // XLA_FFI_Api_Version { struct_size(0), extension_start(8),
        // major(int @16), minor(int @20) }), traits @32.
        private const val OFF_METADATA_API_MAJOR = 8L + 16L
        private const val OFF_METADATA_API_MINOR = 8L + 20L
        private const val XLA_FFI_API_MAJOR = 0
        private const val XLA_FFI_API_MINOR = 3

        /** The XLA_FFI_Handler the upcall stub points at. Must be @JvmStatic:
         * the Linker binds it via a MethodHandle to a plain static method. */
        @JvmStatic
        fun ffiHandlerExecute(framePtr: MemorySegment): MemorySegment {
            val frame = framePtr.reinterpret(128)
            // Metadata query? Walk the frame's extension chain first.
            var ext = frame.get(ADDRESS, OFF_FRAME_EXTENSION_START)
            while (ext.address() != 0L) {
                val node = ext.reinterpret(64)
                if (node.get(JAVA_INT, OFF_EXT_TYPE) == XLA_FFI_EXTENSION_METADATA) {
                    metadataQueries.incrementAndGet()
                    val metadata = node.get(ADDRESS, OFF_METADATA_EXT_METADATA_PTR).reinterpret(64)
                    metadata.set(JAVA_INT, OFF_METADATA_API_MAJOR, XLA_FFI_API_MAJOR)
                    metadata.set(JAVA_INT, OFF_METADATA_API_MINOR, XLA_FFI_API_MINOR)
                    return MemorySegment.NULL
                }
                ext = node.get(ADDRESS, OFF_EXT_NEXT)
            }
            observedStages.add(frame.get(JAVA_INT, OFF_FRAME_STAGE))
            return MemorySegment.NULL // NULL XLA_FFI_Error* = success.
        }
    }

    // PJRT_Extension_Base offsets (identical layout to the XLA_FFI one).
    private val extTypeOff = 8L
    private val extNextOff = 16L
    private val pjrtExtensionTypeGpuCustomCall = 0

    private data class ExtensionChain(val types: List<Int>, val gpuCustomCallExt: MemorySegment?)

    /** Re-resolves `GetPjrtApi` (dlopen refcount bump on the already-loaded
     * plugin — same `PJRT_Api*` [PjrtFfm.load] sees) and walks the extension
     * chain from offset 8. */
    private fun walkExtensionChain(pluginPath: Path, arena: Arena): ExtensionChain {
        val lookup = SymbolLookup.libraryLookup(pluginPath, arena)
        val getPjrtApi = PjrtFfm.LINKER.downcallHandle(
            lookup.find("GetPjrtApi").orElseThrow { error("plugin does not export GetPjrtApi") },
            FunctionDescriptor.of(ADDRESS),
        )
        val apiPtr = (getPjrtApi.invokeExact() as MemorySegment).reinterpret(PjrtFfm.PJRT_API_OBSERVED_SIZE)
        val types = mutableListOf<Int>()
        var gpuExt: MemorySegment? = null
        var ext = apiPtr.get(ADDRESS, 8L)
        while (ext.address() != 0L) {
            val node = ext.reinterpret(64)
            val type = node.get(JAVA_INT, extTypeOff)
            types += type
            if (type == pjrtExtensionTypeGpuCustomCall) gpuExt = node
            ext = node.get(ADDRESS, extNextOff)
        }
        return ExtensionChain(types, gpuExt)
    }

    /** Registers [handlerStub] as a typed-FFI (api_version=1) execute handler
     * under [name] via `PJRT_Gpu_Register_Custom_Call`. Throws with the
     * plugin's error message on failure. */
    private fun registerTypedFfiHandler(
        gpuExt: MemorySegment,
        name: String,
        handlerStub: MemorySegment,
        pluginPath: Path,
        arena: Arena,
    ) {
        val registerFnPtr = gpuExt.get(ADDRESS, 24L).reinterpret(Long.MAX_VALUE)
        val registerFn = PjrtFfm.LINKER.downcallHandle(registerFnPtr, FunctionDescriptor.of(ADDRESS, ADDRESS))

        val nameSeg = handlerArena.allocateFrom(name) // NUL-terminated UTF-8, process lifetime
        val args = arena.allocate(64)
        args.set(JAVA_LONG, 0L, 64L)                  // struct_size
        args.set(ADDRESS, 8L, nameSeg)                // function_name
        args.set(JAVA_LONG, 16L, name.length.toLong()) // function_name_size
        args.set(JAVA_INT, 24L, 1)                    // api_version = 1 (typed FFI)
        args.set(ADDRESS, 32L, MemorySegment.NULL)    // handler_instantiate
        args.set(ADDRESS, 40L, MemorySegment.NULL)    // handler_prepare
        args.set(ADDRESS, 48L, MemorySegment.NULL)    // handler_initialize
        args.set(ADDRESS, 56L, handlerStub)           // handler_execute

        val errorPtr = registerFn.invokeExact(args) as MemorySegment
        if (errorPtr.address() != 0L) {
            throw IllegalStateException(
                "PJRT_Gpu_Register_Custom_Call failed: ${readPjrtError(pluginPath, errorPtr, arena)}",
            )
        }
    }

    /** Minimal PJRT_Error decode (message + destroy) using [PjrtFfm]'s
     * offset table — the test can't reach PjrtApi's private checkError. */
    private fun readPjrtError(pluginPath: Path, errorPtr: MemorySegment, arena: Arena): String {
        val lookup = SymbolLookup.libraryLookup(pluginPath, arena)
        val getPjrtApi = PjrtFfm.LINKER.downcallHandle(
            lookup.find("GetPjrtApi").orElseThrow(),
            FunctionDescriptor.of(ADDRESS),
        )
        val apiPtr = (getPjrtApi.invokeExact() as MemorySegment).reinterpret(PjrtFfm.PJRT_API_OBSERVED_SIZE)
        val errorFull = errorPtr.reinterpret(Long.MAX_VALUE)

        val msgFn = PjrtFfm.LINKER.downcallHandle(
            apiPtr.get(ADDRESS, PjrtFfm.OFFSET_PJRT_Error_Message).reinterpret(Long.MAX_VALUE),
            FunctionDescriptor.ofVoid(ADDRESS),
        )
        val msgArgs = arena.allocate(PjrtFfm.PJRT_Error_Message_Args_LAYOUT)
        msgArgs.set(JAVA_LONG, PjrtFfm.OFF_ErrorMessage_StructSize, PjrtFfm.SZ_ErrorMessage)
        msgArgs.set(ADDRESS, PjrtFfm.OFF_ErrorMessage_Error, errorFull)
        msgFn.invokeExact(msgArgs) as Unit
        val msgPtr = msgArgs.get(ADDRESS, PjrtFfm.OFF_ErrorMessage_Message).reinterpret(Long.MAX_VALUE)
        val msgSize = msgArgs.get(JAVA_LONG, PjrtFfm.OFF_ErrorMessage_MessageSize)
        val bytes = ByteArray(msgSize.toInt()) { msgPtr.get(java.lang.foreign.ValueLayout.JAVA_BYTE, it.toLong()) }

        val destroyFn = PjrtFfm.LINKER.downcallHandle(
            apiPtr.get(ADDRESS, PjrtFfm.OFFSET_PJRT_Error_Destroy).reinterpret(Long.MAX_VALUE),
            FunctionDescriptor.ofVoid(ADDRESS),
        )
        val destroyArgs = arena.allocate(PjrtFfm.PJRT_Error_Destroy_Args_LAYOUT)
        destroyArgs.set(JAVA_LONG, PjrtFfm.OFF_ErrorDestroy_StructSize, PjrtFfm.SZ_ErrorDestroy)
        destroyArgs.set(ADDRESS, PjrtFfm.OFF_ErrorDestroy_Error, errorFull)
        destroyFn.invokeExact(destroyArgs) as Unit

        return String(bytes, Charsets.UTF_8)
    }

    private fun customCallMlir(target: String): String = """
        func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
          %0 = stablehlo.custom_call @$target(%arg0) {api_version = 4 : i32} : (tensor<4xf32>) -> tensor<4xf32>
          return %0 : tensor<4xf32>
        }
    """.trimIndent()

    @Test
    fun registerTypedFfiHandlerAndDispatchThroughCompiledExecutable() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val pluginPath = PjrtBinaries.pluginPath!!

        Arena.ofShared().use { arena ->
            // 1. The plugin must expose the GPU custom-call extension at all.
            val chain = walkExtensionChain(pluginPath, arena)
            println("[kptx-spike] PJRT extension types exposed by $pluginPath: ${chain.types}")
            assumeTrue(
                chain.gpuCustomCallExt != null,
                "plugin exposes no PJRT_Gpu_Custom_Call extension (types=${chain.types}) — " +
                    "KPTX registration would need a self-built plugin.",
            )

            // 2. Register a Kotlin upcall stub as the typed-FFI execute handler.
            val handlerMh = MethodHandles.lookup().findStatic(
                PjrtCustomCallRegistrationSpikeTest::class.java,
                "ffiHandlerExecute",
                MethodType.methodType(MemorySegment::class.java, MemorySegment::class.java),
            )
            val handlerStub = PjrtFfm.LINKER.upcallStub(
                handlerMh, FunctionDescriptor.of(ADDRESS, ADDRESS), handlerArena,
            )
            registerTypedFfiHandler(chain.gpuCustomCallExt!!, "tlaloc_kptx_spike", handlerStub, pluginPath, arena)
            println("[kptx-spike] registered typed-FFI handler 'tlaloc_kptx_spike' (upcall stub at 0x${handlerStub.address().toString(16)})")

            // 3. Compile + execute a program whose only op is the custom call.
            PjrtFfm.load(pluginPath, arena).createClient().use { client ->
                val device = client.addressableDevices().first()

                // Negative control first: an unregistered symbol must be
                // rejected — proves the positive path below isn't vacuous.
                val rejected = assertFailsWith<PjrtRuntimeException> {
                    client.compile(customCallMlir("tlaloc_kptx_never_registered")).use { exec ->
                        client.bufferFromHostF32(device, floatArrayOf(1f, 2f, 3f, 4f), listOf(4)).use { input ->
                            exec.execute(listOf(input), device).forEach { it.close() }
                        }
                    }
                }
                println("[kptx-spike] negative control rejected as expected: ${rejected.message?.take(200)}")

                client.compile(customCallMlir("tlaloc_kptx_spike")).use { exec ->
                    client.bufferFromHostF32(device, floatArrayOf(1f, 2f, 3f, 4f), listOf(4)).use { input ->
                        val outputs = exec.execute(listOf(input), device)
                        outputs.forEach { it.close() }
                    }
                }
            }

            println(
                "[kptx-spike] handler observed stages=$observedStages " +
                    "(3 = EXECUTE), metadata queries=${metadataQueries.get()}",
            )
            assertTrue(
                observedStages.contains(3),
                "expected the registered handler to be invoked at EXECUTE stage (3); " +
                    "observed stages=$observedStages, metadata queries=${metadataQueries.get()}",
            )
        }
    }
}
