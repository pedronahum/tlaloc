package io.tlaloc.runtime.pjrt.ffm

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemoryLayout
import java.lang.foreign.MemoryLayout.PathElement.groupElement
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_BYTE
import java.lang.foreign.ValueLayout.JAVA_LONG
import java.lang.invoke.MethodHandle
import java.nio.file.Path

/**
 * §0.4.303 — hand-written FFM bindings for the OpenXLA PJRT C API
 * (xla/pjrt/c/pjrt_c_api.h, observed at API version 0.106 upstream;
 * the bundled `xla_cuda_plugin.so` reports 0.104, both ABI-compatible
 * for the surface this commit binds).
 *
 * **Why FFM, not JNI**: PJRT is a stable C ABI exposed via plugin shared
 * libraries (`xla_cuda_plugin.so`, `pjrt_plugin_xla_cpu.so`, …). FFM in
 * JDK 21 (preview) lets us bind directly from Kotlin — no C/C++ source,
 * no CMake, no JNI marshalling glue. JDK 22 (stable) is the migration
 * target after the bindings are working — see the §0.4.30N follow-up.
 *
 * # Kotlin + FFM gotcha (and why this file uses MemorySegment.set/get
 *   instead of VarHandles)
 *
 * VarHandle.set / get are *signature-polymorphic* methods. javac compiles
 * each call site with the actual argument types baked into the bytecode;
 * kotlinc emits a generic `invokevirtual VarHandle.set([Ljava/lang/Object;)V`
 * instead. At runtime the JVM raises WrongMethodTypeException because the
 * call site's signature doesn't match the VarHandle's typed view.
 *
 * The fix is to use [MemorySegment.set] / [MemorySegment.get] which take
 * a [ValueLayout] + byte offset + value as regular (non-polymorphic) method
 * arguments. We compute byte offsets once from each [MemoryLayout] and use
 * them as `Long` constants throughout. Cleaner than VarHandle juggling
 * anyway — and JDK 22 doesn't change this story.
 *
 * # PJRT_Api struct field offsets
 *
 * Hand-computed from `xla/pjrt/c/pjrt_c_api.h:2930`. The struct opens with
 * (struct_size: size_t, extension_start: ptr*, pjrt_api_version:
 * PJRT_Api_Version) where PJRT_Api_Version = (size_t struct_size + ptr*
 * extension_start + 2 ints) = 24 bytes. So the first function pointer is
 * at offset 8 + 8 + 24 = 40, and each subsequent fn-ptr is +8 bytes
 * (aarch64 / x86_64). v1 surface used here:
 *
 *   PJRT_Error_Destroy            offset 40
 *   PJRT_Error_Message            offset 48
 *   PJRT_Client_Create            offset 120
 *   PJRT_Client_Destroy           offset 128
 *   PJRT_Client_PlatformName      offset 136
 */
object PjrtFfm {

    /** Loads `xla_cuda_plugin.so` (or any PJRT plugin) and returns a [PjrtApi]
     * handle for downstream calls. Caller owns [arena] — pass `Arena.ofShared()`
     * for process-lifetime, `Arena.ofConfined()` for scoped use. */
    fun load(pluginPath: Path, arena: Arena): PjrtApi {
        val lookup = SymbolLookup.libraryLookup(pluginPath, arena)
        val getPjrtApiAddr = lookup.find("GetPjrtApi")
            .orElseThrow { error("PJRT plugin at $pluginPath does not export GetPjrtApi (not a PJRT-compatible shared library)") }
        val getPjrtApi: MethodHandle = LINKER.downcallHandle(
            getPjrtApiAddr,
            FunctionDescriptor.of(ADDRESS),
        )
        val pjrtApiPtr = (getPjrtApi.invokeExact() as MemorySegment).reinterpret(PJRT_API_OBSERVED_SIZE)
        return PjrtApi(pjrtApiPtr, arena)
    }

    // =========================================================================
    // Linker + low-level config.
    // =========================================================================

    internal val LINKER: Linker = Linker.nativeLinker()

    /** v1 covers function pointers up to PJRT_Client_PlatformName at offset
     * 136. We `reinterpret(256)` to cover the whole prefix safely; future
     * commits widen this as we wire more fields. */
    internal const val PJRT_API_OBSERVED_SIZE: Long = 256

    internal const val OFFSET_PJRT_Error_Destroy: Long = 40
    internal const val OFFSET_PJRT_Error_Message: Long = 48
    internal const val OFFSET_PJRT_Client_Create: Long = 120
    internal const val OFFSET_PJRT_Client_Destroy: Long = 128
    internal const val OFFSET_PJRT_Client_PlatformName: Long = 136

    // =========================================================================
    // Args struct layouts. Every Args struct opens with:
    //   struct_size: size_t   (set by caller to total struct size)
    //   extension_start: ptr* (NULL for v1; PJRT extensions are unused)
    // followed by per-call fields (some [in], some [out]).
    // =========================================================================

    internal val PJRT_Error_Destroy_Args_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("error"),
    )

    internal val PJRT_Error_Message_Args_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("error"),
        ADDRESS.withName("message"),
        JAVA_LONG.withName("message_size"),
    )

    internal val PJRT_Client_Create_Args_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("create_options"),
        JAVA_LONG.withName("num_options"),
        ADDRESS.withName("kv_get_callback"),
        ADDRESS.withName("kv_get_user_arg"),
        ADDRESS.withName("kv_put_callback"),
        ADDRESS.withName("kv_put_user_arg"),
        ADDRESS.withName("client"),
        ADDRESS.withName("kv_try_get_callback"),
        ADDRESS.withName("kv_try_get_user_arg"),
    )

    internal val PJRT_Client_Destroy_Args_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("client"),
    )

    internal val PJRT_Client_PlatformName_Args_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("client"),
        ADDRESS.withName("platform_name"),
        JAVA_LONG.withName("platform_name_size"),
    )

    // Field byte-offsets, computed once from each layout.

    internal val OFF_ClientCreate_StructSize: Long = PJRT_Client_Create_Args_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val OFF_ClientCreate_Client: Long = PJRT_Client_Create_Args_LAYOUT.byteOffset(groupElement("client"))
    internal val SZ_ClientCreate: Long = PJRT_Client_Create_Args_LAYOUT.byteSize()

    internal val OFF_ClientDestroy_StructSize: Long = PJRT_Client_Destroy_Args_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val OFF_ClientDestroy_Client: Long = PJRT_Client_Destroy_Args_LAYOUT.byteOffset(groupElement("client"))
    internal val SZ_ClientDestroy: Long = PJRT_Client_Destroy_Args_LAYOUT.byteSize()

    internal val OFF_PlatformName_StructSize: Long = PJRT_Client_PlatformName_Args_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val OFF_PlatformName_Client: Long = PJRT_Client_PlatformName_Args_LAYOUT.byteOffset(groupElement("client"))
    internal val OFF_PlatformName_Message: Long = PJRT_Client_PlatformName_Args_LAYOUT.byteOffset(groupElement("platform_name"))
    internal val OFF_PlatformName_MessageSize: Long = PJRT_Client_PlatformName_Args_LAYOUT.byteOffset(groupElement("platform_name_size"))
    internal val SZ_PlatformName: Long = PJRT_Client_PlatformName_Args_LAYOUT.byteSize()

    internal val OFF_ErrorDestroy_StructSize: Long = PJRT_Error_Destroy_Args_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val OFF_ErrorDestroy_Error: Long = PJRT_Error_Destroy_Args_LAYOUT.byteOffset(groupElement("error"))
    internal val SZ_ErrorDestroy: Long = PJRT_Error_Destroy_Args_LAYOUT.byteSize()

    internal val OFF_ErrorMessage_StructSize: Long = PJRT_Error_Message_Args_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val OFF_ErrorMessage_Error: Long = PJRT_Error_Message_Args_LAYOUT.byteOffset(groupElement("error"))
    internal val OFF_ErrorMessage_Message: Long = PJRT_Error_Message_Args_LAYOUT.byteOffset(groupElement("message"))
    internal val OFF_ErrorMessage_MessageSize: Long = PJRT_Error_Message_Args_LAYOUT.byteOffset(groupElement("message_size"))
    internal val SZ_ErrorMessage: Long = PJRT_Error_Message_Args_LAYOUT.byteSize()

    // Function descriptors for each downcall.
    internal val FD_ClientCreate: FunctionDescriptor = FunctionDescriptor.of(ADDRESS, ADDRESS)
    internal val FD_ClientDestroy: FunctionDescriptor = FunctionDescriptor.of(ADDRESS, ADDRESS)
    internal val FD_ClientPlatformName: FunctionDescriptor = FunctionDescriptor.of(ADDRESS, ADDRESS)
    internal val FD_ErrorDestroy: FunctionDescriptor = FunctionDescriptor.ofVoid(ADDRESS)
    internal val FD_ErrorMessage: FunctionDescriptor = FunctionDescriptor.ofVoid(ADDRESS)
}

/**
 * High-level handle wrapping a `PJRT_Api*`. Instances are produced by
 * [PjrtFfm.load]; [arena] is the lifetime owner.
 */
class PjrtApi internal constructor(
    private val pjrtApiPtr: MemorySegment,
    private val arena: Arena,
) {

    private val clientCreate = PjrtFfm.LINKER.downcallHandle(fnPtrAt(PjrtFfm.OFFSET_PJRT_Client_Create), PjrtFfm.FD_ClientCreate)
    private val clientDestroy = PjrtFfm.LINKER.downcallHandle(fnPtrAt(PjrtFfm.OFFSET_PJRT_Client_Destroy), PjrtFfm.FD_ClientDestroy)
    private val clientPlatformName = PjrtFfm.LINKER.downcallHandle(fnPtrAt(PjrtFfm.OFFSET_PJRT_Client_PlatformName), PjrtFfm.FD_ClientPlatformName)
    private val errorDestroy = PjrtFfm.LINKER.downcallHandle(fnPtrAt(PjrtFfm.OFFSET_PJRT_Error_Destroy), PjrtFfm.FD_ErrorDestroy)
    private val errorMessage = PjrtFfm.LINKER.downcallHandle(fnPtrAt(PjrtFfm.OFFSET_PJRT_Error_Message), PjrtFfm.FD_ErrorMessage)

    private fun fnPtrAt(offset: Long): MemorySegment =
        pjrtApiPtr.get(ADDRESS, offset).reinterpret(Long.MAX_VALUE)

    /** Create a PJRT client (one per process / device family). Throws
     * [PjrtRuntimeException] if the plugin returns a PJRT_Error*. */
    fun createClient(): PjrtClient {
        val args = arena.allocate(PjrtFfm.PJRT_Client_Create_Args_LAYOUT)
        args.set(JAVA_LONG, PjrtFfm.OFF_ClientCreate_StructSize, PjrtFfm.SZ_ClientCreate)
        val errorPtr = clientCreate.invokeExact(args) as MemorySegment
        checkError(errorPtr)
        val clientPtr = args.get(ADDRESS, PjrtFfm.OFF_ClientCreate_Client).reinterpret(Long.MAX_VALUE)
        require(clientPtr.address() != 0L) { "PJRT_Client_Create returned null client without an error" }
        return PjrtClient(clientPtr, this)
    }

    internal fun destroyClient(clientPtr: MemorySegment) {
        Arena.ofConfined().use { scoped ->
            val args = scoped.allocate(PjrtFfm.PJRT_Client_Destroy_Args_LAYOUT)
            args.set(JAVA_LONG, PjrtFfm.OFF_ClientDestroy_StructSize, PjrtFfm.SZ_ClientDestroy)
            args.set(ADDRESS, PjrtFfm.OFF_ClientDestroy_Client, clientPtr)
            val errorPtr = clientDestroy.invokeExact(args) as MemorySegment
            checkError(errorPtr)
        }
    }

    internal fun clientPlatformName(clientPtr: MemorySegment): String {
        Arena.ofConfined().use { scoped ->
            val args = scoped.allocate(PjrtFfm.PJRT_Client_PlatformName_Args_LAYOUT)
            args.set(JAVA_LONG, PjrtFfm.OFF_PlatformName_StructSize, PjrtFfm.SZ_PlatformName)
            args.set(ADDRESS, PjrtFfm.OFF_PlatformName_Client, clientPtr)
            val errorPtr = clientPlatformName.invokeExact(args) as MemorySegment
            checkError(errorPtr)
            val msgPtr = args.get(ADDRESS, PjrtFfm.OFF_PlatformName_Message).reinterpret(Long.MAX_VALUE)
            val msgSize = args.get(JAVA_LONG, PjrtFfm.OFF_PlatformName_MessageSize)
            return readUtf8(msgPtr, msgSize)
        }
    }

    /** Throws a [PjrtRuntimeException] with the plugin's error message if
     * [errorPtr] is non-null. Always destroys the error after reading. */
    private fun checkError(errorPtr: MemorySegment) {
        if (errorPtr.address() == 0L) return
        val errorPtrFull = errorPtr.reinterpret(Long.MAX_VALUE)
        val message = readErrorMessage(errorPtrFull)
        Arena.ofConfined().use { scoped ->
            val destroyArgs = scoped.allocate(PjrtFfm.PJRT_Error_Destroy_Args_LAYOUT)
            destroyArgs.set(JAVA_LONG, PjrtFfm.OFF_ErrorDestroy_StructSize, PjrtFfm.SZ_ErrorDestroy)
            destroyArgs.set(ADDRESS, PjrtFfm.OFF_ErrorDestroy_Error, errorPtrFull)
            errorDestroy.invokeExact(destroyArgs) as Unit
        }
        throw PjrtRuntimeException(message)
    }

    private fun readErrorMessage(errorPtr: MemorySegment): String {
        Arena.ofConfined().use { scoped ->
            val args = scoped.allocate(PjrtFfm.PJRT_Error_Message_Args_LAYOUT)
            args.set(JAVA_LONG, PjrtFfm.OFF_ErrorMessage_StructSize, PjrtFfm.SZ_ErrorMessage)
            args.set(ADDRESS, PjrtFfm.OFF_ErrorMessage_Error, errorPtr)
            errorMessage.invokeExact(args) as Unit
            val msgPtr = args.get(ADDRESS, PjrtFfm.OFF_ErrorMessage_Message).reinterpret(Long.MAX_VALUE)
            val msgSize = args.get(JAVA_LONG, PjrtFfm.OFF_ErrorMessage_MessageSize)
            return readUtf8(msgPtr, msgSize)
        }
    }

    private fun readUtf8(ptr: MemorySegment, size: Long): String {
        if (size <= 0L) return ""
        val bytes = ByteArray(size.toInt())
        for (i in 0 until size.toInt()) bytes[i] = ptr.get(JAVA_BYTE, i.toLong())
        return String(bytes, Charsets.UTF_8)
    }
}

/**
 * Handle to a live `PJRT_Client*`. Use [close] (or `use {}`) to free it via
 * `PJRT_Client_Destroy`. Forgetting to close leaks the GPU client.
 */
class PjrtClient internal constructor(
    private val clientPtr: MemorySegment,
    private val api: PjrtApi,
) : AutoCloseable {
    fun platformName(): String = api.clientPlatformName(clientPtr)
    override fun close() = api.destroyClient(clientPtr)
}

class PjrtRuntimeException(message: String) : RuntimeException(message)
