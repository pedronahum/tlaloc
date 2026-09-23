package io.tlaloc.runtime.pjrt.ffm

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_BYTE
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.ValueLayout.JAVA_LONG
import java.lang.invoke.MethodHandles
import java.lang.invoke.MethodType
import java.nio.file.Path
import java.util.concurrent.ConcurrentHashMap

/**
 * KPTX v1.1 (§0.4.327) — out-of-tree **typed-FFI custom-call handler
 * registration** against a PJRT GPU plugin, in pure Kotlin FFM.
 *
 * Productionisation of the §0.4.326 spike
 * ([PjrtCustomCallRegistrationSpikeTest]): registers Kotlin upcall stubs
 * with the plugin's static XLA FFI registry so that XLA-compiled programs
 * containing `stablehlo.custom_call @<name> {api_version = 4 : i32}`
 * dispatch into Kotlin at execute time. Proven against the stock
 * JAX-shipped `xla_cuda_plugin.so` (jaxlib 0.10.0) — no C shim, no
 * self-built plugin.
 *
 * Registration is **process-global and permanent**: the plugin's FFI
 * registry keeps the handler function pointer forever, so upcall stubs
 * (and the name bytes passed to the plugin) are allocated in a
 * never-closed shared [Arena]. Register before [PjrtFfm.load]-driven
 * compilation of any program naming the target; order relative to client
 * creation is irrelevant (the registry is static inside the plugin .so).
 *
 * # ABI facts (same provenance as the §0.4.326 spike)
 *
 * - `PJRT_Api.extension_start` at offset 8 (matches [PjrtFfm]'s offset
 *   table). Chain nodes are `PJRT_Extension_Base { struct_size(0),
 *   type(int @8), next(@16) }`; `PJRT_Extension_Type_Gpu_Custom_Call = 0`.
 * - `PJRT_Gpu_Custom_Call` extension: base(24 bytes) + `custom_call` fn
 *   ptr @24 (`pjrt_c_api_gpu_extension.h`, GPU extension version 2).
 * - `PJRT_Gpu_Register_Custom_Call_Args` (64 bytes): struct_size(0),
 *   function_name(8), function_name_size(16), api_version(int @24; 1 =
 *   typed FFI), handler_instantiate(32) / prepare(40) / initialize(48) /
 *   execute(56).
 * - XLA FFI handler: `XLA_FFI_Error* (*)(XLA_FFI_CallFrame*)`; NULL
 *   return = success. Frame: struct_size(0), extension_start(8), api(16),
 *   ctx(24), stage(int @32; 3 = EXECUTE).
 * - Metadata protocol: XLA invokes the handler with an
 *   `XLA_FFI_Metadata_Extension` (chain type 1) before first use; the
 *   handler must write `metadata->api_version` (major 0 / minor 3, from
 *   the jaxlib-0.10.0-shipped `jaxlib/include/xla/ffi/api/c_api.h`) and
 *   return success. The registry answers this centrally — [FfiExecuteHandler]s
 *   only ever see genuine execution frames.
 */
object PjrtFfiRegistry {

    /** Handler for the EXECUTE stage of a typed-FFI custom call.
     * [callFrame] is the raw `XLA_FFI_CallFrame*` (decode via the KPTX
     * call-frame decoder once v1.3 lands; until then, hand offsets).
     * Return [MemorySegment.NULL] for success. Invoked on XLA's dispatch
     * thread — implementations must be thread-safe. Metadata queries are
     * answered by the registry and never reach this handler. */
    fun interface FfiExecuteHandler {
        fun execute(callFrame: MemorySegment): MemorySegment
    }

    // XLA_FFI_CallFrame / metadata-extension offsets (jaxlib 0.10.0).
    internal const val OFF_FRAME_EXTENSION_START = 8L
    internal const val OFF_FRAME_STAGE = 32L
    internal const val XLA_FFI_STAGE_EXECUTE = 3
    private const val OFF_EXT_TYPE = 8L
    private const val OFF_EXT_NEXT = 16L
    private const val XLA_FFI_EXTENSION_METADATA = 1
    private const val OFF_METADATA_EXT_METADATA_PTR = 24L
    private const val OFF_METADATA_API_MAJOR = 8L + 16L
    private const val OFF_METADATA_API_MINOR = 8L + 20L
    private const val XLA_FFI_API_MAJOR = 0
    private const val XLA_FFI_API_MINOR = 3

    // PJRT extension-chain offsets (pjrt_c_api.h).
    private const val OFF_PJRT_API_EXTENSION_START = 8L
    private const val PJRT_EXTENSION_TYPE_GPU_CUSTOM_CALL = 0
    private const val OFF_GPU_EXT_REGISTER_FN = 24L

    // PJRT_Gpu_Register_Custom_Call_Args (GPU extension version 2, 64 bytes).
    private const val SZ_REGISTER_ARGS = 64L
    private const val API_VERSION_TYPED_FFI = 1

    /** Never closed: the plugin's FFI registry references the stubs and
     * name bytes for the remaining process lifetime. */
    private val registryArena: Arena = Arena.ofShared()

    /** (pluginPath, name) pairs already registered in this process. The
     * plugin's registry rejects duplicates with ALREADY_EXISTS; this map
     * turns that into a clear Kotlin-side error before the native call. */
    private val registered = ConcurrentHashMap<Pair<Path, String>, FfiExecuteHandler>()

    /** Plugin `PJRT_Api*` cache, with the underlying `libraryLookup` pinned
     * in [registryArena]. **Load-bearing**: looking the plugin up through a
     * short-lived arena dlcloses it on arena close — if that drops the
     * dlopen refcount to zero, the plugin (and its static FFI registry,
     * including every handler registered so far) is torn down, and the next
     * load starts from an empty registry ("No FFI handler registered for
     * …"). The registry therefore holds the plugin open for the process
     * lifetime, matching the permanence of the registrations themselves. */
    private val apiPtrCache = ConcurrentHashMap<Path, MemorySegment>()

    /** Dispatcher bound per registration: answers metadata queries, routes
     * genuine frames to the delegate. Must be a class with a plain virtual
     * method — the Linker binds it via a MethodHandle. */
    internal class Dispatcher(private val delegate: FfiExecuteHandler) {
        fun dispatch(framePtr: MemorySegment): MemorySegment {
            val frame = framePtr.reinterpret(128)
            var ext = frame.get(ADDRESS, OFF_FRAME_EXTENSION_START)
            while (ext.address() != 0L) {
                val node = ext.reinterpret(64)
                if (node.get(JAVA_INT, OFF_EXT_TYPE) == XLA_FFI_EXTENSION_METADATA) {
                    val metadata = node.get(ADDRESS, OFF_METADATA_EXT_METADATA_PTR).reinterpret(64)
                    metadata.set(JAVA_INT, OFF_METADATA_API_MAJOR, XLA_FFI_API_MAJOR)
                    metadata.set(JAVA_INT, OFF_METADATA_API_MINOR, XLA_FFI_API_MINOR)
                    return MemorySegment.NULL
                }
                ext = node.get(ADDRESS, OFF_EXT_NEXT)
            }
            return delegate.execute(framePtr)
        }
    }

    private val dispatchMethod = MethodHandles.lookup().findVirtual(
        Dispatcher::class.java,
        "dispatch",
        MethodType.methodType(MemorySegment::class.java, MemorySegment::class.java),
    )

    /** The PJRT extension types exposed by the plugin at [pluginPath], in
     * chain order. Diagnostic companion to [isGpuCustomCallSupported]. */
    fun extensionTypes(pluginPath: Path): List<Int> {
        val types = mutableListOf<Int>()
        var ext = pjrtApiPtr(pluginPath).get(ADDRESS, OFF_PJRT_API_EXTENSION_START)
        while (ext.address() != 0L) {
            val node = ext.reinterpret(64)
            types += node.get(JAVA_INT, OFF_EXT_TYPE)
            ext = node.get(ADDRESS, OFF_EXT_NEXT)
        }
        return types
    }

    /** Whether the plugin exposes `PJRT_Gpu_Custom_Call` — the precondition
     * for [registerExecuteHandler]. False means KPTX registration would
     * need a self-built plugin on this host. */
    fun isGpuCustomCallSupported(pluginPath: Path): Boolean =
        extensionTypes(pluginPath).contains(PJRT_EXTENSION_TYPE_GPU_CUSTOM_CALL)

    /**
     * Registers [handler] as the typed-FFI EXECUTE handler for custom-call
     * target [name] on the plugin at [pluginPath] (platform CUDA for the
     * JAX GPU plugin). After this, XLA programs compiled by that plugin
     * may name `stablehlo.custom_call @<name> {api_version = 4 : i32}`.
     *
     * Permanent for the process lifetime; a second registration of the
     * same (plugin, name) throws [IllegalStateException]. Throws
     * [PjrtRuntimeException] with the plugin's message if the native
     * registration fails.
     */
    fun registerExecuteHandler(pluginPath: Path, name: String, handler: FfiExecuteHandler) {
        val key = pluginPath to name
        if (registered.putIfAbsent(key, handler) != null) {
            throw IllegalStateException(
                "custom-call target '$name' is already registered for $pluginPath " +
                    "(the plugin's FFI registry is process-global and permanent)",
            )
        }
        try {
            registerNative(pluginPath, name, handler)
        } catch (t: Throwable) {
            registered.remove(key)
            throw t
        }
    }

    private fun registerNative(pluginPath: Path, name: String, handler: FfiExecuteHandler) {
        Arena.ofConfined().use { arena ->
            val apiPtr = pjrtApiPtr(pluginPath)
            var gpuExt: MemorySegment? = null
            var ext = apiPtr.get(ADDRESS, OFF_PJRT_API_EXTENSION_START)
            while (ext.address() != 0L) {
                val node = ext.reinterpret(64)
                if (node.get(JAVA_INT, OFF_EXT_TYPE) == PJRT_EXTENSION_TYPE_GPU_CUSTOM_CALL) {
                    gpuExt = node
                    break
                }
                ext = node.get(ADDRESS, OFF_EXT_NEXT)
            }
            checkNotNull(gpuExt) {
                "plugin at $pluginPath exposes no PJRT_Gpu_Custom_Call extension " +
                    "(types=${extensionTypes(pluginPath)}) — cannot register '$name'"
            }
            val registerFn = PjrtFfm.LINKER.downcallHandle(
                gpuExt.get(ADDRESS, OFF_GPU_EXT_REGISTER_FN).reinterpret(Long.MAX_VALUE),
                FunctionDescriptor.of(ADDRESS, ADDRESS),
            )

            // Stub + name bytes outlive this call by design: registryArena.
            val stub = PjrtFfm.LINKER.upcallStub(
                dispatchMethod.bindTo(Dispatcher(handler)),
                FunctionDescriptor.of(ADDRESS, ADDRESS),
                registryArena,
            )
            val nameSeg = registryArena.allocateFrom(name)

            val args = arena.allocate(SZ_REGISTER_ARGS)
            args.set(JAVA_LONG, 0L, SZ_REGISTER_ARGS)
            args.set(ADDRESS, 8L, nameSeg)
            args.set(JAVA_LONG, 16L, name.length.toLong())
            args.set(JAVA_INT, 24L, API_VERSION_TYPED_FFI)
            args.set(ADDRESS, 32L, MemorySegment.NULL) // handler_instantiate
            args.set(ADDRESS, 40L, MemorySegment.NULL) // handler_prepare
            args.set(ADDRESS, 48L, MemorySegment.NULL) // handler_initialize
            args.set(ADDRESS, 56L, stub)               // handler_execute

            val errorPtr = registerFn.invokeExact(args) as MemorySegment
            if (errorPtr.address() != 0L) {
                throw PjrtRuntimeException(
                    "PJRT_Gpu_Register_Custom_Call('$name') failed: " +
                        readAndDestroyError(apiPtr, errorPtr, arena),
                )
            }
        }
    }

    /** Returns the same `PJRT_Api*` that [PjrtFfm.load] sees, with the
     * library pinned open in [registryArena] for the process lifetime (see
     * [apiPtrCache] for why this must never use a short-lived arena). */
    private fun pjrtApiPtr(pluginPath: Path): MemorySegment =
        apiPtrCache.computeIfAbsent(pluginPath) { path ->
            val lookup = SymbolLookup.libraryLookup(path, registryArena)
            val getPjrtApi = PjrtFfm.LINKER.downcallHandle(
                lookup.find("GetPjrtApi")
                    .orElseThrow { IllegalStateException("$path does not export GetPjrtApi") },
                FunctionDescriptor.of(ADDRESS),
            )
            PjrtFfm.checkedApi(getPjrtApi.invokeExact() as MemorySegment, path)
        }

    /** Minimal PJRT_Error message read + destroy via [PjrtFfm]'s offset
     * table (mirrors PjrtApi.checkError, which is private to its class). */
    private fun readAndDestroyError(apiPtr: MemorySegment, errorPtr: MemorySegment, arena: Arena): String {
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
        val bytes = ByteArray(msgSize.toInt()) { msgPtr.get(JAVA_BYTE, it.toLong()) }

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
}
