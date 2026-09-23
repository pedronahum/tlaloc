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
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.ValueLayout.JAVA_LONG
import java.lang.invoke.MethodHandle
import java.nio.charset.StandardCharsets
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
        val pjrtApiPtr = checkedApi(getPjrtApi.invokeExact() as MemorySegment, pluginPath)
        return PjrtApi(pjrtApiPtr, arena)
    }

    /** The PJRT C API major version these bindings are written against. A
     * major bump may reorder the function table, so any other major is refused. */
    internal const val PJRT_API_MAJOR_SUPPORTED: Int = 0

    /** Highest `PJRT_Api` function-pointer offset any binding in this module
     * reads. The plugin's `PJRT_Api.struct_size` must cover it. */
    internal const val PJRT_API_HIGHEST_OFFSET_USED: Long = 600 // OFFSET_PJRT_Buffer_ToHostBuffer; a test pins the two equal

    /** Bytes of `PJRT_Api` a plugin must provide: through the last pointer read. */
    internal const val PJRT_API_MIN_STRUCT_SIZE: Long = PJRT_API_HIGHEST_OFFSET_USED + 8

    // `PJRT_Api` header: struct_size @0, extension_start @8, then the embedded
    // PJRT_Api_Version (struct_size @16, extension_start @24, major @32, minor @36).
    private const val OFF_Api_StructSize: Long = 0
    private const val OFF_Api_VersionMajor: Long = 32
    private const val OFF_Api_VersionMinor: Long = 36
    private const val API_HEADER_SIZE: Long = 40

    /**
     * Validates the `PJRT_Api*` a plugin's `GetPjrtApi` returned before any
     * fixed offset is read from it, and returns it sized to the plugin's own
     * `struct_size`. Refuses a NULL table, a major version other than
     * [PJRT_API_MAJOR_SUPPORTED], and a table too short to hold every
     * function pointer these bindings call.
     */
    internal fun checkedApi(raw: MemorySegment, pluginPath: Path): MemorySegment {
        if (raw.address() == 0L) {
            throw PjrtRuntimeException("PJRT plugin at $pluginPath returned NULL from GetPjrtApi")
        }
        val header = raw.reinterpret(API_HEADER_SIZE)
        val structSize = header.get(JAVA_LONG, OFF_Api_StructSize)
        val major = header.get(JAVA_INT, OFF_Api_VersionMajor)
        val minor = header.get(JAVA_INT, OFF_Api_VersionMinor)
        checkApiCompatible(pluginPath.toString(), structSize, major, minor)
        return raw.reinterpret(structSize)
    }

    /** The version/size rule behind [checkedApi], separate so it is testable
     * without a plugin. */
    internal fun checkApiCompatible(plugin: String, structSize: Long, major: Int, minor: Int) {
        if (major != PJRT_API_MAJOR_SUPPORTED) {
            throw PjrtRuntimeException(
                "PJRT plugin at $plugin implements PJRT C API $major.$minor; Tlaloc binds " +
                    "API major version $PJRT_API_MAJOR_SUPPORTED and cannot call a plugin with a " +
                    "different major version. Use a plugin built for PJRT C API $PJRT_API_MAJOR_SUPPORTED.x.",
            )
        }
        if (structSize < PJRT_API_MIN_STRUCT_SIZE) {
            throw PjrtRuntimeException(
                "PJRT plugin at $plugin implements PJRT C API $major.$minor with a PJRT_Api table of " +
                    "$structSize bytes; Tlaloc calls function pointers up to byte offset " +
                    "$PJRT_API_HIGHEST_OFFSET_USED and needs at least $PJRT_API_MIN_STRUCT_SIZE bytes. " +
                    "The plugin is too old; use a newer one.",
            )
        }
    }

    // =========================================================================
    // Linker + low-level config.
    // =========================================================================

    internal val LINKER: Linker = Linker.nativeLinker()

    // A new PJRT call bound past PJRT_Buffer_ToHostBuffer must move
    // PJRT_API_HIGHEST_OFFSET_USED with it, so [checkedApi] keeps refusing
    // plugins whose table does not reach it.

    // PJRT_Api function-pointer offsets. The struct opens with
    // (struct_size: 8 + extension_start: 8 + pjrt_api_version: 24) = 40 bytes
    // of prefix; each function pointer thereafter is 8 bytes (aarch64/x86_64).
    // Indices counted from `xla/pjrt/c/pjrt_c_api.h:2930`.
    internal const val OFFSET_PJRT_Error_Destroy: Long = 40
    internal const val OFFSET_PJRT_Error_Message: Long = 48
    internal const val OFFSET_PJRT_Event_Destroy: Long = 80
    internal const val OFFSET_PJRT_Event_Await: Long = 104
    internal const val OFFSET_PJRT_Client_Create: Long = 120
    internal const val OFFSET_PJRT_Client_Destroy: Long = 128
    internal const val OFFSET_PJRT_Client_PlatformName: Long = 136
    internal const val OFFSET_PJRT_Client_AddressableDevices: Long = 168
    internal const val OFFSET_PJRT_Client_Compile: Long = 200
    internal const val OFFSET_PJRT_Client_BufferFromHostBuffer: Long = 216
    internal const val OFFSET_PJRT_Executable_Destroy: Long = 360
    internal const val OFFSET_PJRT_Executable_NumOutputs: Long = 392
    internal const val OFFSET_PJRT_LoadedExecutable_Destroy: Long = 440
    internal const val OFFSET_PJRT_LoadedExecutable_GetExecutable: Long = 448
    internal const val OFFSET_PJRT_LoadedExecutable_Execute: Long = 480
    internal const val OFFSET_PJRT_Buffer_Destroy: Long = 504
    internal const val OFFSET_PJRT_Buffer_OnDeviceSizeInBytes: Long = 552
    internal const val OFFSET_PJRT_Buffer_ToHostBuffer: Long = 600

    // Enum values from PJRT_Buffer_Type (xla/pjrt/c/pjrt_c_api.h:907).
    internal const val PJRT_BUFFER_TYPE_F32: Int = 11
    // §0.4.354 — F64 joins the marshalling surface (PJRT_Buffer_Type.F64).
    internal const val PJRT_BUFFER_TYPE_F64: Int = 12
    // §0.4.457 (G1c) — BF16 joins: PJRT_Buffer_Type_BF16 sits directly after
    // F64 in the header enum (verified against xla/pjrt/c/pjrt_c_api.h:934).
    // Host representation is the :core convention — a ShortArray of raw
    // upper-16-bit f32 patterns, never numbers (§0.4.455).
    internal const val PJRT_BUFFER_TYPE_BF16: Int = 13

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

    /**
     * §0.4.333 — `PJRT_NamedValue` (pjrt_c_api.h:229), the entry format for
     * `PJRT_Client_Create_Args.create_options`. Hand-computed offsets
     * (aarch64/x86_64 LP64):
     *
     *   struct_size      @ 0   (size_t)
     *   extension_start  @ 8   (ptr)
     *   name             @ 16  (ptr, not NUL-terminated on the read side)
     *   name_size        @ 24  (size_t)
     *   type             @ 32  (enum, i32; +4 bytes padding — the union that
     *                           follows is 8-byte aligned)
     *   value union      @ 40  (8 bytes: string ptr / i64 / i64* / f32 / bool)
     *   value_size       @ 48  (size_t; 1 for scalar values per the header)
     *
     * total 56 bytes. Enum values from `PJRT_NamedValue_Type`:
     * kString=0, kInt64=1, kInt64List=2, kFloat=3, kBool=4.
     */
    internal val PJRT_NamedValue_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("name"),
        JAVA_LONG.withName("name_size"),
        JAVA_INT.withName("type"),
        MemoryLayout.paddingLayout(4),
        JAVA_LONG.withName("value"),
        JAVA_LONG.withName("value_size"),
    )

    internal const val PJRT_NAMED_VALUE_TYPE_INT64: Int = 1
    internal const val PJRT_NAMED_VALUE_TYPE_FLOAT: Int = 3
    internal const val PJRT_NAMED_VALUE_TYPE_BOOL: Int = 4

    internal val OFF_NamedValue_StructSize: Long = PJRT_NamedValue_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val OFF_NamedValue_Name: Long = PJRT_NamedValue_LAYOUT.byteOffset(groupElement("name"))
    internal val OFF_NamedValue_NameSize: Long = PJRT_NamedValue_LAYOUT.byteOffset(groupElement("name_size"))
    internal val OFF_NamedValue_Type: Long = PJRT_NamedValue_LAYOUT.byteOffset(groupElement("type"))
    internal val OFF_NamedValue_Value: Long = PJRT_NamedValue_LAYOUT.byteOffset(groupElement("value"))
    internal val OFF_NamedValue_ValueSize: Long = PJRT_NamedValue_LAYOUT.byteOffset(groupElement("value_size"))
    internal val SZ_NamedValue: Long = PJRT_NamedValue_LAYOUT.byteSize()

    /**
     * §0.4.333 — marshal [options] as a `PJRT_NamedValue` array
     * (`memory_fraction`: kFloat, `preallocate`: kBool) allocated in [arena].
     * Returns the array segment to be stored in
     * `PJRT_Client_Create_Args.create_options` (with `num_options =`
     * [PjrtClientOptions.namedValueCount]). Extracted from
     * [PjrtApi.createClient] so the byte layout is pinned by a GPU-less
     * unit test.
     *
     * §0.4.461 (G3a-2) — when [options] declares a multi-node group
     * (`numNodes > 1`), two kInt64 entries follow: `node_id` and
     * `num_nodes`, the names the XLA GPU plugin's create path parses
     * (verified against xla/pjrt/c/pjrt_c_api_gpu_internal.cc, openxla/xla
     * main 2026-09-21). The single-node encoding is BYTE-IDENTICAL to the
     * §0.4.333 two-entry form — the distributed fields add nothing until
     * they are asked for. `coordinatorAddress` is deliberately NOT
     * marshalled: the PJRT C API has no such option — the coordinator
     * backs the kv-store callbacks in `PJRT_Client_Create_Args`
     * (kv_get/kv_try_get/kv_put), which Tlaloc does not implement yet
     * (G4; see docs/MULTIHOST_DESIGN.md).
     */
    internal fun marshalCreateOptions(arena: Arena, options: PjrtClientOptions): MemorySegment {
        val count = options.namedValueCount
        val array = arena.allocate(SZ_NamedValue * count)

        fun header(index: Int, name: String, type: Int): Long {
            val base = index * SZ_NamedValue
            val nameBytes = name.toByteArray(StandardCharsets.UTF_8)
            val nameSeg = arena.allocate(nameBytes.size + 1L)
            MemorySegment.copy(nameBytes, 0, nameSeg, JAVA_BYTE, 0L, nameBytes.size)
            array.set(JAVA_LONG, base + OFF_NamedValue_StructSize, SZ_NamedValue)
            array.set(ADDRESS, base + OFF_NamedValue_Name, nameSeg)
            array.set(JAVA_LONG, base + OFF_NamedValue_NameSize, nameBytes.size.toLong())
            array.set(JAVA_INT, base + OFF_NamedValue_Type, type)
            array.set(JAVA_LONG, base + OFF_NamedValue_ValueSize, 1L)
            return base
        }

        val fracBase = header(0, "memory_fraction", PJRT_NAMED_VALUE_TYPE_FLOAT)
        array.set(ValueLayout.JAVA_FLOAT, fracBase + OFF_NamedValue_Value, options.memoryFraction)

        val preallocBase = header(1, "preallocate", PJRT_NAMED_VALUE_TYPE_BOOL)
        array.set(JAVA_BYTE, preallocBase + OFF_NamedValue_Value, if (options.preallocate) 1 else 0)

        if (options.numNodes > 1) {
            val nodeIdBase = header(2, "node_id", PJRT_NAMED_VALUE_TYPE_INT64)
            array.set(JAVA_LONG, nodeIdBase + OFF_NamedValue_Value, options.nodeId.toLong())

            val numNodesBase = header(3, "num_nodes", PJRT_NAMED_VALUE_TYPE_INT64)
            array.set(JAVA_LONG, numNodesBase + OFF_NamedValue_Value, options.numNodes.toLong())
        }

        return array
    }

    /**
     * §0.4.461 (G3a-2) — the multi-node create refusal, extracted pure so it
     * certifies GPU-less. A `numNodes > 1` client CANNOT be created today:
     * the GPU plugin rendezvouses multi-node clients through the kv-store
     * callbacks in `PJRT_Client_Create_Args` (kv_get/kv_try_get/kv_put over
     * the coordinator's store — how JAX's distributed service does it), and
     * Tlaloc leaves those NULL. Passing `num_nodes > 1` with a NULL kv store
     * would fail or hang inside the plugin — a loud named refusal beats
     * either. Lifting this is the G4 kv-store upcall work
     * (docs/MULTIHOST_DESIGN.md §4).
     */
    internal fun requireKvStoreForMultiNode(options: PjrtClientOptions?) {
        require(options == null || options.numNodes == 1) {
            "PJRT client create: options declare a multi-node group " +
                "(node_id=${options!!.nodeId}, num_nodes=${options.numNodes}, " +
                "coordinator=${options.coordinatorAddress}) but the kv-store callbacks " +
                "(PJRT_Client_Create_Args.kv_get/kv_try_get/kv_put) are not implemented — " +
                "a multi-node client cannot rendezvous without them. Multi-host init is " +
                "G4 surface; see docs/MULTIHOST_DESIGN.md. Marshalling of node_id/" +
                "num_nodes is certified; client CREATION at num_nodes > 1 is refused by name."
        }
    }

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
    internal val OFF_ClientCreate_CreateOptions: Long = PJRT_Client_Create_Args_LAYOUT.byteOffset(groupElement("create_options"))
    internal val OFF_ClientCreate_NumOptions: Long = PJRT_Client_Create_Args_LAYOUT.byteOffset(groupElement("num_options"))
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

    // -------------------------------------------------------------------------
    // §0.4.304 — extended Args struct layouts for compile / buffer / execute.
    // -------------------------------------------------------------------------

    internal val PJRT_Event_Destroy_Args_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("event"),
    )
    internal val OFF_EventDestroy_StructSize: Long = PJRT_Event_Destroy_Args_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val OFF_EventDestroy_Event: Long = PJRT_Event_Destroy_Args_LAYOUT.byteOffset(groupElement("event"))
    internal val SZ_EventDestroy: Long = PJRT_Event_Destroy_Args_LAYOUT.byteSize()

    internal val PJRT_Event_Await_Args_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("event"),
    )
    internal val OFF_EventAwait_StructSize: Long = PJRT_Event_Await_Args_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val OFF_EventAwait_Event: Long = PJRT_Event_Await_Args_LAYOUT.byteOffset(groupElement("event"))
    internal val SZ_EventAwait: Long = PJRT_Event_Await_Args_LAYOUT.byteSize()

    /** PJRT_Program ([2382] header line 714) — wraps the StableHLO bytes
     * passed to PJRT_Client_Compile. */
    internal val PJRT_Program_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("code"),
        JAVA_LONG.withName("code_size"),
        ADDRESS.withName("format"),
        JAVA_LONG.withName("format_size"),
    )
    internal val OFF_Program_StructSize: Long = PJRT_Program_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val OFF_Program_Code: Long = PJRT_Program_LAYOUT.byteOffset(groupElement("code"))
    internal val OFF_Program_CodeSize: Long = PJRT_Program_LAYOUT.byteOffset(groupElement("code_size"))
    internal val OFF_Program_Format: Long = PJRT_Program_LAYOUT.byteOffset(groupElement("format"))
    internal val OFF_Program_FormatSize: Long = PJRT_Program_LAYOUT.byteOffset(groupElement("format_size"))
    internal val SZ_Program: Long = PJRT_Program_LAYOUT.byteSize()

    internal val PJRT_Client_Compile_Args_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("client"),
        ADDRESS.withName("program"),
        ADDRESS.withName("compile_options"),
        JAVA_LONG.withName("compile_options_size"),
        ADDRESS.withName("executable"),  // out
    )
    internal val OFF_Compile_StructSize: Long = PJRT_Client_Compile_Args_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val OFF_Compile_Client: Long = PJRT_Client_Compile_Args_LAYOUT.byteOffset(groupElement("client"))
    internal val OFF_Compile_Program: Long = PJRT_Client_Compile_Args_LAYOUT.byteOffset(groupElement("program"))
    internal val OFF_Compile_Executable: Long = PJRT_Client_Compile_Args_LAYOUT.byteOffset(groupElement("executable"))
    internal val SZ_Compile: Long = PJRT_Client_Compile_Args_LAYOUT.byteSize()

    internal val PJRT_Client_AddressableDevices_Args_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("client"),
        ADDRESS.withName("addressable_devices"),  // out
        JAVA_LONG.withName("num_addressable_devices"),  // out
    )
    internal val OFF_AddressableDevices_StructSize: Long = PJRT_Client_AddressableDevices_Args_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val OFF_AddressableDevices_Client: Long = PJRT_Client_AddressableDevices_Args_LAYOUT.byteOffset(groupElement("client"))
    internal val OFF_AddressableDevices_Devices: Long = PJRT_Client_AddressableDevices_Args_LAYOUT.byteOffset(groupElement("addressable_devices"))
    internal val OFF_AddressableDevices_NumDevices: Long = PJRT_Client_AddressableDevices_Args_LAYOUT.byteOffset(groupElement("num_addressable_devices"))
    internal val SZ_AddressableDevices: Long = PJRT_Client_AddressableDevices_Args_LAYOUT.byteSize()

    /** PJRT_Client_BufferFromHostBuffer_Args (header line 1184). */
    internal val PJRT_Client_BufferFromHostBuffer_Args_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("client"),
        ADDRESS.withName("data"),
        JAVA_INT.withName("type"),
        MemoryLayout.paddingLayout(4),
        ADDRESS.withName("dims"),
        JAVA_LONG.withName("num_dims"),
        ADDRESS.withName("byte_strides"),
        JAVA_LONG.withName("num_byte_strides"),
        JAVA_INT.withName("host_buffer_semantics"),
        MemoryLayout.paddingLayout(4),
        ADDRESS.withName("device"),
        ADDRESS.withName("memory"),
        ADDRESS.withName("device_layout"),
        ADDRESS.withName("done_with_host_buffer"),  // out
        ADDRESS.withName("buffer"),                 // out
    )
    internal val OFF_BufferFromHost_StructSize: Long = PJRT_Client_BufferFromHostBuffer_Args_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val OFF_BufferFromHost_Client: Long = PJRT_Client_BufferFromHostBuffer_Args_LAYOUT.byteOffset(groupElement("client"))
    internal val OFF_BufferFromHost_Data: Long = PJRT_Client_BufferFromHostBuffer_Args_LAYOUT.byteOffset(groupElement("data"))
    internal val OFF_BufferFromHost_Type: Long = PJRT_Client_BufferFromHostBuffer_Args_LAYOUT.byteOffset(groupElement("type"))
    internal val OFF_BufferFromHost_Dims: Long = PJRT_Client_BufferFromHostBuffer_Args_LAYOUT.byteOffset(groupElement("dims"))
    internal val OFF_BufferFromHost_NumDims: Long = PJRT_Client_BufferFromHostBuffer_Args_LAYOUT.byteOffset(groupElement("num_dims"))
    internal val OFF_BufferFromHost_HostSemantics: Long = PJRT_Client_BufferFromHostBuffer_Args_LAYOUT.byteOffset(groupElement("host_buffer_semantics"))
    internal val OFF_BufferFromHost_Device: Long = PJRT_Client_BufferFromHostBuffer_Args_LAYOUT.byteOffset(groupElement("device"))
    internal val OFF_BufferFromHost_DoneEvent: Long = PJRT_Client_BufferFromHostBuffer_Args_LAYOUT.byteOffset(groupElement("done_with_host_buffer"))
    internal val OFF_BufferFromHost_Buffer: Long = PJRT_Client_BufferFromHostBuffer_Args_LAYOUT.byteOffset(groupElement("buffer"))
    internal val SZ_BufferFromHost: Long = PJRT_Client_BufferFromHostBuffer_Args_LAYOUT.byteSize()

    /** PJRT_HostBufferSemantics enum value 0 = kImmutableOnlyDuringCall —
     * the simplest semantics: PJRT may not hold the host buffer past the
     * Compile call, so we don't need an event to know when it's safe to
     * free the host data. */
    internal const val HOST_BUFFER_SEMANTICS_IMMUTABLE_ONLY_DURING_CALL: Int = 0

    /** PJRT_Buffer_OnDeviceSizeInBytes_Args (header line 2406). */
    internal val PJRT_Buffer_OnDeviceSizeInBytes_Args_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("buffer"),
        JAVA_LONG.withName("on_device_size_in_bytes"),  // out
    )
    internal val OFF_BufferSize_StructSize: Long = PJRT_Buffer_OnDeviceSizeInBytes_Args_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val OFF_BufferSize_Buffer: Long = PJRT_Buffer_OnDeviceSizeInBytes_Args_LAYOUT.byteOffset(groupElement("buffer"))
    internal val OFF_BufferSize_Size: Long = PJRT_Buffer_OnDeviceSizeInBytes_Args_LAYOUT.byteOffset(groupElement("on_device_size_in_bytes"))
    internal val SZ_BufferSize: Long = PJRT_Buffer_OnDeviceSizeInBytes_Args_LAYOUT.byteSize()

    /** PJRT_Buffer_ToHostBuffer_Args (header line 2380). */
    internal val PJRT_Buffer_ToHostBuffer_Args_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("src"),
        ADDRESS.withName("host_layout"),
        ADDRESS.withName("dst"),
        JAVA_LONG.withName("dst_size"),
        ADDRESS.withName("event"),  // out
    )
    internal val OFF_ToHost_StructSize: Long = PJRT_Buffer_ToHostBuffer_Args_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val OFF_ToHost_Src: Long = PJRT_Buffer_ToHostBuffer_Args_LAYOUT.byteOffset(groupElement("src"))
    internal val OFF_ToHost_Dst: Long = PJRT_Buffer_ToHostBuffer_Args_LAYOUT.byteOffset(groupElement("dst"))
    internal val OFF_ToHost_DstSize: Long = PJRT_Buffer_ToHostBuffer_Args_LAYOUT.byteOffset(groupElement("dst_size"))
    internal val OFF_ToHost_Event: Long = PJRT_Buffer_ToHostBuffer_Args_LAYOUT.byteOffset(groupElement("event"))
    internal val SZ_ToHost: Long = PJRT_Buffer_ToHostBuffer_Args_LAYOUT.byteSize()

    internal val PJRT_Buffer_Destroy_Args_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("buffer"),
    )
    internal val OFF_BufferDestroy_StructSize: Long = PJRT_Buffer_Destroy_Args_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val OFF_BufferDestroy_Buffer: Long = PJRT_Buffer_Destroy_Args_LAYOUT.byteOffset(groupElement("buffer"))
    internal val SZ_BufferDestroy: Long = PJRT_Buffer_Destroy_Args_LAYOUT.byteSize()

    internal val PJRT_LoadedExecutable_Destroy_Args_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("executable"),
    )
    internal val OFF_LoadedExecDestroy_StructSize: Long = PJRT_LoadedExecutable_Destroy_Args_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val OFF_LoadedExecDestroy_Executable: Long = PJRT_LoadedExecutable_Destroy_Args_LAYOUT.byteOffset(groupElement("executable"))
    internal val SZ_LoadedExecDestroy: Long = PJRT_LoadedExecutable_Destroy_Args_LAYOUT.byteSize()

    internal val PJRT_LoadedExecutable_GetExecutable_Args_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("loaded_executable"),
        ADDRESS.withName("executable"),  // out
    )
    internal val OFF_GetExec_StructSize: Long = PJRT_LoadedExecutable_GetExecutable_Args_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val OFF_GetExec_LoadedExec: Long = PJRT_LoadedExecutable_GetExecutable_Args_LAYOUT.byteOffset(groupElement("loaded_executable"))
    internal val OFF_GetExec_Executable: Long = PJRT_LoadedExecutable_GetExecutable_Args_LAYOUT.byteOffset(groupElement("executable"))
    internal val SZ_GetExec: Long = PJRT_LoadedExecutable_GetExecutable_Args_LAYOUT.byteSize()

    internal val PJRT_Executable_Destroy_Args_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("executable"),
    )
    internal val OFF_ExecDestroy_StructSize: Long = PJRT_Executable_Destroy_Args_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val OFF_ExecDestroy_Executable: Long = PJRT_Executable_Destroy_Args_LAYOUT.byteOffset(groupElement("executable"))
    internal val SZ_ExecDestroy: Long = PJRT_Executable_Destroy_Args_LAYOUT.byteSize()

    internal val PJRT_Executable_NumOutputs_Args_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("executable"),
        JAVA_LONG.withName("num_outputs"),  // out
    )
    internal val OFF_NumOutputs_StructSize: Long = PJRT_Executable_NumOutputs_Args_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val OFF_NumOutputs_Executable: Long = PJRT_Executable_NumOutputs_Args_LAYOUT.byteOffset(groupElement("executable"))
    internal val OFF_NumOutputs_Count: Long = PJRT_Executable_NumOutputs_Args_LAYOUT.byteOffset(groupElement("num_outputs"))
    internal val SZ_NumOutputs: Long = PJRT_Executable_NumOutputs_Args_LAYOUT.byteSize()

    /** PJRT_ExecuteOptions (header line 1946). All-zero fields except struct_size
     * suffice for single-device execute with no callbacks / contexts.
     *
     * §0.4.459 (G2a) backend audit for TPU: this layout is the C header's
     * ABI, not a backend's — every PJRT plugin consumes the same struct.
     * The one padding decision (4 bytes after `launch_id`, an i32, so the
     * following pointer lands 8-aligned) holds on LP64 for both aarch64
     * (this GB10) and x86_64 (Cloud TPU VM hosts). The fields a TPU cares
     * about beyond CUDA are exactly the ones we zero: `launch_id`
     * (cross-host collective matching), `num_tasks`/`task_ids`/
     * `incarnation_ids` and `multi_slice_config` (multi-host/multi-slice) —
     * zero is the documented single-host single-task default, correct for
     * G2b's single-device smoke; the non-zero forms are G3/G4 surface and
     * are deliberately not marshalled yet. */
    internal val PJRT_ExecuteOptions_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("send_callbacks"),
        ADDRESS.withName("recv_callbacks"),
        JAVA_LONG.withName("num_send_ops"),
        JAVA_LONG.withName("num_recv_ops"),
        JAVA_INT.withName("launch_id"),
        MemoryLayout.paddingLayout(4),
        ADDRESS.withName("non_donatable_input_indices"),
        JAVA_LONG.withName("num_non_donatable_input_indices"),
        ADDRESS.withName("context"),
        ADDRESS.withName("call_location"),
        JAVA_LONG.withName("num_tasks"),
        ADDRESS.withName("task_ids"),
        ADDRESS.withName("incarnation_ids"),
        ADDRESS.withName("multi_slice_config"),
    )
    internal val OFF_ExecOpts_StructSize: Long = PJRT_ExecuteOptions_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val SZ_ExecOpts: Long = PJRT_ExecuteOptions_LAYOUT.byteSize()

    /** PJRT_LoadedExecutable_Execute_Args (header line 1998). */
    internal val PJRT_LoadedExecutable_Execute_Args_LAYOUT: MemoryLayout = MemoryLayout.structLayout(
        JAVA_LONG.withName("struct_size"),
        ADDRESS.withName("extension_start"),
        ADDRESS.withName("executable"),
        ADDRESS.withName("options"),
        ADDRESS.withName("argument_lists"),
        JAVA_LONG.withName("num_devices"),
        JAVA_LONG.withName("num_args"),
        ADDRESS.withName("output_lists"),
        ADDRESS.withName("device_complete_events"),
        ADDRESS.withName("execute_device"),
    )
    internal val OFF_Execute_StructSize: Long = PJRT_LoadedExecutable_Execute_Args_LAYOUT.byteOffset(groupElement("struct_size"))
    internal val OFF_Execute_Executable: Long = PJRT_LoadedExecutable_Execute_Args_LAYOUT.byteOffset(groupElement("executable"))
    internal val OFF_Execute_Options: Long = PJRT_LoadedExecutable_Execute_Args_LAYOUT.byteOffset(groupElement("options"))
    internal val OFF_Execute_ArgLists: Long = PJRT_LoadedExecutable_Execute_Args_LAYOUT.byteOffset(groupElement("argument_lists"))
    internal val OFF_Execute_NumDevices: Long = PJRT_LoadedExecutable_Execute_Args_LAYOUT.byteOffset(groupElement("num_devices"))
    internal val OFF_Execute_NumArgs: Long = PJRT_LoadedExecutable_Execute_Args_LAYOUT.byteOffset(groupElement("num_args"))
    internal val OFF_Execute_OutputLists: Long = PJRT_LoadedExecutable_Execute_Args_LAYOUT.byteOffset(groupElement("output_lists"))
    internal val OFF_Execute_DeviceCompleteEvents: Long = PJRT_LoadedExecutable_Execute_Args_LAYOUT.byteOffset(groupElement("device_complete_events"))
    internal val OFF_Execute_ExecuteDevice: Long = PJRT_LoadedExecutable_Execute_Args_LAYOUT.byteOffset(groupElement("execute_device"))
    internal val SZ_Execute: Long = PJRT_LoadedExecutable_Execute_Args_LAYOUT.byteSize()

    // -------------------------------------------------------------------------
    // Function descriptors. Each PJRT entry takes a single args-struct pointer
    // and returns PJRT_Error* (ADDRESS). PJRT_Error_Destroy returns void.
    // -------------------------------------------------------------------------

    /**
     * §0.4.304's hand-encoded minimal `CompileOptionsProto`, extracted and
     * re-verified for TPU in §0.4.459 (G2a). Field numbers checked against
     * `xla/pjrt/proto/compile_options.proto` at openxla/xla main
     * (2026-09-21):
     *
     *   CompileOptionsProto.executable_build_options = **field 3**
     *   ExecutableBuildOptionsProto.num_replicas     = **field 4** (int64)
     *   ExecutableBuildOptionsProto.num_partitions   = **field 5** (int64)
     *
     * Bytes: `0x1A 0x04 0x20 0x01 0x28 0x01` —
     *   0x1A = (3<<3)|2 (field 3, length-delimited), 0x04 = inner length,
     *   0x20 = (4<<3)|0 (num_replicas, varint) value 0x01,
     *   0x28 = (5<<3)|0 (num_partitions, varint) value 0x01.
     *
     * **Backend audit**: protobuf wire format is backend-agnostic and the
     * TPU compile path deserialises the same `CompileOptionsProto` message
     * (PJRT_Client_Compile's `compile_options` is that serialised proto for
     * every plugin), so a TPU compile parses these six bytes identically.
     * num_replicas=1 / num_partitions=1 is the single-device shape on TPU
     * exactly as on CUDA; multi-replica TPU topologies are G3/G4 territory
     * (device_assignment, use_spmd_partitioning — fields we deliberately
     * leave unset).
     */
    internal val COMPILE_OPTIONS_PROTO_BYTES: ByteArray =
        byteArrayOf(0x1A, 0x04, 0x20, 0x01, 0x28, 0x01)

    internal val FD_ClientCreate: FunctionDescriptor = FunctionDescriptor.of(ADDRESS, ADDRESS)
    internal val FD_ClientDestroy: FunctionDescriptor = FunctionDescriptor.of(ADDRESS, ADDRESS)
    internal val FD_ClientPlatformName: FunctionDescriptor = FunctionDescriptor.of(ADDRESS, ADDRESS)
    internal val FD_ErrorDestroy: FunctionDescriptor = FunctionDescriptor.ofVoid(ADDRESS)
    internal val FD_ErrorMessage: FunctionDescriptor = FunctionDescriptor.ofVoid(ADDRESS)
    internal val FD_EventDestroy: FunctionDescriptor = FunctionDescriptor.of(ADDRESS, ADDRESS)
    internal val FD_EventAwait: FunctionDescriptor = FunctionDescriptor.of(ADDRESS, ADDRESS)
    internal val FD_AddressableDevices: FunctionDescriptor = FunctionDescriptor.of(ADDRESS, ADDRESS)
    internal val FD_Compile: FunctionDescriptor = FunctionDescriptor.of(ADDRESS, ADDRESS)
    internal val FD_BufferFromHost: FunctionDescriptor = FunctionDescriptor.of(ADDRESS, ADDRESS)
    internal val FD_BufferDestroy: FunctionDescriptor = FunctionDescriptor.of(ADDRESS, ADDRESS)
    internal val FD_BufferSize: FunctionDescriptor = FunctionDescriptor.of(ADDRESS, ADDRESS)
    internal val FD_ToHost: FunctionDescriptor = FunctionDescriptor.of(ADDRESS, ADDRESS)
    internal val FD_LoadedExecDestroy: FunctionDescriptor = FunctionDescriptor.of(ADDRESS, ADDRESS)
    internal val FD_GetExecutable: FunctionDescriptor = FunctionDescriptor.of(ADDRESS, ADDRESS)
    internal val FD_ExecutableDestroy: FunctionDescriptor = FunctionDescriptor.of(ADDRESS, ADDRESS)
    internal val FD_NumOutputs: FunctionDescriptor = FunctionDescriptor.of(ADDRESS, ADDRESS)
    internal val FD_Execute: FunctionDescriptor = FunctionDescriptor.of(ADDRESS, ADDRESS)
}

/**
 * §0.4.333 — GPU-client allocator options passed as `create_options` to
 * `PJRT_Client_Create`.
 *
 * **Why this exists (the 2026-07-18 reboot incident).** With zero
 * create_options the CUDA plugin defaults to `preallocate=true` +
 * `memory_fraction=0.75`: at client create it `cuMemAlloc`s 75% of "device
 * memory" up front for the BFCAllocator. On the GB10 device memory *is*
 * system RAM (128 GB unified, CPU-coherent), so every client pinned a
 * ~98 GB unswappable pool. A test run that created a handful of clients
 * (each PjrtSession / spike test creates its own) stacked those pools,
 * starved the OS of reclaimable memory, and hard-hung the machine — twice,
 * with journald's last words being the moment a process opened the GPU.
 *
 * Defaults here: **no preallocation** (the BFC pool grows on demand and is
 * released at client destroy) and a **0.5 fraction cap** so even a runaway
 * program leaves half the machine to the OS. Overridable per-environment:
 *
 *   `TLALOC_PJRT_MEMORY_FRACTION` — float in (0, 1]
 *   `TLALOC_PJRT_PREALLOCATE`     — "true" / "false"
 *
 * Discrete-GPU hosts (H100 etc.) can safely raise the fraction and turn
 * preallocation back on for benchmark stability; on unified-memory hosts
 * (GB10, Jetson) leave preallocation off.
 *
 * §0.4.459 (G2a) — **these are GPU-plugin options, gated by platform.**
 * `memory_fraction` / `preallocate` are the XLA GPU plugin's BFCAllocator
 * knobs; the TPU plugin (libtpu) neither needs nor is guaranteed to accept
 * them, so a TPU client is created with NO create_options
 * ([PjrtApi.createClient] with null — `create_options = NULL,
 * num_options = 0`). libtpu's own accepted option set is not enumerable
 * here (the TPU plugin headers are not vendored; known-from-framework-source
 * candidates like `ml_framework_name` / `max_inflight_computations` are
 * recorded as UNVERIFIED in docs/TPU_BRINGUP.md and stay unpassed until
 * G2b measures them on real hardware).
 */
data class PjrtClientOptions(
    val memoryFraction: Float,
    val preallocate: Boolean,
    /** §0.4.461 (G3a-2) — this process's rank in a multi-node group.
     * Marshals as the GPU plugin's `node_id` kInt64 create-option when
     * [numNodes] > 1; single-node (the default) marshals nothing new. */
    val nodeId: Int = 0,
    /** §0.4.461 (G3a-2) — group size. 1 (the default) is the single-node
     * client every certified lane uses today; > 1 marshals `node_id` +
     * `num_nodes` and is REFUSED at client create until the kv-store
     * callbacks exist (G4 — see [PjrtFfm.requireKvStoreForMultiNode]). */
    val numNodes: Int = 1,
    /** §0.4.461 (G3a-2) — `host:port` of node 0's coordination service.
     * NOT a PJRT create-option (the C API has none) — it is the address
     * the G4 kv-store callbacks will dial, carried here so one options
     * object states the whole group contract, and so the pod-group env
     * (`TLALOC_PJRT_COORDINATOR_ADDRESS`, emitted by Maestro's
     * TlalocPodSpecBuilder.buildPodGroup) resolves into it. Required
     * exactly when [numNodes] > 1. */
    val coordinatorAddress: String? = null,
) {
    init {
        require(memoryFraction > 0f && memoryFraction <= 1f) {
            "memoryFraction must be in (0, 1], got $memoryFraction"
        }
        require(numNodes >= 1) { "numNodes must be >= 1, got $numNodes" }
        require(nodeId in 0 until numNodes) {
            "nodeId must be in [0, numNodes), got nodeId=$nodeId with numNodes=$numNodes"
        }
        if (numNodes > 1) {
            require(coordinatorAddress != null) {
                "a multi-node group (numNodes=$numNodes) requires coordinatorAddress " +
                    "(host:port of node 0's coordination service)"
            }
        }
        if (coordinatorAddress != null) {
            val port = coordinatorAddress.substringAfterLast(':', "")
            require(
                coordinatorAddress.substringBeforeLast(':', "").isNotBlank() &&
                    port.toIntOrNull() in 1..65535,
            ) {
                "coordinatorAddress must be host:port with port in [1, 65535], " +
                    "got '$coordinatorAddress'"
            }
        }
    }

    /** Number of `PJRT_NamedValue` entries [PjrtFfm.marshalCreateOptions]
     * emits for these options: 2 single-node (§0.4.333 unchanged), 4 when
     * the group is multi-node (+node_id, +num_nodes). */
    val namedValueCount: Long get() = if (numNodes > 1) 4L else 2L

    companion object {
        /** Resolve from env, falling back to the unified-memory-safe defaults.
         *
         * §0.4.461 (G3a-2) — the distributed trio joins the env surface:
         * `TLALOC_PJRT_NODE_ID`, `TLALOC_PJRT_NUM_NODES`,
         * `TLALOC_PJRT_COORDINATOR_ADDRESS` — exactly the variables a
         * Maestro pod-group member is launched with (TlalocPodSpecBuilder
         * .buildPodGroup). Absent, everything defaults to the single-node
         * client. */
        fun resolve(): PjrtClientOptions = resolve(System::getenv)

        /** [resolve] over an arbitrary variable lookup, so the parsing and its
         * error messages are testable without touching the process environment. */
        internal fun resolve(env: (String) -> String?): PjrtClientOptions {
            val fraction = parseEnv(env, "TLALOC_PJRT_MEMORY_FRACTION", "a number in (0, 1]") { it.toFloatOrNull() }
            val nodeId = parseEnv(env, "TLALOC_PJRT_NODE_ID", "a non-negative integer") { it.toIntOrNull() }
            val numNodes = parseEnv(env, "TLALOC_PJRT_NUM_NODES", "a positive integer") { it.toIntOrNull() }
            val preallocate = parseEnv(env, "TLALOC_PJRT_PREALLOCATE", "true or false") { it.toBooleanStrictOrNull() }
            try {
                return PjrtClientOptions(
                    memoryFraction = fraction ?: 0.5f,
                    preallocate = preallocate ?: false,
                    nodeId = nodeId ?: 0,
                    numNodes = numNodes ?: 1,
                    coordinatorAddress = env("TLALOC_PJRT_COORDINATOR_ADDRESS")?.takeIf { it.isNotBlank() },
                )
            } catch (e: IllegalArgumentException) {
                throw IllegalArgumentException(
                    "PJRT client options from the environment are invalid: ${e.message}. " +
                        "Check TLALOC_PJRT_MEMORY_FRACTION, TLALOC_PJRT_NODE_ID, TLALOC_PJRT_NUM_NODES " +
                        "and TLALOC_PJRT_COORDINATOR_ADDRESS.",
                    e,
                )
            }
        }

        private fun <T : Any> parseEnv(env: (String) -> String?, name: String, expected: String, parse: (String) -> T?): T? {
            val raw = env(name)?.trim()?.takeIf { it.isNotEmpty() } ?: return null
            return parse(raw) ?: throw IllegalArgumentException("$name must be $expected, got '$raw'")
        }
    }
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
    private val eventDestroy = PjrtFfm.LINKER.downcallHandle(fnPtrAt(PjrtFfm.OFFSET_PJRT_Event_Destroy), PjrtFfm.FD_EventDestroy)
    private val eventAwait = PjrtFfm.LINKER.downcallHandle(fnPtrAt(PjrtFfm.OFFSET_PJRT_Event_Await), PjrtFfm.FD_EventAwait)
    private val addressableDevices = PjrtFfm.LINKER.downcallHandle(fnPtrAt(PjrtFfm.OFFSET_PJRT_Client_AddressableDevices), PjrtFfm.FD_AddressableDevices)
    private val clientCompile = PjrtFfm.LINKER.downcallHandle(fnPtrAt(PjrtFfm.OFFSET_PJRT_Client_Compile), PjrtFfm.FD_Compile)
    private val bufferFromHost = PjrtFfm.LINKER.downcallHandle(fnPtrAt(PjrtFfm.OFFSET_PJRT_Client_BufferFromHostBuffer), PjrtFfm.FD_BufferFromHost)
    private val bufferDestroy = PjrtFfm.LINKER.downcallHandle(fnPtrAt(PjrtFfm.OFFSET_PJRT_Buffer_Destroy), PjrtFfm.FD_BufferDestroy)
    private val bufferSize = PjrtFfm.LINKER.downcallHandle(fnPtrAt(PjrtFfm.OFFSET_PJRT_Buffer_OnDeviceSizeInBytes), PjrtFfm.FD_BufferSize)
    private val toHost = PjrtFfm.LINKER.downcallHandle(fnPtrAt(PjrtFfm.OFFSET_PJRT_Buffer_ToHostBuffer), PjrtFfm.FD_ToHost)
    private val loadedExecDestroy = PjrtFfm.LINKER.downcallHandle(fnPtrAt(PjrtFfm.OFFSET_PJRT_LoadedExecutable_Destroy), PjrtFfm.FD_LoadedExecDestroy)
    private val getExecutable = PjrtFfm.LINKER.downcallHandle(fnPtrAt(PjrtFfm.OFFSET_PJRT_LoadedExecutable_GetExecutable), PjrtFfm.FD_GetExecutable)
    private val executableDestroy = PjrtFfm.LINKER.downcallHandle(fnPtrAt(PjrtFfm.OFFSET_PJRT_Executable_Destroy), PjrtFfm.FD_ExecutableDestroy)
    private val numOutputs = PjrtFfm.LINKER.downcallHandle(fnPtrAt(PjrtFfm.OFFSET_PJRT_Executable_NumOutputs), PjrtFfm.FD_NumOutputs)
    private val execute = PjrtFfm.LINKER.downcallHandle(fnPtrAt(PjrtFfm.OFFSET_PJRT_LoadedExecutable_Execute), PjrtFfm.FD_Execute)

    private fun fnPtrAt(offset: Long): MemorySegment =
        pjrtApiPtr.get(ADDRESS, offset).reinterpret(Long.MAX_VALUE)

    /** Create a PJRT client (one per process / device family). Throws
     * [PjrtRuntimeException] if the plugin returns a PJRT_Error*.
     *
     * §0.4.333 — always passes [options] (default: env-resolved
     * [PjrtClientOptions]) so the CUDA plugin's BFCAllocator never
     * preallocates 75% of unified memory (the 2026-07-18 reboot incident;
     * see [PjrtClientOptions] for the full story).
     *
     * §0.4.459 (G2a) — [options] is now nullable, and **null is the
     * deliberate non-CUDA form**: `memory_fraction` / `preallocate` are
     * the XLA *GPU* plugin's allocator options, and a TPU client must not
     * be handed them (an unknown NamedValue is the plugin's to reject —
     * libtpu's accepted option set is not vendored here, so we pass the
     * empty set and record the unknown; see docs/TPU_BRINGUP.md). Null
     * marshals `create_options = NULL, num_options = 0`, the header's
     * spelling for "no options". The §0.4.333 rule is unchanged where it
     * applies: a CUDA client MUST get non-null options — [io.tlaloc.runtime.pjrt.PjrtSession]
     * enforces that pairing by target and refuses the cross-wirings by
     * name. */
    fun createClient(options: PjrtClientOptions? = PjrtClientOptions.resolve()): PjrtClient {
        // §0.4.461 (G3a-2) — multi-node creation refused by name until the
        // kv-store callbacks exist (G4); marshalling alone is certified.
        PjrtFfm.requireKvStoreForMultiNode(options)
        val args = arena.allocate(PjrtFfm.PJRT_Client_Create_Args_LAYOUT)
        args.set(JAVA_LONG, PjrtFfm.OFF_ClientCreate_StructSize, PjrtFfm.SZ_ClientCreate)
        if (options != null) {
            args.set(ADDRESS, PjrtFfm.OFF_ClientCreate_CreateOptions, PjrtFfm.marshalCreateOptions(arena, options))
            args.set(JAVA_LONG, PjrtFfm.OFF_ClientCreate_NumOptions, options.namedValueCount)
        } else {
            args.set(ADDRESS, PjrtFfm.OFF_ClientCreate_CreateOptions, MemorySegment.NULL)
            args.set(JAVA_LONG, PjrtFfm.OFF_ClientCreate_NumOptions, 0L)
        }
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

    /** Normalise Tlaloc's StableHLO emit for XLA's MLIR import. Two
     * transformations:
     *   1. Rename the first `func.func @<sym>` to `func.func @main` — XLA's
     *      pjrt-cuda compile pipeline requires the entry function to be
     *      named `@main` (see error: "conversion requires module with `main`
     *      function"). Tlaloc emits `func.func @<DxirFunction.name>`. The
     *      regex is idempotent: if the function is already `@main`, this
     *      rewrite is a no-op.
     *   2. Wrap a bare `func.func` in `module @tlaloc_emit { … }` if the
     *      input doesn't already start with a module declaration. Tlaloc's
     *      `toStablehlo` produces function-only emits at the top level. */
    private fun normaliseMlirForXla(text: String): String {
        var trimmed = text.trim()
        // Idempotent rename of first func.func @<sym> to @main.
        trimmed = Regex("""func\.func\s+@\w+""").replaceFirst(trimmed, "func.func @main")
        if (!trimmed.startsWith("module")) {
            trimmed = "module @tlaloc_emit {\n$trimmed\n}\n"
        }
        return trimmed
    }

    // =========================================================================
    // §0.4.304 — additional high-level methods on PjrtApi for compile / buffer /
    // execute. Called from PjrtClient methods below; kept on PjrtApi so the
    // downcall handles live with the api lifetime.
    // =========================================================================

    internal fun clientAddressableDevices(clientPtr: MemorySegment): List<MemorySegment> {
        Arena.ofConfined().use { scoped ->
            val args = scoped.allocate(PjrtFfm.PJRT_Client_AddressableDevices_Args_LAYOUT)
            args.set(JAVA_LONG, PjrtFfm.OFF_AddressableDevices_StructSize, PjrtFfm.SZ_AddressableDevices)
            args.set(ADDRESS, PjrtFfm.OFF_AddressableDevices_Client, clientPtr)
            val errorPtr = addressableDevices.invokeExact(args) as MemorySegment
            checkError(errorPtr)
            val devicesPtr = args.get(ADDRESS, PjrtFfm.OFF_AddressableDevices_Devices)
            val nDevices = args.get(JAVA_LONG, PjrtFfm.OFF_AddressableDevices_NumDevices)
            require(nDevices > 0) { "PJRT_Client_AddressableDevices returned 0 devices" }
            // `devicesPtr` points to an array of N PJRT_Device* (8 bytes each).
            val devicesView = devicesPtr.reinterpret(nDevices * 8)
            return List(nDevices.toInt()) { i ->
                devicesView.get(ADDRESS, i * 8L).reinterpret(Long.MAX_VALUE)
            }
        }
    }

    /** Compile [stablehloMlir] into a [PjrtLoadedExecutable]. PJRT_Program's
     * `format` is "mlir" (StableHLO bytecode or text both accepted by XLA's
     * MLIR import).
     *
     * **Minimal CompileOptionsProto** is required by XLA's PJRT-CUDA
     * compile path — `num_replicas` / `num_partitions` default to 0 in C++
     * (not 1), and the `ParseDeviceAssignmentCompileOptions` path
     * `Check fail`s on `replica_count > 0`. We hand-encode the minimal
     * protobuf to set both to 1:
     *
     *   CompileOptionsProto {
     *     executable_build_options = ExecutableBuildOptionsProto {
     *       num_replicas: 1,    // field 4, varint
     *       num_partitions: 1,  // field 5, varint
     *     }                     // field 3 in CompileOptions, length-delimited
     *   }
     *
     * Encoded bytes: `0x1A 0x04 0x20 0x01 0x28 0x01` (6 bytes total).
     *
     *   0x1A = (3 << 3) | 2  — field 3, wire type 2 (length-delimited)
     *   0x04                  — embedded message length (4 bytes)
     *     0x20 = (4 << 3) | 0  — num_replicas, varint
     *     0x01                  — value 1
     *     0x28 = (5 << 3) | 0  — num_partitions, varint
     *     0x01                  — value 1
     */
    internal fun clientCompileMlir(clientPtr: MemorySegment, stablehloMlir: String, scratchArena: Arena): MemorySegment {
        val mlirBytes = normaliseMlirForXla(stablehloMlir).toByteArray(StandardCharsets.UTF_8)
        val codeSeg = scratchArena.allocate((mlirBytes.size + 1).toLong())
        for ((i, b) in mlirBytes.withIndex()) codeSeg.set(JAVA_BYTE, i.toLong(), b)
        codeSeg.set(JAVA_BYTE, mlirBytes.size.toLong(), 0)

        val formatBytes = "mlir".toByteArray(StandardCharsets.UTF_8)
        val formatSeg = scratchArena.allocate((formatBytes.size + 1).toLong())
        for ((i, b) in formatBytes.withIndex()) formatSeg.set(JAVA_BYTE, i.toLong(), b)
        formatSeg.set(JAVA_BYTE, formatBytes.size.toLong(), 0)

        val program = scratchArena.allocate(PjrtFfm.PJRT_Program_LAYOUT)
        program.set(JAVA_LONG, PjrtFfm.OFF_Program_StructSize, PjrtFfm.SZ_Program)
        program.set(ADDRESS, PjrtFfm.OFF_Program_Code, codeSeg)
        program.set(JAVA_LONG, PjrtFfm.OFF_Program_CodeSize, mlirBytes.size.toLong())
        program.set(ADDRESS, PjrtFfm.OFF_Program_Format, formatSeg)
        program.set(JAVA_LONG, PjrtFfm.OFF_Program_FormatSize, formatBytes.size.toLong())

        // Hand-encoded CompileOptionsProto (field numbers verified upstream —
        // see COMPILE_OPTIONS_PROTO_BYTES).
        val optsBytes = PjrtFfm.COMPILE_OPTIONS_PROTO_BYTES
        val optsSeg = scratchArena.allocate(optsBytes.size.toLong())
        for ((i, b) in optsBytes.withIndex()) optsSeg.set(JAVA_BYTE, i.toLong(), b)

        val args = scratchArena.allocate(PjrtFfm.PJRT_Client_Compile_Args_LAYOUT)
        args.set(JAVA_LONG, PjrtFfm.OFF_Compile_StructSize, PjrtFfm.SZ_Compile)
        args.set(ADDRESS, PjrtFfm.OFF_Compile_Client, clientPtr)
        args.set(ADDRESS, PjrtFfm.OFF_Compile_Program, program)
        // compile_options + compile_options_size live at the offsets after
        // Program in the Args struct; set explicitly via direct offset since
        // the layout's group elements are accessible by name.
        args.set(ADDRESS, PjrtFfm.PJRT_Client_Compile_Args_LAYOUT.byteOffset(groupElement("compile_options")), optsSeg)
        args.set(JAVA_LONG, PjrtFfm.PJRT_Client_Compile_Args_LAYOUT.byteOffset(groupElement("compile_options_size")), optsBytes.size.toLong())

        val errorPtr = clientCompile.invokeExact(args) as MemorySegment
        checkError(errorPtr)
        return args.get(ADDRESS, PjrtFfm.OFF_Compile_Executable).reinterpret(Long.MAX_VALUE)
    }

    /** Upload an f32 host buffer to the device, returning a `PJRT_Buffer*`.
     * Uses host_buffer_semantics = kImmutableOnlyDuringCall — PJRT must not
     * hold the host data past this call, so we don't need to track the
     * done_with_host_buffer event. */
    internal fun bufferFromHostF32(
        clientPtr: MemorySegment,
        devicePtr: MemorySegment,
        data: FloatArray,
        dims: List<Int>,
        scratchArena: Arena,
    ): MemorySegment {
        val nElements = if (dims.isEmpty()) 1 else dims.fold(1) { a, b -> a * b }
        require(data.size == nElements) {
            "bufferFromHostF32: dims product $nElements != data.size ${data.size}"
        }

        val dataSeg = scratchArena.allocate((nElements * 4).toLong())
        for (i in 0 until nElements) dataSeg.set(ValueLayout.JAVA_FLOAT, i * 4L, data[i])

        val dimsSeg = scratchArena.allocate((dims.size * 8).toLong())
        for ((i, d) in dims.withIndex()) dimsSeg.set(JAVA_LONG, i * 8L, d.toLong())

        val args = scratchArena.allocate(PjrtFfm.PJRT_Client_BufferFromHostBuffer_Args_LAYOUT)
        args.set(JAVA_LONG, PjrtFfm.OFF_BufferFromHost_StructSize, PjrtFfm.SZ_BufferFromHost)
        args.set(ADDRESS, PjrtFfm.OFF_BufferFromHost_Client, clientPtr)
        args.set(ADDRESS, PjrtFfm.OFF_BufferFromHost_Data, dataSeg)
        args.set(JAVA_INT, PjrtFfm.OFF_BufferFromHost_Type, PjrtFfm.PJRT_BUFFER_TYPE_F32)
        args.set(ADDRESS, PjrtFfm.OFF_BufferFromHost_Dims, dimsSeg)
        args.set(JAVA_LONG, PjrtFfm.OFF_BufferFromHost_NumDims, dims.size.toLong())
        args.set(JAVA_INT, PjrtFfm.OFF_BufferFromHost_HostSemantics, PjrtFfm.HOST_BUFFER_SEMANTICS_IMMUTABLE_ONLY_DURING_CALL)
        args.set(ADDRESS, PjrtFfm.OFF_BufferFromHost_Device, devicePtr)
        // memory / device_layout / byte_strides / num_byte_strides left zeroed.

        val errorPtr = bufferFromHost.invokeExact(args) as MemorySegment
        checkError(errorPtr)

        // The done_with_host_buffer event must be awaited (or freed) per the
        // header contract. Since semantics=kImmutableOnlyDuringCall, the call
        // itself is synchronous and the event is signalled before return —
        // but we still have to free it.
        val doneEvent = args.get(ADDRESS, PjrtFfm.OFF_BufferFromHost_DoneEvent)
        if (doneEvent.address() != 0L) destroyEvent(doneEvent.reinterpret(Long.MAX_VALUE))

        return args.get(ADDRESS, PjrtFfm.OFF_BufferFromHost_Buffer).reinterpret(Long.MAX_VALUE)
    }

    /** §0.4.354 — f64 twin of [bufferFromHostF32]. */
    internal fun bufferFromHostF64(
        clientPtr: MemorySegment,
        devicePtr: MemorySegment,
        data: DoubleArray,
        dims: List<Int>,
        scratchArena: Arena,
    ): MemorySegment {
        val nElements = if (dims.isEmpty()) 1 else dims.fold(1) { a, b -> a * b }
        require(data.size == nElements) {
            "bufferFromHostF64: dims product $nElements != data.size ${data.size}"
        }

        val dataSeg = scratchArena.allocate((nElements * 8).toLong())
        for (i in 0 until nElements) dataSeg.set(ValueLayout.JAVA_DOUBLE, i * 8L, data[i])

        val dimsSeg = scratchArena.allocate((dims.size * 8).toLong())
        for ((i, d) in dims.withIndex()) dimsSeg.set(JAVA_LONG, i * 8L, d.toLong())

        val args = scratchArena.allocate(PjrtFfm.PJRT_Client_BufferFromHostBuffer_Args_LAYOUT)
        args.set(JAVA_LONG, PjrtFfm.OFF_BufferFromHost_StructSize, PjrtFfm.SZ_BufferFromHost)
        args.set(ADDRESS, PjrtFfm.OFF_BufferFromHost_Client, clientPtr)
        args.set(ADDRESS, PjrtFfm.OFF_BufferFromHost_Data, dataSeg)
        args.set(JAVA_INT, PjrtFfm.OFF_BufferFromHost_Type, PjrtFfm.PJRT_BUFFER_TYPE_F64)
        args.set(ADDRESS, PjrtFfm.OFF_BufferFromHost_Dims, dimsSeg)
        args.set(JAVA_LONG, PjrtFfm.OFF_BufferFromHost_NumDims, dims.size.toLong())
        args.set(JAVA_INT, PjrtFfm.OFF_BufferFromHost_HostSemantics, PjrtFfm.HOST_BUFFER_SEMANTICS_IMMUTABLE_ONLY_DURING_CALL)
        args.set(ADDRESS, PjrtFfm.OFF_BufferFromHost_Device, devicePtr)

        val errorPtr = bufferFromHost.invokeExact(args) as MemorySegment
        checkError(errorPtr)

        val doneEvent = args.get(ADDRESS, PjrtFfm.OFF_BufferFromHost_DoneEvent)
        if (doneEvent.address() != 0L) destroyEvent(doneEvent.reinterpret(Long.MAX_VALUE))

        return args.get(ADDRESS, PjrtFfm.OFF_BufferFromHost_Buffer).reinterpret(Long.MAX_VALUE)
    }

    /** §0.4.457 (G1c) — bf16 twin of [bufferFromHostF32]. [data] is raw bit
     * patterns per the §0.4.455 host convention (a Short is a 16-bit bucket);
     * PJRT receives them verbatim as 2-byte elements typed BF16 — there is
     * no numeric conversion anywhere on this path. */
    internal fun bufferFromHostBf16(
        clientPtr: MemorySegment,
        devicePtr: MemorySegment,
        data: ShortArray,
        dims: List<Int>,
        scratchArena: Arena,
    ): MemorySegment {
        val nElements = if (dims.isEmpty()) 1 else dims.fold(1) { a, b -> a * b }
        require(data.size == nElements) {
            "bufferFromHostBf16: dims product $nElements != data.size ${data.size}"
        }

        val dataSeg = scratchArena.allocate((nElements * 2).toLong())
        for (i in 0 until nElements) dataSeg.set(ValueLayout.JAVA_SHORT, i * 2L, data[i])

        val dimsSeg = scratchArena.allocate((dims.size * 8).toLong())
        for ((i, d) in dims.withIndex()) dimsSeg.set(JAVA_LONG, i * 8L, d.toLong())

        val args = scratchArena.allocate(PjrtFfm.PJRT_Client_BufferFromHostBuffer_Args_LAYOUT)
        args.set(JAVA_LONG, PjrtFfm.OFF_BufferFromHost_StructSize, PjrtFfm.SZ_BufferFromHost)
        args.set(ADDRESS, PjrtFfm.OFF_BufferFromHost_Client, clientPtr)
        args.set(ADDRESS, PjrtFfm.OFF_BufferFromHost_Data, dataSeg)
        args.set(JAVA_INT, PjrtFfm.OFF_BufferFromHost_Type, PjrtFfm.PJRT_BUFFER_TYPE_BF16)
        args.set(ADDRESS, PjrtFfm.OFF_BufferFromHost_Dims, dimsSeg)
        args.set(JAVA_LONG, PjrtFfm.OFF_BufferFromHost_NumDims, dims.size.toLong())
        args.set(JAVA_INT, PjrtFfm.OFF_BufferFromHost_HostSemantics, PjrtFfm.HOST_BUFFER_SEMANTICS_IMMUTABLE_ONLY_DURING_CALL)
        args.set(ADDRESS, PjrtFfm.OFF_BufferFromHost_Device, devicePtr)

        val errorPtr = bufferFromHost.invokeExact(args) as MemorySegment
        checkError(errorPtr)

        val doneEvent = args.get(ADDRESS, PjrtFfm.OFF_BufferFromHost_DoneEvent)
        if (doneEvent.address() != 0L) destroyEvent(doneEvent.reinterpret(Long.MAX_VALUE))

        return args.get(ADDRESS, PjrtFfm.OFF_BufferFromHost_Buffer).reinterpret(Long.MAX_VALUE)
    }

    internal fun bufferDestroy(bufferPtr: MemorySegment) {
        Arena.ofConfined().use { scoped ->
            val args = scoped.allocate(PjrtFfm.PJRT_Buffer_Destroy_Args_LAYOUT)
            args.set(JAVA_LONG, PjrtFfm.OFF_BufferDestroy_StructSize, PjrtFfm.SZ_BufferDestroy)
            args.set(ADDRESS, PjrtFfm.OFF_BufferDestroy_Buffer, bufferPtr)
            val errorPtr = bufferDestroy.invokeExact(args) as MemorySegment
            checkError(errorPtr)
        }
    }

    internal fun bufferDeviceSize(bufferPtr: MemorySegment): Long {
        Arena.ofConfined().use { scoped ->
            val args = scoped.allocate(PjrtFfm.PJRT_Buffer_OnDeviceSizeInBytes_Args_LAYOUT)
            args.set(JAVA_LONG, PjrtFfm.OFF_BufferSize_StructSize, PjrtFfm.SZ_BufferSize)
            args.set(ADDRESS, PjrtFfm.OFF_BufferSize_Buffer, bufferPtr)
            val errorPtr = bufferSize.invokeExact(args) as MemorySegment
            checkError(errorPtr)
            return args.get(JAVA_LONG, PjrtFfm.OFF_BufferSize_Size)
        }
    }

    /** Pull a device buffer back to host as an f32 array. [nFloats] is the
     * expected element count (caller knows the type/shape from compile-time
     * info). */
    internal fun bufferToHostF32(bufferPtr: MemorySegment, nFloats: Int): FloatArray {
        Arena.ofConfined().use { scoped ->
            val sizeBytes = (nFloats * 4).toLong()
            val dst = scoped.allocate(sizeBytes)
            val args = scoped.allocate(PjrtFfm.PJRT_Buffer_ToHostBuffer_Args_LAYOUT)
            args.set(JAVA_LONG, PjrtFfm.OFF_ToHost_StructSize, PjrtFfm.SZ_ToHost)
            args.set(ADDRESS, PjrtFfm.OFF_ToHost_Src, bufferPtr)
            args.set(ADDRESS, PjrtFfm.OFF_ToHost_Dst, dst)
            args.set(JAVA_LONG, PjrtFfm.OFF_ToHost_DstSize, sizeBytes)
            val errorPtr = toHost.invokeExact(args) as MemorySegment
            checkError(errorPtr)
            // Wait for the copy to complete.
            val event = args.get(ADDRESS, PjrtFfm.OFF_ToHost_Event)
            if (event.address() != 0L) {
                val eventFull = event.reinterpret(Long.MAX_VALUE)
                awaitEvent(eventFull)
                destroyEvent(eventFull)
            }
            return FloatArray(nFloats) { dst.get(ValueLayout.JAVA_FLOAT, it * 4L) }
        }
    }

    /** §0.4.354 — f64 twin of [bufferToHostF32]. */
    internal fun bufferToHostF64(bufferPtr: MemorySegment, nDoubles: Int): DoubleArray {
        Arena.ofConfined().use { scoped ->
            val sizeBytes = (nDoubles * 8).toLong()
            val dst = scoped.allocate(sizeBytes)
            val args = scoped.allocate(PjrtFfm.PJRT_Buffer_ToHostBuffer_Args_LAYOUT)
            args.set(JAVA_LONG, PjrtFfm.OFF_ToHost_StructSize, PjrtFfm.SZ_ToHost)
            args.set(ADDRESS, PjrtFfm.OFF_ToHost_Src, bufferPtr)
            args.set(ADDRESS, PjrtFfm.OFF_ToHost_Dst, dst)
            args.set(JAVA_LONG, PjrtFfm.OFF_ToHost_DstSize, sizeBytes)
            val errorPtr = toHost.invokeExact(args) as MemorySegment
            checkError(errorPtr)
            val event = args.get(ADDRESS, PjrtFfm.OFF_ToHost_Event)
            if (event.address() != 0L) {
                val eventFull = event.reinterpret(Long.MAX_VALUE)
                awaitEvent(eventFull)
                destroyEvent(eventFull)
            }
            return DoubleArray(nDoubles) { dst.get(ValueLayout.JAVA_DOUBLE, it * 8L) }
        }
    }

    /** §0.4.457 (G1c) — bf16 twin of [bufferToHostF32]: raw 16-bit patterns
     * out, no numeric conversion (widening is the caller's explicit act via
     * `bf16BitsToFloatArray`). */
    internal fun bufferToHostBf16(bufferPtr: MemorySegment, nElements: Int): ShortArray {
        Arena.ofConfined().use { scoped ->
            val sizeBytes = (nElements * 2).toLong()
            val dst = scoped.allocate(sizeBytes)
            val args = scoped.allocate(PjrtFfm.PJRT_Buffer_ToHostBuffer_Args_LAYOUT)
            args.set(JAVA_LONG, PjrtFfm.OFF_ToHost_StructSize, PjrtFfm.SZ_ToHost)
            args.set(ADDRESS, PjrtFfm.OFF_ToHost_Src, bufferPtr)
            args.set(ADDRESS, PjrtFfm.OFF_ToHost_Dst, dst)
            args.set(JAVA_LONG, PjrtFfm.OFF_ToHost_DstSize, sizeBytes)
            val errorPtr = toHost.invokeExact(args) as MemorySegment
            checkError(errorPtr)
            val event = args.get(ADDRESS, PjrtFfm.OFF_ToHost_Event)
            if (event.address() != 0L) {
                val eventFull = event.reinterpret(Long.MAX_VALUE)
                awaitEvent(eventFull)
                destroyEvent(eventFull)
            }
            return ShortArray(nElements) { dst.get(ValueLayout.JAVA_SHORT, it * 2L) }
        }
    }

    internal fun loadedExecDestroy(execPtr: MemorySegment) {
        Arena.ofConfined().use { scoped ->
            val args = scoped.allocate(PjrtFfm.PJRT_LoadedExecutable_Destroy_Args_LAYOUT)
            args.set(JAVA_LONG, PjrtFfm.OFF_LoadedExecDestroy_StructSize, PjrtFfm.SZ_LoadedExecDestroy)
            args.set(ADDRESS, PjrtFfm.OFF_LoadedExecDestroy_Executable, execPtr)
            val errorPtr = loadedExecDestroy.invokeExact(args) as MemorySegment
            checkError(errorPtr)
        }
    }

    internal fun loadedExecGetNumOutputs(loadedExecPtr: MemorySegment): Int {
        Arena.ofConfined().use { scoped ->
            // First convert to PJRT_Executable, then ask for num_outputs.
            val getExecArgs = scoped.allocate(PjrtFfm.PJRT_LoadedExecutable_GetExecutable_Args_LAYOUT)
            getExecArgs.set(JAVA_LONG, PjrtFfm.OFF_GetExec_StructSize, PjrtFfm.SZ_GetExec)
            getExecArgs.set(ADDRESS, PjrtFfm.OFF_GetExec_LoadedExec, loadedExecPtr)
            val errorPtr = getExecutable.invokeExact(getExecArgs) as MemorySegment
            checkError(errorPtr)
            val execPtr = getExecArgs.get(ADDRESS, PjrtFfm.OFF_GetExec_Executable).reinterpret(Long.MAX_VALUE)
            try {
                val numArgs = scoped.allocate(PjrtFfm.PJRT_Executable_NumOutputs_Args_LAYOUT)
                numArgs.set(JAVA_LONG, PjrtFfm.OFF_NumOutputs_StructSize, PjrtFfm.SZ_NumOutputs)
                numArgs.set(ADDRESS, PjrtFfm.OFF_NumOutputs_Executable, execPtr)
                val ne = numOutputs.invokeExact(numArgs) as MemorySegment
                checkError(ne)
                return numArgs.get(JAVA_LONG, PjrtFfm.OFF_NumOutputs_Count).toInt()
            } finally {
                // Free the PJRT_Executable handle returned by GetExecutable.
                val destroyArgs = scoped.allocate(PjrtFfm.PJRT_Executable_Destroy_Args_LAYOUT)
                destroyArgs.set(JAVA_LONG, PjrtFfm.OFF_ExecDestroy_StructSize, PjrtFfm.SZ_ExecDestroy)
                destroyArgs.set(ADDRESS, PjrtFfm.OFF_ExecDestroy_Executable, execPtr)
                val edError = executableDestroy.invokeExact(destroyArgs) as MemorySegment
                checkError(edError)
            }
        }
    }

    /** Single-device execute. [argBuffers] are the input PJRT_Buffer*s in
     * order; returns a list of [nOutputs] output PJRT_Buffer*s the caller
     * must destroy. */
    internal fun loadedExecExecuteSingleDevice(
        loadedExecPtr: MemorySegment,
        argBuffers: List<MemorySegment>,
        nOutputs: Int,
        executeDevicePtr: MemorySegment,
    ): List<MemorySegment> {
        Arena.ofConfined().use { scoped ->
            // Build argument_lists: PJRT_Buffer* const* const*. For 1 device,
            // a single-element outer array containing one pointer to the inner
            // (input PJRT_Buffer**) array.
            val innerArgs = scoped.allocate((argBuffers.size * 8).toLong())
            for ((i, b) in argBuffers.withIndex()) innerArgs.set(ADDRESS, i * 8L, b)
            val outerArgs = scoped.allocate(8)
            outerArgs.set(ADDRESS, 0L, innerArgs)

            // Build output_lists: PJRT_Buffer** const*. Outer is a 1-element
            // array of (PJRT_Buffer**); the inner is an nOutputs-sized array
            // of PJRT_Buffer* slots (zeroed; written by execute).
            val innerOutputs = scoped.allocate((nOutputs * 8).toLong())
            val outerOutputs = scoped.allocate(8)
            outerOutputs.set(ADDRESS, 0L, innerOutputs)

            // ExecuteOptions — all-zero except struct_size.
            val options = scoped.allocate(PjrtFfm.PJRT_ExecuteOptions_LAYOUT)
            options.set(JAVA_LONG, PjrtFfm.OFF_ExecOpts_StructSize, PjrtFfm.SZ_ExecOpts)

            // Slot for the device-complete event.
            val deviceCompleteEvents = scoped.allocate(8)

            val args = scoped.allocate(PjrtFfm.PJRT_LoadedExecutable_Execute_Args_LAYOUT)
            args.set(JAVA_LONG, PjrtFfm.OFF_Execute_StructSize, PjrtFfm.SZ_Execute)
            args.set(ADDRESS, PjrtFfm.OFF_Execute_Executable, loadedExecPtr)
            args.set(ADDRESS, PjrtFfm.OFF_Execute_Options, options)
            args.set(ADDRESS, PjrtFfm.OFF_Execute_ArgLists, outerArgs)
            args.set(JAVA_LONG, PjrtFfm.OFF_Execute_NumDevices, 1L)
            args.set(JAVA_LONG, PjrtFfm.OFF_Execute_NumArgs, argBuffers.size.toLong())
            args.set(ADDRESS, PjrtFfm.OFF_Execute_OutputLists, outerOutputs)
            args.set(ADDRESS, PjrtFfm.OFF_Execute_DeviceCompleteEvents, deviceCompleteEvents)
            args.set(ADDRESS, PjrtFfm.OFF_Execute_ExecuteDevice, executeDevicePtr)

            val errorPtr = execute.invokeExact(args) as MemorySegment
            checkError(errorPtr)

            // Await + destroy the device-complete event so subsequent buffer
            // reads observe the freshly-computed values.
            val event = deviceCompleteEvents.get(ADDRESS, 0L)
            if (event.address() != 0L) {
                val eventFull = event.reinterpret(Long.MAX_VALUE)
                awaitEvent(eventFull)
                destroyEvent(eventFull)
            }

            return List(nOutputs) { i ->
                innerOutputs.get(ADDRESS, i * 8L).reinterpret(Long.MAX_VALUE)
            }
        }
    }

    /**
     * §0.4.309 — fast-path single-device execute that takes pre-allocated
     * args / inner-args / inner-outputs / device-complete-events segments
     * from the caller (typically a [io.tlaloc.runtime.pjrt.PjrtSession]'s
     * long-lived arena). Saves ~10 µs/call vs [loadedExecExecuteSingleDevice]
     * by avoiding `Arena.ofConfined()` create+close per dispatch.
     *
     * Caller responsibilities:
     *   - [argsSegment] must be PJRT_LoadedExecutable_Execute_Args_LAYOUT-sized,
     *     with `executable`, `options`, `argument_lists`, `num_devices=1`,
     *     `output_lists`, `device_complete_events`, `execute_device` fields
     *     pre-populated. Only `num_args` is set per call.
     *   - [innerArgsSegment] is `nInputs * 8` bytes; caller fills with the
     *     current PJRT_Buffer pointers per call.
     *   - [innerOutputsSegment] is `nOutputs * 8` bytes; written by PJRT.
     *   - [deviceCompleteEventSlot] is an 8-byte pointer slot (in args).
     *
     * Returns the [nOutputs] output PJRT_Buffer pointers, freshly populated.
     * The device-complete event is awaited + destroyed before return.
     */
    internal fun executeReusable(
        argsSegment: MemorySegment,
        innerOutputsSegment: MemorySegment,
        deviceCompleteEventSlot: MemorySegment,
        nInputs: Int,
        nOutputs: Int,
    ): List<MemorySegment> {
        // Set per-call num_args; everything else is pre-populated.
        argsSegment.set(JAVA_LONG, PjrtFfm.OFF_Execute_NumArgs, nInputs.toLong())
        // Zero out output pointers from any previous call.
        for (i in 0 until nOutputs) innerOutputsSegment.set(ADDRESS, i * 8L, MemorySegment.NULL)
        // Zero out the device-complete-event slot.
        deviceCompleteEventSlot.set(ADDRESS, 0L, MemorySegment.NULL)

        val errorPtr = execute.invokeExact(argsSegment) as MemorySegment
        checkError(errorPtr)

        val event = deviceCompleteEventSlot.get(ADDRESS, 0L)
        if (event.address() != 0L) {
            val eventFull = event.reinterpret(Long.MAX_VALUE)
            awaitEvent(eventFull)
            destroyEvent(eventFull)
        }

        return List(nOutputs) { i ->
            innerOutputsSegment.get(ADDRESS, i * 8L).reinterpret(Long.MAX_VALUE)
        }
    }

    private fun awaitEvent(eventPtr: MemorySegment) {
        Arena.ofConfined().use { scoped ->
            val args = scoped.allocate(PjrtFfm.PJRT_Event_Await_Args_LAYOUT)
            args.set(JAVA_LONG, PjrtFfm.OFF_EventAwait_StructSize, PjrtFfm.SZ_EventAwait)
            args.set(ADDRESS, PjrtFfm.OFF_EventAwait_Event, eventPtr)
            val errorPtr = eventAwait.invokeExact(args) as MemorySegment
            checkError(errorPtr)
        }
    }

    internal fun destroyEvent(eventPtr: MemorySegment) {
        Arena.ofConfined().use { scoped ->
            val args = scoped.allocate(PjrtFfm.PJRT_Event_Destroy_Args_LAYOUT)
            args.set(JAVA_LONG, PjrtFfm.OFF_EventDestroy_StructSize, PjrtFfm.SZ_EventDestroy)
            args.set(ADDRESS, PjrtFfm.OFF_EventDestroy_Event, eventPtr)
            val errorPtr = eventDestroy.invokeExact(args) as MemorySegment
            checkError(errorPtr)
        }
    }
}

/**
 * Handle to a live `PJRT_Client*`. Use [close] (or `use {}`) to free it via
 * `PJRT_Client_Destroy`. Forgetting to close leaks the GPU client.
 */
class PjrtClient internal constructor(
    internal val clientPtr: MemorySegment,
    internal val api: PjrtApi,
) : AutoCloseable {
    fun platformName(): String {
        checkOpen("PjrtClient")
        return api.clientPlatformName(clientPtr)
    }

    /** §0.4.304 — list of addressable devices on this client. Element 0 is
     * the typical "default" device for single-GPU hosts. The returned [PjrtDevice]
     * handles share lifetime with [PjrtClient] (PJRT owns them; no destroy call). */
    fun addressableDevices(): List<PjrtDevice> {
        checkOpen("PjrtClient")
        return api.clientAddressableDevices(clientPtr).map { PjrtDevice(it) }
    }

    /** Compile [stablehloMlir] into a [PjrtLoadedExecutable]. The MLIR can be
     * either StableHLO bytecode or text — XLA's MLIR import accepts both
     * via the "mlir" PJRT_Program format. Caller takes ownership of the
     * returned executable and must `close()` it. */
    fun compile(stablehloMlir: String): PjrtLoadedExecutable {
        checkOpen("PjrtClient")
        // A confined arena lives only for the duration of the Compile call; the
        // PJRT plugin doesn't retain pointers into it past the return.
        Arena.ofConfined().use { scratch ->
            val execPtr = api.clientCompileMlir(clientPtr, stablehloMlir, scratch)
            return PjrtLoadedExecutable(execPtr, this)
        }
    }

    /** Stage an f32 host array onto [device] as a PJRT buffer. Caller takes
     * ownership of the returned [PjrtBuffer] and must `close()` it. */
    fun bufferFromHostF32(device: PjrtDevice, data: FloatArray, dims: List<Int>): PjrtBuffer {
        checkOpen("PjrtClient")
        Arena.ofConfined().use { scratch ->
            val bufPtr = api.bufferFromHostF32(clientPtr, device.devicePtr, data, dims, scratch)
            return PjrtBuffer(bufPtr, this)
        }
    }

    /** §0.4.354 — f64 twin of [bufferFromHostF32]. */
    fun bufferFromHostF64(device: PjrtDevice, data: DoubleArray, dims: List<Int>): PjrtBuffer {
        checkOpen("PjrtClient")
        Arena.ofConfined().use { scratch ->
            val bufPtr = api.bufferFromHostF64(clientPtr, device.devicePtr, data, dims, scratch)
            return PjrtBuffer(bufPtr, this)
        }
    }

    /** §0.4.457 (G1c) — bf16 twin of [bufferFromHostF32]. [data] is raw bit
     * patterns (the §0.4.455 ShortArray convention). */
    fun bufferFromHostBf16(device: PjrtDevice, data: ShortArray, dims: List<Int>): PjrtBuffer {
        checkOpen("PjrtClient")
        Arena.ofConfined().use { scratch ->
            val bufPtr = api.bufferFromHostBf16(clientPtr, device.devicePtr, data, dims, scratch)
            return PjrtBuffer(bufPtr, this)
        }
    }

    /** True once [close] has run. This client's own methods, and the buffers
     * and executables it made, check it, so a use after the client is gone
     * fails by name instead of calling into freed native memory. */
    @Volatile
    var isClosed: Boolean = false
        private set

    internal fun checkOpen(what: String) {
        check(!isClosed) { "$what used after its PJRT client was closed" }
    }

    override fun close() {
        synchronized(this) {
            if (isClosed) return
            isClosed = true
        }
        api.destroyClient(clientPtr)
    }
}

/** Handle to a `PJRT_Device*`. Owned by the [PjrtClient] — does not need
 * its own destroy call. */
class PjrtDevice internal constructor(internal val devicePtr: MemorySegment)

/** Handle to a `PJRT_Buffer*`. AutoCloseable — `close()` calls
 * `PJRT_Buffer_Destroy`. */
class PjrtBuffer internal constructor(
    internal val bufferPtr: MemorySegment,
    private val client: PjrtClient,
) : AutoCloseable {
    /** Pulls the buffer's contents back to host as an f32 array of [nFloats]
     * elements. Caller knows the expected size from compile-time type info. */
    fun toFloatArray(nFloats: Int): FloatArray = usable().api.bufferToHostF32(bufferPtr, nFloats)

    /** §0.4.354 — f64 twin of [toFloatArray]. */
    fun toDoubleArray(nDoubles: Int): DoubleArray = usable().api.bufferToHostF64(bufferPtr, nDoubles)

    /** §0.4.457 (G1c) — bf16 twin of [toFloatArray]: raw 16-bit patterns. */
    fun toBf16Array(nElements: Int): ShortArray = usable().api.bufferToHostBf16(bufferPtr, nElements)

    /** Size in bytes of the buffer's on-device storage (after layout +
     * padding). Useful for cross-checking against caller's expected size. */
    fun deviceSizeInBytes(): Long = usable().api.bufferDeviceSize(bufferPtr)

    private val closed = java.util.concurrent.atomic.AtomicBoolean(false)

    /** The owning client, after checking that neither this buffer nor the
     * client has been closed. */
    internal fun usable(): PjrtClient {
        check(!closed.get()) { "PjrtBuffer used after close()" }
        client.checkOpen("PjrtBuffer")
        return client
    }

    /** Destroys the device buffer. Idempotent. Does nothing once the owning
     * client is closed: the client released the buffer's memory with it. */
    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        if (client.isClosed) return
        client.api.bufferDestroy(bufferPtr)
    }
}

/** Handle to a `PJRT_LoadedExecutable*`. AutoCloseable — `close()` calls
 * `PJRT_LoadedExecutable_Destroy`. */
class PjrtLoadedExecutable internal constructor(
    internal val execPtr: MemorySegment,
    private val client: PjrtClient,
) : AutoCloseable {

    /** Number of outputs the executable produces per device. Cached on first
     * call so subsequent execute() invocations don't pay another round-trip. */
    val numOutputs: Int by lazy {
        client.checkOpen("PjrtLoadedExecutable")
        client.api.loadedExecGetNumOutputs(execPtr)
    }

    private val closed = java.util.concurrent.atomic.AtomicBoolean(false)

    /** Single-device execute. [argBuffers] are the input device buffers in
     * order; returns one output buffer per executable result (caller takes
     * ownership and must close). */
    fun execute(argBuffers: List<PjrtBuffer>, device: PjrtDevice): List<PjrtBuffer> {
        check(!closed.get()) { "PjrtLoadedExecutable used after close()" }
        client.checkOpen("PjrtLoadedExecutable")
        argBuffers.forEach { it.usable() }
        val outputPtrs = client.api.loadedExecExecuteSingleDevice(
            execPtr,
            argBuffers.map { it.bufferPtr },
            numOutputs,
            device.devicePtr,
        )
        return outputPtrs.map { PjrtBuffer(it, client) }
    }

    /** Destroys the executable. Idempotent. Does nothing once the owning
     * client is closed: the client released the executable with it. */
    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        if (client.isClosed) return
        client.api.loadedExecDestroy(execPtr)
    }
}

class PjrtRuntimeException(message: String) : RuntimeException(message)
