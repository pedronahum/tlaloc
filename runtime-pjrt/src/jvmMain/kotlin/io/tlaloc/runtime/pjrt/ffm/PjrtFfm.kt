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
        val pjrtApiPtr = (getPjrtApi.invokeExact() as MemorySegment).reinterpret(PJRT_API_OBSERVED_SIZE)
        return PjrtApi(pjrtApiPtr, arena)
    }

    // =========================================================================
    // Linker + low-level config.
    // =========================================================================

    internal val LINKER: Linker = Linker.nativeLinker()

    /** §0.4.304 widens the observed PJRT_Api prefix to 1024 bytes — enough
     * for every function pointer we use through PJRT_Buffer_ToHostBuffer
     * (offset 600). Subsequent commits can widen further as new PJRT calls
     * (DmaMap, AsyncTransferManager, …) come online. */
    internal const val PJRT_API_OBSERVED_SIZE: Long = 1024

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
     * suffice for single-device execute with no callbacks / contexts. */
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

        // Hand-encoded CompileOptionsProto.
        val optsBytes = byteArrayOf(0x1A, 0x04, 0x20, 0x01, 0x28, 0x01)
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
    fun platformName(): String = api.clientPlatformName(clientPtr)

    /** §0.4.304 — list of addressable devices on this client. Element 0 is
     * the typical "default" device for single-GPU hosts. The returned [PjrtDevice]
     * handles share lifetime with [PjrtClient] (PJRT owns them; no destroy call). */
    fun addressableDevices(): List<PjrtDevice> =
        api.clientAddressableDevices(clientPtr).map { PjrtDevice(it) }

    /** Compile [stablehloMlir] into a [PjrtLoadedExecutable]. The MLIR can be
     * either StableHLO bytecode or text — XLA's MLIR import accepts both
     * via the "mlir" PJRT_Program format. Caller takes ownership of the
     * returned executable and must `close()` it. */
    fun compile(stablehloMlir: String): PjrtLoadedExecutable {
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
        Arena.ofConfined().use { scratch ->
            val bufPtr = api.bufferFromHostF32(clientPtr, device.devicePtr, data, dims, scratch)
            return PjrtBuffer(bufPtr, this)
        }
    }

    override fun close() = api.destroyClient(clientPtr)
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
    fun toFloatArray(nFloats: Int): FloatArray = client.api.bufferToHostF32(bufferPtr, nFloats)

    /** Size in bytes of the buffer's on-device storage (after layout +
     * padding). Useful for cross-checking against caller's expected size. */
    fun deviceSizeInBytes(): Long = client.api.bufferDeviceSize(bufferPtr)

    override fun close() = client.api.bufferDestroy(bufferPtr)
}

/** Handle to a `PJRT_LoadedExecutable*`. AutoCloseable — `close()` calls
 * `PJRT_LoadedExecutable_Destroy`. */
class PjrtLoadedExecutable internal constructor(
    internal val execPtr: MemorySegment,
    private val client: PjrtClient,
) : AutoCloseable {

    /** Number of outputs the executable produces per device. Cached on first
     * call so subsequent execute() invocations don't pay another round-trip. */
    val numOutputs: Int by lazy { client.api.loadedExecGetNumOutputs(execPtr) }

    /** Single-device execute. [argBuffers] are the input device buffers in
     * order; returns one output buffer per executable result (caller takes
     * ownership and must close). */
    fun execute(argBuffers: List<PjrtBuffer>, device: PjrtDevice): List<PjrtBuffer> {
        val outputPtrs = client.api.loadedExecExecuteSingleDevice(
            execPtr,
            argBuffers.map { it.bufferPtr },
            numOutputs,
            device.devicePtr,
        )
        return outputPtrs.map { PjrtBuffer(it, client) }
    }

    override fun close() = client.api.loadedExecDestroy(execPtr)
}

class PjrtRuntimeException(message: String) : RuntimeException(message)
