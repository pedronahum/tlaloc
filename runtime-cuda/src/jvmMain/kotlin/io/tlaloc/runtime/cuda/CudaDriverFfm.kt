package io.tlaloc.runtime.cuda

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_BYTE
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.ValueLayout.JAVA_LONG
import java.lang.invoke.MethodHandle
import java.nio.charset.StandardCharsets

/**
 * Hand-written FFM bindings for the **CUDA driver
 * API** (`libcuda.so.1`). The KPTX kernel tier loads hand-written /
 * DSL-emitted PTX through the driver JIT (`cuModuleLoadData`) and launches
 * with `cuLaunchKernel` — no CUDA *toolkit* dependency at runtime, only the
 * driver the GPU host already has, so no Python is needed at runtime.
 *
 * Conventions follow [io.tlaloc.runtime.pjrt.ffm.PjrtFfm] exactly:
 * `MemorySegment.set/get` (never VarHandles — the kotlinc
 * signature-polymorphism gotcha documented there), `invokeExact` with an
 * `as` cast, offsets/descriptors as constants.
 *
 * # Symbol versioning
 *
 * `libcuda` exports legacy 32-bit-era entry points under the original
 * names; the CUDA headers `#define` the modern ones to `_v2` suffixes
 * (`cuMemAlloc` → `cuMemAlloc_v2`, …). Binding the unversioned memory
 * symbols silently corrupts pointers — this file binds the `_v2` names
 * for every call that has one (`cuMemAlloc_v2`, `cuMemFree_v2`,
 * `cuMemcpyHtoD_v2`, `cuMemcpyDtoH_v2`). `CUdeviceptr` is
 * `unsigned long long` → [JAVA_LONG].
 *
 * # Contexts
 *
 * Standalone use (tests, future native runtime) drives the **primary
 * context** (`cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent`) — the same
 * context CUDA runtime-API users (and XLA) share, so module handles are
 * usable across both worlds. In the PJRT dispatch path, XLA's context is
 * already current on the FFI callback thread; handlers must *not* switch
 * contexts, only launch onto XLA's stream.
 */
class CudaDriverFfm private constructor(private val arena: Arena) {

    companion object {
        private val LINKER: Linker = Linker.nativeLinker()

        /** Loads `libcuda.so.1` and calls `cuInit(0)`. [arena] owns the
         * library lifetime — pass a long-lived arena (`Arena.ofShared()`);
         * unloading libcuda mid-process is never useful. Throws
         * [CudaDriverException] if the driver is missing or init fails. */
        fun load(arena: Arena): CudaDriverFfm {
            val cuda = CudaDriverFfm(arena)
            cuda.check("cuInit", cuda.cuInit.invokeExact(0) as Int)
            return cuda
        }
    }

    private val lookup: SymbolLookup = try {
        SymbolLookup.libraryLookup("libcuda.so.1", arena)
    } catch (e: IllegalArgumentException) {
        throw CudaDriverException("libcuda.so.1 not found — no NVIDIA driver on this host", e)
    }

    private fun handle(symbol: String, descriptor: FunctionDescriptor): MethodHandle =
        LINKER.downcallHandle(
            lookup.find(symbol).orElseThrow { CudaDriverException("libcuda.so.1 does not export $symbol") },
            descriptor,
        )

    // CUresult fn(...) — every driver call returns an int status.
    private val cuInit = handle("cuInit", FunctionDescriptor.of(JAVA_INT, JAVA_INT))
    private val cuDeviceGet = handle("cuDeviceGet", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT))
    private val cuDevicePrimaryCtxRetain = handle("cuDevicePrimaryCtxRetain", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT))
    private val cuDevicePrimaryCtxRelease = handle("cuDevicePrimaryCtxRelease_v2", FunctionDescriptor.of(JAVA_INT, JAVA_INT))
    private val cuCtxSetCurrent = handle("cuCtxSetCurrent", FunctionDescriptor.of(JAVA_INT, ADDRESS))
    private val cuCtxSynchronize = handle("cuCtxSynchronize", FunctionDescriptor.of(JAVA_INT))
    private val cuModuleLoadData = handle("cuModuleLoadData", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS))
    private val cuModuleUnload = handle("cuModuleUnload", FunctionDescriptor.of(JAVA_INT, ADDRESS))
    private val cuModuleGetFunction = handle("cuModuleGetFunction", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS))
    private val cuFuncSetAttribute = handle("cuFuncSetAttribute", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT))
    private val cuLaunchKernel = handle(
        "cuLaunchKernel",
        FunctionDescriptor.of(
            JAVA_INT,
            ADDRESS,                                  // CUfunction
            JAVA_INT, JAVA_INT, JAVA_INT,             // grid x, y, z
            JAVA_INT, JAVA_INT, JAVA_INT,             // block x, y, z
            JAVA_INT,                                 // sharedMemBytes
            ADDRESS,                                  // CUstream
            ADDRESS,                                  // void** kernelParams
            ADDRESS,                                  // void** extra
        ),
    )
    private val cuMemAlloc = handle("cuMemAlloc_v2", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG))
    private val cuMemFree = handle("cuMemFree_v2", FunctionDescriptor.of(JAVA_INT, JAVA_LONG))
    private val cuMemcpyHtoD = handle("cuMemcpyHtoD_v2", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG))
    private val cuMemcpyDtoH = handle("cuMemcpyDtoH_v2", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG))
    private val cuGetErrorName = handle("cuGetErrorName", FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS))
    private val cuGetErrorString = handle("cuGetErrorString", FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS))

    /** CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES (cuda.h enum value 8)
     * — required before launching kernels using > 48 KB dynamic SMEM. */
    val FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: Int = 8

    // -------------------------------------------------------------------------
    // Error decode.
    // -------------------------------------------------------------------------

    private fun errorText(code: Int): String {
        Arena.ofConfined().use { scoped ->
            val nameSlot = scoped.allocate(ADDRESS)
            val strSlot = scoped.allocate(ADDRESS)
            val nameOk = (cuGetErrorName.invokeExact(code, nameSlot) as Int) == 0
            val strOk = (cuGetErrorString.invokeExact(code, strSlot) as Int) == 0
            val name = if (nameOk) readCString(nameSlot.get(ADDRESS, 0L)) else "CUDA_ERROR_$code"
            val str = if (strOk) readCString(strSlot.get(ADDRESS, 0L)) else "unknown error"
            return "$name ($code): $str"
        }
    }

    private fun readCString(ptr: MemorySegment): String {
        if (ptr.address() == 0L) return ""
        val full = ptr.reinterpret(Long.MAX_VALUE)
        var len = 0L
        while (full.get(JAVA_BYTE, len) != 0.toByte()) len++
        val bytes = ByteArray(len.toInt()) { full.get(JAVA_BYTE, it.toLong()) }
        return String(bytes, StandardCharsets.UTF_8)
    }

    private fun check(call: String, result: Int) {
        if (result != 0) throw CudaDriverException("$call failed: ${errorText(result)}")
    }

    // -------------------------------------------------------------------------
    // Device / context.
    // -------------------------------------------------------------------------

    /** Returns the CUdevice handle (an int) for [ordinal]. */
    fun deviceGet(ordinal: Int): Int {
        Arena.ofConfined().use { scoped ->
            val slot = scoped.allocate(JAVA_INT)
            check("cuDeviceGet", cuDeviceGet.invokeExact(slot, ordinal) as Int)
            return slot.get(JAVA_INT, 0L)
        }
    }

    /** Retains the device's primary context and makes it current on this
     * thread. Pair with [primaryCtxRelease]. Returns the CUcontext. */
    fun primaryCtxRetainAndSetCurrent(device: Int): MemorySegment {
        Arena.ofConfined().use { scoped ->
            val slot = scoped.allocate(ADDRESS)
            check("cuDevicePrimaryCtxRetain", cuDevicePrimaryCtxRetain.invokeExact(slot, device) as Int)
            val ctx = slot.get(ADDRESS, 0L)
            check("cuCtxSetCurrent", cuCtxSetCurrent.invokeExact(ctx) as Int)
            return ctx
        }
    }

    fun primaryCtxRelease(device: Int) =
        check("cuDevicePrimaryCtxRelease", cuDevicePrimaryCtxRelease.invokeExact(device) as Int)

    /** Blocks until all work in the current context completes. */
    fun ctxSynchronize() = check("cuCtxSynchronize", cuCtxSynchronize.invokeExact() as Int)

    // -------------------------------------------------------------------------
    // Modules / functions.
    // -------------------------------------------------------------------------

    /** Driver-JIT compiles [ptx] (NUL-terminated internally) and returns the
     * CUmodule. The PTX `.target` must be ≤ the device's compute capability;
     * the driver JITs up to the device's actual SASS. */
    fun moduleLoadPtx(ptx: String): MemorySegment {
        Arena.ofConfined().use { scoped ->
            val bytes = ptx.toByteArray(StandardCharsets.UTF_8)
            val image = scoped.allocate(bytes.size + 1L)
            for ((i, b) in bytes.withIndex()) image.set(JAVA_BYTE, i.toLong(), b)
            image.set(JAVA_BYTE, bytes.size.toLong(), 0)
            val slot = scoped.allocate(ADDRESS)
            check("cuModuleLoadData", cuModuleLoadData.invokeExact(slot, image) as Int)
            return slot.get(ADDRESS, 0L)
        }
    }

    fun moduleUnload(module: MemorySegment) =
        check("cuModuleUnload", cuModuleUnload.invokeExact(module) as Int)

    /** Resolves the kernel [name] (the `.entry` symbol) in [module]. */
    fun moduleGetFunction(module: MemorySegment, name: String): MemorySegment {
        Arena.ofConfined().use { scoped ->
            val slot = scoped.allocate(ADDRESS)
            val nameSeg = scoped.allocateFrom(name)
            check("cuModuleGetFunction", cuModuleGetFunction.invokeExact(slot, module, nameSeg) as Int)
            return slot.get(ADDRESS, 0L)
        }
    }

    fun funcSetAttribute(function: MemorySegment, attribute: Int, value: Int) =
        check("cuFuncSetAttribute", cuFuncSetAttribute.invokeExact(function, attribute, value) as Int)

    // -------------------------------------------------------------------------
    // Launch.
    // -------------------------------------------------------------------------

    /**
     * `cuLaunchKernel`. [kernelParams] is the `void**` array: one 8-byte
     * pointer per kernel parameter, each pointing at that parameter's
     * *value* (for a device-pointer param: a slot holding the CUdeviceptr).
     * Pass [MemorySegment.NULL] for [stream] to use the legacy default
     * stream; in the PJRT dispatch path always pass XLA's stream.
     */
    fun launchKernel(
        function: MemorySegment,
        gridX: Int, gridY: Int, gridZ: Int,
        blockX: Int, blockY: Int, blockZ: Int,
        sharedMemBytes: Int,
        stream: MemorySegment,
        kernelParams: MemorySegment,
    ) = check(
        "cuLaunchKernel",
        cuLaunchKernel.invokeExact(
            function,
            gridX, gridY, gridZ,
            blockX, blockY, blockZ,
            sharedMemBytes,
            stream,
            kernelParams,
            MemorySegment.NULL,
        ) as Int,
    )

    // -------------------------------------------------------------------------
    // Memory (standalone / test / future-native-runtime use; the PJRT path
    // never allocates tensor data — XLA owns those buffers).
    // -------------------------------------------------------------------------

    /** Allocates [bytes] of device memory; returns the CUdeviceptr. */
    fun memAlloc(bytes: Long): Long {
        Arena.ofConfined().use { scoped ->
            val slot = scoped.allocate(JAVA_LONG)
            check("cuMemAlloc_v2", cuMemAlloc.invokeExact(slot, bytes) as Int)
            return slot.get(JAVA_LONG, 0L)
        }
    }

    fun memFree(devicePtr: Long) = check("cuMemFree_v2", cuMemFree.invokeExact(devicePtr) as Int)

    fun memcpyHtoD(devicePtr: Long, src: MemorySegment, bytes: Long) =
        check("cuMemcpyHtoD_v2", cuMemcpyHtoD.invokeExact(devicePtr, src, bytes) as Int)

    fun memcpyDtoH(dst: MemorySegment, devicePtr: Long, bytes: Long) =
        check("cuMemcpyDtoH_v2", cuMemcpyDtoH.invokeExact(dst, devicePtr, bytes) as Int)
}

class CudaDriverException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)
