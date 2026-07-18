package io.tlaloc.runtime.pjrt.kptx

import io.tlaloc.runtime.cuda.CudaDriverFfm
import io.tlaloc.runtime.pjrt.ffm.PjrtFfiRegistry
import io.tlaloc.runtime.pjrt.ffm.XlaFfi
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_LONG
import java.nio.file.Path
import java.util.concurrent.ConcurrentHashMap

/**
 * KPTX v1.4 (§0.4.330) — the kernel launch registry: name →
 * [LaunchConfig], dispatched inside XLA-compiled PJRT executables.
 *
 * [registerKernel] wires the full path proven piecewise in §0.4.326–329:
 * a [PjrtFfiRegistry] typed-FFI handler that, per execution, decodes the
 * [XlaFfi] call frame, resolves the driver-JIT'd CUfunction (cached),
 * marshals the `void**` kernel-params array — **inputs then outputs**,
 * each a pointer to that buffer's device pointer, the pyptx convention —
 * and `cuLaunchKernel`s **on XLA's stream**.
 *
 * Design decisions (see docs/KPTX_PLAN.md):
 * - **Per-kernel-name registration**, not pyptx's single-handler +
 *   handle-attribute: Tlaloc's emitter names kernels symbolically
 *   (`stablehlo.custom_call @<kernelName>`, §0.4.261), so the name *is*
 *   the natural key and the MLIR stays free of process-local handles.
 * - **Framework owns memory and streams**: the handler never allocates
 *   tensor data, never synchronizes, never switches CUDA context.
 * - **Lazy module load in the handler thread**: CUmodule handles are
 *   per-CUcontext, and XLA's context is only guaranteed current on the
 *   FFI callback thread — loading at registration time (primary context)
 *   would yield handles unusable from XLA's context. First-call latency
 *   is the driver-JIT cost; subsequent calls hit [functionCache].
 * - **Failures become `XLA_FFI_Error`s** ([XlaFfi.Frame.createError]),
 *   failing the one execution instead of tearing down the JVM.
 */
object KptxKernelRegistry {

    /** Grid dimensions, possibly derived from the runtime shapes of the
     * call's input buffers. */
    data class Dim3(val x: Int, val y: Int = 1, val z: Int = 1)

    /**
     * Everything needed to launch one kernel.
     *
     * @param ptx PTX text; driver-JIT compiled on first dispatch
     *   (`.target` must be ≤ the device's compute capability).
     * @param entryName the `.visible .entry` symbol inside [ptx].
     * @param grid grid-dims from the decoded input buffers (runtime shapes).
     * @param block CTA dims.
     * @param sharedMemBytes dynamic SMEM per CTA; values > 48 KB
     *   automatically set CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES.
     * @param trailingI32Params extra scalar i32 params appended after the
     *   buffer pointers (e.g. element counts), derived per call from the
     *   decoded frame. Empty by default: kernels that can read shapes from
     *   a fixed signature need nothing else at v1.
     */
    data class LaunchConfig(
        val ptx: String,
        val entryName: String,
        val grid: (args: List<XlaFfi.Buffer>) -> Dim3,
        val block: Dim3,
        val sharedMemBytes: Int = 0,
        val trailingI32Params: (args: List<XlaFfi.Buffer>) -> IntArray = { IntArray(0) },
    )

    private val driverArena: Arena = Arena.ofShared()
    private val cuda: CudaDriverFfm by lazy { CudaDriverFfm.load(driverArena) }

    /** (ptx-identity, entry) → CUfunction, resolved lazily in the handler
     * thread (XLA's context current). Modules are never unloaded — they
     * live as long as the registration, i.e. the process. */
    private val functionCache = ConcurrentHashMap<Pair<Int, String>, MemorySegment>()

    private val registered = ConcurrentHashMap<Pair<Path, String>, LaunchConfig>()

    /**
     * Registers [config] under custom-call target [name] on the PJRT
     * plugin at [pluginPath]. After this, XLA programs may name
     * `stablehlo.custom_call @<name> {api_version = 4 : i32}` with any
     * number of tensor operands/results; the kernel receives their device
     * pointers in operand order followed by result order.
     *
     * Process-permanent, like the underlying FFI registration; duplicate
     * (plugin, name) throws [IllegalStateException].
     */
    fun registerKernel(pluginPath: Path, name: String, config: LaunchConfig) {
        val key = pluginPath to name
        if (registered.putIfAbsent(key, config) != null) {
            throw IllegalStateException("KPTX kernel '$name' is already registered for $pluginPath")
        }
        try {
            PjrtFfiRegistry.registerExecuteHandler(pluginPath, name) { framePtr ->
                dispatch(name, config, framePtr)
            }
        } catch (t: Throwable) {
            registered.remove(key)
            throw t
        }
    }

    private fun dispatch(name: String, config: LaunchConfig, framePtr: MemorySegment): MemorySegment {
        val frame = XlaFfi.decode(framePtr)
        return try {
            val stream = frame.streamGet()
            val function = functionCache.computeIfAbsent(System.identityHashCode(config.ptx) to config.entryName) {
                val module = cuda.moduleLoadPtx(config.ptx)
                val fn = cuda.moduleGetFunction(module, config.entryName)
                if (config.sharedMemBytes > 48 * 1024) {
                    cuda.funcSetAttribute(
                        fn, cuda.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, config.sharedMemBytes,
                    )
                }
                fn
            }

            Arena.ofConfined().use { scoped ->
                val buffers = frame.args + frame.rets
                val scalars = config.trailingI32Params(frame.args)
                val nParams = buffers.size + scalars.size
                val params = scoped.allocate(nParams * 8L)
                for ((i, buf) in buffers.withIndex()) {
                    val slot = scoped.allocate(JAVA_LONG).also { it.set(JAVA_LONG, 0L, buf.dataAddress) }
                    params.set(ADDRESS, i * 8L, slot)
                }
                for ((i, v) in scalars.withIndex()) {
                    val slot = scoped.allocate(java.lang.foreign.ValueLayout.JAVA_INT).also {
                        it.set(java.lang.foreign.ValueLayout.JAVA_INT, 0L, v)
                    }
                    params.set(ADDRESS, (buffers.size + i) * 8L, slot)
                }

                val grid = config.grid(frame.args)
                cuda.launchKernel(
                    function,
                    grid.x, grid.y, grid.z,
                    config.block.x, config.block.y, config.block.z,
                    config.sharedMemBytes,
                    stream,
                    params,
                )
            }
            MemorySegment.NULL
        } catch (t: Throwable) {
            frame.createError("KPTX kernel '$name' dispatch failed: ${t.message ?: t.javaClass.simpleName}")
        }
    }
}
