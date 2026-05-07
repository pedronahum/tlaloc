package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirFunction
import io.tlaloc.runtime.pjrt.ffm.PjrtApi
import io.tlaloc.runtime.pjrt.ffm.PjrtClient
import io.tlaloc.runtime.pjrt.ffm.PjrtDevice
import io.tlaloc.runtime.pjrt.ffm.PjrtFfm
import io.tlaloc.runtime.pjrt.ffm.PjrtLoadedExecutable
import io.tlaloc.stablehlo.toStablehlo
import java.lang.foreign.Arena
import java.nio.file.Path
import java.util.concurrent.ConcurrentHashMap

/**
 * §0.4.307 — long-lived holder that amortises the PJRT init cost across many
 * dispatches. Without this, each `runOnPjrt` call pays ~500 ms of XLA service
 * init + CUDA context create + plugin dlopen. With this, the steady-state
 * per-call cost is dominated by the actual compute (sub-ms for tiny
 * workloads).
 *
 * Session-scoped resources:
 *   - one [java.lang.foreign.Arena] that lives for the whole session,
 *   - one [PjrtApi] (plugin dlopen + GetPjrtApi),
 *   - one [PjrtClient] (XLA service init + CUDA context create),
 *   - one [PjrtDevice] (the first addressable device — the typical
 *     "default GPU" for single-GPU hosts),
 *   - a [ConcurrentHashMap] compile-cache keyed by StableHLO MLIR text:
 *     the first call with a new DxirFunction compiles + caches; subsequent
 *     calls reuse the [PjrtLoadedExecutable].
 *
 * # Usage shapes
 *
 * One-shot dispatch (no caching benefit, but composes cleanly):
 *
 * ```
 * PjrtSession().use { session ->
 *     val outputs = session.runOn(fn, inputs)
 * }
 * ```
 *
 * Training loop / benchmark (the whole point of this class):
 *
 * ```
 * PjrtSession().use { session ->
 *     repeat(1000) { session.runOn(fn, inputs) }   // first iter compiles, rest reuse
 * }
 * ```
 *
 * Multi-fn (each unique DxirFunction caches independently):
 *
 * ```
 * PjrtSession().use { session ->
 *     val fwd = session.runOn(forwardFn, inputs)
 *     val grads = session.runOn(backwardFn, gradInputs)   // separate cache slot
 * }
 * ```
 *
 * # Thread-safety
 *
 * The compile-cache uses [ConcurrentHashMap.computeIfAbsent], so concurrent
 * callers with the same MLIR will compile exactly once and the others wait.
 * Different MLIR strings compile in parallel. Dispatch (`execute`) itself is
 * not parallelised — PJRT serialises GPU work onto the device's stream — so
 * concurrent callers will block each other on the underlying device, but
 * the compile-cache is the right shape for that workload.
 *
 * # Lifetime
 *
 * `close()` walks: cached executables → client → arena. The arena's close
 * releases all MemorySegments allocated against it; client/exec destroy
 * downcalls happen before that, so the args structs they marshal are still
 * valid memory at the time of the call.
 *
 * # v1 limitations (carried from runOnPjrt)
 *
 *   - F32 only.
 *   - Single-device dispatch (the first addressable device).
 *   - Plugin distribution depends on a JAX install or
 *     `TLALOC_PJRT_PLUGIN_PATH`; CPU plugin is "build from source"
 *     (per §0.4.306 doc).
 */
class PjrtSession(
    plugin: Path = PjrtBinaries.pluginPath
        ?: error("PJRT plugin not resolved; set TLALOC_PJRT_PLUGIN_PATH or `pip install jax[cuda12]`"),
    val target: PjrtTarget = PjrtTarget.Cuda,
) : AutoCloseable {

    private val arena: Arena = Arena.ofShared()
    private val api: PjrtApi = PjrtFfm.load(plugin, arena)
    private val client: PjrtClient = api.createClient()

    /** First addressable device — the "default" GPU for single-GPU hosts. */
    val device: PjrtDevice = client.addressableDevices().firstOrNull()
        ?: error("PJRT client has no addressable devices for $target")

    private val executableCache = ConcurrentHashMap<String, PjrtLoadedExecutable>()

    @Volatile
    private var closed = false

    /**
     * Compile [fn] (or fetch from cache) and execute against [inputs]. Returns
     * one [FloatArray] per [DxirFunction.returns], sized by `return.type.elementCount`.
     *
     * The cache key is the StableHLO MLIR text [fn.toStablehlo] produces —
     * structurally identical DxirFunctions hit the same cache slot.
     */
    fun runOn(fn: DxirFunction, inputs: List<FloatArray>): List<FloatArray> {
        check(!closed) { "PjrtSession is closed" }
        require(fn.params.size == inputs.size) {
            "PjrtSession.runOn: param count ${fn.params.size} != input count ${inputs.size}"
        }
        for ((i, p) in fn.params.withIndex()) {
            require(p.type.dtype == F32) {
                "PjrtSession.runOn: param '${p.name}' dtype is ${p.type.dtype}; v1 only supports F32"
            }
            val expected = p.type.elementCount.toInt()
            require(inputs[i].size == expected) {
                "PjrtSession.runOn: param '${p.name}' expects size $expected (type ${p.type}) " +
                    "but received input of size ${inputs[i].size}"
            }
        }
        for ((i, r) in fn.returns.withIndex()) {
            require(r.type.dtype == F32) {
                "PjrtSession.runOn: return[$i] dtype is ${r.type.dtype}; v1 only supports F32"
            }
        }

        val mlir = fn.toStablehlo("")
        val exec = executableCache.computeIfAbsent(mlir) { client.compile(mlir) }
        require(exec.numOutputs == fn.returns.size) {
            "PjrtSession.runOn: PJRT executable reports numOutputs=${exec.numOutputs}; " +
                "DxirFunction declares ${fn.returns.size} returns"
        }

        val inputBuffers = fn.params.zip(inputs).map { (p, arr) ->
            client.bufferFromHostF32(device, arr, p.type.dims)
        }
        try {
            val outputs = exec.execute(inputBuffers, device)
            try {
                return outputs.zip(fn.returns).map { (buf, ret) ->
                    buf.toFloatArray(ret.type.elementCount.toInt())
                }
            } finally {
                outputs.forEach { it.close() }
            }
        } finally {
            inputBuffers.forEach { it.close() }
        }
    }

    /**
     * Force-compile [fn] now without dispatching. Useful for benchmark setup
     * where you want the compile cost outside the timing loop. Idempotent
     * (a second call with structurally identical [fn] is a no-op).
     */
    fun prepare(fn: DxirFunction) {
        check(!closed) { "PjrtSession is closed" }
        val mlir = fn.toStablehlo("")
        executableCache.computeIfAbsent(mlir) { client.compile(mlir) }
    }

    /** Number of executables currently in the compile cache. Useful for
     * tests asserting cache hit/miss behaviour. */
    val cacheSize: Int get() = executableCache.size

    override fun close() {
        if (closed) return
        closed = true
        executableCache.values.forEach { runCatching { it.close() } }
        executableCache.clear()
        runCatching { client.close() }
        runCatching { arena.close() }
    }
}
