package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirFunction
import io.tlaloc.runtime.pjrt.ffm.PjrtApi
import io.tlaloc.runtime.pjrt.ffm.PjrtBuffer
import io.tlaloc.runtime.pjrt.ffm.PjrtClient
import io.tlaloc.runtime.pjrt.ffm.PjrtDevice
import io.tlaloc.runtime.pjrt.ffm.PjrtFfm
import io.tlaloc.runtime.pjrt.ffm.PjrtLoadedExecutable
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_LONG
import io.tlaloc.stablehlo.toStablehlo
import io.tlaloc.ir.recognizer.kernel.lowerMosaicKernels
import java.lang.foreign.Arena
import java.nio.file.Path
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.locks.ReentrantReadWriteLock
import io.tlaloc.runtime.pjrt.ffm.PjrtClientOptions

/**
 * Long-lived holder that amortises the PJRT init cost across many
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
 * Every public method may be called from any number of threads at once.
 *
 *   - The compile cache uses [ConcurrentHashMap.computeIfAbsent], so
 *     concurrent callers with the same program compile it once and the
 *     others wait; different programs compile in parallel.
 *   - [runOn], [runOnF64] and [runOnBf16] marshal through per-call scratch
 *     memory and share no mutable state between calls.
 *   - [executeOn] reuses one pre-allocated argument block per executable.
 *     Calls on the same executable are serialised on that block (fill,
 *     execute, await, read back the output pointers); calls on different
 *     executables do not wait for each other on the JVM side. The device
 *     still orders the GPU work itself.
 *
 * # Lifetime
 *
 * [close] waits for every in-flight call on this session to return, then
 * destroys the cached executables, the client and the arena, in that order.
 * A call that starts after [close] fails with "PjrtSession is closed".
 *
 * [PjrtBuffer]s handed out by [bufferFromHostF32], [bufferFromHostBf16] and
 * [executeOn] belong to the caller and are not tracked by the session:
 * close them before closing the session. After the session is closed,
 * reading such a buffer throws and closing it does nothing (the client
 * that owned the device memory has already released it).
 *
 * # Limitations
 *
 *   - [runOn] takes F32 and I32 (float-encoded), [runOnF64] all-F64,
 *     [runOnBf16] all-BF16. Programs that compute in bf16
 *     ride [runOn] with in-graph CASTs (the cast-at-boundary pattern).
 *   - Single-device dispatch (the first addressable device).
 *   - Plugin distribution depends on a JAX install or
 *     `TLALOC_PJRT_PLUGIN_PATH`; a CPU plugin has to be built from XLA
 *     source.
 */
class PjrtSession(
    plugin: Path = PjrtBinaries.requireCudaPlugin("PjrtSession"),
    val target: PjrtTarget = PjrtTarget.Cuda,
    /** Allocator options for the underlying client. The
     * env-resolved default (`preallocate=false`, fraction 0.5) is the
     * unified-memory-safe choice; benchmark sessions may opt
     * into a bounded preallocated pool for allocation-latency-free
     * dispatch (see [io.tlaloc.runtime.pjrt.ffm.PjrtClientOptions]).
     *
     * Nullable, and the nullability is exactly the
     * platform gate: `memory_fraction` / `preallocate` are GPU-plugin
     * allocator options, so a [PjrtTarget.Tpu] session defaults to (and
     * must keep) null — the client is created with zero create_options —
     * while every other target defaults to (and must keep) the
     * env-resolved options. Both cross-wirings refuse by name in init. */
    private val options: io.tlaloc.runtime.pjrt.ffm.PjrtClientOptions? =
        if (target == PjrtTarget.Tpu) null
        else io.tlaloc.runtime.pjrt.ffm.PjrtClientOptions.resolve(),
) : AutoCloseable {

    /**
     * Lower f32 dots that the emitter spells as an explicit dot algorithm as
     * `precision = [HIGHEST, HIGHEST]` instead ([io.tlaloc.stablehlo.portableF32Dots]).
     * On by default for a TPU, whose accepted dot algorithms differ from the
     * GPU's. Set it before the first call; programs already compiled keep
     * the spelling they were compiled with.
     */
    @Volatile
    var portableF32Dots: Boolean = target == PjrtTarget.Tpu

    init {
        // §0.4.459 (G2a) — the platform gate, refused by name in BOTH
        // directions and BEFORE any FFM work (this block precedes the
        // arena/api/client property initializers):
        require(target != PjrtTarget.Tpu || options == null) {
            "PjrtSession: PjrtClientOptions (memory_fraction/preallocate) are the XLA GPU " +
                "plugin's allocator options — a PjrtTarget.Tpu client must not be handed " +
                "them (pass options = null; libtpu create-options are not wired, " +
                "see docs/TPU_BRINGUP.md)"
        }
        require(target == PjrtTarget.Tpu || options != null) {
            "PjrtSession: a $target client must always carry PjrtClientOptions — creating " +
                "a CUDA client with zero create_options preallocates 75% of GPU memory, which " +
                "on a unified-memory machine can exhaust system RAM; null is reserved " +
                "for PjrtTarget.Tpu"
        }
    }

    /** The native handles, opened together so that a failure part-way
     * (the plugin will not load, client creation fails, no device) releases
     * whatever was already opened instead of leaking the arena or client. */
    internal class Handles(val arena: Arena, val api: PjrtApi, val client: PjrtClient, val device: PjrtDevice)

    private val handles: Handles = openHandles(plugin, target, options)
    private val arena: Arena get() = handles.arena
    private val api: PjrtApi get() = handles.api
    private val client: PjrtClient get() = handles.client

    /** First addressable device — the "default" GPU for single-GPU hosts. */
    val device: PjrtDevice get() = handles.device

    /**
     * Guards the session's lifetime. Every public operation holds the read
     * lock for its whole duration, so any number of them run concurrently;
     * [close] takes the write lock, so it waits for in-flight calls to return
     * and no call can start while it frees native memory.
     */
    private val lifecycle = ReentrantReadWriteLock()

    /** Runs [block] as an in-flight session call. Tests use it to hold a
     * call open while [close] is attempted. */
    internal fun <T> asInFlightCall(block: () -> T): T = live(block)

    private inline fun <T> live(block: () -> T): T {
        val read = lifecycle.readLock()
        read.lock()
        try {
            check(!closed) { "PjrtSession is closed" }
            return block()
        } finally {
            read.unlock()
        }
    }

    private val executableCache = ConcurrentHashMap<String, PjrtLoadedExecutable>()

    /**
     * Pre-allocated execute context per cached executable.
     * Holds the args struct + inner args/outputs arrays + options + event slot
     * in the session arena so [executeOn] doesn't allocate-and-free a confined
     * arena on each dispatch. Lazily built on first [executeOn] (we don't know
     * the input count from the executable alone).
     */
    private class ExecuteContext(
        val argsSegment: MemorySegment,
        val innerArgsSegment: MemorySegment,
        val innerOutputsSegment: MemorySegment,
        val deviceCompleteEventSlot: MemorySegment,
        val nInputs: Int,
        val nOutputs: Int,
    )

    private val executeContextCache = ConcurrentHashMap<String, ExecuteContext>()

    @Volatile
    private var closed = false

    /**
     * The cheap front door to the compile cache.
     *
     * The back-stop key is the emitted StableHLO text, which is correct
     * (structurally identical programs hash together) but costs a full
     * re-emission of the program on every lookup. At decode rates that is the
     * one thing a cache is supposed to avoid. A caller that can name its
     * program with a short string — `DecodeGraphSpec.executableCacheKey`,
     * which is `modelHash / kind / bucket / dtypes` — passes it as `cacheKey`
     * and the emission happens ONCE per key.
     *
     * The contract on that string is absolute: **the key must fully determine
     * the program.** On a hit nothing is re-emitted, so there is no place to
     * notice that a key was reused for a different graph — which is exactly
     * why `executableCacheKey` carries a content-addressed model hash rather
     * than a friendly name.
     */
    private val keyedMlir = ConcurrentHashMap<String, String>()

    private fun lower(fn: DxirFunction, cacheKey: String?): String =
        if (cacheKey == null) emit(fn) else keyedMlir.computeIfAbsent(cacheKey) { emit(fn) }

    /**
     * [fn] as StableHLO for this session's [target]. MOSAIC_KERNEL ops are
     * resolved first: claimed as `tpu_custom_call` on a TPU, replaced by
     * their reference elsewhere when they declare a fallback, refused by
     * name otherwise (see [lowerMosaicKernels]). With [portableF32Dots], f32
     * dot algorithms are then spelled as `HIGHEST` precision.
     */
    private fun emit(fn: DxirFunction): String {
        val mlir = lowerMosaicKernels(fn, target.kernelTarget).toStablehlo("")
        return if (portableF32Dots) io.tlaloc.stablehlo.portableF32Dots(mlir) else mlir
    }

    /** Number of distinct caller-supplied cache keys seen. Tests assert that a
     *  repeated key does not re-emit. */
    val keyedLoweringCount: Int get() = keyedMlir.size

    /**
     * Compile [fn] (or fetch from cache) and execute against [inputs]. Returns
     * one [FloatArray] per [DxirFunction.returns], sized by `return.type.elementCount`.
     *
     * Params and returns are F32 or I32. An I32 value travels as a
     * [FloatArray] of whole numbers, the interpreter's convention, so the
     * same bindings run on [io.tlaloc.ir.passes.DxirInterpreter] and here;
     * each must be an integer below 2²⁴ in magnitude.
     *
     * The cache key is the StableHLO MLIR text [fn.toStablehlo] produces —
     * structurally identical DxirFunctions hit the same cache slot. Pass
     * [cacheKey] to skip the re-emission on a hit; see [keyedMlir].
     */
    fun runOn(fn: DxirFunction, inputs: List<FloatArray>, cacheKey: String? = null): List<FloatArray> = live {
        require(fn.params.size == inputs.size) {
            "PjrtSession.runOn: param count ${fn.params.size} != input count ${inputs.size}"
        }
        for ((i, p) in fn.params.withIndex()) {
            require(p.type.dtype == F32 || p.type.dtype == I32) {
                "PjrtSession.runOn: param '${p.name}' dtype is ${p.type.dtype}; runOn takes F32 and I32"
            }
            val expected = p.type.elementCount.toInt()
            require(inputs[i].size == expected) {
                "PjrtSession.runOn: param '${p.name}' expects size $expected (type ${p.type}) " +
                    "but received input of size ${inputs[i].size}"
            }
        }
        for ((i, r) in fn.returns.withIndex()) {
            require(r.type.dtype == F32 || r.type.dtype == I32) {
                "PjrtSession.runOn: return[$i] dtype is ${r.type.dtype}; runOn returns F32 and I32"
            }
        }

        val mlir = lower(fn, cacheKey)
        val exec = executableCache.computeIfAbsent(mlir) { client.compile(mlir) }
        require(exec.numOutputs == fn.returns.size) {
            "PjrtSession.runOn: PJRT executable reports numOutputs=${exec.numOutputs}; " +
                "DxirFunction declares ${fn.returns.size} returns"
        }

        val inputBuffers = ArrayList<PjrtBuffer>(inputs.size)
        try {
            for ((p, arr) in fn.params.zip(inputs)) {
                inputBuffers += if (p.type.dtype == I32) {
                    client.bufferFromHostI32(device, floatEncodedInts(p.name, arr), p.type.dims)
                } else {
                    client.bufferFromHostF32(device, arr, p.type.dims)
                }
            }
            val outputs = exec.execute(inputBuffers, device)
            try {
                outputs.zip(fn.returns).map { (buf, ret) ->
                    val n = ret.type.elementCount.toInt()
                    if (ret.type.dtype == I32) buf.toIntArray(n).let { v -> FloatArray(n) { v[it].toFloat() } }
                    else buf.toFloatArray(n)
                }
            } finally {
                outputs.forEach { it.close() }
            }
        } finally {
            inputBuffers.forEach { it.close() }
        }
    }

    /**
     * [runOn] with each param and return at its own dtype: a [FloatArray] for F32, a
     * [DoubleArray] for F64, an [IntArray] for I32, a [ByteArray] of codes for I8, U8 and
     * F8E4M3FN, in any combination. F64 values reach
     * the device and come back as Doubles, with no F32 step. Other dtypes are refused by
     * name. [cacheKey] works as in [runOn].
     */
    fun runOnHost(fn: DxirFunction, inputs: List<Any>, cacheKey: String? = null): List<Any> = live {
        require(fn.params.size == inputs.size) {
            "PjrtSession.runOnHost: param count ${fn.params.size} != input count ${inputs.size}"
        }
        for ((i, p) in fn.params.withIndex()) {
            val expected = p.type.elementCount.toInt()
            val size = when (val a = inputs[i]) {
                is FloatArray -> a.size.also { requireHostDtype(p.name, p.type.dtype, F32, "FloatArray") }
                is DoubleArray -> a.size.also { requireHostDtype(p.name, p.type.dtype, io.tlaloc.core.F64, "DoubleArray") }
                is IntArray -> a.size.also { requireHostDtype(p.name, p.type.dtype, I32, "IntArray") }
                is ByteArray -> a.size.also {
                    require(p.type.dtype.isOneByteCode()) {
                        "PjrtSession.runOnHost: param '${p.name}' is ${p.type.dtype}; a ByteArray is for one-byte codes (I8, U8, F8E4M3FN)"
                    }
                }
                else -> throw IllegalArgumentException(
                    "PjrtSession.runOnHost: param '${p.name}' input is a ${a::class.simpleName}; " +
                        "runOnHost takes FloatArray (F32), DoubleArray (F64), IntArray (I32) and ByteArray (I8, U8, F8E4M3FN codes)",
                )
            }
            require(size == expected) {
                "PjrtSession.runOnHost: param '${p.name}' expects size $expected (type ${p.type}) but received $size"
            }
        }
        for ((i, r) in fn.returns.withIndex()) {
            require(r.type.dtype == F32 || r.type.dtype == io.tlaloc.core.F64 || r.type.dtype == I32 || r.type.dtype.isOneByteCode()) {
                "PjrtSession.runOnHost: return[$i] dtype is ${r.type.dtype}; runOnHost returns F32, F64, I32 " +
                    "and one-byte codes (I8, U8, F8E4M3FN) as ByteArray"
            }
        }
        val mlir = lower(fn, cacheKey)
        val exec = executableCache.computeIfAbsent(mlir) { client.compile(mlir) }
        require(exec.numOutputs == fn.returns.size) {
            "PjrtSession.runOnHost: PJRT executable reports numOutputs=${exec.numOutputs}; " +
                "DxirFunction declares ${fn.returns.size} returns"
        }
        val inputBuffers = ArrayList<PjrtBuffer>(inputs.size)
        try {
            for ((p, a) in fn.params.zip(inputs)) {
                inputBuffers += when (a) {
                    is FloatArray -> client.bufferFromHostF32(device, a, p.type.dims)
                    is DoubleArray -> client.bufferFromHostF64(device, a, p.type.dims)
                    is ByteArray -> client.bufferFromHostBytes(device, a, p.type.dims, p.type.dtype)
                    else -> client.bufferFromHostI32(device, a as IntArray, p.type.dims)
                }
            }
            val outputs = exec.execute(inputBuffers, device)
            try {
                outputs.zip(fn.returns).map { (buf, ret) ->
                    val n = ret.type.elementCount.toInt()
                    when (ret.type.dtype) {
                        io.tlaloc.core.F64 -> buf.toDoubleArray(n)
                        I32 -> buf.toIntArray(n)
                        F32 -> buf.toFloatArray(n)
                        else -> buf.toByteArray(n)
                    }
                }
            } finally {
                outputs.forEach { it.close() }
            }
        } finally {
            inputBuffers.forEach { it.close() }
        }
    }

    private fun io.tlaloc.core.DType.isOneByteCode() = sizeBytes == 1 && this != io.tlaloc.core.Bool

    private fun requireHostDtype(param: String, declared: io.tlaloc.core.DType, carried: io.tlaloc.core.DType, array: String) {
        require(declared == carried) {
            "PjrtSession.runOnHost: param '$param' is ${declared.name} but its input is a $array (${carried.name})"
        }
    }

    /**
     * F64 twin of [runOn]: every param and return must be F64
     * (mixed-dtype programs are not supported on this lane). f64
     * throughput on a GB10 is modest, but
     * scientific kernels and gradient checks want the precision.
     */
    fun runOnF64(fn: DxirFunction, inputs: List<DoubleArray>): List<DoubleArray> = live {
        require(fn.params.size == inputs.size) {
            "PjrtSession.runOnF64: param count ${fn.params.size} != input count ${inputs.size}"
        }
        for ((i, p) in fn.params.withIndex()) {
            require(p.type.dtype == io.tlaloc.core.F64) {
                "PjrtSession.runOnF64: param '${p.name}' dtype is ${p.type.dtype}; expected F64"
            }
            val expected = p.type.elementCount.toInt()
            require(inputs[i].size == expected) {
                "PjrtSession.runOnF64: param '${p.name}' expects size $expected but got ${inputs[i].size}"
            }
        }
        for ((i, r) in fn.returns.withIndex()) {
            require(r.type.dtype == io.tlaloc.core.F64) {
                "PjrtSession.runOnF64: return[$i] dtype is ${r.type.dtype}; expected F64"
            }
        }

        val mlir = emit(fn)
        val exec = executableCache.computeIfAbsent(mlir) { client.compile(mlir) }
        val inputBuffers = fn.params.zip(inputs).map { (p, arr) ->
            client.bufferFromHostF64(device, arr, p.type.dims)
        }
        try {
            val outputs = exec.execute(inputBuffers, device)
            try {
                outputs.zip(fn.returns).map { (buf, ret) ->
                    buf.toDoubleArray(ret.type.elementCount.toInt())
                }
            } finally {
                outputs.forEach { it.close() }
            }
        } finally {
            inputBuffers.forEach { it.close() }
        }
    }

    /**
     * BF16 twin of [runOn]: every param and return must be
     * BF16, and host arrays are RAW BIT PATTERNS per the ShortArray
     * convention (a Short is a 16-bit bucket, never a number — narrow/widen
     * explicitly with `floatArrayToBf16Bits`/`bf16BitsToFloatArray`). The
     * device stores and computes TRUE bf16; this lane does no numeric
     * conversion in either direction, so what XLA rounds is what you read.
     * Mixed-dtype programs (f32 params casting into bf16 compute) ride the
     * plain [runOn] lane instead — the cast-at-boundary pattern.
     */
    fun runOnBf16(fn: DxirFunction, inputs: List<ShortArray>): List<ShortArray> = live {
        require(fn.params.size == inputs.size) {
            "PjrtSession.runOnBf16: param count ${fn.params.size} != input count ${inputs.size}"
        }
        for ((i, p) in fn.params.withIndex()) {
            require(p.type.dtype == io.tlaloc.core.BF16) {
                "PjrtSession.runOnBf16: param '${p.name}' dtype is ${p.type.dtype}; expected BF16"
            }
            val expected = p.type.elementCount.toInt()
            require(inputs[i].size == expected) {
                "PjrtSession.runOnBf16: param '${p.name}' expects size $expected but got ${inputs[i].size}"
            }
        }
        for ((i, r) in fn.returns.withIndex()) {
            require(r.type.dtype == io.tlaloc.core.BF16) {
                "PjrtSession.runOnBf16: return[$i] dtype is ${r.type.dtype}; expected BF16"
            }
        }

        val mlir = emit(fn)
        val exec = executableCache.computeIfAbsent(mlir) { client.compile(mlir) }
        val inputBuffers = fn.params.zip(inputs).map { (p, arr) ->
            client.bufferFromHostBf16(device, arr, p.type.dims)
        }
        try {
            val outputs = exec.execute(inputBuffers, device)
            try {
                outputs.zip(fn.returns).map { (buf, ret) ->
                    buf.toBf16Array(ret.type.elementCount.toInt())
                }
            } finally {
                outputs.forEach { it.close() }
            }
        } finally {
            inputBuffers.forEach { it.close() }
        }
    }

    /** Stages a raw bf16 pattern array onto [device]. Caller
     * owns the returned [PjrtBuffer] and must close it. */
    fun bufferFromHostBf16(data: ShortArray, dims: List<Int>): PjrtBuffer = live {
        client.bufferFromHostBf16(device, data, dims)
    }

    /** Stages raw bytes onto [device] as a U8 buffer. Caller owns the returned [PjrtBuffer]. */
    fun bufferFromHostU8(data: ByteArray, dims: List<Int>): PjrtBuffer = live {
        client.bufferFromHostU8(device, data, dims)
    }

    /**
     * Force-compile [fn] now without dispatching. Useful for benchmark setup
     * where you want the compile cost outside the timing loop. Idempotent
     * (a second call with structurally identical [fn] is a no-op).
     *
     * [cacheKey] is the serving-side warm-up path: a plugin
     * walks `DecodeBucketPolicy.allBuckets` at startup and prepares one
     * executable per bucket, so the first real request never pays a compile.
     */
    fun prepare(fn: DxirFunction, cacheKey: String? = null): Unit = live {
        val mlir = lower(fn, cacheKey)
        executableCache.computeIfAbsent(mlir) { client.compile(mlir) }
        Unit
    }

    /** Number of executables currently in the compile cache. Useful for
     * tests asserting cache hit/miss behaviour. */
    val cacheSize: Int get() = executableCache.size

    /** The platform string the loaded plugin reports via
     * `PJRT_Client_PlatformName` ("cuda"/"gpu" for the CUDA plugin, "tpu"
     * for libtpu). The TPU smoke suite asserts this so a mis-resolved
     * plugin can never silently certify the wrong backend. */
    fun platformName(): String = live {
        client.platformName()
    }

    // =========================================================================
    // §0.4.308 — lower-level methods for benchmark loops.
    //
    // [runOn]'s FloatArray-in/out shape is convenient but pays host↔device
    // transfer cost on every call (allocate input buffers, upload, allocate
    // output buffers, download). For benchmarking we want to amortise that
    // staging cost across many dispatches — pre-stage inputs once, time
    // execute-only, drop outputs without copying back to host.
    //
    // [bufferFromHostF32] returns a [PjrtBuffer] the caller owns + closes;
    // [executeOn] runs a cached executable against pre-staged input buffers
    // and returns output buffers (caller closes after reading or discarding).
    //
    // Apples-to-apples with JAX's `arr.block_until_ready()` benchmark pattern
    // (host buffers not allocated each iter; sync via PJRT's
    // device-complete event which executeOn awaits internally).
    // =========================================================================

    /** Stage a host f32 buffer onto [device]. Caller owns the returned
     * [PjrtBuffer] and must close it. */
    fun bufferFromHostF32(data: FloatArray, dims: List<Int>): PjrtBuffer = live {
        client.bufferFromHostF32(device, data, dims)
    }

    /** Stage a host i32 buffer (token ids, page tables) onto [device]. Caller owns and closes it. */
    fun bufferFromHostI32(data: IntArray, dims: List<Int>): PjrtBuffer = live {
        client.bufferFromHostI32(device, data, dims)
    }

    /** Execute a previously-prepared (or first-time-compiled) executable
     * against [stagedInputs]. Returns one [PjrtBuffer] per executable
     * output; **caller must close each output** after use.
     *
     * Uses a pre-allocated [ExecuteContext] cached per executable
     * so the per-call cost is the FFM downcall + GPU work, no per-call
     * arena allocation. [cacheKey] skips re-emitting [fn] on each call, as
     * for [runOn]; the same contract applies (the key must fully determine
     * the program). `@JvmOverloads` keeps the two-argument JVM signature.
     */
    @JvmOverloads
    fun executeOn(fn: DxirFunction, stagedInputs: List<PjrtBuffer>, cacheKey: String? = null): List<PjrtBuffer> = live {
        executeCompiled(lower(fn, cacheKey), stagedInputs)
    }

    /**
     * [executeOn] for a program that is already StableHLO text (a serving
     * artifact's body): compiled once per distinct text, as written, and run
     * against [stagedInputs]. The caller closes each returned buffer. An input
     * the program donates to an output is consumed by the call; close it
     * afterwards and do not pass it again.
     */
    fun executeStablehlo(stablehlo: String, stagedInputs: List<PjrtBuffer>): List<PjrtBuffer> = live {
        executeCompiled(stablehlo, stagedInputs)
    }

    /** Compile [stablehlo] now, without running it (see [executeStablehlo]). Idempotent. */
    fun prepareStablehlo(stablehlo: String): Unit = live {
        executableCache.computeIfAbsent(stablehlo) { client.compile(stablehlo) }
    }

    private fun executeCompiled(mlir: String, stagedInputs: List<PjrtBuffer>): List<PjrtBuffer> {
        val exec = executableCache.computeIfAbsent(mlir) { client.compile(mlir) }
        val ctx = executeContextCache.computeIfAbsent(mlir) {
            buildExecuteContext(exec, nInputs = stagedInputs.size)
        }
        require(ctx.nInputs == stagedInputs.size) {
            "PjrtSession.executeOn: cached context expects ${ctx.nInputs} inputs but got ${stagedInputs.size}"
        }

        // The context is one argument block shared by every call on this
        // executable: fill, execute, await and read back under its monitor so
        // two threads never interleave their input or output pointers.
        // A closed buffer (or one from a closed client) refuses by name here
        // rather than handing PJRT a freed pointer.
        stagedInputs.forEach { it.usable() }
        val outputPtrs = synchronized(ctx) {
            for ((i, buf) in stagedInputs.withIndex()) {
                ctx.innerArgsSegment.set(ADDRESS, i * 8L, buf.bufferPtr)
            }
            api.executeReusable(
                argsSegment = ctx.argsSegment,
                innerOutputsSegment = ctx.innerOutputsSegment,
                deviceCompleteEventSlot = ctx.deviceCompleteEventSlot,
                nInputs = ctx.nInputs,
                nOutputs = ctx.nOutputs,
            )
        }
        return outputPtrs.map { PjrtBuffer(it, client) }
    }

    private fun buildExecuteContext(exec: PjrtLoadedExecutable, nInputs: Int): ExecuteContext {
        val nOutputs = exec.numOutputs

        // Allocate everything in the session arena.
        val argsSegment = arena.allocate(PjrtFfm.PJRT_LoadedExecutable_Execute_Args_LAYOUT)
        val optionsSegment = arena.allocate(PjrtFfm.PJRT_ExecuteOptions_LAYOUT)
        val outerArgsSegment = arena.allocate(8)                                    // 1 ptr
        val innerArgsSegment = arena.allocate((nInputs * 8).toLong())               // n ptrs
        val outerOutputsSegment = arena.allocate(8)                                 // 1 ptr
        val innerOutputsSegment = arena.allocate((nOutputs * 8).toLong())           // n ptrs
        val deviceCompleteEventSlot = arena.allocate(8)                             // 1 ptr

        // ExecuteOptions: only struct_size needed; rest stays zero.
        optionsSegment.set(JAVA_LONG, PjrtFfm.OFF_ExecOpts_StructSize, PjrtFfm.SZ_ExecOpts)

        // Wire up the pointer chain: args → outer → inner.
        outerArgsSegment.set(ADDRESS, 0L, innerArgsSegment)
        outerOutputsSegment.set(ADDRESS, 0L, innerOutputsSegment)

        // Pre-populate args struct fields that don't change between calls.
        argsSegment.set(JAVA_LONG, PjrtFfm.OFF_Execute_StructSize, PjrtFfm.SZ_Execute)
        argsSegment.set(ADDRESS, PjrtFfm.OFF_Execute_Executable, exec.execPtr)
        argsSegment.set(ADDRESS, PjrtFfm.OFF_Execute_Options, optionsSegment)
        argsSegment.set(ADDRESS, PjrtFfm.OFF_Execute_ArgLists, outerArgsSegment)
        argsSegment.set(JAVA_LONG, PjrtFfm.OFF_Execute_NumDevices, 1L)
        // num_args set per call by executeReusable.
        argsSegment.set(ADDRESS, PjrtFfm.OFF_Execute_OutputLists, outerOutputsSegment)
        argsSegment.set(ADDRESS, PjrtFfm.OFF_Execute_DeviceCompleteEvents, deviceCompleteEventSlot)
        argsSegment.set(ADDRESS, PjrtFfm.OFF_Execute_ExecuteDevice, device.devicePtr)

        return ExecuteContext(
            argsSegment, innerArgsSegment, innerOutputsSegment, deviceCompleteEventSlot,
            nInputs, nOutputs,
        )
    }

    private fun floatEncodedInts(param: String, values: FloatArray): IntArray = IntArray(values.size) { i ->
        val v = values[i]
        require(v == kotlin.math.floor(v) && kotlin.math.abs(v) < 16_777_216f) {
            "PjrtSession.runOn: I32 param '$param' holds ${v} at $i, not an integer below 2^24"
        }
        v.toInt()
    }

    override fun close() {
        val write = lifecycle.writeLock()
        write.lock()
        try {
            if (closed) return
            closed = true
            executableCache.values.forEach { runCatching { it.close() } }
            executableCache.clear()
            executeContextCache.clear()
            keyedMlir.clear()
            runCatching { client.close() }
            runCatching { arena.close() }
        } finally {
            write.unlock()
        }
    }

    internal companion object {
        private val pinnedPlugins: MutableSet<Path> = ConcurrentHashMap.newKeySet()

        /**
         * Keeps [plugin] loaded for the rest of the process: one extra
         * `dlopen` through the global arena, never closed. Without it, the
         * last session's close unloads the plugin (its arena's lookup is the
         * only reference), and an exit handler the plugin registered then
         * runs from unmapped memory when the process exits: a SIGSEGV in
         * libc's exit path, seen with a session opened on a Spark executor
         * thread. [io.tlaloc.runtime.pjrt.ffm.PjrtFfiRegistry] pins its
         * plugins for a related reason.
         */
        internal fun pinPlugin(plugin: Path) {
            val key = plugin.toAbsolutePath().normalize()
            if (key in pinnedPlugins) return
            java.lang.foreign.SymbolLookup.libraryLookup(key, Arena.global())
            pinnedPlugins.add(key)
        }

        /** Opens arena, plugin, client and device; on any failure closes what
         * was opened and rethrows. [newArena] is a seam for tests. */
        internal fun openHandles(
            plugin: Path,
            target: PjrtTarget,
            options: PjrtClientOptions?,
            newArena: () -> Arena = { Arena.ofShared() },
        ): Handles {
            val arena = newArena()
            var client: PjrtClient? = null
            try {
                val api = PjrtFfm.load(plugin, arena)
                pinPlugin(plugin)
                client = api.createClient(options)
                val device = client.addressableDevices().firstOrNull()
                    ?: error("PJRT client has no addressable devices for $target (plugin $plugin)")
                return Handles(arena, api, client, device)
            } catch (e: Throwable) {
                client?.let { c -> runCatching { c.close() }.exceptionOrNull()?.let(e::addSuppressed) }
                runCatching { arena.close() }.exceptionOrNull()?.let(e::addSuppressed)
                throw e
            }
        }
    }
}
