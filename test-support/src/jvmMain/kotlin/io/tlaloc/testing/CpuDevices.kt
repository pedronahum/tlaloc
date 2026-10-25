package io.tlaloc.testing

import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.core.I32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit

/**
 * A StableHLO module run as one SPMD program on N virtual XLA:CPU devices
 * (`harness/python/run_on_cpu_devices.py`, jaxlib in the oracle venv), with its results
 * recorded by [Goldens] so the comparison runs where jax is not installed. The key is
 * the module text and the inputs: a change to the emitter that changes the module makes
 * the recording stale.
 *
 * This is where Tlaloc's collectives and sharded programs execute on more than one
 * device; the GB10 has one GPU.
 */
object CpuDevices {

    private val script: Path get() = Goldens.root.toPath().resolve("harness/python/run_on_cpu_devices.py")

    private fun dtype(t: DxirType) = when (t.dtype) {
        F32 -> "f32"
        F64 -> "f64"
        I32 -> "i32"
        else -> error("CpuDevices: ${t.dtype} arguments are not supported")
    }

    private fun list(v: DoubleArray) = v.joinToString(",", "[", "]") { if (it == Math.rint(it) && kotlin.math.abs(it) < 1e15) it.toLong().toString() else it.toString() }

    /** jaxlib compiles the module's `main`; PJRT takes the function by any name. */
    private fun asMain(fn: DxirFunction, mlir: String): String {
        val entry = "@${fn.name}("
        require(entry in mlir) { "CpuDevices: no function $entry in the module" }
        return mlir.replace(entry, "@main(")
    }

    private fun run(suite: String, case: String, mlir: String, request: String): JsonObject {
        val dir = Files.createTempDirectory("tlaloc-cpu-devices-")
        try {
            Files.writeString(dir.resolve("module.mlir"), mlir)
            Files.writeString(dir.resolve("request.json"), request)
            val text = Goldens.script(suite, case, script, dir, "", "jax", envVar = "TLALOC_JAX_PYTHON") { python ->
                val out = Files.createTempFile("tlaloc-cpu-devices-", ".json")
                try {
                    val p = ProcessBuilder(
                        python, "-I", script.toString(),
                        "--mlir", dir.resolve("module.mlir").toString(),
                        "--request", dir.resolve("request.json").toString(),
                        "--output", out.toString(),
                    ).redirectErrorStream(true).start()
                    val log = p.inputStream.bufferedReader().readText()
                    check(p.waitFor(600, TimeUnit.SECONDS) && p.exitValue() == 0) { "run_on_cpu_devices.py failed:\n${log.takeLast(4000)}" }
                    Files.readString(out)
                } finally {
                    Files.deleteIfExists(out)
                }
            }
            return parseJson(text) as JsonObject
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    private fun doubles(v: Any?): DoubleArray = (v as JsonArray).elements.map { (it as JsonNumber).value.toDouble() }.toDoubleArray()

    /**
     * [fn]'s module on [devices] devices, each with its own buffer of every argument
     * ([perDevice][arg][device]); the results as [result][device].
     */
    fun runPerDevice(
        suite: String,
        case: String,
        fn: DxirFunction,
        mlir: String,
        devices: Int,
        perDevice: List<List<DoubleArray>>,
    ): List<List<DoubleArray>> {
        require(perDevice.size == fn.params.size) { "CpuDevices: ${perDevice.size} arguments for ${fn.params.size} parameters" }
        val args = fn.params.zip(perDevice).joinToString(",") { (p, bufs) ->
            require(bufs.size == devices) { "CpuDevices: ${bufs.size} buffers of '${p.name}' for $devices devices" }
            """{"dtype": "${dtype(p.type)}", "dims": ${p.type.dims}, "per_device": [${bufs.joinToString(",") { list(it) }}]}"""
        }
        val request = """{"devices": $devices, "mode": "per-device", "args": [$args]}"""
        val out = run(suite, case, asMain(fn, mlir), request)
        return (out.fields["results"] as JsonArray).elements.map { r -> (r as JsonArray).elements.map { doubles(it) } }
    }

    /** [fn]'s sharded module on [devices] devices from whole arguments; the results whole. */
    fun runGlobal(
        suite: String,
        case: String,
        fn: DxirFunction,
        mlir: String,
        devices: Int,
        args: List<DoubleArray>,
    ): List<DoubleArray> {
        require(args.size == fn.params.size) { "CpuDevices: ${args.size} arguments for ${fn.params.size} parameters" }
        val a = fn.params.zip(args).joinToString(",") { (p, v) ->
            """{"dtype": "${dtype(p.type)}", "dims": ${p.type.dims}, "values": ${list(v)}}"""
        }
        val request = """{"devices": $devices, "mode": "global", "result_dims": ${fn.returns.map { it.type.dims }}, "args": [$a]}"""
        val out = run(suite, case, asMain(fn, mlir), request)
        return (out.fields["results"] as JsonArray).elements.map { doubles(it) }
    }
}

/**
 * Copies [fn]'s body into this builder with its parameters bound to [args]; the copies
 * of its returns. Single-result ops without regions only.
 */
fun DxirBuilder.inline(fn: DxirFunction, args: List<DxirNode>): List<DxirNode> {
    require(args.size == fn.params.size) { "inline: ${args.size} arguments for ${fn.params.size} parameters of '${fn.name}'" }
    val map = HashMap<Int, DxirNode>()
    fn.params.zip(args).forEach { (p, a) -> map[p.id] = a }
    for (n in fn.body) {
        map[n.id] = when (n) {
            is DxirConst -> const(n.value, n.type, n.sharding)
            is DxirOp -> {
                require(n.regions.isEmpty() && n.types.size == 1) { "inline: ${n.op} (id=${n.id}) has regions or several results" }
                op(n.op, n.operands.map { map[it.id] ?: error("inline: operand ${it.id} of ${n.op} is not defined") }, n.type, n.attrs, n.sharding, emptyList())
            }
            else -> error("inline: ${n::class.simpleName} (id=${n.id})")
        }
    }
    return fn.returns.map { map[it.id] ?: error("inline: return ${it.id} is not defined") }
}
