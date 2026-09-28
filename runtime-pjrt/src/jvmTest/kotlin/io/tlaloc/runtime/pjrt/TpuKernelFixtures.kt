package io.tlaloc.runtime.pjrt

import io.tlaloc.core.BF16
import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirEmitter
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.mosaicKernel
import io.tlaloc.ir.recognizer.kernel.MosaicKernel
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit

/**
 * The TPU kernel payloads in `src/jvmTest/resources/tpu-kernels/`, written by
 * `harness/python/export_tpu_kernels.py`: one directory per kernel with a
 * `manifest.json` (Mosaic body, custom-call config, shapes, versions), the
 * Mosaic text, and `outN.bin` numpy references (little-endian f32).
 *
 * Inputs are not stored: [genValues] is the exporter's integer formula, so
 * both sides produce the same bits.
 */
internal object TpuKernelFixtures {

    val NAMES: List<String> = listOf(
        "rmsnorm_f32",
        "rmsnorm_bf16",
        "matmul_f32",
        "norm_swiglu_f32",
        "kv_update_inplace_f32",
        "rmsnorm_f32_kmosaic",
        "rmsnorm_bf16_kmosaic",
    )

    class Tensor(val name: String, val dtype: DType, val shape: List<Int>, val seed: Int, val scale: Double) {
        val type: DxirType get() = DxirType(dtype, shape)
        val size: Int get() = shape.fold(1) { a, d -> a * d }
        fun values(): FloatArray = genValues(size, seed, scale)
    }

    class Fixture(
        val name: String,
        val manifest: Map<String, Any?>,
        val inputs: List<Tensor>,
        val outputTypes: List<DxirType>,
        val expected: List<FloatArray>,
        val mosaicText: String,
    ) {
        val source: String get() = manifest["source"] as String
        val tolerance: Double get() = (manifest["tolerance"] as Number).toDouble()
        val eps: Float get() = (manifest["eps"] as Number).toFloat()

        @Suppress("UNCHECKED_CAST")
        val versions: Map<String, Any?> get() = manifest["versions"] as Map<String, Any?>

        @Suppress("UNCHECKED_CAST")
        fun kernel(): MosaicKernel = MosaicKernel(
            kernelName = manifest["kernel_name"] as String,
            bodyBase64 = manifest["body_base64"] as String,
            customCallConfig = manifest["custom_call_config"] as Map<String, Any>,
            inputOutputAliases = (manifest["input_output_aliases"] as Map<String, Any?>)
                .entries.associate { (k, v) -> k.toInt() to (v as Number).toInt() },
            hasSideEffect = manifest["has_side_effect"] as Boolean,
            provenance = versions.entries.associate { (k, v) -> k to v.toString() } +
                mapOf("source" to source, "stable_mosaic_version" to manifest["stable_mosaic_version"].toString()),
        )

        /** One MOSAIC_KERNEL over the fixture's inputs, returning every result. */
        fun program(referenceFallback: Boolean): DxirFunction = DxirBuilder.function(name) {
            val params = inputs.map { param(it.name, it.type) }
            val call = mosaicKernel(params, kernel(), reference(), referenceFallback)
            List(call.numResults) { call.result(it) }
        }

        /** The same computation in DXIR ops. */
        fun reference(): DxirFunction = DxirBuilder.function("${name}_reference") {
            val p = inputs.map { param(it.name, it.type) }
            when (name.removeSuffix("_kmosaic")) {
                "rmsnorm_f32" -> listOf(rmsNorm(p[0], p[1], eps))
                "rmsnorm_bf16" -> {
                    val x = op(OpKind.CAST, listOf(p[0]), DxirType(F32, p[0].type.dims))
                    val w = op(OpKind.CAST, listOf(p[1]), DxirType(F32, p[1].type.dims))
                    val y = rmsNorm(x, w, eps)
                    listOf(op(OpKind.CAST, listOf(y), DxirType(BF16, y.type.dims)))
                }
                "matmul_f32" -> listOf(
                    op(OpKind.MATMUL, listOf(p[0], p[1]), DxirType(F32, listOf(p[0].type.dims[0], p[1].type.dims[1]))),
                )
                "norm_swiglu_f32" -> {
                    val h = rmsNorm(p[0], p[1], eps)
                    val ff = DxirType(F32, listOf(p[0].type.dims[0], p[2].type.dims[1]))
                    val g = op(OpKind.MATMUL, listOf(h, p[2]), ff)
                    val u = op(OpKind.MATMUL, listOf(h, p[3]), ff)
                    listOf(h, op(OpKind.MUL, listOf(op(OpKind.SILU, listOf(g), ff), u), ff))
                }
                "kv_update_inplace_f32" -> {
                    val cache = p[0]
                    val new = p[1]
                    val rows = cache.type.dims[0]
                    val width = cache.type.dims[1]
                    val n = new.type.dims[0]
                    val offset = KV_OFFSET
                    fun rowsOf(from: Int, to: Int) = op(
                        OpKind.SLICE, listOf(cache), DxirType(F32, listOf(to - from, width)),
                        attrs = mapOf(
                            "start_indices" to listOf(from, 0),
                            "limit_indices" to listOf(to, width),
                            "strides" to listOf(1, 1),
                        ),
                    )
                    listOf(
                        op(
                            OpKind.CONCAT, listOf(rowsOf(0, offset), new, rowsOf(offset + n, rows)),
                            cache.type, attrs = mapOf("dimension" to 0),
                        ),
                    )
                }
                else -> error("no reference for fixture $name")
            }
        }
    }

    /** Rows the kv_update kernel writes from (export_tpu_kernels.KV_OFFSET). */
    const val KV_OFFSET: Int = 8

    /** `x * rsqrt(mean(x^2) + eps) * w` over the last axis of `[rows, hidden]`, w `[1, hidden]`. */
    private fun DxirEmitter.rmsNorm(x: DxirNode, w: DxirNode, eps: Float): DxirNode {
        val dims = x.type.dims
        val tX = DxirType(F32, dims)
        val tRow = DxirType(F32, listOf(dims[0], 1))
        val sq = op(OpKind.MUL, listOf(x, x), tX)
        val mean = op(OpKind.MEAN, listOf(sq), tRow, attrs = mapOf("reduction_dims" to listOf(1)))
        val epsC = const(FloatArray(dims[0]) { eps }, tRow)
        val inv = op(OpKind.RSQRT, listOf(op(OpKind.ADD, listOf(mean, epsC), tRow)), tRow)
        val invB = op(OpKind.BROADCAST, listOf(inv), tX, attrs = mapOf("broadcast_dimensions" to listOf(0, 1)))
        val wB = op(OpKind.BROADCAST, listOf(w), tX, attrs = mapOf("broadcast_dimensions" to listOf(0, 1)))
        return op(OpKind.MUL, listOf(op(OpKind.MUL, listOf(x, invB), tX), wB), tX)
    }

    /** `((i*37 + seed*101) mod 257 - 128) / 128 * scale`, exact in f32 and bf16. */
    fun genValues(n: Int, seed: Int, scale: Double): FloatArray = FloatArray(n) { i ->
        val k = ((i.toLong() * 37 + seed.toLong() * 101) % 257 - 128).toDouble()
        (k / 128.0 * scale).toFloat()
    }

    fun load(name: String): Fixture {
        val dir = "/tpu-kernels/$name"
        @Suppress("UNCHECKED_CAST")
        val manifest = MiniJson.parse(resourceText("$dir/manifest.json")) as Map<String, Any?>
        @Suppress("UNCHECKED_CAST")
        val inputs = (manifest["inputs"] as List<Map<String, Any?>>).map {
            Tensor(
                name = it["name"] as String,
                dtype = dtypeOf(it["dtype"] as String),
                shape = (it["shape"] as List<Any?>).map { d -> (d as Number).toInt() },
                seed = (it["seed"] as Number).toInt(),
                scale = (it["scale"] as Number).toDouble(),
            )
        }
        @Suppress("UNCHECKED_CAST")
        val outputs = manifest["outputs"] as List<Map<String, Any?>>
        val types = outputs.map { o ->
            DxirType(dtypeOf(o["dtype"] as String), (o["shape"] as List<Any?>).map { (it as Number).toInt() })
        }
        val expected = outputs.map { o -> readF32(resourceBytes("$dir/${o["file"]}")) }
        return Fixture(name, manifest, inputs, types, expected, resourceText("$dir/mosaic.mlir"))
    }

    private fun dtypeOf(s: String): DType = when (s) {
        "f32" -> F32
        "bf16" -> BF16
        else -> error("fixture dtype '$s' is not handled")
    }

    private fun resourceBytes(path: String): ByteArray =
        TpuKernelFixtures::class.java.getResourceAsStream(path)?.use { it.readBytes() }
            ?: error("missing test resource $path (run harness/python/export_tpu_kernels.py)")

    private fun resourceText(path: String): String = resourceBytes(path).decodeToString()

    private fun readF32(bytes: ByteArray): FloatArray {
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buf.remaining()).also { buf.get(it) }
    }

    fun maxAbsDiff(a: FloatArray, b: FloatArray): Double {
        require(a.size == b.size) { "size ${a.size} != ${b.size}" }
        var m = 0.0
        for (i in a.indices) m = maxOf(m, kotlin.math.abs(a[i].toDouble() - b[i].toDouble()))
        return m
    }

    fun maxAbs(a: FloatArray): Double = a.maxOf { kotlin.math.abs(it.toDouble()) }

    /** The iree venv python, when it can import jax and jaxlib's TPU dialect. */
    fun jaxPython(): String? {
        val candidates = listOfNotNull(
            System.getenv("TLALOC_JAX_PYTHON"),
            System.getProperty("user.home")?.let { "$it/.local/venvs/iree/bin/python" },
        )
        val python = candidates.firstOrNull { Files.isExecutable(Path.of(it)) } ?: return null
        val pb = ProcessBuilder(python, "-c", "import jax, jaxlib; from jax._src.lib import tpu")
        pb.environment()["JAX_PLATFORMS"] = "cpu"
        pb.redirectErrorStream(true)
        val ok = runCatching {
            val p = pb.start()
            p.inputStream.readAllBytes()
            p.waitFor(60, TimeUnit.SECONDS) && p.exitValue() == 0
        }.getOrElse { false }
        return if (ok) python else null
    }

    /** harness/python/<name>, resolved from the module directory Gradle runs tests in. */
    fun harnessScript(name: String): Path =
        Path.of("..", "harness", "python", name).toAbsolutePath().normalize()
}
