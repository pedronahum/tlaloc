package io.tlaloc.benchmarks

import io.tlaloc.kptx.KptxKernels
import io.tlaloc.kptx.emitPtx
import io.tlaloc.runtime.pjrt.kptx.KptxKernelRegistry
import java.nio.file.Path

/**
 * KPTX v1.8/1.9 (§0.4.336/§0.4.337) — registration glue for the KPTX
 * kernels the benchmarks-module tests use: idempotent per-JVM
 * registration (the registry is process-permanent and throws on
 * duplicate (plugin, name); multiple tests in the same Gradle worker
 * JVM share one registration).
 *
 * §0.4.344 — kernel *text* now comes from the :kptx production kernel
 * library (v2 task 15 closed the "kernels live in test sources" era).
 */
internal object KptxTestKernels {

    /**
     * §0.4.344 — RMS-norm forward, eps-operand variant, now sourced from
     * the :kptx production kernel library ([io.tlaloc.kptx.KptxKernels.rmsNormEps]
     * specialized at block=256): a kernel written in pure Kotlin executing
     * inside XLA executables. The §0.4.336 GPU E2E + §0.4.337 bench
     * re-certify the DSL transcription numerically on every run.
     */
    val rmsNormEpsPtx = KptxKernels.rmsNormEps
        .specialize(shapes = mapOf("block" to 256))
        .emitPtx()

    /** §0.4.349 — RoPE forward from the :kptx library (SUB-form/cos-first,
     * the shape [io.tlaloc.ir.recognizer.kernel.RopeKernel] claims). */
    val ropePtx = KptxKernels.rope.specialize().emitPtx()

    private val registeredRopePlugins = HashSet<Path>()

    /** Register `@kptx_rope` on [pluginPath] exactly once per JVM. */
    @Synchronized
    fun ensureRopeRegistered(pluginPath: Path) {
        if (!registeredRopePlugins.add(pluginPath)) return
        KptxKernelRegistry.registerKernel(
            pluginPath,
            "kptx_rope",
            KptxKernelRegistry.LaunchConfig(
                ptx = ropePtx,
                entryName = "kptx_rope",
                grid = { args ->
                    val n = args[0].dims.fold(1L) { a, d -> a * d }.toInt()
                    KptxKernelRegistry.Dim3((n + 255) / 256)
                },
                block = KptxKernelRegistry.Dim3(256),
                trailingI32Params = { args ->
                    intArrayOf(args[0].dims.fold(1L) { a, d -> a * d }.toInt())
                },
            ),
        )
    }

    /** §0.4.351 — CrossEntropy two-stage chain from the :kptx library. */
    val crossEntropyPtx = KptxKernels.crossEntropyModule(block = 256).emitPtx()

    private val registeredCePlugins = HashSet<Path>()

    /** Register the `@kptx_cross_entropy` launch chain exactly once per JVM. */
    @Synchronized
    fun ensureCrossEntropyRegistered(pluginPath: Path) {
        if (!registeredCePlugins.add(pluginPath)) return
        KptxKernelRegistry.registerKernelChain(
            pluginPath,
            "kptx_cross_entropy",
            crossEntropyPtx,
            listOf(
                KptxKernelRegistry.Stage(
                    entryName = "kptx_cross_entropy_rows",
                    grid = { args, _ -> KptxKernelRegistry.Dim3(args[0].dims[0].toInt()) },
                    block = KptxKernelRegistry.Dim3(256),
                    paramBuffers = { args, rets -> listOf(args[0], args[1], rets[1]) },
                    trailingI32Params = { args, _ -> intArrayOf(args[0].dims[1].toInt()) },
                ),
                KptxKernelRegistry.Stage(
                    entryName = "kptx_cross_entropy_sum",
                    grid = { _, _ -> KptxKernelRegistry.Dim3(1) },
                    block = KptxKernelRegistry.Dim3(256),
                    paramBuffers = { _, rets -> listOf(rets[1], rets[0]) },
                    trailingI32Params = { args, _ -> intArrayOf(args[0].dims[0].toInt()) },
                ),
            ),
        )
    }

    private val registeredPlugins = HashSet<Path>()

    /** Register `@kptx_rms_norm` on [pluginPath] exactly once per JVM. */
    @Synchronized
    fun ensureRmsNormRegistered(pluginPath: Path) {
        if (!registeredPlugins.add(pluginPath)) return
        KptxKernelRegistry.registerKernel(
            pluginPath,
            "kptx_rms_norm",
            KptxKernelRegistry.LaunchConfig(
                ptx = rmsNormEpsPtx,
                entryName = "kptx_rms_norm",
                grid = { args -> KptxKernelRegistry.Dim3(args[0].dims[0].toInt()) }, // one CTA per row
                block = KptxKernelRegistry.Dim3(256),
                trailingI32Params = { args -> intArrayOf(args[0].dims[1].toInt()) },
            ),
        )
    }
}
