package io.tlaloc.benchmarks

import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.coarsener.decomposeCoarsened
import io.tlaloc.ir.recognizer.recognizeAll
import io.tlaloc.runtime.iree.IreeBenchmark
import io.tlaloc.runtime.iree.IreeBinaries
import io.tlaloc.runtime.iree.IreeRuntime
import io.tlaloc.runtime.iree.IreeTarget
import io.tlaloc.runtime.iree.NpyWriter
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.300 — sweep candidate IREE compile flags on the LlamaDecoder medium
 * forward + backward. The §0.4.299 PJRT-XLA spike pinned the gap at IREE's
 * GPU codegen; this test measures how much of that gap each flag set
 * recovers.
 *
 * Flag sets exercised on both forward and backward:
 *   - "default"            — IreeTarget.Cuda's base flags (the §0.4.291 path).
 *   - "aggressive-fusion"  — adds `--iree-dispatch-creation-enable-aggressive-fusion=true`
 *                            + `--iree-dispatch-creation-fuse-multi-use=true`.
 *   - "tile-and-fuse"      — adds `--iree-codegen-llvmgpu-use-tile-and-fuse-matmul=true`
 *                            + `--iree-codegen-llvmgpu-test-tile-and-fuse-vectorize=true`.
 *   - "kitchen-sink"       — both flag sets above combined.
 *
 * Each compiled VMFB is benchmarked via iree-benchmark-module on the same
 * Kotlin-seed-42 .npy inputs the §0.4.294 / §0.4.295 / §0.4.299 benches
 * use — apples-to-apples within this sweep.
 *
 * Self-skips when no IREE binaries / no CUDA. Writes
 * `harness-results-tlaloc-iree-cuda-flag-sweep-medium.csv` under
 * `benchmarks/build/`.
 */
class LlamaDecoderIreeFlagSweepTest {

    private val flagSets: Map<String, List<String>> = linkedMapOf(
        "default" to emptyList(),
        "aggressive-fusion" to listOf(
            "--iree-dispatch-creation-enable-aggressive-fusion=true",
            "--iree-dispatch-creation-fuse-multi-use=true",
        ),
        "tile-and-fuse" to listOf(
            "--iree-codegen-llvmgpu-use-tile-and-fuse-matmul=true",
            "--iree-codegen-llvmgpu-test-tile-and-fuse-vectorize=true",
        ),
        "kitchen-sink" to listOf(
            "--iree-dispatch-creation-enable-aggressive-fusion=true",
            "--iree-dispatch-creation-fuse-multi-use=true",
            "--iree-dispatch-creation-enable-fuse-horizontal-contractions=true",
            "--iree-dispatch-creation-element-wise-fuse-multi-reduction=true",
            "--iree-codegen-llvmgpu-use-tile-and-fuse-matmul=true",
            "--iree-codegen-llvmgpu-test-tile-and-fuse-vectorize=true",
            "--iree-opt-aggressively-propagate-transposes=true",
            "--iree-global-opt-propagate-transposes=true",
        ),
    )

    private fun stageInputsOnce(workDir: java.nio.file.Path, fn: io.tlaloc.ir.DxirFunction): List<String> {
        val inputs = llamaSynthesizeInputs(seed = 42L, fn)
        return fn.params.zip(inputs).mapIndexed { i, (p, arr) ->
            val target = workDir.resolve("input_${"%03d".format(i)}_${p.name}.npy")
            NpyWriter.writeFloat32(target, arr, p.type.dims)
            target.toFile().deleteOnExit()
            "@$target"
        }
    }

    private data class SweepRow(
        val flagSet: String,
        val mode: String,
        val medianNs: Long?,         // null = compile or run failed
        val minNs: Long?,
        val p99Ns: Long?,
        val itersPerRep: Int?,
        val errorClass: String? = null,
    )

    @Test
    fun sweepsIreeCudaFlagSetsOnLlamaMediumForwardAndBackward() {
        assumeTrue(
            IreeBinaries.available && IreeBinaries.ireeBenchmarkModule != null,
            "IREE binaries / iree-benchmark-module not resolved — skipping.",
        )
        assumeTrue(
            IreeBinaries.cudaAvailable,
            "no CUDA device detected via `nvidia-smi -L` — skipping flag sweep.",
        )

        val raw = LlamaDecoderPrimal.build(LlamaDecoderConfig.medium)
        val coarsened = coarsenRecognizedPatterns(raw, recognizeAll(raw))
        val fwd = decomposeCoarsened(coarsened)
        val gradRaw = DxirReverseTransform.apply(coarsened)
        val bwd = decomposeCoarsened(gradRaw)
        val fwdMlir = fwd.toStablehlo("")
        val bwdMlir = bwd.toStablehlo("")

        val workDir = Files.createTempDirectory("tlaloc-llama-flag-sweep-")
        workDir.toFile().deleteOnExit()
        // Inputs are shared across all flag sets and both modes (params have
        // the same names + types in fwd and bwd).
        val npyArgs = stageInputsOnce(workDir, fwd)

        val rows = mutableListOf<SweepRow>()
        for ((label, flags) in flagSets) {
            for ((mode, fn, mlir) in listOf(
                Triple("forward", fwd, fwdMlir),
                Triple("backward", bwd, bwdMlir),
            )) {
                val rowOrError: SweepRow = try {
                    val module = IreeRuntime.compile(
                        mlir,
                        target = IreeTarget.Cuda,
                        timeoutSeconds = 600,
                        extraCompileFlags = flags,
                    )
                    val stats = IreeBenchmark.run(
                        module = module,
                        function = fn.name,
                        inputs = npyArgs,
                        repetitions = 3,
                        minTimePerRep = "0.5s",
                        timeoutSeconds = 600,
                    )
                    SweepRow(
                        label, mode,
                        stats.medianNanos, stats.minNanos, stats.p99Nanos,
                        stats.iterationsPerRepetition.toInt(),
                    )
                } catch (e: Exception) {
                    println("[llama-medium-iree-cuda flag=$label mode=$mode] FAILED: ${e::class.simpleName}: ${e.message?.take(200)}")
                    SweepRow(label, mode, null, null, null, null, errorClass = e::class.simpleName)
                }
                rows += rowOrError
                if (rowOrError.medianNs != null) {
                    println(
                        "[llama-medium-iree-cuda flag=$label mode=$mode] median=${rowOrError.medianNs / 1_000} us " +
                            "min=${rowOrError.minNs!! / 1_000} us p99=${rowOrError.p99Ns!! / 1_000} us iters=${rowOrError.itersPerRep}",
                    )
                }
            }
        }

        // Sanity: at least the default rows must succeed; otherwise the gate is broken.
        val defaultRows = rows.filter { it.flagSet == "default" }
        assertTrue(defaultRows.all { it.medianNs != null }, "default flag set must compile + run on both modes")

        val outputDir = java.io.File("build")
        outputDir.mkdirs()
        java.io.File(outputDir, "harness-results-tlaloc-iree-cuda-flag-sweep-medium.csv").writeText(
            buildString {
                append("flag_set,mode,median_ns,min_ns,p99_ns,iters_per_rep,error\n")
                for (r in rows) {
                    append(r.flagSet); append(',')
                    append(r.mode); append(',')
                    append(r.medianNs ?: ""); append(',')
                    append(r.minNs ?: ""); append(',')
                    append(r.p99Ns ?: ""); append(',')
                    append(r.itersPerRep ?: ""); append(',')
                    append(r.errorClass ?: ""); append('\n')
                }
            },
        )
        java.io.File(outputDir, "harness-results-tlaloc-iree-cuda-flag-sweep-medium.md").writeText(
            buildString {
                append("# IREE-CUDA flag sweep on LlamaDecoder medium forward + backward\n\n")
                append("| flag set | forward (µs) | backward (µs) | step (ms) | speedup vs default (step) |\n")
                append("|---|---:|---:|---:|---:|\n")
                val byFlag = rows.groupBy { it.flagSet }
                val defaultStep = run {
                    val r = byFlag["default"] ?: error("missing default rows")
                    val f = r.first { it.mode == "forward" }.medianNs ?: error("default forward row failed")
                    val b = r.first { it.mode == "backward" }.medianNs ?: error("default backward row failed")
                    f + b
                }
                for ((label, _) in flagSets) {
                    val r = byFlag[label] ?: continue
                    val f = r.first { it.mode == "forward" }.medianNs
                    val b = r.first { it.mode == "backward" }.medianNs
                    if (f == null || b == null) {
                        val errF = r.first { it.mode == "forward" }.errorClass ?: "—"
                        val errB = r.first { it.mode == "backward" }.errorClass ?: "—"
                        append("| $label | ${if (f == null) "FAIL ($errF)" else "${f / 1_000}"} | ${if (b == null) "FAIL ($errB)" else "${b / 1_000}"} | — | — |\n")
                    } else {
                        val step = f + b
                        val speedup = defaultStep.toDouble() / step
                        append("| $label | ${f / 1_000} | ${b / 1_000} | %.3f | %.2fx |\n".format(step / 1_000_000.0, speedup))
                    }
                }
            },
        )
    }
}
