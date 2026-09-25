package io.tlaloc.maestro.serving

import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.HfCheckpoint
import java.nio.file.Path

/**
 * Entry point of
 * `./gradlew :maestro:exportHfServingArtifact -PckptDir=… -PoutDir=… [-PnumLayers=2]`
 * (also registered as `exportLlamaServingArtifact`), for any supported
 * [io.tlaloc.ir.inference.HfModelFamily].
 * Lives in the `tools` compilation, which is not published: the `:maestro` jar
 * carries [HfServingExport] but no `main`.
 *
 * Arguments, positionally: checkpoint dir, output dir, then optional
 * `numLayers`, `maxBatch`, `maxContext`, `blockSize`, `numBlocks`, `prefill`
 * (`true` or `false`, default `true`: one prefill entry per context bucket),
 * `modelName` (default [HfServingExport.modelNameFor] of the checkpoint dir),
 * `windowedKv` (`true` or `false`, default `true`: a model with sliding-window
 * layers keeps their KV in a windowed pool; `false` gives them full-history
 * pools), `prefillMaxBatch` (default `maxBatch`: prefill entries for every
 * batch of the ladder up to it, so the prompts of several sequences that
 * arrive together are prefilled in one call), `weightDType` (`f32` or `bf16`,
 * default the family's: f32 for Llama and Qwen3, bf16 for Muse Glimmer; bf16
 * keeps a bf16 checkpoint's weights as stored, half the bytes of f32, and
 * every projection then rounds its input to bf16 and sums in f32).
 *
 * The defaults are a **small demo ladder**, and the runbook says so: one
 * batch size and one modest context, because every extra ladder point is
 * another full XLA compile of a 22-layer model and the point of the demo is
 * that it serves, not that it scales.
 */
fun main(args: Array<String>) {
    require(args.size >= 2) {
        "usage: ExportLlamaServingArtifactKt <checkpointDir> <outDir> " +
            "[numLayers] [maxBatch] [maxContext] [blockSize] [numBlocks] [prefill] [modelName] [windowedKv] " +
            "[prefillMaxBatch] [weightDType]"
    }
    fun arg(i: Int, d: Int) = args.getOrNull(i)?.takeIf { it.isNotBlank() }?.toInt() ?: d
    val ckptDir = Path.of(args[0])
    val outDir = Path.of(args[1])
    val maxBatch = arg(3, 1)
    val maxContext = arg(4, 64)
    val blockSize = arg(5, 16)
    val numBlocks = arg(6, HfServingExport.DEFAULT_NUM_BLOCKS)
    val prefill = when (val p = args.getOrNull(7)?.trim().orEmpty()) {
        "", "true" -> true
        "false" -> false
        else -> throw IllegalArgumentException("prefill must be true or false, got '$p'")
    }
    val windowedKv = when (val w = args.getOrNull(9)?.trim().orEmpty()) {
        "", "true" -> true
        "false" -> false
        else -> throw IllegalArgumentException("windowedKv must be true or false, got '$w'")
    }

    val prefillMaxBatch = arg(10, maxBatch)
    val weightDType = when (val w = args.getOrNull(11)?.trim()?.lowercase().orEmpty()) {
        "" -> null
        "f32" -> io.tlaloc.core.F32
        "bf16" -> io.tlaloc.core.BF16
        else -> throw IllegalArgumentException("weightDType must be f32 or bf16, got '$w'")
    }

    HfCheckpoint.open(ckptDir).use { ckpt ->
        val layers = arg(2, ckpt.config.numLayers)
        val config = ckpt.config.copy(numLayers = layers).let {
            if (weightDType == null) it else it.copy(weightDType = weightDType)
        }
        val policy = DecodeBucketPolicy(
            maxBatch = maxBatch, maxContext = maxContext,
            blockSize = blockSize, minContext = maxContext,
        )
        // Tensors the decoder does not read (a multimodal checkpoint's vision
        // encoder) are listed, not silently dropped.
        val unread = ckpt.verifyInventory()
        if (unread.isNotEmpty()) {
            println(
                "${unread.size} checkpoint tensors are not read by the ${config.family} decoder, " +
                    "e.g. ${unread.take(3).joinToString()}",
            )
        }
        println("weights staged as ${config.weightDType}")
        val t0 = System.nanoTime()
        val manifest = HfServingExport.export(
            ckpt = ckpt, dir = outDir, config = config, policy = policy,
            numBlocks = numBlocks,
            prefill = prefill,
            windowedKv = windowedKv,
            prefillMaxBatch = prefillMaxBatch,
            modelName = args.getOrNull(8)?.takeIf { it.isNotBlank() }
                ?: HfServingExport.modelNameFor(ckptDir),
        )
        val secs = (System.nanoTime() - t0) / 1e9
        val bytes = manifest.weights.table.sumOf { it.byteLength }
        println(
            "wrote ${manifest.entries.size} entries and ${manifest.weights.table.size} staged " +
                "weights (${bytes / (1024 * 1024)} MiB) to ${outDir.toAbsolutePath()} " +
                "in %.1fs".format(secs),
        )
        println("  model ${manifest.modelName}  hash ${manifest.modelHash}")
        manifest.model.windowedKv?.let {
            println(
                "  windowed KV pool: layers ${it.layers}, window ${it.window}, a ring of " +
                    "${it.ringPages} pages per sequence, ${it.numBlocks} pages",
            )
        }
        for (e in manifest.entries) println("  ${e.entryId}  ${e.bodyPath}")
    }
}
