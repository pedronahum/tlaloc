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
 * `modelName` (default [HfServingExport.modelNameFor] of the checkpoint dir).
 *
 * The defaults are a **small demo ladder**, and the runbook says so: one
 * batch size and one modest context, because every extra ladder point is
 * another full XLA compile of a 22-layer model and the point of the demo is
 * that it serves, not that it scales.
 */
fun main(args: Array<String>) {
    require(args.size >= 2) {
        "usage: ExportLlamaServingArtifactKt <checkpointDir> <outDir> " +
            "[numLayers] [maxBatch] [maxContext] [blockSize] [numBlocks] [prefill] [modelName]"
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

    HfCheckpoint.open(ckptDir).use { ckpt ->
        val layers = arg(2, ckpt.config.numLayers)
        val config = ckpt.config.copy(numLayers = layers)
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
        for (e in manifest.entries) println("  ${e.entryId}  ${e.bodyPath}")
    }
}
