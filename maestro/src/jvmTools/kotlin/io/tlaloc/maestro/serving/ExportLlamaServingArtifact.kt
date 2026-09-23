package io.tlaloc.maestro.serving

import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.HfLlamaCheckpoint
import java.nio.file.Path

/**
 * Entry point of
 * `./gradlew :maestro:exportLlamaServingArtifact -PckptDir=… -PoutDir=… [-PnumLayers=2]`.
 * Lives in the `tools` compilation, which is not published: the `:maestro` jar
 * carries [HfLlamaServingExport] but no `main`.
 *
 * Arguments, positionally: checkpoint dir, output dir, then optional
 * `numLayers`, `maxBatch`, `maxContext`, `blockSize`, `numBlocks`.
 *
 * The defaults are a **small demo ladder**, and the runbook says so: one
 * batch size and one modest context, because every extra ladder point is
 * another full XLA compile of a 22-layer model and the point of the demo is
 * that it serves, not that it scales.
 */
fun main(args: Array<String>) {
    require(args.size >= 2) {
        "usage: ExportLlamaServingArtifactKt <checkpointDir> <outDir> " +
            "[numLayers] [maxBatch] [maxContext] [blockSize] [numBlocks]"
    }
    fun arg(i: Int, d: Int) = args.getOrNull(i)?.takeIf { it.isNotBlank() }?.toInt() ?: d
    val ckptDir = Path.of(args[0])
    val outDir = Path.of(args[1])
    val maxBatch = arg(3, 1)
    val maxContext = arg(4, 64)
    val blockSize = arg(5, 16)
    val numBlocks = arg(6, HfLlamaServingExport.DEFAULT_NUM_BLOCKS)

    HfLlamaCheckpoint.open(ckptDir).use { ckpt ->
        val layers = arg(2, ckpt.config.numLayers)
        val config = ckpt.config.copy(numLayers = layers)
        val policy = DecodeBucketPolicy(
            maxBatch = maxBatch, maxContext = maxContext,
            blockSize = blockSize, minContext = maxContext,
        )
        val t0 = System.nanoTime()
        val manifest = HfLlamaServingExport.export(
            ckpt = ckpt, dir = outDir, config = config, policy = policy,
            numBlocks = numBlocks,
            modelName = ckptDir.fileName.toString(),
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
