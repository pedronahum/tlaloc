import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.HfLlamaCheckpoint
import io.tlaloc.maestro.serving.HfLlamaServingExport
import io.tlaloc.maestro.serving.ReferenceDecodeGraph
import io.tlaloc.maestro.serving.ServingManifest
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.fileSize
import kotlin.io.path.isDirectory
import kotlin.io.path.name
import kotlin.streams.asSequence

/**
 * HALF ONE OF TWO — the compile side.
 *
 * This process reads a model and writes a **directory**. It never loads a PJRT
 * plugin, never opens a CUDA context and never runs the graph it just built;
 * it does not depend on `io.github.pedronahum:tlaloc-runtime-pjrt` at all. When `main` returns,
 * Kotlin's involvement in serving this model is over, permanently.
 *
 * That is the claim the whole example exists to make legible: **the artifact is
 * the deployment.** Half two ([serve.py]) is a Python process that imports no
 * jax, no torch and no numpy, `dlopen`s one PJRT plugin `.so`, and runs what
 * this directory contains. Nothing downstream calls back into the JVM.
 *
 * Two models are exportable here and they are the same code path:
 *
 *  * the **reference decode graph** (default) — one attention layer, an 11-word
 *    vocabulary, weights from a fixed LCG. It exists so the seam can be
 *    certified by a graph small enough that a disagreement is attributable.
 *    It needs no checkpoint, no download and no GPU to export.
 *  * a **real HuggingFace Llama** (`--checkpoint <dir>`) — all 22 of
 *    TinyLlama-1.1B's layers, 201 tensors read by role from its own
 *    `model.safetensors`, staged out as raw little-endian operand files.
 *
 * The argument handling below is the ONLY difference between them. Everything
 * after `ServingArtifactWriter.export` — the manifest, the content-addressed
 * bodies, the loader, the padding contract, the ladder — is identical, which is
 * why the fast lane says something about the slow one.
 */
fun main(args: Array<String>) {
    val opts = Opts.parse(args)
    Files.createDirectories(opts.outDir)

    val t0 = System.nanoTime()
    val manifest =
        if (opts.checkpoint == null) exportReference(opts) else exportLlama(opts, opts.checkpoint)
    val secs = (System.nanoTime() - t0) / 1e9

    println()
    println("wrote the artifact in %.1fs".format(secs))
    inventory(opts.outDir, manifest)
    println()
    println("Half one is done. Nothing below this line is Kotlin's business:")
    println("    python3 serve.py --artifact ${opts.outDir.toAbsolutePath()}")
}

// ---------------------------------------------------------------------------
// The reference decode graph — no checkpoint, no download, no GPU.
// ---------------------------------------------------------------------------

private fun exportReference(opts: Opts): ServingManifest {
    val m = ReferenceDecodeGraph.MODEL
    println("model     ${ReferenceDecodeGraph.MODEL_NAME} (the reference decode graph)")
    println("           vocab ${m.vocabSize}  hidden ${m.hiddenSize}  layers ${m.numLayers}")
    println("           ${m.numHeads} heads / ${m.numKvHeads} kv-heads x headDim ${m.headDim}")
    println("           KV pool ${m.numBlocks} pages x blockSize ${m.blockSize}")
    println("weights   in-body constants from a fixed LCG (so two exports are byte-identical)")
    println("out       ${opts.outDir.toAbsolutePath()}")
    println()

    // One call. The ladder, the six compiled entries, the manifest and the
    // StableHLO bodies all land inside it.
    return ReferenceDecodeGraph.exportTo(opts.outDir)
}

// ---------------------------------------------------------------------------
// A real HuggingFace Llama checkpoint.
// ---------------------------------------------------------------------------

private fun exportLlama(opts: Opts, ckptDir: Path): ServingManifest =
    HfLlamaCheckpoint.open(ckptDir).use { ckpt ->
        // `--layers 2` is the cheap lane: a prefix of the real layers, the same
        // arithmetic on both sides, and the exact reduced model the repo
        // certifies against HuggingFace transformers. It is one integer, not a
        // second code path.
        val config = ckpt.config.copy(numLayers = opts.layers ?: ckpt.config.numLayers)

        // ONE ladder point, deliberately. Every extra (batch, context) bucket is
        // another full XLA compile of a 22-layer model, and this example is
        // about the seam, not about scaling. `minContext = maxContext` collapses
        // the context ladder to its top rung.
        val policy = DecodeBucketPolicy(
            maxBatch = 1,
            maxContext = opts.context,
            blockSize = opts.blockSize,
            minContext = opts.context,
        )

        println("model     ${ckptDir.name}")
        println("           vocab ${config.vocabSize}  hidden ${config.hiddenSize}  " +
            "layers ${config.numLayers} of ${ckpt.config.numLayers}")
        println("           ${config.numHeads} heads / ${config.numKvHeads} kv-heads")
        println("weights   read by ROLE from the checkpoint's own safetensors, widened to f32")
        println("           and TRANSPOSED host-side (HF stores nn.Linear as [out, in])")
        println("ladder    batch 1 x context ${opts.context}, blockSize ${opts.blockSize}")
        println("out       ${opts.outDir.toAbsolutePath()}")
        println()
        println("(reading 2.2 GiB and writing ~4.2 GiB — this takes a few seconds)")

        HfLlamaServingExport.export(
            ckpt = ckpt,
            dir = opts.outDir,
            config = config,
            policy = policy,
            numBlocks = opts.numBlocks,
            // The HF repo id, NOT a pretty label: it is how a frontend finds the
            // tokenizer and config.json that go with these weights. An artifact
            // deliberately contains no tokenizer.
            modelName = ckptDir.name,
        )
    }

// ---------------------------------------------------------------------------
// What landed on disk. This is the part worth reading twice.
// ---------------------------------------------------------------------------

private fun inventory(dir: Path, manifest: ServingManifest) {
    println()
    println("  ${dir.toAbsolutePath()}/")
    for (child in Files.list(dir).asSequence().sortedBy { it.name }) {
        if (child.isDirectory()) {
            val files = Files.list(child).use { it.asSequence().toList() }
            val bytes = files.sumOf { it.fileSize() }
            println("    %-14s %3d files  %s".format(child.name + "/", files.size, human(bytes)))
        } else {
            println("    %-14s %14s".format(child.name, human(child.fileSize())))
        }
    }
    println()
    println("  tlaloc-serving.json   the manifest: model shape, the KV-pool axis order,")
    println("                        the bucket ladder, and one entry per compiled point")
    println("  bodies/<sha256>.mlir  textual StableHLO, content-addressed and de-duplicated.")
    println("                        The name is a hash of the body, not a timestamp, so two")
    println("                        exports of one model are byte-identical and `grep` is a")
    println("                        legitimate debugging tool on a deployment.")
    if (manifest.weights.table.isNotEmpty()) {
        val bytes = manifest.weights.table.sumOf { it.byteLength }
        println("  weights/NNNN_<slot>.bin  ${manifest.weights.table.size} raw little-endian, dense")
        println("                        row-major files, ${human(bytes)} total, NO HEADER: the file")
        println("                        IS the operand. The loader's whole job is open, readinto,")
        println("                        upload — it never makes a Python number out of a weight.")
    }
    println()
    println("  model ${manifest.modelName}")
    println("  hash  ${manifest.modelHash}")
    println("  ${manifest.entries.size} compiled " +
        (if (manifest.entries.size == 1) "entry" else "entries") + ":")
    for (e in manifest.entries) {
        println("    ${e.entryId}  ->  ${e.bodyPath}")
    }
}

private fun human(bytes: Long): String = when {
    bytes >= 1L shl 30 -> "%.1f GiB".format(bytes / (1L shl 30).toDouble())
    bytes >= 1L shl 20 -> "%.1f MiB".format(bytes / (1L shl 20).toDouble())
    bytes >= 1L shl 10 -> "%.1f KiB".format(bytes / (1L shl 10).toDouble())
    else -> "$bytes B"
}

// ---------------------------------------------------------------------------

private class Opts(
    val outDir: Path,
    val checkpoint: Path?,
    val layers: Int?,
    val context: Int,
    val blockSize: Int,
    val numBlocks: Int,
) {
    companion object {
        fun parse(args: Array<String>): Opts {
            var out = Path.of("build/artifact")
            var ckpt: Path? = null
            var reference = false
            var layers: Int? = null
            var context = 64
            var blockSize = 16
            var numBlocks = 64
            var i = 0
            while (i < args.size) {
                val a = args[i]
                fun next(): String = args.getOrNull(++i)
                    ?: error("$a needs a value")
                when (a) {
                    "--out" -> out = Path.of(next())
                    "--checkpoint" -> ckpt = Path.of(next())
                    "--reference" -> reference = true
                    "--layers" -> layers = next().toInt()
                    "--context" -> context = next().toInt()
                    "--block-size" -> blockSize = next().toInt()
                    "--num-blocks" -> numBlocks = next().toInt()
                    else -> error(
                        "unknown argument '$a'. usage: [--out DIR] [--checkpoint DIR] [--reference] " +
                            "[--layers N] [--context N] [--block-size N] [--num-blocks N]",
                    )
                }
                i++
            }
            require(ckpt == null || ckpt.isDirectory()) {
                "--checkpoint $ckpt is not a directory. Fetch TinyLlama with the command in " +
                    "this example's README, or pass --reference to export the toy graph."
            }
            // No --checkpoint given: look where the repo's own tooling caches
            // checkpoints. If a real TinyLlama is sitting there, USE IT — the
            // default should be the demo that says "Paris", not the toy graph.
            // `--reference` opts back out; so does simply not having the model.
            if (ckpt == null && !reference) ckpt = defaultCheckpoint()
            return Opts(out, ckpt, layers, context, blockSize, numBlocks)
        }

        /**
         * The checkpoint cache `scripts/` and the serving tests already use.
         * Returns null when there is nothing there — in which case this example
         * exports the reference decode graph and says so, which still runs end
         * to end and needs no download.
         */
        fun defaultCheckpoint(): Path? {
            val dir = Path.of(System.getProperty("user.home"))
                .resolve(".cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0")
            // A checkpoint is only usable if the weights are actually there.
            val hasWeights = dir.isDirectory() &&
                Files.list(dir).use { s -> s.anyMatch { it.fileName.toString().endsWith(".safetensors") } }
            if (hasWeights) return dir
            val nested = dir.resolve("snapshot")
            return if (nested.isDirectory()) nested else null
        }
    }
}
