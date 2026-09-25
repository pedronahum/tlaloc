import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.HfCheckpoint
import io.tlaloc.maestro.serving.HfServingExport
import io.tlaloc.maestro.serving.TritonModelRepository
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.isDirectory

/**
 * The Kotlin half of the example: a HuggingFace checkpoint in, a Triton model
 * repository out.
 *
 * Two library calls do the work:
 *
 *  1. [HfServingExport.export] reads the checkpoint, builds the decoder's
 *     prefill and decode graphs (embedding, RMSNorm, RoPE, paged attention
 *     over a KV cache, SwiGLU, LM head), lowers each to StableHLO, and writes
 *     a serving artifact: a manifest, one StableHLO file per compiled entry,
 *     and the weights as raw files.
 *  2. [TritonModelRepository.write] turns that artifact into a Triton model:
 *     a `config.pbtxt` in sequence mode (Triton's sequence batcher, the
 *     backend keeps each sequence's KV pages) and a version directory that
 *     holds the artifact's files.
 *
 * This process never loads a PJRT plugin and never touches the GPU. The
 * server that runs the result is Triton with `libtriton_tlaloc.so`, a C++
 * backend; there is no JVM in it.
 *
 *     run --args="--model qwen3 --checkpoint <dir> --out <dir>"
 */
fun main(args: Array<String>) {
    val opts = Opts.parse(args)
    val model = MODELS[opts.model] ?: error(
        "unknown --model '${opts.model}'; one of ${MODELS.keys.joinToString()}",
    )
    val artifactDir = opts.out.resolve("artifact")
    val repositoryDir = opts.out.resolve("repository")
    // A stale export must not survive under a new one.
    opts.out.toFile().deleteRecursively()
    Files.createDirectories(opts.out)

    HfCheckpoint.open(opts.checkpoint).use { ckpt ->
        val config = ckpt.config
        println("checkpoint  ${opts.checkpoint}")
        println("family      ${config.family.id}, ${config.numLayers} layers, hidden ${config.hiddenSize}, " +
            "vocab ${config.vocabSize}, weights as ${config.weightDType.name}")

        // Tensors the text decoder does not read (a multimodal checkpoint's
        // vision encoder) are listed rather than dropped silently.
        val unread = ckpt.verifyInventory()
        if (unread.isNotEmpty()) {
            println("            ${unread.size} checkpoint tensors are not part of the text decoder " +
                "and are not read")
        }

        // The compiled shapes. Each (batch, context) point is one XLA compile
        // when the server loads the model. The context ladder is 64, 128 and
        // 256: the server runs each request on the smallest entry that holds
        // the sequence, so a short conversation does not pay for 256
        // positions of attention. Each context gets a decode entry and a
        // prefill entry (a prompt runs as one call).
        val policy = DecodeBucketPolicy(
            maxBatch = model.maxBatch,
            maxContext = MAX_CONTEXT,
            blockSize = BLOCK_SIZE,
            minContext = MIN_CONTEXT,
        )
        println("entries     decode at batch 1..${model.maxBatch} and prefill at batch 1, " +
            "for contexts ${policy.contextLadder.joinToString()}; KV pages of $BLOCK_SIZE tokens")

        val t0 = System.nanoTime()
        val manifest = HfServingExport.export(
            ckpt = ckpt,
            dir = artifactDir,
            policy = policy,
            prefill = true,
        )
        val secs = (System.nanoTime() - t0) / 1e9
        val bytes = manifest.weights.table.sumOf { it.byteLength }
        println("artifact    ${manifest.entries.size} entries, ${manifest.weights.table.size} weight files " +
            "(${bytes shr 20} MiB) in %.1f s".format(secs))
        for (e in manifest.entries) println("              ${e.entryId}")

        // Sequence mode is the default for an artifact with a KV cache. A
        // sequence left idle for 60 s loses its pages.
        val modelDir = TritonModelRepository.write(artifactDir, repositoryDir, opts.model)
        println("triton      ${modelDir.resolve("config.pbtxt")}")
    }
}

/** KV page size in tokens. */
private const val BLOCK_SIZE = 16

/** The context ladder: a chat prompt plus its answer fits in [MAX_CONTEXT]. */
private const val MIN_CONTEXT = 64
private const val MAX_CONTEXT = 256

/**
 * The largest decode batch compiled for each model. The example asks one
 * question at a time, so batch 1 is all it needs; the server batches the
 * decode steps of concurrent sequences up to this size.
 */
private class ModelShape(val maxBatch: Int)

private val MODELS = mapOf(
    "qwen3" to ModelShape(maxBatch = 1),
    "tinyllama" to ModelShape(maxBatch = 1),
    // 56 GB of bf16 weights; every extra entry is another compile of a 30B model.
    "muse-glimmer" to ModelShape(maxBatch = 1),
)

private class Opts(val model: String, val checkpoint: Path, val out: Path) {
    companion object {
        fun parse(args: Array<String>): Opts {
            var model = "qwen3"
            var checkpoint: Path? = null
            var out: Path? = null
            var i = 0
            while (i < args.size) {
                val a = args[i]
                fun next(): String = args.getOrNull(++i) ?: error("$a needs a value")
                when (a) {
                    "--model" -> model = next()
                    "--checkpoint" -> checkpoint = Path.of(next())
                    "--out" -> out = Path.of(next())
                    else -> error("unknown argument '$a'. usage: --model NAME --checkpoint DIR --out DIR")
                }
                i++
            }
            requireNotNull(checkpoint) { "--checkpoint is required (run.sh finds it in the HuggingFace cache)" }
            require(checkpoint.isDirectory()) { "--checkpoint $checkpoint is not a directory" }
            return Opts(model, checkpoint, out ?: Path.of("build", model))
        }
    }
}
