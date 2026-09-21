package io.tlaloc.maestro.serving

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.DecodeGraphSpec
import io.tlaloc.ir.inference.HfLlamaCheckpoint
import io.tlaloc.ir.inference.HfLlamaConfig
import io.tlaloc.ir.inference.HfLlamaDecodeGraph
import io.tlaloc.ir.inference.HfLlamaStagedWeights
import java.nio.file.Path

/**
 * §0.4.480 — Phase H3c-3: export a **real HuggingFace Llama checkpoint** as a
 * serving artifact.
 *
 * This is the other end of the line [ServingArtifactWriter] draws.
 * §0.4.478 taught the repo to read a real TinyLlama by role, §0.4.479 turned
 * those tensors into a decode graph whose logits match HuggingFace
 * transformers, and the thing both of them stopped short of was the
 * ARTIFACT — because the manifest had nowhere to put a weight table and the
 * loader had no way to bind one. Both now do, and this object is the
 * composition: checkpoint in, deployment directory out.
 *
 * ## The reduced-layer knob, and why it is a parameter
 *
 * [export] takes `numLayers`, defaulting to the checkpoint's own. A reduced
 * copy is the exact object §0.4.479 certified against transformers (both
 * sides reduced by the same arithmetic), so the cheap lane and the real lane
 * are **the same code path with one integer different** — which is the only
 * way a fast test says anything about the slow one. It is NOT a trimming
 * heuristic: layers are taken as a prefix, 0..n-1, and every other dimension
 * must still match the file (§0.4.479's `loadFor` refuses per tensor if it
 * does not).
 *
 * ## What is NOT here
 *
 * The **tokenizer**. An artifact is weights and programs; turning text into
 * token ids is the frontend's job and vLLM/`transformers` already do it from
 * the checkpoint directory that [ServingManifest.modelName] names. This is
 * the reason `modelName` is set to the HF repo id rather than to a pretty
 * label: it is what a serving frontend needs in order to find the tokenizer
 * and `config.json` that go with these weights.
 */
object HfLlamaServingExport {

    /**
     * Pages in the pool. Not derived from the ladder, and deliberately so:
     * the number of KV pages is an ALLOCATOR budget (how many sequences of
     * what length fit at once), not a property of the compiled shape, and
     * vLLM overrides it through `num_gpu_blocks_override` off the manifest.
     * The exporter's job is to state the one the bodies were compiled for.
     */
    const val DEFAULT_NUM_BLOCKS: Int = 64

    /**
     * Build the decode ladder for [ckpt] under [config] and write it to [dir].
     *
     * @param config usually `ckpt.config`, or a `copy(numLayers = n)` of it.
     */
    fun export(
        ckpt: HfLlamaCheckpoint,
        dir: Path,
        config: HfLlamaConfig = ckpt.config,
        policy: DecodeBucketPolicy,
        numBlocks: Int = DEFAULT_NUM_BLOCKS,
        modelName: String = ckpt.dir.fileName.toString(),
    ): ServingManifest {
        val model = config.toDecodeModelShape(numBlocks = numBlocks, blockSize = policy.blockSize)
        val specs = policy.allBuckets.map { HfLlamaDecodeGraph.spec(config, model, it) }
        val build: (DecodeGraphSpec) -> DxirFunction = { spec ->
            HfLlamaDecodeGraph.build(spec, config, ServingArtifactWriter.ENTRY_POINT)
        }
        // The slot ORDER is the contract (a loader binds by index), so the
        // exporter resolves a slot to its index in the very list the spec was
        // built from rather than re-deriving one. An index lookup by name is
        // O(n) per slot and n is ~200; a map keeps the export linear.
        val slotIndex = HfLlamaDecodeGraph.weightSlots(config)
            .withIndex().associate { (i, s) -> s.name to i }
        return ServingArtifactWriter.export(
            dir = dir,
            modelName = modelName,
            // The content address of "these weights + this architecture".
            // The layer count is IN it: a 2-layer reduction of TinyLlama is a
            // different model, and an executable cache that thought otherwise
            // would serve the wrong program.
            modelHash = "hf-llama:$modelName:L${config.numLayers}:" +
                "h${config.hiddenSize}:v${config.vocabSize}",
            model = model,
            ladder = ServingArtifactWriter.ladderOf(policy),
            specs = specs,
            stageWeight = { slot ->
                val i = slotIndex[slot.name]
                    ?: error("HfLlamaServingExport: no weight slot named '${slot.name}'")
                HfLlamaStagedWeights.stageAt(ckpt, config, i)
            },
            build = build,
        )
    }
}

/**
 * `./gradlew :maestro:exportLlamaServingArtifact -PckptDir=… -PoutDir=… [-PnumLayers=2]`
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
        "usage: HfLlamaServingExportKt <checkpointDir> <outDir> " +
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
