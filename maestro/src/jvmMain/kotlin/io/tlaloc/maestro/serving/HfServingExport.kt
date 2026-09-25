package io.tlaloc.maestro.serving

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodeGraphSpec
import io.tlaloc.ir.inference.DecodeModelShape
import io.tlaloc.ir.inference.HfCheckpoint
import io.tlaloc.ir.inference.HfDecoderConfig
import io.tlaloc.ir.inference.HfDecoderGraph
import io.tlaloc.ir.inference.HfStagedWeights
import java.nio.file.Path

/**
 * Export a **real HuggingFace decoder checkpoint** (any [io.tlaloc.ir.inference.HfModelFamily]) as a
 * serving artifact.
 *
 * This is the other end of the line [ServingArtifactWriter] draws. It
 * composes the checkpoint reader (tensors located by role), the decode
 * graph built from them (logits match HuggingFace transformers), and the
 * staged weight table: checkpoint in, deployment directory out.
 *
 * ## The reduced-layer knob, and why it is a parameter
 *
 * [export] takes `numLayers`, defaulting to the checkpoint's own. A reduced
 * copy is the exact object certified against transformers (both
 * sides reduced by the same arithmetic), so the cheap lane and the real lane
 * are **the same code path with one integer different** — which is the only
 * way a fast test says anything about the slow one. It is NOT a trimming
 * heuristic: layers are taken as a prefix, 0..n-1, and every other dimension
 * must still match the file (`loadFor` refuses per tensor if it
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
object HfServingExport {

    /**
     * Pages in the pool. Not derived from the ladder, and deliberately so:
     * the number of KV pages is an ALLOCATOR budget (how many sequences of
     * what length fit at once), not a property of the compiled shape, and
     * vLLM overrides it through `num_gpu_blocks_override` off the manifest.
     * The exporter's job is to state the one the bodies were compiled for.
     */
    const val DEFAULT_NUM_BLOCKS: Int = 64

    /**
     * The model name for a checkpoint directory. A HuggingFace cache snapshot
     * (`…/models--Qwen--Qwen3-0.6B/snapshots/<revision>`) gives the repo id,
     * `Qwen/Qwen3-0.6B`; any other directory gives its own name.
     */
    fun modelNameFor(dir: Path): String {
        val abs = dir.toAbsolutePath().normalize()
        val parent = abs.parent
        val repo = parent?.parent?.fileName?.toString()
        if (parent?.fileName?.toString() == "snapshots" && repo != null && repo.startsWith("models--")) {
            val parts = repo.removePrefix("models--").split("--", limit = 2)
            if (parts.size == 2 && parts.all { it.isNotEmpty() }) return parts[0] + "/" + parts[1]
        }
        return abs.fileName.toString()
    }

    /**
     * The entries of an artifact: a decode entry per bucket of [policy], then
     * a prefill entry per batch of the batch ladder up to [prefillMaxBatch]
     * (none when it is 0) and per context bucket.
     *
     * A prefill entry of batch `b` takes the prompts of up to `b` sequences
     * in one call, each right-aligned in its own row of the token axis with
     * its own positions, block table and slots, and returns each row's
     * last-token logits. Rows do not read each other: the causal length of a
     * token is its position plus one, and a padding token's slot is -1, so
     * its KV is never written. A row prefilled in a batch gets, in the
     * reference interpreter, the logits and KV it gets alone.
     *
     * With [prefillChunk] a prefill entry takes at most that many tokens
     * per sequence (its context when that is smaller), and a longer prompt is
     * prefilled in several calls. Each token of a call attends over the whole
     * context, so a call's attention memory grows with chunk times context;
     * a long context needs a chunk shorter than itself.
     *
     * Refuses by name a [prefillMaxBatch] that is negative or above the
     * largest decode batch: a Triton model's `max_batch_size` is the largest
     * decode batch, and no batch of requests is larger.
     */
    fun specs(
        config: HfDecoderConfig,
        model: DecodeModelShape,
        policy: DecodeBucketPolicy,
        prefillMaxBatch: Int = policy.maxBatch,
        prefillChunk: Int? = null,
    ): List<DecodeGraphSpec> {
        require(prefillMaxBatch in 0..policy.maxBatch) {
            "HfServingExport: prefillMaxBatch $prefillMaxBatch must be between 0 (no prefill " +
                "entries) and the largest decode batch ${policy.maxBatch}"
        }
        require(prefillChunk == null || prefillChunk >= 1) {
            "HfServingExport: prefillChunk must be >= 1, got $prefillChunk"
        }
        val decode = policy.allBuckets.map { HfDecoderGraph.spec(config, model, it) }
        val batches = policy.batchLadder.filter { it <= prefillMaxBatch }
        val prefill = batches.flatMap { b ->
            policy.contextLadder.map { c ->
                HfDecoderGraph.spec(config, model, DecodeBucket(b, c), DecodeGraphKind.PREFILL, prefillChunk)
            }
        }
        return decode + prefill
    }

    /**
     * Build the decode ladder for [ckpt] under [config] and write it to [dir].
     *
     * With [prefill] (the default) the artifact also gets prefill entries
     * (see [specs]): a chunk of `context` tokens per sequence that writes
     * their KV and returns each sequence's last-token logits in one call. A
     * server prefills a prompt of up to `context` tokens with one call instead
     * of one decode step per token, and the prompts of up to
     * [prefillMaxBatch] sequences that arrive together with one call.
     *
     * With [windowedKv] (the default) a config with sliding-window layers
     * gets a windowed KV pool ([HfDecoderConfig.windowedKvPool]): those
     * layers keep a ring of pages per sequence, bounded by the window, and
     * the artifact is `tlaloc-serving-v3`. Without it, or without sliding
     * layers, every layer keeps full-history pages.
     *
     * [prefillChunk] caps the tokens of a prefill call (see [specs]); the
     * windowed ring is then sized so that a call of that many tokens fits
     * past the window ([HfDecoderConfig.windowedKvPool]).
     *
     * @param config usually `ckpt.config`, or a `copy(numLayers = n)` of it.
     */
    fun export(
        ckpt: HfCheckpoint,
        dir: Path,
        config: HfDecoderConfig = ckpt.config,
        policy: DecodeBucketPolicy,
        numBlocks: Int = DEFAULT_NUM_BLOCKS,
        modelName: String = modelNameFor(ckpt.dir),
        prefill: Boolean = true,
        windowedKv: Boolean = true,
        prefillMaxBatch: Int = policy.maxBatch,
        prefillChunk: Int? = null,
    ): ServingManifest {
        val window = if (!windowedKv) null else config.windowedKvPool(
            blockSize = policy.blockSize, maxContext = policy.contextLadder.last(), fullNumBlocks = numBlocks,
            prefillChunk = if (prefill) prefillChunk else null,
        )
        val model = config.toDecodeModelShape(numBlocks = numBlocks, blockSize = policy.blockSize, windowedKv = window)
        val specs = specs(config, model, policy, if (prefill) prefillMaxBatch else 0, prefillChunk)
        val build: (DecodeGraphSpec) -> DxirFunction = { spec ->
            HfDecoderGraph.build(spec, config, ServingArtifactWriter.ENTRY_POINT)
        }
        // The slot ORDER is the contract (a loader binds by index), so the
        // exporter resolves a slot to its index in the very list the spec was
        // built from rather than re-deriving one. An index lookup by name is
        // O(n) per slot and n is ~200; a map keeps the export linear.
        val slotIndex = HfDecoderGraph.weightSlots(config)
            .withIndex().associate { (i, s) -> s.name to i }
        return ServingArtifactWriter.export(
            dir = dir,
            modelName = modelName,
            // The content address of "these weights + this architecture".
            // The layer count is IN it: a 2-layer reduction of TinyLlama is a
            // different model, and an executable cache that thought otherwise
            // would serve the wrong program.
            modelHash = "hf-${config.family.id}:$modelName:L${config.numLayers}:" +
                "h${config.hiddenSize}:v${config.vocabSize}" +
                // Existing f32 hashes are unchanged; a bf16 weight table is a
                // different program input and says so.
                (if (config.weightDType == io.tlaloc.core.F32) "" else ":w${config.weightDType.name}") +
                // A tied head that reads the embedding table binds one weight
                // fewer than one staged as a copy: a different signature.
                (if (HfDecoderGraph.headReadsEmbedding(config)) ":tiedHead" else ""),
            model = model,
            ladder = ServingArtifactWriter.ladderOf(policy),
            specs = specs,
            // Streamed slot by slot: a bf16 Muse Glimmer table is 56 GB and
            // its embedding table alone is larger than a JVM array.
            writeWeight = { slot, out ->
                val i = slotIndex[slot.name]
                    ?: error("HfServingExport: no weight slot named '${slot.name}'")
                HfStagedWeights.writeSlot(ckpt, config, i, out)
            },
            build = build,
            // A multimodal checkpoint's image and video placeholders: the
            // text-only graph must not be fed them, and a server refuses them.
            refusedTokenIds = config.refusedTokenIds,
        )
    }
}
