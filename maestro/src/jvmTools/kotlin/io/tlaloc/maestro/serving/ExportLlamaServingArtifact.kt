package io.tlaloc.maestro.serving

import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.HfCheckpoint
import io.tlaloc.ir.inference.WeightQuant
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
 * every projection then rounds its input to bf16 and sums in f32),
 * `contextLadder` (comma-separated context buckets, e.g. `512,2048,8192`;
 * default the one context `maxContext`; when given, `maxContext` may be
 * blank and otherwise must equal its largest), `prefillChunk` (default
 * none: a prefill entry takes its whole context; with it, at most that many
 * tokens per sequence per call, see [HfServingExport.specs]) and
 * `weightQuant` (`none`, the default, or `int8`: the layers' Linear weights
 * as int8 codes with one f32 scale per output channel, see
 * [io.tlaloc.ir.inference.WeightQuant]; it changes the model's numerics and
 * is never on by default), `stateSlots` (the linear-attention state pools'
 * slots), `kvDtype` (the KV pools' dtype: blank for the activations',
 * or `fp8` for e4m3fn), `mtpDraftTokens` (0, the default, or the drafts
 * per step of speculative entries with the checkpoint's MTP head; each
 * sequence then holds `mtpDraftTokens + 2` state slots) and
 * `cudaKernels` (`true` emits attention and the Gated DeltaNet recurrence as
 * the CUDA kernels of libtlaloc_kernels.so where they apply) and `mtpDraftHeadQuant` (blank, or
 * `fp8`/`int8`/`nvfp4`: the MTP drafts read a quantized copy of the LM head;
 * the target's tokens keep the full head) and `headQuant` (blank, or a format
 * for the LM head itself, which changes the outputs) and `mtpDraftVocab` (blank,
 * or the drafts are chosen from the first that many token ids; `N+M` adds the
 * last M ids, where a vocabulary keeps its special tokens; blank is
 * [DEFAULT_DRAFT_VOCAB] when the draft head is NVFP4 and the vocabulary holds it,
 * `0` the whole vocabulary).
 *
 * The defaults are a **small demo ladder**, and the runbook says so: one
 * batch size and one modest context, because every extra ladder point is
 * another full XLA compile of a 22-layer model and the point of the demo is
 * that it serves, not that it scales.
 */
/**
 * The drafts' vocabulary an export with an NVFP4 draft head takes by default:
 * the first 65,536 token ids and the last 288. In Qwen3.6-35B-A3B's generated
 * text they hold about 97% of the tokens, and each draft reads 27% of the head.
 */
const val DEFAULT_DRAFT_VOCAB: String = "65536+288"
private const val DEFAULT_DRAFT_VOCAB_ROWS: Int = 65536 + 288

fun main(args: Array<String>) {
    require(args.size >= 2) {
        "usage: ExportLlamaServingArtifactKt <checkpointDir> <outDir> " +
            "[numLayers] [maxBatch] [maxContext] [blockSize] [numBlocks] [prefill] [modelName] [windowedKv] " +
            "[prefillMaxBatch] [weightDType] [contextLadder] [prefillChunk] [weightQuant] [stateSlots] [kvDtype] [mtpDraftTokens] [cudaKernels] [mtpDraftHeadQuant] [headQuant] [mtpDraftVocab]"
    }
    fun arg(i: Int, d: Int) = args.getOrNull(i)?.takeIf { it.isNotBlank() }?.toInt() ?: d
    val ckptDir = Path.of(args[0])
    val outDir = Path.of(args[1])
    val maxBatch = arg(3, 1)
    val ladder = args.getOrNull(12)?.trim().orEmpty().takeIf { it.isNotEmpty() }?.split(',')?.map {
        it.trim().toIntOrNull()
            ?: throw IllegalArgumentException("contextLadder must be comma-separated integers, got '${args[12]}'")
    }
    val maxContext = arg(4, ladder?.max() ?: 64)
    require(ladder == null || ladder.max() == maxContext) {
        "contextLadder $ladder ends at ${ladder!!.max()} but maxContext is $maxContext"
    }
    // One chunk, or a comma-separated list: the largest is the chunk a long
    // prompt is prefilled in, the others smaller entries for short requests.
    val chunks = args.getOrNull(13)?.takeIf { it.isNotBlank() }?.split(',')?.map { it.trim().toInt() }
    val prefillChunk = chunks?.max()
    val extraChunks = chunks.orEmpty().filter { it != prefillChunk }.distinct()
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

    val weightQuant = args.getOrNull(14)?.takeIf { it.isNotBlank() }
        ?.let { WeightQuant.parse(it) } ?: WeightQuant.NONE

    val kvDtype = when (val k = args.getOrNull(16)?.trim()?.lowercase().orEmpty()) {
        "" -> null
        "fp8" -> io.tlaloc.core.F8E4M3FN
        else -> throw IllegalArgumentException("kvDtype must be fp8 or blank, got '$k'")
    }

    HfCheckpoint.open(ckptDir).use { ckpt ->
        val layers = arg(2, ckpt.config.numLayers)
        val config = ckpt.config.copy(numLayers = layers).let {
            if (weightDType == null) it else it.copy(weightDType = weightDType)
        }.copy(
            weightQuant = weightQuant, mtpDraftTokens = arg(17, 0),
            mtpDraftHeadQuant = args.getOrNull(19)?.takeIf { it.isNotBlank() }?.let { WeightQuant.parse(it) } ?: WeightQuant.NONE,
            headQuant = args.getOrNull(20)?.takeIf { it.isNotBlank() }?.let { WeightQuant.parse(it) } ?: WeightQuant.NONE,
        ).let {
            // N, or N+M: the first N token ids and the last M; 0 for the whole
            // vocabulary. Blank: DEFAULT_DRAFT_VOCAB with an NVFP4 draft head over
            // a vocabulary that holds it, else the whole vocabulary.
            val v = args.getOrNull(21)?.takeIf { it.isNotBlank() }
                ?: if (it.mtpDraftHeadQuant == WeightQuant.NVFP4 && it.vocabSize >= DEFAULT_DRAFT_VOCAB_ROWS) DEFAULT_DRAFT_VOCAB else "0"
            it.copy(mtpDraftVocab = v.substringBefore('+').toInt(), mtpDraftVocabTail = v.substringAfter('+', "0").toInt())
        }
        val single = DecodeBucketPolicy(
            maxBatch = maxBatch, maxContext = maxContext,
            blockSize = blockSize, minContext = maxContext,
        )
        val policy = if (ladder == null) single else DecodeBucketPolicy.withLadders(
            batchLadder = single.batchLadder, contextLadder = ladder.sorted(), blockSize = blockSize,
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
        println(
            "weights staged as ${config.weightDType}" +
                (if (weightQuant == WeightQuant.NONE) "" else ", layer projections quantized to ${weightQuant.tag}") +
                (if (kvDtype == null) "" else ", KV cache in $kvDtype") +
                (if (config.mtpDraftTokens == 0) "" else ", speculative with ${config.mtpDraftTokens} MTP drafts") +
                (if (config.mtpDraftHeadQuant == WeightQuant.NONE) "" else ", drafts through a ${config.mtpDraftHeadQuant.tag} head") +
                (if (config.mtpDraftVocab == 0) "" else " over the first ${config.mtpDraftVocab} token ids") +
                (if (config.mtpDraftVocabTail == 0) "" else " and the last ${config.mtpDraftVocabTail}") +
                (if (config.headQuant == WeightQuant.NONE) "" else ", LM head in ${config.headQuant.tag}"),
        )
        val t0 = System.nanoTime()
        val manifest = HfServingExport.export(
            ckpt = ckpt, dir = outDir, config = config, policy = policy,
            numBlocks = numBlocks,
            prefill = prefill,
            windowedKv = windowedKv,
            prefillMaxBatch = prefillMaxBatch,
            prefillChunk = prefillChunk,
            stateSlots = arg(15, HfServingExport.DEFAULT_STATE_SLOTS),
            extraPrefillChunks = extraChunks,
            kvDtype = kvDtype,
            cudaKernels = args.getOrNull(18)?.trim()?.lowercase() == "true",
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
