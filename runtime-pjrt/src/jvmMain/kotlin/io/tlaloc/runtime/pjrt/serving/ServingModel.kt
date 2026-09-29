package io.tlaloc.runtime.pjrt.serving

import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodePadding
import io.tlaloc.ir.inference.DecodeSlotRole
import io.tlaloc.maestro.serving.ServingEntry
import io.tlaloc.maestro.serving.ServingManifest
import io.tlaloc.maestro.serving.ServingSlot
import io.tlaloc.maestro.serving.ServingWeightFile
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.PjrtSession
import io.tlaloc.runtime.pjrt.PjrtTarget
import io.tlaloc.runtime.pjrt.ffm.PjrtBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.security.MessageDigest

/**
 * A Tlaloc serving artifact (the directory `HfServingExport` writes), loaded
 * into this JVM and run on the GPU through PJRT, with greedy decoding.
 *
 * ```java
 * try (ServingModel model = ServingModel.load(Path.of("qwen3-artifact"))) {
 *     int[] next = model.generate(promptIds, 16);
 * }
 * ```
 *
 * [load] reads the manifest, checks every body against its SHA-256, and
 * uploads the staged weights to the device once. The KV pools are device
 * buffers for the model's lifetime: each step's output pools (which the
 * program writes in place of its inputs) are the next step's inputs, so no
 * KV data crosses to the host. A request never reads a KV slot it did not
 * write first (positions start at 0 and a step writes its token before it
 * attends), so the pools are not cleared between requests.
 *
 * Prompts go through one prefill call when the artifact has a prefill entry
 * that holds them, and through decode steps otherwise. Generation is greedy:
 * the highest logit, the lowest id on a tie.
 *
 * Methods are `synchronized`: one request runs at a time per model. Use one
 * [ServingModel] per thread for parallel requests, or [generateBatch].
 *
 * Refused by name: windowed KV pools (`tlaloc-serving-v3`), quantized KV
 * pools, and weight or pool dtypes other than f32 and bf16.
 */
class ServingModel private constructor(
    private val root: Path,
    private val manifest: ServingManifest,
    private val session: PjrtSession,
) : AutoCloseable {

    /** The model name the artifact records (a Hugging Face repo id for an exported checkpoint). */
    val modelName: String get() = manifest.modelName

    /** The artifact's model hash. */
    val modelHash: String get() = manifest.modelHash

    val vocabSize: Int get() = manifest.model.vocabSize

    /** The longest sequence (prompt plus generated tokens) the artifact holds. */
    val maxContext: Int get() = minOf(manifest.bucketLadder.maxContext, (manifest.model.numBlocks - 1) * blockSize)

    /** The most sequences one [generateBatch] step runs together. */
    val maxBatch: Int get() = manifest.bucketLadder.maxBatch

    /** The PJRT platform (`cuda`). */
    val platformName: String get() = session.platformName()

    /** How many distinct programs this model has compiled so far. */
    val compileCount: Int get() = compiled.size

    private val blockSize = manifest.bucketLadder.blockSize
    private val bodies = HashMap<String, String>()
    private val compiled = HashSet<String>()
    private var weights: Map<String, PjrtBuffer> = emptyMap()
    private var pools: List<PjrtBuffer> = emptyList()
    private var closed = false

    private fun open() {
        for (e in manifest.entries) {
            val bytes = Files.readAllBytes(root.resolve(e.bodyPath))
            val hash = sha256(bytes)
            require(hash == e.bodyHash) {
                "ServingModel: ${e.bodyPath} hashes to $hash but the manifest says ${e.bodyHash}"
            }
            bodies[e.bodyPath] = String(bytes, Charsets.UTF_8)
        }
        val staged = LinkedHashMap<String, PjrtBuffer>()
        try {
            for (w in manifest.weights.table) staged[w.name] = stageWeight(w)
            weights = staged
            pools = freshPools()
        } catch (e: Throwable) {
            staged.values.forEach { runCatching { it.close() } }
            throw e
        }
    }

    // ------------------------------------------------------------------
    // Requests.
    // ------------------------------------------------------------------

    /** Greedy continuation of [prompt]: up to [maxNewTokens] ids. */
    fun generate(prompt: IntArray, maxNewTokens: Int): IntArray = generate(prompt, maxNewTokens, IntArray(0))

    /**
     * Greedy continuation of [prompt]: up to [maxNewTokens] ids, stopping
     * after the first id in [stopTokenIds] (which is included).
     */
    @Synchronized
    fun generate(prompt: IntArray, maxNewTokens: Int, stopTokenIds: IntArray): IntArray =
        run(listOf(prompt), maxNewTokens, stopTokenIds.toSet(), null).single()

    /**
     * [generate] for several prompts, decoded together (up to [maxBatch] per
     * step, and as many as the KV pool's pages hold). Row `i` of the result
     * continues `prompts[i]`.
     */
    @Synchronized
    fun generateBatch(prompts: Array<IntArray>, maxNewTokens: Int, stopTokenIds: IntArray): Array<IntArray> {
        require(prompts.isNotEmpty()) { "ServingModel.generateBatch: no prompts" }
        val need = prompts.maxOf { it.size } + maxNewTokens
        val pagesPerSeq = (need + blockSize - 1) / blockSize
        val perGroup = minOf(maxBatch, (manifest.model.numBlocks - 1) / maxOf(pagesPerSeq, 1))
        require(perGroup >= 1) {
            "ServingModel: a sequence of $need tokens needs $pagesPerSeq pages of $blockSize; " +
                "the KV pool has ${manifest.model.numBlocks - 1} (page 0 is scratch)"
        }
        val stops = stopTokenIds.toSet()
        return prompts.toList().chunked(perGroup).flatMap { run(it, maxNewTokens, stops, null) }.toTypedArray()
    }

    /** [generateBatch] without stop tokens. */
    fun generateBatch(prompts: Array<IntArray>, maxNewTokens: Int): Array<IntArray> =
        generateBatch(prompts, maxNewTokens, IntArray(0))

    /** The logits for the token after [prompt]: [vocabSize] floats. */
    @Synchronized
    fun nextTokenLogits(prompt: IntArray): FloatArray {
        var logits: FloatArray? = null
        run(listOf(prompt), 1, emptySet()) { logits = it }
        return logits!!
    }

    // ------------------------------------------------------------------

    private class Row(val tokens: MutableList<Int>, val promptLength: Int, val pages: List<Int>) {
        /** The position of the next token to feed. */
        var pos = 0
        var done = false
        val generated: Int get() = tokens.size - promptLength
    }

    private fun run(
        prompts: List<IntArray>,
        maxNewTokens: Int,
        stops: Set<Int>,
        firstLogits: ((FloatArray) -> Unit)?,
    ): List<IntArray> {
        check(!closed) { "ServingModel is closed" }
        require(maxNewTokens >= 1) { "ServingModel: maxNewTokens must be at least 1 (got $maxNewTokens)" }
        require(prompts.size <= maxBatch) { "ServingModel: ${prompts.size} prompts exceed maxBatch $maxBatch" }
        for ((i, p) in prompts.withIndex()) {
            require(p.isNotEmpty()) { "ServingModel: prompt $i is empty" }
            require(p.size + maxNewTokens <= maxContext) {
                "ServingModel: prompt $i has ${p.size} tokens; with $maxNewTokens new that is " +
                    "${p.size + maxNewTokens}, beyond the artifact's $maxContext"
            }
            checkTokens(p)
        }
        val pagesPerSeq = (prompts.maxOf { it.size } + maxNewTokens + blockSize - 1) / blockSize
        require(1 + prompts.size * pagesPerSeq <= manifest.model.numBlocks) {
            "ServingModel: ${prompts.size} sequences of $pagesPerSeq pages need ${prompts.size * pagesPerSeq} " +
                "pages; the KV pool has ${manifest.model.numBlocks - 1}"
        }
        // Page 0 is scratch (padding rows write there); sequence i owns a run of pages after it.
        val rows = prompts.mapIndexed { i, p ->
            Row(p.toMutableList(), p.size, List(pagesPerSeq) { 1 + i * pagesPerSeq + it })
        }

        fun accept(row: Row, logits: FloatArray, offset: Int) {
            if (firstLogits != null && row.generated == 0) firstLogits(logits.copyOfRange(offset, offset + vocabSize))
            val next = argmax(logits, offset)
            row.tokens += next
            if (row.generated >= maxNewTokens || next in stops) row.done = true
        }

        // The prompts in one prefill call when an entry holds them all.
        val prefill = prefillEntry(rows.size, rows.maxOf { it.promptLength }, rows.maxOf { it.promptLength })
        if (prefill != null && rows.all { it.promptLength >= 2 }) {
            val logits = prefill(prefill, rows)
            val stride = logits.size / prefill.batch
            for ((i, row) in rows.withIndex()) {
                accept(row, logits, i * stride + stride - vocabSize)
                row.pos = row.promptLength
            }
        }
        // Decode steps: a row feeds its token at `pos`; when that is its last
        // token, the step's logits give the next one.
        while (true) {
            val active = rows.filter { !it.done }
            if (active.isEmpty()) break
            val logits = decode(active)
            for ((i, row) in active.withIndex()) {
                val feedingLast = row.pos == row.tokens.size - 1
                row.pos++
                if (feedingLast) accept(row, logits, i * (logits.size / decodeBatchOf(active.size)))
            }
        }
        return rows.map { r -> r.tokens.subList(r.promptLength, r.tokens.size).toIntArray() }
    }

    private fun decodeBatchOf(n: Int) = manifest.bucketLadder.batch.first { it >= n }

    private fun decode(active: List<Row>): FloatArray {
        val n = active.size
        val b = decodeBatchOf(n)
        val context = active.maxOf { it.pos + 1 }
        val c = manifest.bucketLadder.context.firstOrNull { it >= context }
            ?: throw IllegalArgumentException("ServingModel: context $context exceeds the artifact's ${manifest.bucketLadder.maxContext}")
        val entry = manifest.entries.firstOrNull { it.kind == DecodeGraphKind.DECODE && it.batch == b && it.context == c }
            ?: throw IllegalStateException("ServingModel: the artifact has no decode entry for (batch=$b, context=$c)")
        val mbs = entry.maxBlocksPerSeq
        val args = mapOf(
            DecodeSlotRole.TOKEN_IDS to IntArray(b) { if (it < n) active[it].tokens[active[it].pos] else DecodePadding.PADDING_TOKEN_ID },
            DecodeSlotRole.POSITIONS to IntArray(b) { if (it < n) active[it].pos else DecodePadding.PADDING_POSITION },
            DecodeSlotRole.SEQ_LENS to IntArray(b) { if (it < n) active[it].pos + 1 else DecodePadding.PADDING_SEQ_LEN },
            DecodeSlotRole.SLOT_MAPPING to IntArray(b) {
                if (it < n) slotOf(active[it], active[it].pos) else DecodePadding.PADDING_SLOT
            },
            DecodeSlotRole.BLOCK_TABLES to IntArray(b * mbs) { k ->
                val (row, j) = k / mbs to k % mbs
                if (row < n && j <= active[row].pos / blockSize) active[row].pages[j] else DecodePadding.PADDING_BLOCK
            },
        )
        return execute(entry, args)
    }

    private fun prefillEntry(batch: Int, context: Int, tokens: Int): ServingEntry? =
        manifest.entries
            .filter { it.kind == DecodeGraphKind.PREFILL && it.batch >= batch && it.context >= context && it.tokensPerSeq >= tokens }
            .minByOrNull { it.batch * it.context }

    /** One prefill call: each prompt right-aligned in its row, padding first (slot -1, dropped). */
    private fun prefill(entry: ServingEntry, rows: List<Row>): FloatArray {
        val (b, t, mbs) = Triple(entry.batch, entry.tokensPerSeq, entry.maxBlocksPerSeq)
        val tokens = IntArray(b * t) { DecodePadding.PADDING_TOKEN_ID }
        val positions = IntArray(b * t) { DecodePadding.PADDING_POSITION }
        val slots = IntArray(b * t) { DecodePadding.PADDING_SLOT }
        val tables = IntArray(b * mbs) { DecodePadding.PADDING_BLOCK }
        val lens = IntArray(b) { DecodePadding.PADDING_SEQ_LEN }
        for ((i, row) in rows.withIndex()) {
            val pad = t - row.promptLength
            for (j in 0 until row.promptLength) {
                tokens[i * t + pad + j] = row.tokens[j]
                positions[i * t + pad + j] = j
                slots[i * t + pad + j] = slotOf(row, j)
            }
            val pages = (row.promptLength + blockSize - 1) / blockSize
            for (j in 0 until minOf(pages, mbs)) tables[i * mbs + j] = row.pages[j]
            lens[i] = row.promptLength
        }
        return execute(entry, mapOf(
            DecodeSlotRole.TOKEN_IDS to tokens, DecodeSlotRole.POSITIONS to positions,
            DecodeSlotRole.SLOT_MAPPING to slots, DecodeSlotRole.BLOCK_TABLES to tables,
            DecodeSlotRole.SEQ_LENS to lens,
        ))
    }

    private fun slotOf(row: Row, pos: Int) = row.pages[pos / blockSize] * blockSize + pos % blockSize

    /**
     * Binds [args] (by role), the pools and the weights (by name) to [entry],
     * runs it, keeps the output pools, and returns the logits.
     */
    private fun execute(entry: ServingEntry, args: Map<DecodeSlotRole, IntArray>): FloatArray {
        val body = bodies.getValue(entry.bodyPath)
        val inputs = ArrayList<PjrtBuffer>(entry.inputs.size)
        val mine = ArrayList<PjrtBuffer>()
        val poolInputIndex = ArrayList<Int>()
        var poolCursor = 0
        try {
            for ((index, slot) in entry.inputs.withIndex()) {
                when (slot.role) {
                    DecodeSlotRole.WEIGHT -> inputs += weights[slot.name] ?: throw IllegalStateException(
                        "ServingModel: ${entry.entryId} binds weight '${slot.name}', which the weight table does not name",
                    )
                    DecodeSlotRole.KV_POOL_IN -> {
                        inputs += pools[poolCursor++]
                        poolInputIndex += index
                    }
                    DecodeSlotRole.TOKEN_IDS, DecodeSlotRole.POSITIONS, DecodeSlotRole.SEQ_LENS,
                    DecodeSlotRole.SLOT_MAPPING, DecodeSlotRole.BLOCK_TABLES -> {
                        val values = args.getValue(slot.role)
                        require(slot.type.dtype == "i32" && values.size == count(slot)) {
                            "ServingModel: ${entry.entryId} operand '${slot.name}' is ${slot.type.dtype} ${slot.type.dims}; " +
                                "the runner built ${values.size} i32 values"
                        }
                        session.bufferFromHostI32(values, slot.type.dims).also { inputs += it; mine += it }
                    }
                    else -> throw IllegalArgumentException("ServingModel: ${entry.entryId} has a ${slot.role} operand, which this runner does not bind")
                }
            }
            require(poolCursor == pools.size) {
                "ServingModel: ${entry.entryId} takes $poolCursor KV pools; the model has ${pools.size}"
            }
            compiled += entry.cacheKey
            val outs = try {
                session.executeStablehlo(body, inputs)
            } catch (e: Throwable) {
                // The pools may have been donated to a call that failed: start over with fresh ones.
                resetPools()
                throw e
            }
            try {
                // The updated pools: the output each pool input is donated to, else the k-th pool output.
                val poolOutputs = entry.outputs.withIndex().filter { it.value.role == DecodeSlotRole.KV_POOL_OUT }.map { it.index }
                val byInput = entry.donationPairs.associate { it[0] to it[1] }
                val newPools = poolInputIndex.mapIndexed { k, inIndex -> byInput[inIndex] ?: poolOutputs[k] }
                val logitsSlot = entry.outputs.first { it.role == DecodeSlotRole.LOGITS }
                val logitsIndex = entry.outputs.indexOf(logitsSlot)
                val logits = when (logitsSlot.type.dtype) {
                    "f32" -> outs[logitsIndex].toFloatArray(count(logitsSlot))
                    "bf16" -> outs[logitsIndex].toBf16Array(count(logitsSlot)).let { s -> FloatArray(s.size) { Float.fromBits(s[it].toInt() shl 16) } }
                    else -> throw IllegalArgumentException("ServingModel: logits dtype ${logitsSlot.type.dtype} is not read")
                }
                pools.forEach { runCatching { it.close() } }
                pools = newPools.map { outs[it] }
                for ((i, o) in outs.withIndex()) if (i !in newPools) o.close()
                return logits
            } catch (e: Throwable) {
                outs.forEach { runCatching { it.close() } }
                resetPools()
                throw e
            }
        } finally {
            mine.forEach { runCatching { it.close() } }
        }
    }

    private fun resetPools() {
        pools.forEach { runCatching { it.close() } }
        pools = freshPools()
    }

    private fun freshPools(): List<PjrtBuffer> {
        val m = manifest.model
        val n = m.kvPoolDims.fold(1L) { a, d -> a * d }
        require(n <= Int.MAX_VALUE) { "ServingModel: a KV pool of $n elements does not fit one host array" }
        val out = ArrayList<PjrtBuffer>(2 * m.numLayers)
        try {
            repeat(2 * m.numLayers) {
                out += when (m.kvDtype) {
                    "f32" -> session.bufferFromHostF32(FloatArray(n.toInt()), m.kvPoolDims)
                    "bf16" -> session.bufferFromHostBf16(ShortArray(n.toInt()), m.kvPoolDims)
                    else -> throw IllegalArgumentException("ServingModel: KV pool dtype ${m.kvDtype} is not staged")
                }
            }
        } catch (e: Throwable) {
            out.forEach { runCatching { it.close() } }
            throw e
        }
        return out
    }

    private fun stageWeight(w: ServingWeightFile): PjrtBuffer {
        val path = root.resolve(w.path)
        val size = Files.size(path)
        require(size == w.byteLength) { "ServingModel: ${w.path} has $size bytes; the manifest says ${w.byteLength}" }
        require(w.count <= Int.MAX_VALUE) { "ServingModel: weight '${w.name}' has ${w.count} elements, more than one host array holds" }
        val n = w.count.toInt()
        return when (w.dtype) {
            "f32" -> {
                val values = FloatArray(n)
                readLittleEndian(path, 4L * n) { buf, at -> buf.asFloatBuffer().get(values, at, buf.remaining() / 4); buf.remaining() / 4 }
                session.bufferFromHostF32(values, w.dims)
            }
            "bf16" -> {
                val values = ShortArray(n)
                readLittleEndian(path, 2L * n) { buf, at -> buf.asShortBuffer().get(values, at, buf.remaining() / 2); buf.remaining() / 2 }
                session.bufferFromHostBf16(values, w.dims)
            }
            else -> throw IllegalArgumentException(
                "ServingModel: weight '${w.name}' is ${w.dtype}; the runner stages f32 and bf16 (a quantized export is not served here)",
            )
        }
    }

    /** Reads [bytes] bytes of [path] in chunks, handing each little-endian chunk and the element offset to [sink]. */
    private fun readLittleEndian(path: Path, bytes: Long, sink: (ByteBuffer, Int) -> Int) {
        FileChannel.open(path, StandardOpenOption.READ).use { ch ->
            val buf = ByteBuffer.allocateDirect(1 shl 24).order(ByteOrder.LITTLE_ENDIAN)
            var read = 0L
            var element = 0
            while (read < bytes) {
                buf.clear()
                buf.limit(minOf(buf.capacity().toLong(), bytes - read).toInt())
                while (buf.hasRemaining()) {
                    if (ch.read(buf) < 0) throw IllegalStateException("ServingModel: $path ended after $read bytes")
                }
                buf.flip()
                read += buf.remaining()
                element += sink(buf, element)
            }
        }
    }

    private fun checkTokens(ids: IntArray) {
        val refused = manifest.model.refusedTokens.associate { it.id to it.configKey }
        for ((i, t) in ids.withIndex()) {
            require(t in 0 until vocabSize) { "ServingModel: token $i is $t, outside the vocabulary of $vocabSize" }
            refused[t]?.let { throw IllegalArgumentException("ServingModel: token $i is $t, the model's $it placeholder, which the text decoder refuses") }
        }
    }

    private fun argmax(values: FloatArray, offset: Int): Int {
        var best = 0
        for (c in 1 until vocabSize) if (values[offset + c] > values[offset + best]) best = c
        return best
    }

    private fun count(slot: ServingSlot): Int = slot.type.dims.fold(1) { a, d -> a * d }

    /** Closes the device buffers and the PJRT session. */
    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        pools.forEach { runCatching { it.close() } }
        weights.values.forEach { runCatching { it.close() } }
        session.close()
    }

    companion object {
        /**
         * Loads the artifact in [dir] onto the CUDA device of the PJRT plugin
         * that `PjrtBinaries` resolves (`TLALOC_PJRT_PLUGIN_PATH` or the
         * default search). Refuses an artifact this runner does not serve,
         * by name, before touching the device.
         */
        @JvmStatic
        fun load(dir: Path): ServingModel {
            val file = dir.resolve(ServingManifest.FILE_NAME)
            require(Files.isRegularFile(file)) { "ServingModel: $dir is not a Tlaloc serving artifact (no ${ServingManifest.FILE_NAME})" }
            val manifest = ServingManifest.fromJson(Files.readString(file))
            require(manifest.model.windowedKv == null) {
                "ServingModel: $dir has a windowed KV pool (${manifest.schemaVersion}), which this runner does not fill; " +
                    "export it with windowedKv = false"
            }
            require(manifest.model.kvQuant == null) { "ServingModel: $dir has a quantized KV pool, which this runner does not fill" }
            require(manifest.entries.any { it.kind == DecodeGraphKind.DECODE }) { "ServingModel: $dir has no decode entries" }
            check(PjrtBinaries.available && PjrtBinaries.cudaAvailable) {
                "ServingModel: no PJRT CUDA plugin and device:\n" + PjrtBinaries.pluginSearchReport
            }
            val session = PjrtSession(target = PjrtTarget.Cuda)
            val model = ServingModel(dir, manifest, session)
            try {
                model.open()
            } catch (e: Throwable) {
                runCatching { session.close() }
                throw e
            }
            return model
        }

        /** [load] for a path string. */
        @JvmStatic
        fun load(dir: String): ServingModel = load(Path.of(dir))

        private fun sha256(bytes: ByteArray): String =
            MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    }
}
