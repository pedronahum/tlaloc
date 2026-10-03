package io.tlaloc.maestro.serving

import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodeSlotRole
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption

/**
 * Writes a serving artifact as a model in an NVIDIA Triton Inference Server
 * model repository, served by the `tlaloc` backend (`triton/README.md`).
 *
 * The model directory is
 *
 * ```
 *   <repository>/<model>/config.pbtxt
 *   <repository>/<model>/1/tlaloc-serving.json, bodies/, programs/, weights/
 * ```
 *
 * The version directory `1/` holds the artifact's files unchanged.
 *
 * A bounded-program artifact (`tlaloc-bounded.json`) is written for the backend's bounded
 * mode instead; see [boundedConfig].
 *
 * ## Sequence mode (the default for an artifact with KV pools)
 *
 * The generated model uses Triton's sequence batcher (oldest strategy) and
 * the backend's sequence mode: `config.pbtxt` names the manifest
 * (`serving_manifest`) and declares one input, `TOKENS` (INT32 `[-1]`), two
 * outputs, `LOGITS` (FP32 `[vocab]`) and `KV_PAGES` (INT32 `[2]`, the pages
 * the sequence holds in the full-history and the windowed pools; a client
 * asks for it or not), and the START, END and CORRID control inputs. A client sends a sequence's token ids with a correlation ID; the
 * backend allocates the sequence's KV pages on START, frees them on END (or,
 * when pages run short, once the sequence has been idle for twice
 * `max_sequence_idle_microseconds` plus a queueing allowance),
 * runs a request of several tokens through the smallest prefill entry that
 * fits (the requests of several sequences in one batch together, when the
 * artifact has prefill entries of batch > 1) and a one-token request through
 * a decode entry together with the other sequences' steps in the same batch,
 * and answers with the last token's logits. `max_batch_size` is the artifact's largest decode batch.
 * For an artifact with a windowed KV pool the backend also keeps each
 * sequence's ring of windowed pages and splits a request into calls the ring
 * can hold. See [SequenceOptions] for the knobs.
 *
 * ## Client mode
 *
 * With [KvMode.CLIENT] (and always for an artifact without KV pools) the
 * `config.pbtxt` maps each slot of the manifest by its role:
 *
 * - `TOKEN_IDS`, `POSITIONS`, `BLOCK_TABLES`, `SEQ_LENS` and `SLOT_MAPPING`
 *   become Triton inputs and `LOGITS` becomes a Triton output, under the
 *   slot's own name. A dimension that differs between the artifact's entries
 *   (the batch size, the block-table width) is `-1`; the backend runs each
 *   request on the entry compiled for exactly its shapes.
 * - `WEIGHT` slots are read by the backend from their staged files when the
 *   model loads and stay on the device. They are not request tensors.
 * - `KV_POOL_IN` slots are backend state: zero-filled when a model instance
 *   starts, and replaced after every request by the `KV_POOL_OUT` result the
 *   manifest's `donationPairs` pair them with. The client allocates pages,
 *   as it would for any paged KV cache, by what it sends in `blockTables` and
 *   `slotMapping`. Because the pools belong to a model instance, a model with
 *   KV pools gets one instance.
 *
 * The positional order of the entry function is written into the `arguments`
 * and `results` parameters, so the backend binds every argument by name. A
 * model with KV pools also gets `kv_block_size` and `kv_num_blocks`
 * parameters, which the backend ignores and a client reads to lay out pages.
 */
object TritonModelRepository {

    /** The Triton backend name the generated configuration selects. */
    const val BACKEND: String = "tlaloc"

    /** The model version directory the artifact is written into. */
    const val VERSION: String = "1"

    private val MODEL_NAME = Regex("[A-Za-z0-9_][A-Za-z0-9_.-]*")

    /** How the generated model manages an artifact's KV pools. */
    enum class KvMode {
        /** The backend keeps each sequence's pages, keyed by correlation ID. */
        SEQUENCE,

        /** The client names pages and slots in every request. */
        CLIENT,
    }

    /**
     * Sequence-batcher settings of a [KvMode.SEQUENCE] model.
     *
     * @param maxSequenceIdleMicros how long a sequence may go without a request
     *   before Triton ends it and the backend frees its pages.
     * @param maxQueueDelayMicros how long Triton's sequence batcher may hold a
     *   sequence's request to batch it with other sequences' steps before the
     *   backend sees it. The backend forms its own batches (it waits for the
     *   sequences that decoded in its previous batch, and a lone sequence does
     *   not wait), so the default is 0: a delay here is added to every step
     *   of fewer sequences than `preferred_batch_size`.
     */
    data class SequenceOptions(
        val maxSequenceIdleMicros: Long = 60_000_000,
        val maxQueueDelayMicros: Long = 0,
    ) {
        init {
            require(maxSequenceIdleMicros >= 1 && maxQueueDelayMicros >= 0) {
                "TritonModelRepository.SequenceOptions: maxSequenceIdleMicros must be >= 1 and " +
                    "maxQueueDelayMicros >= 0, got $maxSequenceIdleMicros and $maxQueueDelayMicros"
            }
        }
    }

    /** The token-ids input of a [KvMode.SEQUENCE] model. */
    const val TOKENS: String = "TOKENS"

    /** The logits output of a [KvMode.SEQUENCE] model. */
    const val LOGITS: String = "LOGITS"

    private val REQUEST_INPUT_ROLES = setOf(
        DecodeSlotRole.TOKEN_IDS, DecodeSlotRole.POSITIONS, DecodeSlotRole.BLOCK_TABLES,
        DecodeSlotRole.SEQ_LENS, DecodeSlotRole.SLOT_MAPPING,
        DecodeSlotRole.WINDOW_BLOCK_TABLES, DecodeSlotRole.WINDOW_SLOT_MAPPING, DecodeSlotRole.STATE_SLOTS,
    )

    private val POOL_IN_ROLES =
        setOf(DecodeSlotRole.KV_POOL_IN, DecodeSlotRole.WINDOW_KV_POOL_IN, DecodeSlotRole.STATE_POOL_IN)

    /** The input role each pool output updates. */
    private val POOL_OUT_TO_IN = mapOf(
        DecodeSlotRole.KV_POOL_OUT to DecodeSlotRole.KV_POOL_IN,
        DecodeSlotRole.WINDOW_KV_POOL_OUT to DecodeSlotRole.WINDOW_KV_POOL_IN,
        DecodeSlotRole.STATE_POOL_OUT to DecodeSlotRole.STATE_POOL_IN,
    )

    /**
     * The optional output of a [KvMode.SEQUENCE] model: INT32 `[2]`, the
     * pages the sequence holds after the request in the full-history pool
     * and in the windowed pool (0 without one).
     */
    const val KV_PAGES: String = "KV_PAGES"

    /**
     * The optional greedy output of a [KvMode.SEQUENCE] model: INT32 `[1]`,
     * the first index of the largest logit, chosen by the backend so that a
     * greedy client is sent four bytes instead of the logits row.
     */
    const val NEXT_TOKEN: String = "NEXT_TOKEN"

    /**
     * The `config.pbtxt` text for [manifest] served as the Triton model
     * [modelName].
     *
     * Refuses, by name, an artifact whose entries disagree about their slot
     * names, roles, dtypes or ranks, whose entries name different entry
     * functions, a slot dtype Triton cannot carry, a `KV_POOL_OUT` without a
     * donation pair, and a `WEIGHT` slot the weight table does not name.
     */
    fun config(
        manifest: ServingManifest,
        modelName: String,
        kv: KvMode = KvMode.SEQUENCE,
        options: SequenceOptions = SequenceOptions(),
    ): String {
        require(MODEL_NAME.matches(modelName)) {
            "TritonModelRepository: '$modelName' is not a usable Triton model name; use letters, " +
                "digits, '_', '.' and '-', not starting with '.' or '-'"
        }
        val entries = manifest.entries
        require(entries.isNotEmpty()) { "TritonModelRepository: the manifest has no entries" }
        val first = entries.first()
        for (e in entries) {
            require(e.entryPoint == first.entryPoint) {
                "TritonModelRepository: entry ${e.entryId} is called through @${e.entryPoint} " +
                    "but ${first.entryId} through @${first.entryPoint}; one Triton model serves " +
                    "one entry function name"
            }
            sameSlots(first, e, "input", first.inputs, e.inputs)
            sameSlots(first, e, "output", first.outputs, e.outputs)
        }

        val weightPaths = manifest.weights.table.associate { it.name to it.path }
        val arguments = first.inputs.map { slot ->
            when (slot.role) {
                DecodeSlotRole.WEIGHT -> "weight:" + (
                    weightPaths[slot.name]
                        ?: throw IllegalArgumentException(
                            "TritonModelRepository: WEIGHT slot '${slot.name}' is not in the " +
                                "manifest's weight table, so there is no file to load it from",
                        )
                    )
                in POOL_IN_ROLES -> "state:${slot.name}"
                in REQUEST_INPUT_ROLES -> "input:${slot.name}"
                else -> throw IllegalArgumentException(
                    "TritonModelRepository: input slot '${slot.name}' has role ${slot.role}, " +
                        "which is not an input role",
                )
            }
        }
        val pairedInput = first.donationPairs.associate { it[1] to it[0] }
        val results = first.outputs.mapIndexed { j, slot ->
            when (slot.role) {
                DecodeSlotRole.LOGITS -> "output:${slot.name}"
                in POOL_OUT_TO_IN.keys -> {
                    val i = pairedInput[j]
                        ?: throw IllegalArgumentException(
                            "TritonModelRepository: KV_POOL_OUT '${slot.name}' (result $j) has no " +
                                "donation pair, so no KV_POOL_IN it updates",
                        )
                    val input = first.inputs[i]
                    val wantRole = POOL_OUT_TO_IN.getValue(slot.role)
                    require(input.role == wantRole && input.type == slot.type) {
                        "TritonModelRepository: ${slot.role} '${slot.name}' is paired with input " +
                            "'${input.name}' (${input.role}, ${input.type.dtype}${input.type.dims}), " +
                            "not a $wantRole of type ${slot.type.dtype}${slot.type.dims}"
                    }
                    "state:${input.name}"
                }
                else -> throw IllegalArgumentException(
                    "TritonModelRepository: output slot '${slot.name}' has role ${slot.role}, " +
                        "which is not an output role",
                )
            }
        }
        val hasState = first.inputs.any { it.role in POOL_IN_ROLES }
        if (hasState && kv == KvMode.SEQUENCE) return sequenceConfig(manifest, modelName, options)

        fun tensor(index: Int, slot: ServingSlot, side: (ServingEntry) -> List<ServingSlot>): String {
            val dims = slot.type.dims.indices.map { d ->
                val sizes = entries.map { side(it)[index].type.dims[d] }.toSet()
                if (sizes.size == 1) sizes.single() else -1
            }
            // Triton has no rank-0 tensors in a non-batching model; the backend
            // maps dims [ 1 ] to a rank-0 argument.
            val shown = dims.ifEmpty { listOf(1) }
            return "  { name: \"${slot.name}\" data_type: ${tritonType(slot)} " +
                "dims: [ ${shown.joinToString(", ")} ] }"
        }

        val inputs = first.inputs.withIndex()
            .filter { it.value.role in REQUEST_INPUT_ROLES }
            .map { tensor(it.index, it.value) { e -> e.inputs } }
        val outputs = first.outputs.withIndex()
            .filter { it.value.role == DecodeSlotRole.LOGITS }
            .map { tensor(it.index, it.value) { e -> e.outputs } }
        require(inputs.isNotEmpty() && outputs.isNotEmpty()) {
            "TritonModelRepository: the artifact has ${inputs.size} request inputs and " +
                "${outputs.size} LOGITS outputs; a Triton model needs at least one of each"
        }
        val bodies = entries.map { it.bodyPath }.distinct()

        return buildString {
            append("# Written by Tlaloc from the serving artifact '${manifest.modelName}'\n")
            append("# (${manifest.modelHash}), ${entries.size} ")
            append(if (entries.size == 1) "entry" else "entries")
            append(": ${entries.joinToString(", ") { it.entryId }}.\n")
            if (manifest.weights.table.isNotEmpty()) {
                append("# ${manifest.weights.table.size} staged weights are loaded onto the device ")
                append("when the model loads.\n")
            }
            if (hasState) {
                append("# The KV pools are state of the model instance: zero at start, updated by ")
                append("every\n# request. The client allocates pages through blockTables and ")
                append("slotMapping.\n")
            }
            append("name: \"$modelName\"\n")
            append("backend: \"$BACKEND\"\n")
            append("max_batch_size: 0\n")
            append("input [\n").append(inputs.joinToString(",\n")).append("\n]\n")
            append("output [\n").append(outputs.joinToString(",\n")).append("\n]\n")
            append("instance_group [ { kind: KIND_GPU count: 1 gpus: [ 0 ] } ]\n")
            parameter("artifact", bodies.joinToString(", "))
            parameter("entry", first.entryPoint)
            parameter("arguments", arguments.joinToString(", "))
            parameter("results", results.joinToString(", "))
            if (hasState) {
                // Read by clients, not by the backend: the page geometry a
                // client needs to fill blockTables and slotMapping.
                parameter("kv_block_size", manifest.model.blockSize.toString())
                parameter("kv_num_blocks", manifest.model.numBlocks.toString())
                windowParameters(manifest)
            }
        }
    }

    /** The windowed pool's geometry, for clients; nothing without a windowed pool. */
    private fun StringBuilder.windowParameters(manifest: ServingManifest) {
        val w = manifest.model.windowedKv ?: return
        parameter("kv_window", w.window.toString())
        parameter("kv_window_layers", w.layers.joinToString(","))
        parameter("kv_window_num_blocks", w.numBlocks.toString())
        parameter("kv_window_ring_pages", w.ringPages.toString())
    }

    private fun sequenceConfig(manifest: ServingManifest, modelName: String, options: SequenceOptions): String {
        val decode = manifest.entries.filter { it.kind == DecodeGraphKind.DECODE }
        require(decode.isNotEmpty()) {
            "TritonModelRepository: the artifact has no decode entry; a sequence-mode model " +
                "generates tokens with one"
        }
        val maxBatch = decode.maxOf { it.batch }
        val prefill = manifest.entries.filter { it.kind == DecodeGraphKind.PREFILL }
        val vocab = manifest.model.vocabSize
        return buildString {
            append("# Written by Tlaloc from the serving artifact '${manifest.modelName}'\n")
            append("# (${manifest.modelHash}), ${manifest.entries.size} ")
            append(if (manifest.entries.size == 1) "entry" else "entries")
            append(": ${manifest.entries.joinToString(", ") { it.entryId }}.\n")
            append("# Sequence mode: a request sends one sequence's token ids (with its correlation\n")
            append("# ID and START/END flags) and gets the last token's logits. The backend keeps each\n")
            append("# sequence's KV pages; a request of several tokens runs as ")
            append(if (prefill.isEmpty()) "decode steps" else "a prefill chunk")
            append(",\n# one token as a decode step batched with other sequences' steps. Triton hands\n")
            append("# each request over at once; the backend batches the steps of the sequences that\n")
            append("# decode together.\n")
            val chunk = prefill.maxOfOrNull { it.tokensPerSeq } ?: 0
            if (prefill.any { it.tokensPerSeq < it.context }) {
                append("# A prefill call takes at most $chunk tokens per sequence; a longer request\n")
                append("# runs as several calls.\n")
            }
            val prefillBatch = prefill.maxOfOrNull { it.batch } ?: 0
            if (prefillBatch > 1) {
                append("# The prompts of up to $prefillBatch sequences in one batch share a prefill call.\n")
            }
            append("name: \"$modelName\"\n")
            append("backend: \"$BACKEND\"\n")
            append("max_batch_size: $maxBatch\n")
            append("input [\n")
            append("  { name: \"$TOKENS\" data_type: TYPE_INT32 dims: [ -1 ] allow_ragged_batch: true }\n")
            append("]\n")
            append("output [\n")
            append("  { name: \"$LOGITS\" data_type: TYPE_FP32 dims: [ $vocab ] },\n")
            append("  { name: \"$KV_PAGES\" data_type: TYPE_INT32 dims: [ 2 ] },\n")
            append("  { name: \"$NEXT_TOKEN\" data_type: TYPE_INT32 dims: [ 1 ] }\n")
            append("]\n")
            append("sequence_batching {\n")
            append("  max_sequence_idle_microseconds: ${options.maxSequenceIdleMicros}\n")
            append("  control_input [\n")
            append("    { name: \"START\" control [ { kind: CONTROL_SEQUENCE_START int32_false_true: [ 0, 1 ] } ] },\n")
            append("    { name: \"END\" control [ { kind: CONTROL_SEQUENCE_END int32_false_true: [ 0, 1 ] } ] },\n")
            append("    { name: \"CORRID\" control [ { kind: CONTROL_SEQUENCE_CORRID data_type: TYPE_UINT64 } ] }\n")
            append("  ]\n")
            append("  oldest {\n")
            // Every live sequence holds at least one page of each pool, and
            // page 0 is the padding page, so no more than numBlocks - 1 can be
            // live at once; with linear-attention state, no more than its slots.
            val pages = minOf(manifest.model.numBlocks, manifest.model.windowedKv?.numBlocks ?: Int.MAX_VALUE)
            val live = minOf(maxOf(1, pages - 1), manifest.model.linearState?.numSlots ?: Int.MAX_VALUE)
            append("    max_candidate_sequences: $live\n")
            append("    preferred_batch_size: [ $maxBatch ]\n")
            append("    max_queue_delay_microseconds: ${options.maxQueueDelayMicros}\n")
            append("  }\n")
            append("}\n")
            append("instance_group [ { kind: KIND_GPU count: 1 gpus: [ 0 ] } ]\n")
            parameter("serving_manifest", ServingManifest.FILE_NAME)
            // Read by clients, not by the backend.
            parameter("kv_block_size", manifest.model.blockSize.toString())
            parameter("kv_num_blocks", manifest.model.numBlocks.toString())
            parameter("max_context", decode.maxOf { it.context }.toString())
            windowParameters(manifest)
        }
    }

    /**
     * Write [artifactDir] (a serving artifact directory) into [repositoryDir]
     * as the Triton model [modelName], and return the model directory.
     *
     * Files are hard-linked when [repositoryDir] is on the same file system as
     * the artifact, and copied otherwise; either way the model directory holds
     * real files, so it can be mounted into a container on its own. Files of a
     * previous write with the same names are replaced.
     */
    fun write(
        artifactDir: Path,
        repositoryDir: Path,
        modelName: String,
        kv: KvMode = KvMode.SEQUENCE,
        options: SequenceOptions = SequenceOptions(),
    ): Path {
        val manifestFile = artifactDir.resolve(ServingManifest.FILE_NAME)
        if (!Files.exists(manifestFile) && Files.isRegularFile(artifactDir.resolve(BoundedManifest.FILE_NAME))) {
            require(kv == KvMode.SEQUENCE && options == SequenceOptions()) {
                "TritonModelRepository: $artifactDir is a bounded-program artifact; KV and sequence options do not apply"
            }
            return writeBounded(artifactDir, repositoryDir, modelName)
        }
        require(Files.isRegularFile(manifestFile)) {
            "TritonModelRepository: $artifactDir is not a serving artifact (no " +
                "${ServingManifest.FILE_NAME})"
        }
        val manifest = ServingManifest.fromJson(Files.readString(manifestFile))
        val config = config(manifest, modelName, kv, options)

        val files = buildList {
            add(ServingManifest.FILE_NAME)
            for (e in manifest.entries) {
                add(e.bodyPath)
                add(e.programPath)
            }
            for (w in manifest.weights.table) add(w.path)
        }.distinct()
        return place(artifactDir, repositoryDir, modelName, files, config)
    }

    /**
     * The `config.pbtxt` of a bounded-program artifact (docs/design/bounded-dims.md): the
     * backend's bounded mode (`bounded_manifest`). One input per DATA input of the manifest and
     * its one output, by name, with `-1` for each bounded axis; `max_batch_size: 0`. The backend
     * picks the bucket, pads, fills the masks and lengths and slices the output itself.
     */
    fun boundedConfig(manifest: BoundedManifest, modelName: String): String {
        require(MODEL_NAME.matches(modelName)) {
            "TritonModelRepository: model name '$modelName' must match ${MODEL_NAME.pattern}"
        }
        require(manifest.outputs.size == 1) {
            "TritonModelRepository: bounded program '${manifest.name}' has ${manifest.outputs.size} outputs; the " +
                "tlaloc backend's bounded mode serves one"
        }
        val scalars = (manifest.inputs.filter { it.role == BoundedManifest.ROLE_DATA } + manifest.outputs).filter { it.axes.isEmpty() }
        require(scalars.isEmpty()) {
            "TritonModelRepository: bounded program '${manifest.name}' has rank-0 tensors ${scalars.map { it.name }}; " +
                "a Triton config cannot declare them without a reshape, which the bounded mode does not read"
        }
        fun decl(t: TensorDecl): String {
            val type = if (t.dtype == "i32") "TYPE_INT32" else "TYPE_FP32"
            val dims = t.axes.joinToString(", ") { if (it.bound != null) "-1" else it.size.toString() }
            return "  { name: \"${t.name}\" data_type: $type dims: [ $dims ] }"
        }
        return buildString {
            append("# Tlaloc bounded program '${manifest.name}' (${manifest.schemaVersion}): ")
            append(manifest.bounds.joinToString("; ") { "${it.name} <= ${it.max}, buckets ${it.buckets}" })
            append(".\n# A -1 dim is bounded; the backend pads each request to the smallest bucket that holds it.\n")
            append("name: \"$modelName\"\n")
            append("backend: \"$BACKEND\"\n")
            append("max_batch_size: 0\n")
            append("input [\n")
            append(manifest.inputs.filter { it.role == BoundedManifest.ROLE_DATA }.joinToString(",\n") { decl(it) })
            append("\n]\n")
            append("output [\n")
            append(decl(manifest.outputs.single()))
            append("\n]\n")
            append("instance_group [ { kind: KIND_GPU count: 1 gpus: [ 0 ] } ]\n")
            parameter("bounded_manifest", BoundedManifest.FILE_NAME)
        }
    }

    private fun writeBounded(artifactDir: Path, repositoryDir: Path, modelName: String): Path {
        // load checks every body against its hash, so a changed body is refused here.
        val manifest = BoundedProgramExport.load(artifactDir)
        val files = buildList {
            add(BoundedManifest.FILE_NAME)
            for (e in manifest.entries) {
                add(e.bodyPath)
                add(e.programPath)
            }
        }.distinct()
        return place(artifactDir, repositoryDir, modelName, files, boundedConfig(manifest, modelName))
    }

    private fun place(artifactDir: Path, repositoryDir: Path, modelName: String, files: List<String>, config: String): Path {
        val modelDir = repositoryDir.resolve(modelName)
        val versionDir = modelDir.resolve(VERSION)
        for (relative in files) {
            val source = artifactDir.resolve(relative).normalize()
            require(source.startsWith(artifactDir.normalize()) && Files.isRegularFile(source)) {
                "TritonModelRepository: the manifest names '$relative', which is not a file " +
                    "inside $artifactDir"
            }
            val target = versionDir.resolve(relative)
            Files.createDirectories(target.parent)
            Files.deleteIfExists(target)
            try {
                Files.createLink(target, source)
            } catch (_: IOException) {
                Files.copy(source, target, StandardCopyOption.REPLACE_EXISTING)
            } catch (_: UnsupportedOperationException) {
                Files.copy(source, target, StandardCopyOption.REPLACE_EXISTING)
            }
        }
        Files.writeString(modelDir.resolve("config.pbtxt"), config)
        return modelDir
    }

    private fun StringBuilder.parameter(key: String, value: String) {
        append("parameters: { key: \"$key\" value: { string_value: \"$value\" } }\n")
    }

    private fun sameSlots(
        first: ServingEntry,
        other: ServingEntry,
        what: String,
        a: List<ServingSlot>,
        b: List<ServingSlot>,
    ) {
        val same = a.size == b.size && a.zip(b).all { (x, y) ->
            x.name == y.name && x.role == y.role && x.type.dtype == y.type.dtype &&
                x.type.dims.size == y.type.dims.size
        }
        require(same) {
            "TritonModelRepository: entries ${first.entryId} and ${other.entryId} have different " +
                "$what slots (names, roles, dtypes or ranks); one Triton model binds one signature"
        }
    }

    private fun tritonType(slot: ServingSlot): String = when (slot.type.dtype) {
        "f32" -> "TYPE_FP32"
        "f64" -> "TYPE_FP64"
        "f16" -> "TYPE_FP16"
        "bf16" -> "TYPE_BF16"
        "i8" -> "TYPE_INT8"
        "i32" -> "TYPE_INT32"
        "i64" -> "TYPE_INT64"
        "u8", "ui8" -> "TYPE_UINT8"
        "bool", "i1" -> "TYPE_BOOL"
        else -> throw IllegalArgumentException(
            "TritonModelRepository: slot '${slot.name}' has dtype ${slot.type.dtype}, which the " +
                "tlaloc backend does not serve",
        )
    }
}
