package io.tlaloc.maestro.serving

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
 * The version directory `1/` holds the artifact's files unchanged. The
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

    private val REQUEST_INPUT_ROLES = setOf(
        DecodeSlotRole.TOKEN_IDS, DecodeSlotRole.POSITIONS, DecodeSlotRole.BLOCK_TABLES,
        DecodeSlotRole.SEQ_LENS, DecodeSlotRole.SLOT_MAPPING,
    )

    /**
     * The `config.pbtxt` text for [manifest] served as the Triton model
     * [modelName].
     *
     * Refuses, by name, an artifact whose entries disagree about their slot
     * names, roles, dtypes or ranks, whose entries name different entry
     * functions, a slot dtype Triton cannot carry, a `KV_POOL_OUT` without a
     * donation pair, and a `WEIGHT` slot the weight table does not name.
     */
    fun config(manifest: ServingManifest, modelName: String): String {
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
                DecodeSlotRole.KV_POOL_IN -> "state:${slot.name}"
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
                DecodeSlotRole.KV_POOL_OUT -> {
                    val i = pairedInput[j]
                        ?: throw IllegalArgumentException(
                            "TritonModelRepository: KV_POOL_OUT '${slot.name}' (result $j) has no " +
                                "donation pair, so no KV_POOL_IN it updates",
                        )
                    val input = first.inputs[i]
                    require(input.role == DecodeSlotRole.KV_POOL_IN && input.type == slot.type) {
                        "TritonModelRepository: KV_POOL_OUT '${slot.name}' is paired with input " +
                            "'${input.name}' (${input.role}, ${input.type.dtype}${input.type.dims}), " +
                            "not a KV_POOL_IN of type ${slot.type.dtype}${slot.type.dims}"
                    }
                    "state:${input.name}"
                }
                else -> throw IllegalArgumentException(
                    "TritonModelRepository: output slot '${slot.name}' has role ${slot.role}, " +
                        "which is not an output role",
                )
            }
        }
        val hasState = first.inputs.any { it.role == DecodeSlotRole.KV_POOL_IN }

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
            }
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
    fun write(artifactDir: Path, repositoryDir: Path, modelName: String): Path {
        val manifestFile = artifactDir.resolve(ServingManifest.FILE_NAME)
        require(Files.isRegularFile(manifestFile)) {
            "TritonModelRepository: $artifactDir is not a serving artifact (no " +
                "${ServingManifest.FILE_NAME})"
        }
        val manifest = ServingManifest.fromJson(Files.readString(manifestFile))
        val config = config(manifest, modelName)

        val modelDir = repositoryDir.resolve(modelName)
        val versionDir = modelDir.resolve(VERSION)
        val files = buildList {
            add(ServingManifest.FILE_NAME)
            for (e in manifest.entries) {
                add(e.bodyPath)
                add(e.programPath)
            }
            for (w in manifest.weights.table) add(w.path)
        }.distinct()
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
