package io.tlaloc.maestro.serving

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonException
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodeGraphSpec
import io.tlaloc.ir.inference.DecodeSlot
import io.tlaloc.ir.inference.DecodeSlotRole
import io.tlaloc.maestro.ProgramManifest
import io.tlaloc.maestro.TypeDescriptor

/**
 * §0.4.469 — Phase H3a: the **serving artifact's** top-level manifest.
 *
 * `docs/INFERENCE_SERVING_AUDIT.md` §1 stakes the arc on one property —
 * **no JVM in the serving path**. That property is a claim about an
 * ARTIFACT: Tlaloc AOT-compiles a family of decode/prefill graphs and hands
 * a directory to a Python process, which loads it through jaxlib/PJRT and
 * never calls back. This type is what that directory says about itself.
 *
 * ## What this is NOT
 *
 * It is **not an extension of [ProgramManifest]**, and that is a decision
 * rather than an omission. [ProgramManifest] describes ONE program: its
 * boundary types, its mesh requirement, the content address of its
 * StableHLO bytes, and the L3 backend matrix. A serving artifact is a
 * DEPLOYMENT of a *family* of programs — one per (kind, batch, context)
 * ladder point — that share a model shape, a bucket ladder and a weights
 * pointer, and whose slots additionally carry a serving ROLE (which
 * operand is the block table, which pair of buffers is layer 7's KV pool).
 *
 * The slice was asked to extend [ProgramManifest] "only as needed and
 * justify each field". It needed **zero new fields**, and the justification
 * for each field it did not get is the same: `bucket`, `donationPairs`,
 * `role`, `bucketLadder`, `weights` are meaningless to a `program { }`
 * step that resizes an image or runs a training epoch, and a schema that
 * carries fields most of its instances must leave null is a schema that has
 * stopped describing anything. REJECTED, concretely: putting
 * `DecodeSlotRole` on [TypeDescriptor] — it would make `:maestro`'s
 * boundary-type vocabulary depend on the inference vocabulary for every
 * caller. The composition runs the other way: a [ServingEntry] POINTS AT a
 * `ProgramManifest`, by path, and the writer cross-checks that the two
 * agree about the program's boundary types and its content address.
 *
 * ## The directory
 *
 * ```
 *   <artifact>/
 *     tlaloc-serving.json          this manifest
 *     bodies/<bodyHash>.mlir       StableHLO+SDY, content-addressed
 *     programs/<entryId>.json      one ProgramManifest per entry
 *     weights/…                    pointed at, not necessarily present
 * ```
 *
 * Bodies are named by their own SHA-256, so two ladder points that lower to
 * the same program — which happens the moment a model has one layer and two
 * buckets differ only in a dimension the graph does not use — share one
 * file, and a corrupted body is detectable without trusting the manifest.
 *
 * ## Why a JSON schema and not a serialized protobuf / StableHLO bytecode
 *
 * REJECTED: `stablehlo` bytecode for the bodies. It is the better transport
 * (smaller, version-stable) and it is a **named deferral**; textual MLIR is
 * what `jaxlib.mlir.ir.Module.parse` takes directly, it is what the
 * §0.4.299 spike proved jaxlib consumes verbatim, and a serving artifact a
 * human can `grep` is worth a lot in the slice that first draws the line
 * between the two runtimes.
 *
 * REJECTED: protobuf / kotlinx-serialization for the manifest. `:core`'s
 * strict [parseJson] (§0.4.468, written because a checkpoint is untrusted
 * input) already exists and is strict in exactly the ways that matter here
 * — duplicate keys refused, `4.0` is not an extent of 4 — so the artifact
 * and the checkpoint are read by the same discipline.
 */
data class ServingManifest(
    /** Model identity, for humans and for logs. */
    val modelName: String,
    /**
     * The content address of weights+architecture that
     * [DecodeGraphSpec.executableCacheKey] keys on. Two deployments of the
     * same architecture with different weights are different artifacts, and
     * this is the field that says so.
     */
    val modelHash: String,
    val model: ServingModelShape,
    val bucketLadder: ServingBucketLadder,
    val weights: ServingWeightsPointer,
    /** One per compiled ladder point. Order is the writer's; lookup is by
     *  (kind, batch, context) — see [entryFor]. */
    val entries: List<ServingEntry>,
    val schemaVersion: String = SCHEMA_VERSION,
) {
    init {
        require(schemaVersion == SCHEMA_VERSION) {
            "ServingManifest: schemaVersion '$schemaVersion' is not '$SCHEMA_VERSION' — a " +
                "loader must refuse an artifact it does not know the shape of rather than " +
                "read the fields it recognises and guess at the rest"
        }
        require(modelName.isNotBlank()) { "ServingManifest: modelName must not be blank" }
        require(modelHash.isNotBlank()) {
            "ServingManifest: modelHash must not be blank — it is the executable-cache key's " +
                "model component and two weight sets must not share a compiled step"
        }
        require(entries.isNotEmpty()) {
            "ServingManifest: an artifact with no entries compiles nothing; export refused " +
                "rather than shipping a directory a loader can only fail on"
        }
        val dup = entries.groupBy { Triple(it.kind, it.batch, it.context) }.filterValues { it.size > 1 }
        require(dup.isEmpty()) {
            "ServingManifest: duplicate ladder points ${dup.keys} — a serving loop selecting by " +
                "(kind, batch, context) would have to pick one silently"
        }
        for (e in entries) {
            require(bucketLadder.batch.contains(e.batch) && bucketLadder.context.contains(e.context)) {
                "ServingManifest: entry ${e.entryId} is at (batch=${e.batch}, context=${e.context}) " +
                    "which is not a point of the declared ladder ${bucketLadder.batch} × " +
                    "${bucketLadder.context} — the ladder is what the plugin rounds a request UP to, " +
                    "so a compiled shape outside it is unreachable and an advertised shape with no " +
                    "entry is a runtime failure at request time"
            }
        }
    }

    /** Exact lookup. Bucket SELECTION (rounding a request up) is the
     *  plugin's, against [bucketLadder]; this is the last step of it. */
    fun entryFor(kind: DecodeGraphKind, batch: Int, context: Int): ServingEntry =
        entries.firstOrNull { it.kind == kind && it.batch == batch && it.context == context }
            ?: throw IllegalArgumentException(
                "ServingManifest '$modelName': no ${kind.name.lowercase()} entry for " +
                    "(batch=$batch, context=$context); compiled points are " +
                    entries.filter { it.kind == kind }.joinToString { "(${it.batch},${it.context})" },
            )

    fun toJson(): String = buildString {
        append("{")
        append("\"schemaVersion\":").append(jsonStr(schemaVersion)).append(',')
        append("\"modelName\":").append(jsonStr(modelName)).append(',')
        append("\"modelHash\":").append(jsonStr(modelHash)).append(',')
        append("\"model\":").append(model.toJson()).append(',')
        append("\"bucketLadder\":").append(bucketLadder.toJson()).append(',')
        append("\"weights\":").append(weights.toJson()).append(',')
        append("\"entries\":").append(entries.joinToString(",", "[", "]") { it.toJson() })
        append("}")
    }

    companion object {
        const val SCHEMA_VERSION: String = "tlaloc-serving-v1"

        /** The manifest's filename inside the artifact directory. */
        const val FILE_NAME: String = "tlaloc-serving.json"

        fun fromJson(text: String): ServingManifest {
            val o = parseJson(text) as? JsonObject
                ?: throw JsonException("ServingManifest: top level is not a JSON object")
            return ServingManifest(
                schemaVersion = o.str("schemaVersion"),
                modelName = o.str("modelName"),
                modelHash = o.str("modelHash"),
                model = ServingModelShape.fromJson(o.obj("model")),
                bucketLadder = ServingBucketLadder.fromJson(o.obj("bucketLadder")),
                weights = ServingWeightsPointer.fromJson(o.obj("weights")),
                entries = o.arr("entries").elements.map {
                    ServingEntry.fromJson(it as? JsonObject ?: throw JsonException("entries[] element is not an object"))
                },
            )
        }
    }
}

/**
 * The model-side constants, transport-shaped. A structural mirror of
 * `io.tlaloc.ir.inference.DecodeModelShape` rather than that type itself:
 * the manifest is a wire format that a Python process parses, and pinning
 * a wire format to a Kotlin data class's field set is how a refactor
 * becomes a breaking change to an artifact already on disk.
 *
 * [kvPoolAxisOrder] is stated rather than implied. The pool is
 * `[numBlocks, blockSize, numKvHeads, headDim]` (H1a's operand contract),
 * and a loader that has to infer which axis is which from four integers
 * will get it right until the day two of them are equal.
 */
data class ServingModelShape(
    val vocabSize: Int,
    val hiddenSize: Int,
    val numHeads: Int,
    val numKvHeads: Int,
    val headDim: Int,
    val numLayers: Int,
    val numBlocks: Int,
    val blockSize: Int,
    /** Activation/logits dtype, as a `io.tlaloc.core.DType` name. */
    val dtype: String,
    /** KV-pool dtype as the graph boundary carries it. Under [kvQuant] this is
     *  the CODES' integer dtype, not the quantized format — the format is
     *  [kvQuant]'s business, and the two are deliberately separate fields
     *  (§0.4.472). */
    val kvDtype: String,
    /**
     * §0.4.472 — Phase H5: the KV-quant format, or null for a float pool. The
     * slot §0.4.258 reserved, finally carrying a value.
     *
     * A consumer reads this to learn something no tensor type tells it: that
     * the buffers behind the KV-pool slots are integer CODES read against a
     * scale vector, and which code range they live in.
     */
    val kvQuant: ServingKvQuant? = null,
) {
    val kvPoolAxisOrder: List<String> = listOf("numBlocks", "blockSize", "numKvHeads", "headDim")
    val kvPoolDims: List<Int> = listOf(numBlocks, blockSize, numKvHeads, headDim)

    fun toJson(): String = buildString {
        append("{")
        append("\"vocabSize\":").append(vocabSize).append(',')
        append("\"hiddenSize\":").append(hiddenSize).append(',')
        append("\"numHeads\":").append(numHeads).append(',')
        append("\"numKvHeads\":").append(numKvHeads).append(',')
        append("\"headDim\":").append(headDim).append(',')
        append("\"numLayers\":").append(numLayers).append(',')
        append("\"numBlocks\":").append(numBlocks).append(',')
        append("\"blockSize\":").append(blockSize).append(',')
        append("\"dtype\":").append(jsonStr(dtype)).append(',')
        append("\"kvDtype\":").append(jsonStr(kvDtype)).append(',')
        append("\"kvQuant\":").append(kvQuant?.toJson() ?: "null").append(',')
        append("\"kvPoolAxisOrder\":").append(kvPoolAxisOrder.joinToString(",", "[", "]") { jsonStr(it) }).append(',')
        append("\"kvPoolDims\":").append(kvPoolDims.joinToString(",", "[", "]"))
        append("}")
    }

    companion object {
        fun fromJson(o: JsonObject): ServingModelShape = ServingModelShape(
            vocabSize = o.int("vocabSize"), hiddenSize = o.int("hiddenSize"),
            numHeads = o.int("numHeads"), numKvHeads = o.int("numKvHeads"),
            headDim = o.int("headDim"), numLayers = o.int("numLayers"),
            numBlocks = o.int("numBlocks"), blockSize = o.int("blockSize"),
            dtype = o.str("dtype"), kvDtype = o.str("kvDtype"),
            // Absent OR null both read as "no KV-quant": an artifact written
            // before this slice is still a legal artifact, and the reader says
            // so instead of failing on a field it did not have.
            kvQuant = (o["kvQuant"] as? JsonObject)?.let { ServingKvQuant.fromJson(it) },
        )
    }
}

/**
 * §0.4.472 — Phase H5: the KV-quant format, as the artifact publishes it.
 *
 * Four fields, and the fourth is the interesting one:
 *
 * - [dtype] — the format tag (`"int8"`, `"int4"`), i.e. what the codes MEAN.
 * - [scaleStrategy] — `"PER_HEAD"` or `"PER_TENSOR"`, the shape of the scale
 *   vector a loader must supply: `[numKvHeads]` or `[1]`.
 * - [codeMax] — the symmetric code bound (127 / 7). Redundant with [dtype] and
 *   stated anyway, because a Python consumer should not have to keep a table
 *   of this house's conventions to validate a pool it is handed.
 * - [codeDtype] — the dtype the codes actually ride at the graph boundary,
 *   `"i32"` in v1. This is the deferral said OUT LOUD, in the artifact, where
 *   a deployment can see it: the contract is quantized but the BYTES are not
 *   yet narrow, because there is no I8 [io.tlaloc.core.DType] (bf16's
 *   §0.4.455–458 tour is what adding one costs). A serving stack sizing a KV
 *   pool reads [codeDtype] for its byte budget and [dtype] for its accuracy
 *   story, and today those two disagree — which is exactly the fact a manifest
 *   exists to carry.
 */
data class ServingKvQuant(
    val dtype: String,
    val scaleStrategy: String,
    val codeMax: Int,
    val codeDtype: String,
) {
    /** Scale-vector length for a pool of [numKvHeads] kv heads. */
    fun scaleCount(numKvHeads: Int): Int = if (scaleStrategy == PER_HEAD) numKvHeads else 1

    fun toJson(): String = buildString {
        append("{")
        append("\"dtype\":").append(jsonStr(dtype)).append(',')
        append("\"scaleStrategy\":").append(jsonStr(scaleStrategy)).append(',')
        append("\"codeMax\":").append(codeMax).append(',')
        append("\"codeDtype\":").append(jsonStr(codeDtype))
        append("}")
    }

    companion object {
        const val PER_HEAD: String = "PER_HEAD"
        const val PER_TENSOR: String = "PER_TENSOR"

        fun fromJson(o: JsonObject): ServingKvQuant = ServingKvQuant(
            dtype = o.str("dtype"),
            scaleStrategy = o.str("scaleStrategy"),
            codeMax = o.int("codeMax"),
            codeDtype = o.str("codeDtype"),
        )
    }
}

/**
 * The compiled ladder, as the artifact advertises it. This is the
 * "manifest field carrying the spec" that H1c named as a deferral to this
 * slice: a deployment now states its own bucket ladder, so the Python side
 * performs bucket SELECTION — round the request up, refuse over-cap — with
 * no access to `DecodeBucketPolicy` and no second copy of the policy's
 * arithmetic hard-coded in Python.
 */
data class ServingBucketLadder(
    val blockSize: Int,
    val batch: List<Int>,
    val context: List<Int>,
) {
    init {
        require(batch.isNotEmpty() && context.isNotEmpty()) {
            "ServingBucketLadder: both ladders must be non-empty"
        }
        require(batch == batch.sorted() && batch.distinct() == batch) {
            "ServingBucketLadder: batch ladder must be strictly ascending, got $batch"
        }
        require(context == context.sorted() && context.distinct() == context) {
            "ServingBucketLadder: context ladder must be strictly ascending, got $context"
        }
        require(context.all { it % blockSize == 0 }) {
            "ServingBucketLadder: every context bucket must be a whole number of pages of " +
                "blockSize $blockSize, got $context (DecodeBucketPolicy's alignment rule, " +
                "restated on the wire so a loader cannot re-derive it differently)"
        }
    }

    val maxBatch: Int get() = batch.last()
    val maxContext: Int get() = context.last()

    fun toJson(): String =
        "{\"blockSize\":$blockSize,\"batch\":${batch.joinToString(",", "[", "]")}," +
            "\"context\":${context.joinToString(",", "[", "]")}}"

    companion object {
        fun fromJson(o: JsonObject): ServingBucketLadder = ServingBucketLadder(
            blockSize = o.int("blockSize"),
            batch = o.arr("batch").asIntList("bucketLadder.batch"),
            context = o.arr("context").asIntList("bucketLadder.context"),
        )
    }
}

/**
 * Where the weights are. A POINTER, not the bytes.
 *
 * v1 exports graphs whose weights are StableHLO `constant`s inside the
 * body, which is why [embedded] exists and is `true` today. That is an
 * honest v1 and a **named deferral**, not a design: a real Llama's weights
 * are gigabytes, a constant-folded body is a body XLA must re-ingest on
 * every compile, and the whole point of H2's per-tensor safetensors reader
 * is to stage them as device buffers instead. When that lands the body's
 * weight constants become graph PARAMETERS, [embedded] goes false, and
 * [path] names the checkpoint the loader stages — the schema does not move.
 *
 * REJECTED: copying the checkpoint into the artifact directory. The
 * artifact is small and content-addressed; a checkpoint is neither, and
 * duplicating 140 GB to make a directory "self-contained" is a worse
 * property than a pointer plus a hash.
 */
data class ServingWeightsPointer(
    /** `"safetensors"`, or `"embedded"` while v1 folds them into the body. */
    val format: String,
    /** Artifact-relative or absolute path; null while [embedded]. */
    val path: String?,
    val embedded: Boolean,
) {
    init {
        require(embedded == (path == null)) {
            "ServingWeightsPointer: embedded=$embedded with path=$path — weights are either " +
                "folded into the bodies (embedded, no path) or staged from a checkpoint " +
                "(a path); an artifact that claims both has not decided"
        }
    }

    fun toJson(): String =
        "{\"format\":${jsonStr(format)},\"path\":${path?.let { jsonStr(it) } ?: "null"}," +
            "\"embedded\":$embedded}"

    companion object {
        /** The v1 shape: constants live in the StableHLO body. */
        fun embedded(): ServingWeightsPointer = ServingWeightsPointer("embedded", null, true)

        fun fromJson(o: JsonObject): ServingWeightsPointer = ServingWeightsPointer(
            format = o.str("format"),
            path = (o["path"] as? io.tlaloc.core.io.JsonString)?.value,
            embedded = (o["embedded"] as? io.tlaloc.core.io.JsonBool)?.value
                ?: throw JsonException("weights.embedded is not a boolean"),
        )
    }
}

/** One compiled ladder point: a program, its shape, and how to call it. */
data class ServingEntry(
    val kind: DecodeGraphKind,
    val batch: Int,
    val context: Int,
    /** 1 for decode, the context width for prefill (H1c's token axis). */
    val tokensPerSeq: Int,
    val maxBlocksPerSeq: Int,
    /** [DecodeGraphSpec.executableCacheKey] — the string a serving process
     *  looks a compiled step up by, written down so the loader's cache and
     *  the JVM's agree by construction rather than by coincidence. */
    val cacheKey: String,
    /** MLIR symbol to invoke. */
    val entryPoint: String,
    /** Artifact-relative path of the StableHLO body. */
    val bodyPath: String,
    /** SHA-256 hex of the body bytes; [bodyPath] is named by it. */
    val bodyHash: String,
    /** Artifact-relative path of this entry's [ProgramManifest]. */
    val programPath: String,
    val inputs: List<ServingSlot>,
    val outputs: List<ServingSlot>,
    /** `(inputIndex, outputIndex)` pairs a runtime may donate. H1c's list. */
    val donationPairs: List<List<Int>>,
) {
    /** Stable id: what the program-manifest file is named after. */
    val entryId: String get() = "${kind.name.lowercase()}_b${batch}_c$context"

    init {
        require(inputs.isNotEmpty() && outputs.isNotEmpty()) {
            "ServingEntry $entryId: a decode entry has at least tokenIds in and logits out"
        }
        require(donationPairs.all { it.size == 2 }) {
            "ServingEntry $entryId: donationPairs entries must be [inputIndex, outputIndex] pairs"
        }
        for ((i, o) in donationPairs.map { it[0] to it[1] }) {
            require(i in inputs.indices && o in outputs.indices) {
                "ServingEntry $entryId: donation pair ($i -> $o) is out of range for " +
                    "${inputs.size} inputs / ${outputs.size} outputs"
            }
            require(inputs[i].type == outputs[o].type) {
                "ServingEntry $entryId: donation pair ($i -> $o) aliases ${inputs[i].type} onto " +
                    "${outputs[o].type} — a donated buffer is written IN PLACE, so the two must " +
                    "be the same shape and dtype or XLA is being told to overwrite the wrong bytes"
            }
        }
    }

    fun toJson(): String = buildString {
        append("{")
        append("\"kind\":").append(jsonStr(kind.name.lowercase())).append(',')
        append("\"batch\":").append(batch).append(',')
        append("\"context\":").append(context).append(',')
        append("\"tokensPerSeq\":").append(tokensPerSeq).append(',')
        append("\"maxBlocksPerSeq\":").append(maxBlocksPerSeq).append(',')
        append("\"cacheKey\":").append(jsonStr(cacheKey)).append(',')
        append("\"entryPoint\":").append(jsonStr(entryPoint)).append(',')
        append("\"bodyPath\":").append(jsonStr(bodyPath)).append(',')
        append("\"bodyHash\":").append(jsonStr(bodyHash)).append(',')
        append("\"programPath\":").append(jsonStr(programPath)).append(',')
        append("\"inputs\":").append(inputs.joinToString(",", "[", "]") { it.toJson() }).append(',')
        append("\"outputs\":").append(outputs.joinToString(",", "[", "]") { it.toJson() }).append(',')
        append("\"donationPairs\":")
        append(donationPairs.joinToString(",", "[", "]") { "[${it[0]},${it[1]}]" })
        append("}")
    }

    companion object {
        fun fromJson(o: JsonObject): ServingEntry = ServingEntry(
            kind = when (val k = o.str("kind")) {
                "decode" -> DecodeGraphKind.DECODE
                "prefill" -> DecodeGraphKind.PREFILL
                else -> throw JsonException(
                    "ServingEntry.kind '$k' is neither 'decode' nor 'prefill'",
                )
            },
            batch = o.int("batch"), context = o.int("context"),
            tokensPerSeq = o.int("tokensPerSeq"), maxBlocksPerSeq = o.int("maxBlocksPerSeq"),
            cacheKey = o.str("cacheKey"), entryPoint = o.str("entryPoint"),
            bodyPath = o.str("bodyPath"), bodyHash = o.str("bodyHash"),
            programPath = o.str("programPath"),
            inputs = o.arr("inputs").elements.map { ServingSlot.fromJson(it.asObj("inputs[]")) },
            outputs = o.arr("outputs").elements.map { ServingSlot.fromJson(it.asObj("outputs[]")) },
            donationPairs = o.arr("donationPairs").elements.map {
                it.asArr("donationPairs[]").asIntList("donationPairs[]")
            },
        )
    }
}

/**
 * A named, ROLE-tagged position in a compiled entry's signature. The role
 * is what lets a plugin bind by meaning — "hand me the block tables" —
 * rather than by an index it would have to keep in sync with H1c by hand.
 */
data class ServingSlot(val name: String, val role: DecodeSlotRole, val type: TypeDescriptor) {
    fun toJson(): String =
        "{\"name\":${jsonStr(name)},\"role\":${jsonStr(role.name)},\"type\":${type.toJson()}}"

    companion object {
        fun fromSlot(s: DecodeSlot): ServingSlot =
            ServingSlot(s.name, s.role, TypeDescriptor.fromDxirType(s.type))

        fun fromJson(o: JsonObject): ServingSlot {
            val roleName = o.str("role")
            val role = DecodeSlotRole.entries.firstOrNull { it.name == roleName }
                ?: throw JsonException(
                    "ServingSlot.role '$roleName' is not a DecodeSlotRole (one of " +
                        DecodeSlotRole.entries.joinToString { it.name } + ")",
                )
            val t = o.obj("type")
            return ServingSlot(
                name = o.str("name"),
                role = role,
                type = TypeDescriptor(
                    dtype = t.str("dtype"),
                    dims = t.arr("dims").asIntList("type.dims"),
                    axisNames = t.arr("axisNames").elements.map {
                        (it as? io.tlaloc.core.io.JsonString)?.value
                    },
                ),
            )
        }
    }
}

// --- small local JSON helpers ------------------------------------------
//
// `:maestro` already hand-rolls its writer side (ProgramManifest), and the
// reader side is `:core`'s strict parser. These are the two adapters in
// between; there is deliberately no third JSON library in the build.

private fun JsonObject.int(key: String): Int =
    (this[key] as? JsonNumber)?.asInt(key) ?: throw JsonException("field '$key' is not a number")

private fun io.tlaloc.core.io.JsonValue.asObj(what: String): JsonObject =
    this as? JsonObject ?: throw JsonException("$what is not a JSON object")

private fun io.tlaloc.core.io.JsonValue.asArr(what: String): JsonArray =
    this as? JsonArray ?: throw JsonException("$what is not a JSON array")

internal fun jsonStr(s: String): String = buildString {
    append('"')
    for (c in s) {
        when (c) {
            '"' -> append("\\\"")
            '\\' -> append("\\\\")
            '\n' -> append("\\n")
            '\r' -> append("\\r")
            '\t' -> append("\\t")
            else -> if (c.code < 0x20) append("\\u").append(c.code.toString(16).padStart(4, '0')) else append(c)
        }
    }
    append('"')
}
