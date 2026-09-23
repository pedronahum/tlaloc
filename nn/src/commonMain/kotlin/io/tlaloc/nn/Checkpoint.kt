/**
 * §0.4.502 (Tier 2 item 7) — **model persistence.** Before this file, Tlaloc
 * could train a model on a GPU (certified: 600 Adam steps on a GB10,
 * `examples/gpu-training`) and could not save the result. `Components.kt` said
 * so in a comment — "minus `store`/`load`, out of scope v1" — and
 * `core/.../Safetensors.kt` was read-only. §0.4.502 wrote the safetensors
 * WRITER; this file is the model-level round trip on top of it.
 *
 * ## The format, and why it is safetensors
 *
 * A checkpoint is ONE safetensors file. Nothing else: no side-car JSON, no
 * zip, no second format to keep in sync. Every quantity a resume needs is
 * either a tensor or a string, and safetensors carries both.
 *
 *   `param.<key>`    one entry per [Trainable.parameters] entry, key verbatim
 *   `buffer.<key>`   one entry per [Stateful.buffers] entry, key verbatim
 *   `opt.<slot>`     the optimizer's tensor state (Adam's `m`/`v`, SGD's
 *                    velocity, RMSprop's mean-square), slot names owned by
 *                    the optimizer
 *   `__metadata__`   `tlaloc.checkpoint.version`, the optimizer's kind and
 *                    its scalar state (step counts), plus the caller's own
 *                    metadata
 *
 * REJECTED: a Tlaloc-specific container format. The repository already reads
 * safetensors (§0.4.468, HuggingFace ingestion), already writes it (§0.4.502),
 * and a checkpoint that `safetensors.torch.load_file` can open is a checkpoint
 * a user can inspect with tools they already have. REJECTED: Java
 * serialization — it ties the file to the JVM and to the exact class shapes,
 * which is the opposite of what a checkpoint is for. REJECTED: a directory of
 * one file per tensor — it multiplies the interrupted-save problem by the
 * parameter count.
 *
 * ## Immutability
 *
 * Loading produces a NEW model. There is no `load(into:)` and no mutation
 * anywhere: [ModelSnapshot.restore] takes the model whose STRUCTURE the
 * checkpoint must match and returns a new instance through
 * [Trainable.withParameters] / [Stateful.withBuffers]. This is not a stylistic
 * choice, it is the module's contract (decision 2 of MODEL_LAYER_PLAN.md): a
 * training step already returns a new model, so a loader that mutated would be
 * the only thing in `:nn` that did.
 *
 * ## What a checkpoint does NOT carry, by name
 *
 * **The model's STRUCTURE.** A checkpoint holds parameter VALUES, and
 * [ModelSnapshot.restore] requires a model whose keys and dims already match —
 * refusing by name, naming the missing and the extra keys, when they do not.
 * Reconstructing a `Sequential(Dense(2,16), ReluLayer, …)` from a file means
 * serialising Kotlin class identities and constructor arguments, which is a
 * different problem (and a much worse one: it makes the file executable). The
 * recipe is the one PyTorch's `state_dict` uses — build the model, then load
 * into it — and it is stated in the refusal message.
 *
 * **Hyperparameters.** An optimizer's learning rate, betas and epsilon, a
 * BatchNorm's `momentum`, a Dropout's `p` and key: all of these are
 * constructor arguments of the value the caller holds, and a checkpoint that
 * overwrote them would change what the reloaded object does without the caller
 * asking. What IS carried is the optimizer's mutable STATE (moments, step
 * counts), which is what a resume needs and what nothing else can reproduce.
 *
 * **A non-`Stateful` container's `Stateful` children.** See [Stateful]'s
 * container contract. [Sequential] honours it; it is the only container in
 * this module that can hold a stateful child at all.
 */
package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.Shape
import io.tlaloc.core.io.JsonException
import io.tlaloc.core.io.LoadedTensor
import io.tlaloc.core.io.Safetensors
import io.tlaloc.core.io.SafetensorsTensor
import io.tlaloc.core.io.SafetensorsWriter

/**
 * An optimizer's state, flattened for a checkpoint: a [kind] tag, the scalar
 * part as strings, and the tensor part keyed by slot.
 *
 * The [kind] is what stops a resume from reading Adam's first moment into
 * RMSprop's mean-square. It is compared on load and a mismatch refuses by
 * name: both state shapes are "one tensor per parameter", so nothing else
 * would notice, and the resulting run would train with a silently wrong
 * preconditioner.
 */
class OptimizerCheckpoint(
    val kind: String,
    val scalars: Map<String, String> = emptyMap(),
    val tensors: Map<String, DTensor<*, F32>> = emptyMap(),
) {
    init {
        require(kind.isNotEmpty()) { "OptimizerCheckpoint: kind must not be empty" }
        require(scalars.keys.none { it.isEmpty() }) { "OptimizerCheckpoint: empty scalar key" }
        require(tensors.keys.none { it.isEmpty() }) { "OptimizerCheckpoint: empty tensor key" }
    }

    fun scalar(key: String): String = scalars[key] ?: throw JsonException(
        "optimizer checkpoint '$kind': no scalar '$key' (have: ${scalars.keys})",
    )

    fun int(key: String): Int = scalar(key).toIntOrNull() ?: throw JsonException(
        "optimizer checkpoint '$kind': scalar '$key' = '${scalar(key)}' is not an integer",
    )

    /**
     * The per-parameter tensor slots under [group], with the group prefix
     * stripped — the shape every optimizer's state has (`"m.0.w"`,
     * `"v.0.w"`, …). An empty result is the legitimate "no key has been
     * visited yet" state, not an error.
     */
    fun group(group: String): Map<String, DTensor<*, F32>> {
        val prefix = "$group."
        return tensors.entries
            .filter { it.key.startsWith(prefix) }
            .associate { it.key.removePrefix(prefix) to it.value }
    }

    /** [group]'s inverse: prefix a per-parameter map for storage. */
    companion object {
        fun grouped(group: String, slots: Map<String, DTensor<*, F32>>): Map<String, DTensor<*, F32>> =
            slots.entries.associate { "$group.${it.key}" to it.value }

        /** The one place a non-persistable optimizer is refused, so the text is one text. */
        fun refuse(optimizer: Optimizer<*>, what: String): Nothing = throw JsonException(
            "checkpoint: optimizer ${optimizer.describeForRefusal()} does not implement " +
                "CheckpointableOptimizer, so its $what cannot be written to or read from a " +
                "checkpoint. Every optimizer shipped in :nn does (FixedLearningRate, SGD, " +
                "RMSprop, Adam, Scheduled); a custom one must implement the three members " +
                "(checkpointKind, saveState, loadState) to be resumable. Saving the MODEL " +
                "alone is always available: ModelCheckpoint.encode(model).",
        )
    }
}

/**
 * The opt-in an optimizer implements to be resumable.
 *
 * A SEPARATE interface rather than three defaulted members on [Optimizer],
 * because a default that throws is a member every implementer inherits and
 * nobody is told about, and a default that guessed would write state it does
 * not understand. Everything shipped in `:nn` implements this; a custom
 * optimizer that does not is refused BY NAME at save time (see
 * [OptimizerCheckpoint.refuse]) rather than saved half-way.
 */
interface CheckpointableOptimizer<S> : Optimizer<S> {
    /**
     * The tag written into the checkpoint and compared on load. It names the
     * STATE SHAPE, not the hyperparameters: two `Adam`s with different
     * learning rates share a kind, and that is correct — the learning rate is
     * the caller's, the moments are the file's.
     */
    val checkpointKind: String

    fun saveState(state: S): OptimizerCheckpoint

    fun loadState(checkpoint: OptimizerCheckpoint): S
}

/** Used only inside refusal messages; `simpleName` is null for an anonymous object. */
internal fun Optimizer<*>.describeForRefusal(): String =
    this::class.simpleName ?: this.toString()

/**
 * What a checkpoint file holds, decoded but not yet bound to a model.
 *
 * The two-step shape ([ModelCheckpoint.decode] then [restore]) is deliberate:
 * a caller can read the metadata — how many steps the run had taken, which
 * optimizer wrote it — BEFORE deciding what model to build, and the alternative
 * one-step `load(path, model)` makes that impossible.
 */
class ModelSnapshot internal constructor(
    val parameters: Map<String, DTensor<*, F32>>,
    val buffers: Map<String, DTensor<*, F32>>,
    val optimizer: OptimizerCheckpoint?,
    /** The caller's own metadata, with Tlaloc's reserved keys removed. */
    val metadata: Map<String, String>,
    val formatVersion: String,
) {

    /** Total scalars across every parameter tensor — the number a log line wants. */
    val parameterScalars: Int get() = parameters.values.sumOf { it.size }

    /**
     * Rebuild [model] from this snapshot, returning a NEW instance.
     *
     * Requires an EXACT structural match: the same parameter keys and the same
     * dims per key. A missing key, an extra key or a changed shape refuses by
     * name and says which — the alternative is a model that loads and is
     * silently part-trained, which is the single worst outcome available here.
     */
    @Suppress("UNCHECKED_CAST")
    fun <M : Trainable<M>> restore(model: M): M {
        matchKeys("parameter", parameters, model.parameters)
        var out = model.withParameters(parameters)
        val stateful = out as? Stateful<*>
        if (stateful != null) {
            matchKeys("buffer", buffers, stateful.buffers)
            if (buffers.isNotEmpty()) {
                out = (out as Stateful<*>).withBuffers(buffers) as M
            }
        } else if (buffers.isNotEmpty()) {
            throw JsonException(
                "checkpoint: the file carries ${buffers.size} buffer(s) ${buffers.keys} but " +
                    "${model::class.simpleName} is not Stateful, so there is nowhere to put them — " +
                    "this checkpoint was written from a different model structure",
            )
        }
        return out
    }

    /**
     * The optimizer state this file carries, as [optimizer]'s own state type.
     * Refuses by name when the file has no optimizer state, when the optimizer
     * cannot be checkpointed, or when the kinds disagree.
     */
    fun <S> restoreOptimizerState(optimizer: Optimizer<S>): S {
        val cp = this.optimizer ?: throw JsonException(
            "checkpoint: this file carries model parameters only — no optimizer state was saved, " +
                "so training cannot resume from it. Save with " +
                "ModelCheckpoint.encode(model, optimizer, state) to make a resumable checkpoint; " +
                "or start from optimizer.initialState(), which restarts the moments.",
        )
        @Suppress("UNCHECKED_CAST")
        val ck = (optimizer as? CheckpointableOptimizer<S>)
            ?: OptimizerCheckpoint.refuse(optimizer, "state")
        if (ck.checkpointKind != cp.kind) {
            throw JsonException(
                "checkpoint: the file's optimizer state was written by '${cp.kind}' but is being " +
                    "loaded into '${ck.checkpointKind}'. Every optimizer here keeps one " +
                    "tensor per parameter, so nothing downstream would have noticed: the run " +
                    "would have trained with another optimizer's preconditioner.",
            )
        }
        return ck.loadState(cp)
    }

    private fun matchKeys(
        what: String,
        have: Map<String, DTensor<*, F32>>,
        want: List<NamedParameter>,
    ) {
        val wantKeys = want.map { it.key }
        val missing = wantKeys - have.keys
        val extra = have.keys - wantKeys.toSet()
        if (missing.isNotEmpty() || extra.isNotEmpty()) {
            throw JsonException(
                "checkpoint: $what keys do not match the model. Missing from the file: " +
                    "${missing.ifEmpty { listOf("(none)") }}; present in the file but not in the " +
                    "model: ${extra.ifEmpty { listOf("(none)") }}. A checkpoint carries VALUES, " +
                    "not structure — build the same model, then restore into it.",
            )
        }
        for (p in want) {
            val t = have.getValue(p.key)
            if (!t.dims.contentEquals(p.tensor.dims)) {
                throw JsonException(
                    "checkpoint: $what '${p.key}' is ${t.dims.toList()} in the file but " +
                        "${p.tensor.dims.toList()} in the model",
                )
            }
        }
    }
}

object ModelCheckpoint {

    const val FORMAT_VERSION: String = "1"

    const val PARAM_PREFIX: String = "param."
    const val BUFFER_PREFIX: String = "buffer."
    const val OPTIMIZER_PREFIX: String = "opt."

    const val VERSION_KEY: String = "tlaloc.checkpoint.version"
    const val OPTIMIZER_KIND_KEY: String = "tlaloc.optimizer.kind"
    const val OPTIMIZER_SCALAR_PREFIX: String = "tlaloc.optimizer."
    const val RESERVED_METADATA_PREFIX: String = "tlaloc."

    /** Encode [model]'s parameters and buffers. No optimizer state: not resumable. */
    fun encode(model: Trainable<*>, metadata: Map<String, String> = emptyMap()): ByteArray =
        build(model, null, metadata)

    /**
     * Encode [model] AND [optimizer]'s [state], producing a checkpoint a
     * training run can resume from without restarting its moments.
     */
    fun <S> encode(
        model: Trainable<*>,
        optimizer: Optimizer<S>,
        state: S,
        metadata: Map<String, String> = emptyMap(),
    ): ByteArray {
        @Suppress("UNCHECKED_CAST")
        val ck = (optimizer as? CheckpointableOptimizer<S>)
            ?: OptimizerCheckpoint.refuse(optimizer, "state")
        return build(model, ck.saveState(state), metadata)
    }

    private fun build(
        model: Trainable<*>,
        optimizer: OptimizerCheckpoint?,
        metadata: Map<String, String>,
    ): ByteArray {
        for (k in metadata.keys) {
            require(!k.startsWith(RESERVED_METADATA_PREFIX)) {
                "ModelCheckpoint: metadata key '$k' is reserved — keys beginning " +
                    "'$RESERVED_METADATA_PREFIX' belong to the checkpoint format itself"
            }
        }
        val tensors = ArrayList<SafetensorsTensor>()
        for (p in model.parameters) {
            tensors.add(SafetensorsTensor.of(PARAM_PREFIX + p.key, p.tensor))
        }
        if (model is Stateful<*>) {
            for (b in model.buffers) {
                tensors.add(SafetensorsTensor.of(BUFFER_PREFIX + b.key, b.tensor))
            }
        }
        val meta = LinkedHashMap<String, String>()
        meta[VERSION_KEY] = FORMAT_VERSION
        if (optimizer != null) {
            meta[OPTIMIZER_KIND_KEY] = optimizer.kind
            for ((k, v) in optimizer.scalars) meta[OPTIMIZER_SCALAR_PREFIX + k] = v
            for ((k, t) in optimizer.tensors) {
                tensors.add(SafetensorsTensor.of(OPTIMIZER_PREFIX + k, t))
            }
        }
        meta.putAll(metadata)
        return SafetensorsWriter.encode(tensors, meta)
    }

    /**
     * Decode a checkpoint. Every tensor name must carry one of the three known
     * prefixes — an unprefixed name means this is somebody else's safetensors
     * file (a HuggingFace checkpoint, say), and reading one as a Tlaloc
     * checkpoint would produce a model with no parameters restored and no
     * complaint.
     */
    fun decode(bytes: ByteArray): ModelSnapshot {
        val n = Safetensors.headerLength(bytes)
        val dataStart = (Safetensors.HEADER_LENGTH_PREFIX_BYTES + n).toInt()
        if (bytes.size < dataStart) {
            throw JsonException(
                "checkpoint: the file is ${bytes.size} bytes but its header alone claims $dataStart",
            )
        }
        val all = Safetensors.readAll(bytes)
        val header = Safetensors.parseHeader(
            bytes.decodeToString(Safetensors.HEADER_LENGTH_PREFIX_BYTES, dataStart),
            dataStart.toLong(),
            (bytes.size - dataStart).toLong(),
        )
        return fromParts(all, header.metadata)
    }

    /** [decode]'s core, for a caller that already has the tensors and metadata. */
    fun fromParts(
        tensors: Map<String, LoadedTensor>,
        metadata: Map<String, String>,
    ): ModelSnapshot {
        val version = metadata[VERSION_KEY] ?: throw JsonException(
            "checkpoint: no '$VERSION_KEY' in the file's metadata — this is a safetensors file " +
                "but not a Tlaloc checkpoint. A foreign checkpoint is read with " +
                "io.tlaloc.core.io.Safetensors / SafetensorsFile, which is what the HuggingFace " +
                "ingestion path uses.",
        )
        if (version != FORMAT_VERSION) {
            throw JsonException(
                "checkpoint: format version '$version' is not '$FORMAT_VERSION'; this build reads " +
                    "version $FORMAT_VERSION only. No migration exists because no other version " +
                    "has ever been written.",
            )
        }
        val params = LinkedHashMap<String, DTensor<*, F32>>()
        val buffers = LinkedHashMap<String, DTensor<*, F32>>()
        val optTensors = LinkedHashMap<String, DTensor<*, F32>>()
        for ((name, t) in tensors) {
            val target = when {
                name.startsWith(PARAM_PREFIX) -> params to name.removePrefix(PARAM_PREFIX)
                name.startsWith(BUFFER_PREFIX) -> buffers to name.removePrefix(BUFFER_PREFIX)
                name.startsWith(OPTIMIZER_PREFIX) -> optTensors to name.removePrefix(OPTIMIZER_PREFIX)
                else -> throw JsonException(
                    "checkpoint: tensor '$name' carries none of the prefixes a Tlaloc checkpoint " +
                        "uses ('$PARAM_PREFIX', '$BUFFER_PREFIX', '$OPTIMIZER_PREFIX') — the file " +
                        "is not a Tlaloc checkpoint, or was written by a newer format",
                )
            }
            if (t.dtype != F32) {
                throw JsonException(
                    "checkpoint: tensor '$name' is ${t.dtype.name}; a Tlaloc checkpoint stores " +
                        "f32 because :nn's parameters ARE f32 master weights (the mixed-precision " +
                        "convention, Training.kt: compute in bf16, store f32). A bf16 checkpoint " +
                        "would lose 16 mantissa bits of every optimizer moment.",
                )
            }
            require(target.second.isNotEmpty()) {
                "checkpoint: tensor '$name' has an empty key after its prefix"
            }
            target.first[target.second] = t.asF32<Shape>()
        }
        val kind = metadata[OPTIMIZER_KIND_KEY]
        if (kind == null && optTensors.isNotEmpty()) {
            throw JsonException(
                "checkpoint: the file holds ${optTensors.size} '$OPTIMIZER_PREFIX' tensor(s) but no " +
                    "'$OPTIMIZER_KIND_KEY' — nothing can say which optimizer's state they are, " +
                    "and every optimizer here keeps one tensor per parameter",
            )
        }
        val optimizer = kind?.let {
            OptimizerCheckpoint(
                kind = it,
                scalars = metadata.entries
                    .filter { e -> e.key.startsWith(OPTIMIZER_SCALAR_PREFIX) && e.key != OPTIMIZER_KIND_KEY }
                    .associate { e -> e.key.removePrefix(OPTIMIZER_SCALAR_PREFIX) to e.value },
                tensors = optTensors,
            )
        }
        val user = metadata.filterKeys { !it.startsWith(RESERVED_METADATA_PREFIX) }
        return ModelSnapshot(params, buffers, optimizer, user, version)
    }
}
