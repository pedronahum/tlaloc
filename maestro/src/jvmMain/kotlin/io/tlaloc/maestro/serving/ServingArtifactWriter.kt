package io.tlaloc.maestro.serving

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirModule
import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodeGraphSpec
import io.tlaloc.ir.inference.DecodeModelShape
import io.tlaloc.ir.inference.DecodeSlot
import io.tlaloc.ir.inference.DecodeSlotRole
import io.tlaloc.maestro.ProgramManifest
import io.tlaloc.maestro.TypeDescriptor
import io.tlaloc.maestro.sha256Hex
import io.tlaloc.stablehlo.toStablehlo
import java.io.BufferedOutputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import java.security.DigestOutputStream
import java.security.MessageDigest

/**
 * **The exporter**. This is the JVM's last act.
 *
 * Everything upstream of here is Kotlin: the graph builder, the recogniser,
 * the coarsener, the StableHLO emitter, the [DecodeGraphSpec] shape contract. Everything
 * downstream is Python + PJRT. The line between them is a DIRECTORY, and
 * this object is what draws it — which is what makes "no JVM in
 * the serving path" a property of the system rather than an intention.
 *
 * ## What it writes
 *
 * For each (kind, batch, context) ladder point it is given: build the graph,
 * **check it against [DecodeGraphSpec.verifySignature]**, emit StableHLO+SDY,
 * content-address the bytes, write the body under its own hash, write that
 * entry's [ProgramManifest], and record a [ServingEntry]. Then one
 * [ServingManifest] over the family.
 *
 * ## Decisions
 *
 * - **`verifySignature` runs on every exported graph, at export time.** The
 *   artifact is consumed by a process that cannot call back; a signature
 *   mismatch discovered in Python is a shape error from inside XLA with the
 *   diagnosis removed. REJECTED: verifying only in tests — the exporter is
 *   the thing a deployment runs, and the check costs nothing.
 * - **Bodies are named by their SHA-256 and DE-DUPLICATED.** Two ladder
 *   points routinely lower to byte-identical programs (a context bucket the
 *   graph does not read, a batch axis that is only a leading dim in a model
 *   small enough to test with), and writing the same megabyte twice under
 *   two names makes the artifact lie about how many programs it has.
 *   REJECTED: naming bodies by entry id — readable, and it discards the one
 *   property that lets a loader verify a body it was handed.
 * - **The entry point is `main`.** XLA's `compile_and_load` wants the
 *   module's entry function, and jaxlib's own path is `@main`; the
 *   alternative is a regex rename, which is a loader
 *   editing a program's text before running it. The exporter names it
 *   correctly instead, and [ServingEntry.entryPoint] states it rather than
 *   leaving Python to assume.
 * - **One module per entry, not one module with N functions.** A serving
 *   process compiles buckets lazily and independently (warm-up policy is
 *   still open); a single module would make every compile pay
 *   for every bucket's text.
 * - **The ProgramManifest is written per entry, unmodified.** Its
 *   `inputs`/`outputs`/`bodyHash` are cross-checked against the
 *   serving-side slots here, so the two descriptions of one program cannot
 *   drift — see [ServingManifest]'s class doc for why the composition runs
 *   this way and not by extending [ProgramManifest] with serving fields.
 */
object ServingArtifactWriter {

    const val BODIES_DIR: String = "bodies"
    const val PROGRAMS_DIR: String = "programs"

    /** Where staged weight operands land. */
    const val WEIGHTS_DIR: String = ServingWeightsPointer.STAGED_DIR

    /**
     * A serving artifact is single-device. The field is
     * [ProgramManifest]'s mesh requirement and it is not optional there, so
     * it is stated explicitly rather than left as a model-class name that
     * would imply a mesh nobody declared. Multi-device serving (tensor
     * parallelism across a mesh, with the SDY sharding the emitter already
     * knows how to write) is not supported.
     */
    const val SINGLE_DEVICE_MESH: String = "single-device"

    /**
     * Export [specs] into [dir], which is created if absent and must
     * otherwise be an empty directory or a previous artifact.
     *
     * @param build produces the graph for one spec. The exporter does not
     *   know how to build a model; it knows how to check one and write it.
     * @param stageWeight produces the host bytes of ONE staged weight slot,
     *   in math layout and the slot's dtype. Required exactly when the specs
     *   declare `weightSlots`. It is a per-slot callback rather than a
     *   `List<FloatArray>` argument on purpose: TinyLlama-1.1B staged as f32
     *   is 4.4 GB, and a list would hold every tensor resident while the
     *   writer is only ever looking at one. Peak heap is the largest single
     *   tensor plus its transpose, not the model.
     * @param writeWeight the streaming form of [stageWeight]: writes ONE slot's
     *   bytes (math layout, little-endian, the slot's dtype, F32 or BF16) to
     *   the stream and returns how many it wrote. For weights too large to
     *   hold as one array. Give at most one of the two.
     * @param refusedTokenIds token ids a server must refuse, with the config
     *   key naming each ([ServingModelShape.refusedTokens]).
     */
    fun export(
        dir: Path,
        modelName: String,
        modelHash: String,
        model: DecodeModelShape,
        ladder: ServingBucketLadder,
        specs: List<DecodeGraphSpec>,
        weights: ServingWeightsPointer = ServingWeightsPointer.embedded(),
        stageWeight: ((DecodeSlot) -> FloatArray)? = null,
        writeWeight: ((DecodeSlot, OutputStream) -> Long)? = null,
        build: (DecodeGraphSpec) -> DxirFunction,
        refusedTokenIds: Map<Int, String> = emptyMap(),
    ): ServingManifest {
        require(specs.isNotEmpty()) {
            "ServingArtifactWriter.export: no specs — an artifact that compiles nothing is a " +
                "directory a loader can only fail on"
        }
        for (s in specs) {
            require(s.model == model) {
                "ServingArtifactWriter.export: spec $s describes model ${s.model} but the " +
                    "artifact declares $model — one artifact is one model"
            }
        }
        // §0.4.480. One artifact is one model, so one weight signature: a
        // ladder whose points disagree about their staged operands would make
        // "the weight table" ambiguous, and the loader binds ONE table across
        // every entry it may select.
        val weightSlots = specs.first().weightSlots
        for (s in specs) {
            require(s.weightSlots == weightSlots) {
                "ServingArtifactWriter.export: ladder point ${s.bucket} declares " +
                    "${s.weightSlots.size} staged weight slots but ${specs.first().bucket} " +
                    "declares ${weightSlots.size} — the weight table is written ONCE for the " +
                    "artifact and bound by every entry, so the signatures must be identical"
            }
        }
        require(stageWeight == null || writeWeight == null) {
            "ServingArtifactWriter.export: give stageWeight or writeWeight, not both"
        }
        val writer: ((DecodeSlot, OutputStream) -> Long)? = writeWeight ?: stageWeight?.let { stage ->
            { slot, out -> writeFloats(stage(slot), out) }
        }
        require(weightSlots.isEmpty() == (writer == null)) {
            if (weightSlots.isEmpty()) {
                "ServingArtifactWriter.export: a stageWeight callback was supplied for specs " +
                    "that declare no weight slots — the bytes would be written into the " +
                    "artifact and bound by nothing"
            } else {
                "ServingArtifactWriter.export: these specs declare ${weightSlots.size} staged " +
                    "weight slots (${weightSlots.take(3).joinToString { it.name }}…) and no " +
                    "stageWeight callback was given. A half-artifact — a manifest promising " +
                    "operands whose files are absent — is refused rather than written"
            }
        }
        Files.createDirectories(dir.resolve(BODIES_DIR))
        Files.createDirectories(dir.resolve(PROGRAMS_DIR))
        val weightsPointer =
            if (writer == null) weights else stageWeights(dir, weightSlots, writer)

        val entries = specs.map { spec ->
            val fn = build(spec)
            spec.verifySignature(fn, "ServingArtifactWriter.export")
            require(fn.name == ENTRY_POINT) {
                "ServingArtifactWriter.export: exported graph is named '${fn.name}' but the " +
                    "artifact's entry point is '$ENTRY_POINT' — XLA compiles the module's entry " +
                    "function, and a loader that renames a function before running it is editing " +
                    "a program it was asked to execute"
            }

            // Each KV_POOL_OUT is written over the KV_POOL_IN it pairs with,
            // so a runtime that donates the pools updates them in place.
            val text = DxirModule(listOf(fn)).toStablehlo(
                outputAliases = mapOf(fn.name to spec.donationPairs.toMap()),
            )
            val bytes = text.toByteArray(Charsets.UTF_8)
            val hash = sha256Hex(bytes)
            val bodyPath = "$BODIES_DIR/$hash.mlir"
            val bodyFile = dir.resolve(bodyPath)
            // De-duplication: identical bytes, identical name, written once.
            if (!Files.exists(bodyFile)) Files.write(bodyFile, bytes)

            val entryId = "${spec.kind.name.lowercase()}_b${spec.bucket.batch}_c${spec.bucket.maxContext}"
            val program = ProgramManifest(
                name = entryId,
                inputs = spec.inputs.map { TypeDescriptor.fromDxirType(it.type) },
                outputs = spec.outputs.map { TypeDescriptor.fromDxirType(it.type) },
                meshRequirement = SINGLE_DEVICE_MESH,
                bodyHash = hash,
            )
            val programPath = "$PROGRAMS_DIR/$entryId.json"
            Files.write(dir.resolve(programPath), program.toJson().toByteArray(Charsets.UTF_8))

            val inputs = spec.inputs.map { ServingSlot.fromSlot(it) }
            val outputs = spec.outputs.map { ServingSlot.fromSlot(it) }
            // The cross-check that keeps the two descriptions of one program
            // from drifting. Cheap, and it is the whole reason the serving
            // manifest is allowed to restate the boundary types at all.
            require(program.inputs == inputs.map { it.type } && program.outputs == outputs.map { it.type }) {
                "ServingArtifactWriter.export: entry $entryId's ProgramManifest boundary types " +
                    "disagree with the serving slots derived from the same spec"
            }

            ServingEntry(
                kind = spec.kind,
                batch = spec.bucket.batch,
                context = spec.bucket.maxContext,
                tokensPerSeq = spec.tokensPerSeq,
                maxBlocksPerSeq = spec.maxBlocksPerSeq,
                cacheKey = spec.executableCacheKey(modelHash),
                entryPoint = ENTRY_POINT,
                bodyPath = bodyPath,
                bodyHash = hash,
                programPath = programPath,
                inputs = inputs,
                outputs = outputs,
                donationPairs = spec.donationPairs.map { listOf(it.first, it.second) },
            )
        }

        val manifest = ServingManifest(
            modelName = modelName,
            modelHash = modelHash,
            model = ServingModelShape(
                vocabSize = model.vocabSize, hiddenSize = model.hiddenSize,
                numHeads = model.numHeads, numKvHeads = model.numKvHeads,
                headDim = model.headDim, numLayers = model.numLayers,
                numBlocks = model.numBlocks, blockSize = model.blockSize,
                dtype = model.dtype.name, kvDtype = model.kvDtype.name,
                // §0.4.472 — Phase H5: the reserved slot, fed from the spec's
                // own config so the artifact cannot claim a format the graphs
                // were not built for.
                kvQuant = model.kvQuant?.let {
                    ServingKvQuant(
                        dtype = it.dtype.nameTag,
                        scaleStrategy = it.scaleStrategy.name,
                        codeMax = it.dtype.codeMax,
                        codeDtype = model.kvDtype.name,
                    )
                },
                windowedKv = model.windowedKv?.let {
                    ServingWindowedKv(it.window, it.layers, it.numBlocks, it.ringPages)
                },
                refusedTokens = refusedTokenIds.entries.sortedBy { it.key }
                    .map { ServingRefusedToken(it.key, it.value) },
            ),
            bucketLadder = ladder,
            weights = weightsPointer,
            entries = entries,
        )
        Files.write(
            dir.resolve(ServingManifest.FILE_NAME),
            manifest.toJson().toByteArray(Charsets.UTF_8),
        )
        return manifest
    }

    /**
     * Write one raw file per staged weight slot and describe them.
     *
     * **Little-endian, dense row-major, no header.** The file is the operand,
     * byte for byte, in the form PJRT's `BufferFromHostBuffer` takes — so the
     * loader's whole job is `open`, `readinto`, upload, and it never has to
     * interpret a format. The dims, the dtype and the length are in the
     * manifest, which is where the artifact's shape vocabulary already lives;
     * duplicating them into a per-file header would create a second place for
     * them to be wrong.
     *
     * Little-endian is stated rather than "native": the artifact is a
     * deployment format and aarch64 being LE is a fact about this machine.
     *
     * Names are `weights/NNNN_<slot>.bin`, zero-padded to the slot INDEX, so
     * `ls` sorts into call order and two exports of one model are
     * byte-identical directories, weights included.
     */
    private fun stageWeights(
        dir: Path,
        slots: List<DecodeSlot>,
        write: (DecodeSlot, OutputStream) -> Long,
    ): ServingWeightsPointer {
        Files.createDirectories(dir.resolve(WEIGHTS_DIR))
        val table = slots.mapIndexed { i, slot ->
            require(slot.role == DecodeSlotRole.WEIGHT) {
                "ServingArtifactWriter: slot '${slot.name}' in the weight signature has role " +
                    "${slot.role}, not WEIGHT"
            }
            val width = when (slot.type.dtype) {
                F32 -> 4L
                BF16 -> 2L
                else -> throw IllegalArgumentException(
                    "ServingArtifactWriter: staged weight '${slot.name}' is ${slot.type.dtype}; " +
                        "the serving artifact holds F32 or BF16 weights",
                )
            }
            val want = slot.type.dims.fold(1L) { a, b -> a * b } * width
            val name = "$WEIGHTS_DIR/${i.toString().padStart(4, '0')}_${slot.name}.bin"
            val digest = MessageDigest.getInstance("SHA-256")
            val written = DigestOutputStream(
                BufferedOutputStream(Files.newOutputStream(dir.resolve(name)), 1 shl 20), digest,
            ).use { out -> write(slot, out) }
            val onDisk = Files.size(dir.resolve(name))
            require(written == want && onDisk == want) {
                "ServingArtifactWriter: staged weight '${slot.name}' wrote $written bytes " +
                    "($onDisk on disk) but the slot declares ${slot.type.dims} x ${slot.type.dtype} " +
                    "= $want"
            }
            ServingWeightFile(
                name = slot.name,
                path = name,
                dtype = slot.type.dtype.name,
                dims = slot.type.dims,
                byteLength = want,
                sha256 = digest.digest().joinToString("") { b ->
                    ((b.toInt() and 0xFF) + 0x100).toString(16).substring(1)
                },
            )
        }
        return ServingWeightsPointer(
            format = ServingWeightsPointer.STAGED_FORMAT,
            path = WEIGHTS_DIR,
            embedded = false,
            table = table,
        )
    }

    /** Little-endian f32 bytes of [data], written in pieces; returns the byte count. */
    private fun writeFloats(data: FloatArray, out: OutputStream): Long {
        val piece = 65536
        val bb = ByteBuffer.allocate(4 * piece).order(ByteOrder.LITTLE_ENDIAN)
        var i = 0
        while (i < data.size) {
            val n = minOf(piece, data.size - i)
            bb.clear()
            for (k in 0 until n) bb.putFloat(data[i + k])
            out.write(bb.array(), 0, 4 * n)
            i += n
        }
        return 4L * data.size
    }

    /** The MLIR symbol every exported entry is called through. */
    const val ENTRY_POINT: String = "main"

    /** Project a [DecodeBucketPolicy] onto the wire form the artifact carries. */
    fun ladderOf(policy: DecodeBucketPolicy): ServingBucketLadder = ServingBucketLadder(
        blockSize = policy.blockSize,
        batch = policy.batchLadder,
        context = policy.contextLadder,
    )

    /** Every decode ladder point of [policy] as a spec over [model]. */
    fun decodeSpecs(model: DecodeModelShape, policy: DecodeBucketPolicy): List<DecodeGraphSpec> =
        policy.allBuckets.map { b: DecodeBucket -> DecodeGraphSpec(model, b, DecodeGraphKind.DECODE) }
}
