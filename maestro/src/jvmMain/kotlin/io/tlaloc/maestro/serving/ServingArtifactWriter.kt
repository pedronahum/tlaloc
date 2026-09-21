package io.tlaloc.maestro.serving

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirModule
import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodeGraphSpec
import io.tlaloc.ir.inference.DecodeModelShape
import io.tlaloc.maestro.ProgramManifest
import io.tlaloc.maestro.TypeDescriptor
import io.tlaloc.maestro.sha256Hex
import io.tlaloc.stablehlo.toStablehlo
import java.nio.file.Files
import java.nio.file.Path

/**
 * §0.4.469 — Phase H3a: **the exporter**. This is the JVM's last act.
 *
 * Everything upstream of here is Kotlin: the graph builder, the recogniser,
 * the coarsener, the StableHLO emitter, H1c's shape contract. Everything
 * downstream is Python + PJRT. The line between them is a DIRECTORY, and
 * this object is what draws it — which is what makes the audit's "no JVM in
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
 *   alternative is the §0.4.325 spike's regex rename, which is a loader
 *   editing a program's text before running it. The exporter names it
 *   correctly instead, and [ServingEntry.entryPoint] states it rather than
 *   leaving Python to assume.
 * - **One module per entry, not one module with N functions.** A serving
 *   process compiles buckets lazily and independently (warm-up policy is
 *   still open, H1c named it); a single module would make every compile pay
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

    /**
     * A serving artifact is single-device in H3a. The field is
     * [ProgramManifest]'s mesh requirement and it is not optional there, so
     * it is stated honestly rather than left as a model-class name that
     * would imply a mesh nobody declared. Multi-device serving (tensor
     * parallelism across a mesh, with the SDY sharding the emitter already
     * knows how to write) is a NAMED DEFERRAL of this slice.
     */
    const val SINGLE_DEVICE_MESH: String = "single-device"

    /**
     * Export [specs] into [dir], which is created if absent and must
     * otherwise be an empty directory or a previous artifact.
     *
     * @param build produces the graph for one spec. The exporter does not
     *   know how to build a model; it knows how to check one and write it.
     */
    fun export(
        dir: Path,
        modelName: String,
        modelHash: String,
        model: DecodeModelShape,
        ladder: ServingBucketLadder,
        specs: List<DecodeGraphSpec>,
        weights: ServingWeightsPointer = ServingWeightsPointer.embedded(),
        build: (DecodeGraphSpec) -> DxirFunction,
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
        Files.createDirectories(dir.resolve(BODIES_DIR))
        Files.createDirectories(dir.resolve(PROGRAMS_DIR))

        val entries = specs.map { spec ->
            val fn = build(spec)
            spec.verifySignature(fn, "ServingArtifactWriter.export")
            require(fn.name == ENTRY_POINT) {
                "ServingArtifactWriter.export: exported graph is named '${fn.name}' but the " +
                    "artifact's entry point is '$ENTRY_POINT' — XLA compiles the module's entry " +
                    "function, and a loader that renames a function before running it is editing " +
                    "a program it was asked to execute"
            }

            val text = DxirModule(listOf(fn)).toStablehlo()
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
            ),
            bucketLadder = ladder,
            weights = weights,
            entries = entries,
        )
        Files.write(
            dir.resolve(ServingManifest.FILE_NAME),
            manifest.toJson().toByteArray(Charsets.UTF_8),
        )
        return manifest
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
