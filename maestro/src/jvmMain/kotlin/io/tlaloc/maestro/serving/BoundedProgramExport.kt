package io.tlaloc.maestro.serving

import io.tlaloc.autograd.AxisSpec
import io.tlaloc.autograd.BoundedInputRole
import io.tlaloc.autograd.BoundedProgram
import io.tlaloc.autograd.BucketLadders
import io.tlaloc.autograd.TensorSpec
import io.tlaloc.core.DimBound
import io.tlaloc.core.ExperimentalTlalocApi
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirModule
import io.tlaloc.maestro.ProgramManifest
import io.tlaloc.maestro.TypeDescriptor
import io.tlaloc.stablehlo.toStablehlo
import java.nio.file.Files
import java.nio.file.Path

/**
 * Writes a [BoundedProgram] as a bounded-program artifact: one StableHLO body per combination
 * of buckets, and a [BoundedManifest] (docs/design/bounded-dims.md).
 *
 * Before writing anything it compares, in the reference interpreter, the padded result with
 * the exact-size result at the edges of every bucket ([BoundedProgram.checkPadding]; or at
 * `paddingCheckSizes`), and
 * refuses the export when they differ by more than the tolerance: a program that reads a
 * padded position without masking it would otherwise serve wrong answers for every size that
 * is not a bucket size.
 */
@ExperimentalTlalocApi
object BoundedProgramExport {
    const val BODIES_DIR = "bodies"
    const val PROGRAMS_DIR = "programs"
    const val ENTRY_POINT = "main"

    fun export(
        program: BoundedProgram,
        dir: Path,
        ladders: BucketLadders = BucketLadders.powersOfTwo(program.bounds),
        tolerance: Float = 1e-5f,
        paddingCheckSizes: List<Map<DimBound, Int>> = program.edgeSizes(ladders),
    ): BoundedManifest {
        for (b in program.bounds) ladders.ladder(b)
        require(paddingCheckSizes.isNotEmpty()) { "BoundedProgramExport '${program.name}': no sizes to check padding at" }
        val report = program.checkPadding(ladders, paddingCheckSizes, tolerance = tolerance)
        require(report.passed) {
            "BoundedProgramExport '${program.name}': at sizes ${report.worstSizes?.describe()} the padded result " +
                "differs from the exact one by ${report.maxDifference} (relative to max(1, max |exact|)), over the " +
                "tolerance $tolerance. A value that depends on how many positions are real (a mean, a softmax, an " +
                "attention) must read ctx.validMask or ctx.validLength; export refused"
        }

        Files.createDirectories(dir.resolve(BODIES_DIR))
        Files.createDirectories(dir.resolve(PROGRAMS_DIR))

        val combos = program.bounds.fold(listOf(emptyMap<DimBound, Int>())) { acc, b ->
            acc.flatMap { m -> ladders.ladder(b).map { m + (b to it) } }
        }
        var parameters: List<Pair<BoundedInputRole, DimBound?>>? = null
        val entries = combos.map { sizes ->
            val trace = program.trace(sizes)
            if (parameters == null) parameters = trace.parameters
            require(trace.parameters == parameters) {
                "BoundedProgramExport '${program.name}': the program asks for different extra inputs at " +
                    "${sizes.describe()} (${trace.parameters}) than at the first bucket ($parameters); every bucket " +
                    "must have one signature"
            }
            val fn = trace.function.renamed(ENTRY_POINT)
            val bytes = DxirModule(listOf(fn)).toStablehlo().toByteArray(Charsets.UTF_8)
            val hash = sha256(bytes)
            val bodyPath = "$BODIES_DIR/$hash.mlir"
            if (!Files.exists(dir.resolve(bodyPath))) Files.write(dir.resolve(bodyPath), bytes)
            val id = program.bounds.joinToString("_") { "${it.boundName}${sizes.getValue(it)}" }
            val manifest = ProgramManifest(
                name = id,
                inputs = fn.params.map { TypeDescriptor.fromDxirType(it.type) },
                outputs = fn.returns.map { TypeDescriptor.fromDxirType(it.type) },
                meshRequirement = ServingArtifactWriter.SINGLE_DEVICE_MESH,
                bodyHash = hash,
            )
            val programPath = "$PROGRAMS_DIR/$id.json"
            Files.write(dir.resolve(programPath), manifest.toJson().toByteArray(Charsets.UTF_8))
            BoundedEntry(
                id = id,
                sizes = sizes.entries.associate { it.key.boundName to it.value },
                bodyPath = bodyPath,
                bodyHash = hash,
                programPath = programPath,
                entryPoint = ENTRY_POINT,
            )
        }

        var dataIndex = 0
        val inputs = parameters!!.map { (role, bound) ->
            when (role) {
                BoundedInputRole.DATA -> {
                    val k = dataIndex++
                    program.inputs[k].decl("x$k")
                }
                BoundedInputRole.VALID_MASK -> TensorDecl(
                    "validMask_${bound!!.boundName}", BoundedManifest.ROLE_VALID_MASK, "f32",
                    listOf(AxisDecl(bound = bound.boundName)), bound.boundName,
                )
                BoundedInputRole.VALID_LENGTH -> TensorDecl(
                    "validLength_${bound!!.boundName}", BoundedManifest.ROLE_VALID_LENGTH, "f32", emptyList(), bound.boundName,
                )
            }
        }
        val manifest = BoundedManifest(
            name = program.name,
            bounds = program.bounds.map { BoundDecl(it.boundName, it.max, ladders.ladder(it)) },
            inputs = inputs,
            outputs = program.outputs.mapIndexed { i, o -> o.decl("y$i") },
            entries = entries,
            paddingCheck = PaddingCheck(
                sizes = report.sizesChecked.map { m -> m.entries.associate { it.key.boundName to it.value } },
                maxDifference = report.maxDifference.toDouble(),
                tolerance = tolerance.toDouble(),
            ),
        )
        // What a loader reads is what was written.
        val json = manifest.toJson()
        check(BoundedManifest.fromJson(json) == manifest) { "BoundedProgramExport: the manifest does not read back equal" }
        Files.write(dir.resolve(BoundedManifest.FILE_NAME), json.toByteArray(Charsets.UTF_8))
        return manifest
    }

    /** Reads an artifact's manifest and checks every body against its hash. */
    fun load(dir: Path): BoundedManifest {
        val file = dir.resolve(BoundedManifest.FILE_NAME)
        require(Files.exists(file)) {
            if (Files.exists(dir.resolve(ServingManifest.FILE_NAME))) {
                "BoundedProgramExport.load: $dir holds ${ServingManifest.FILE_NAME}, a language-model serving " +
                    "artifact, not a bounded-program artifact"
            } else {
                "BoundedProgramExport.load: no ${BoundedManifest.FILE_NAME} in $dir"
            }
        }
        val manifest = BoundedManifest.fromJson(Files.readString(file))
        for (e in manifest.entries) {
            val bytes = Files.readAllBytes(dir.resolve(e.bodyPath))
            require(sha256(bytes) == e.bodyHash) { "BoundedProgramExport.load: ${e.bodyPath} does not match its hash" }
        }
        return manifest
    }

    private fun TensorSpec.decl(name: String) = TensorDecl(
        name = name,
        role = BoundedManifest.ROLE_DATA,
        dtype = if (dtype == I32) "i32" else "f32",
        axes = axes.map {
            when (it) {
                is AxisSpec.Fixed -> AxisDecl(size = it.size)
                is AxisSpec.Bounded -> AxisDecl(bound = it.bound.boundName)
            }
        },
    )

    private fun DxirFunction.renamed(name: String) = DxirFunction(name, params, body, returns, meshes)

    private fun sha256(bytes: ByteArray): String = io.tlaloc.maestro.sha256Hex(bytes)

    private fun Map<DimBound, Int>.describe(): String =
        entries.joinToString(", ", "{", "}") { "${it.key.boundName}=${it.value}" }
}
