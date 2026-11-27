package io.tlaloc.runtime.pjrt.serving

import io.tlaloc.autograd.captureMulti
import io.tlaloc.core.BF16
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostBf16Storage
import io.tlaloc.core.HostBytesStorage
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.Shape
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodeSlotRole
import io.tlaloc.maestro.ProgramManifest
import io.tlaloc.maestro.TypeDescriptor
import io.tlaloc.maestro.serving.ServingArtifactWriter
import io.tlaloc.maestro.serving.ServingBucketLadder
import io.tlaloc.maestro.serving.ServingEntry
import io.tlaloc.maestro.serving.ServingLinearState
import io.tlaloc.maestro.serving.ServingManifest
import io.tlaloc.maestro.serving.ServingModelShape
import io.tlaloc.maestro.serving.ServingSlot
import io.tlaloc.maestro.serving.ServingWeightFile
import io.tlaloc.maestro.serving.ServingWeightsPointer
import io.tlaloc.nn.InklingCheckpoint
import io.tlaloc.nn.InklingLoop
import io.tlaloc.stablehlo.toStablehlo
import java.io.BufferedOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import java.security.DigestOutputStream
import java.security.MessageDigest

/**
 * Inkling 2 as a serving artifact the Triton backend's sequence mode serves (`tlaloc-serving-v4`):
 * every layer application's K/V caches and convolution histories are state pools of [numSlots]
 * slots ([InklingLoop.poolDims] at the largest context), listed in `model.linearState.poolDims`,
 * and each entry is [InklingLoop.batchChunk] for a batch and a context, the context being the span
 * of the caches its rows attend. There is no KV pool: the backend still allocates pages for its
 * slot mapping, which the program reads only for its sign (padding tokens are negative).
 *
 * The experts are kept quantized ([InklingCheckpoint.forEachWeight] with quantized experts) and the
 * f32 dots run at HIGHEST ([InklingModel.highestF32Dots]). The logits are f32 `[B, 1, V]` at each
 * row's last token.
 */
object InklingServingExport {

    /** Writes the artifact into [dir] (created) from the checkpoint [ck]; returns its manifest. */
    fun export(
        dir: Path,
        ck: InklingCheckpoint,
        modelName: String = "inkling-2-mini",
        contexts: List<Int> = listOf(2048, 8192),
        decodeBatches: List<Int> = listOf(1, 2, 4),
        prefillChunk: Int = 256,
        prefillBatches: List<Int> = listOf(1),
        numSlots: Int = 4,
        blockSize: Int = 16,
        routed: Boolean = true,
        bf16: Boolean = true,
        log: (String) -> Unit = {},
    ): ServingManifest {
        val c = ck.config
        val maxCtx = contexts.max()
        require(contexts == contexts.sorted().distinct() && contexts.all { it % blockSize == 0 }) { "InklingServingExport: contexts $contexts" }
        require(prefillChunk in 2..contexts.min()) { "InklingServingExport: prefill chunk $prefillChunk" }
        Files.createDirectories(dir.resolve(ServingArtifactWriter.BODIES_DIR))
        Files.createDirectories(dir.resolve(ServingArtifactWriter.PROGRAMS_DIR))
        Files.createDirectories(dir.resolve(ServingArtifactWriter.WEIGHTS_DIR))

        // The weights: written as they are read, only their specs kept.
        val names = ArrayList<String>()
        val specs = ArrayList<DTensor<*, *>>()
        val table = ArrayList<ServingWeightFile>()
        ck.forEachWeight(bf16 = bf16, log = log, quantizedExperts = true) { k, t ->
            val i = names.size
            val slot = "w$i"
            val path = "${ServingArtifactWriter.WEIGHTS_DIR}/${i.toString().padStart(4, '0')}_$slot.bin"
            val digest = MessageDigest.getInstance("SHA-256")
            val bytes = DigestOutputStream(BufferedOutputStream(Files.newOutputStream(dir.resolve(path)), 1 shl 20), digest).use { out ->
                when (val s = t.storage) {
                    is HostBytesStorage -> { out.write(s.data); s.data.size.toLong() }
                    is HostBf16Storage -> {
                        val bb = ByteBuffer.allocate(s.data.size * 2).order(ByteOrder.LITTLE_ENDIAN)
                        bb.asShortBuffer().put(s.data)
                        out.write(bb.array()); bb.capacity().toLong()
                    }
                    is HostF32Storage -> {
                        val bb = ByteBuffer.allocate(s.data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                        bb.asFloatBuffer().put(s.data)
                        out.write(bb.array()); bb.capacity().toLong()
                    }
                    else -> error("InklingServingExport: weight $k has ${s::class.simpleName} storage")
                }
            }
            table += ServingWeightFile(
                name = slot, path = path, dtype = t.dtype.name, dims = t.dims.toList(), byteLength = bytes,
                sha256 = digest.digest().joinToString("") { b -> ((b.toInt() and 0xFF) + 0x100).toString(16).substring(1) },
            )
            names += k
            specs += when (val s = t.storage) {
                is HostBytesStorage -> DTensor<Shape, io.tlaloc.core.DType>(HostBytesStorage(ByteArray(0)), t.dims, t.dtype)
                is HostBf16Storage -> DTensor<Shape, io.tlaloc.core.DType>(HostBf16Storage(ShortArray(0)), t.dims, BF16)
                else -> DTensor<Shape, io.tlaloc.core.DType>(HostF32Storage(FloatArray(0)), t.dims, F32).also { check(s is HostF32Storage) }
            }
        }
        log("inkling serving: ${table.size} weights staged, ${table.sumOf { it.byteLength } shr 20} MiB")

        val pools = InklingLoop.poolDims(c, maxCtx, numSlots)
        val apps = InklingLoop.applications(c)
        fun i32(dims: IntArray) = DTensor<Shape, I32>(HostI32Storage(IntArray(dims.fold(1) { a, b -> a * b })), dims, I32)
        val v = c.unpaddedVocabSize

        fun entry(kind: DecodeGraphKind, b: Int, ctx: Int, t: Int): ServingEntry {
            val maxBlocks = ctx / blockSize
            val requestDims = listOf(intArrayOf(b, t), intArrayOf(b, t), intArrayOf(b, maxBlocks), intArrayOf(b), intArrayOf(b * t), intArrayOf(b))
            val requestNames = listOf("tokenIds", "positions", "blockTables", "seqLens", "slotMapping", "stateSlots")
            val requestRoles = listOf(
                DecodeSlotRole.TOKEN_IDS, DecodeSlotRole.POSITIONS, DecodeSlotRole.BLOCK_TABLES, DecodeSlotRole.SEQ_LENS,
                DecodeSlotRole.SLOT_MAPPING, DecodeSlotRole.STATE_SLOTS,
            )
            val ins = requestDims.map { i32(it) } + pools.map { DTensor<Shape, F32>(HostF32Storage(FloatArray(0)), it, F32) } + specs
            val nReq = requestDims.size
            val fn = captureMulti(ins, ServingArtifactWriter.ENTRY_POINT, symbolic = true) { l ->
                val w = InklingCheckpoint.Leaves(names.withIndex().associate { (i, k) -> k to l[nReq + pools.size + i] }, if (bf16) BF16 else F32)
                InklingLoop.batchChunk(c, maxCtx, l[0], l[1], l[4], l[5], l.subList(nReq, nReq + pools.size), w, routed, span = ctx)
            }
            val donation = pools.indices.map { j -> (nReq + j) to (1 + j) }
            val text = InklingModel.highestF32Dots(fn.toStablehlo(outputAliases = donation.toMap()))
            val bytes = text.toByteArray(Charsets.UTF_8)
            val hash = sha256(bytes)
            val bodyPath = "${ServingArtifactWriter.BODIES_DIR}/$hash.mlir"
            if (!Files.exists(dir.resolve(bodyPath))) Files.write(dir.resolve(bodyPath), bytes)

            val poolName = { j: Int -> "pool${j / InklingLoop.STATE_PER_APPLICATION}_${POOL_PARTS[j % InklingLoop.STATE_PER_APPLICATION]}" }
            val inputs = requestNames.indices.map { ServingSlot(requestNames[it], requestRoles[it], TypeDescriptor(I32.name, requestDims[it].toList())) } +
                pools.indices.map { ServingSlot(poolName(it), DecodeSlotRole.STATE_POOL_IN, TypeDescriptor(F32.name, pools[it].toList())) } +
                table.map { ServingSlot(it.name, DecodeSlotRole.WEIGHT, TypeDescriptor(it.dtype, it.dims)) }
            val outputs = listOf(ServingSlot("logits", DecodeSlotRole.LOGITS, TypeDescriptor(F32.name, listOf(b, 1, v)))) +
                pools.indices.map { ServingSlot(poolName(it) + "Out", DecodeSlotRole.STATE_POOL_OUT, TypeDescriptor(F32.name, pools[it].toList())) }
            require(fn.params.map { TypeDescriptor.fromDxirType(it.type) } == inputs.map { it.type }) {
                "InklingServingExport: the program's parameters do not match its slots"
            }
            val entryId = "${kind.name.lowercase()}_b${b}_c$ctx" + (if (kind == DecodeGraphKind.PREFILL) "_t$t" else "")
            val programPath = "${ServingArtifactWriter.PROGRAMS_DIR}/$entryId.json"
            val program = ProgramManifest(entryId, inputs.map { it.type }, outputs.map { it.type }, ServingArtifactWriter.SINGLE_DEVICE_MESH, hash)
            Files.write(dir.resolve(programPath), program.toJson().toByteArray(Charsets.UTF_8))
            log("inkling serving: $entryId (${bytes.size shr 10} KiB)")
            return ServingEntry(
                kind = kind, batch = b, context = ctx, tokensPerSeq = t, maxBlocksPerSeq = maxBlocks,
                cacheKey = "inkling2:$hash", entryPoint = ServingArtifactWriter.ENTRY_POINT,
                bodyPath = bodyPath, bodyHash = hash, programPath = programPath,
                inputs = inputs, outputs = outputs, donationPairs = donation.map { listOf(it.first, it.second) },
            )
        }

        val entries = contexts.flatMap { ctx ->
            decodeBatches.map { entry(DecodeGraphKind.DECODE, it, ctx, 1) } +
                prefillBatches.map { entry(DecodeGraphKind.PREFILL, it, ctx, prefillChunk) }
        }
        val weightsHash = sha256(table.joinToString("\n") { it.sha256 }.toByteArray(Charsets.UTF_8)).take(16)
        val manifest = ServingManifest(
            modelName = modelName,
            modelHash = "inkling2:$modelName:weights-$weightsHash",
            model = ServingModelShape(
                vocabSize = v, hiddenSize = c.hidden, numHeads = c.heads, numKvHeads = c.kvHeads, headDim = c.headDim,
                numLayers = c.layers,
                // Pages for the backend's slot mapping (no pool behind them): every slot's whole context, plus the padding page.
                numBlocks = numSlots * maxCtx / blockSize + 1, blockSize = blockSize,
                dtype = F32.name, kvDtype = F32.name,
                linearState = ServingLinearState(
                    layers = (0 until pools.size / 2).toList(), numSlots = numSlots,
                    convStateDims = listOf(numSlots, c.convKernel - 1, c.hidden),
                    recurrentStateDims = listOf(numSlots, 1, 1, 1),
                    poolDims = pools.map { it.toList() },
                ),
            ),
            bucketLadder = ServingBucketLadder(blockSize, decodeBatches.sorted(), contexts),
            weights = ServingWeightsPointer(ServingWeightsPointer.STAGED_FORMAT, ServingWeightsPointer.STAGED_DIR, false, table),
            entries = entries,
        )
        check(apps.size * InklingLoop.STATE_PER_APPLICATION == pools.size)
        Files.write(dir.resolve(ServingManifest.FILE_NAME), manifest.toJson().toByteArray(Charsets.UTF_8))
        return manifest
    }

    private val POOL_PARTS = listOf("k", "v", "kHist", "vHist", "attnHist", "mlpHist")

    private fun sha256(b: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(b).joinToString("") { x -> ((x.toInt() and 0xFF) + 0x100).toString(16).substring(1) }
}
