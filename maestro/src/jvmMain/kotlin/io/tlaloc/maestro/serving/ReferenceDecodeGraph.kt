package io.tlaloc.maestro.serving

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.DecodeGraphSpec
import io.tlaloc.ir.inference.DecodeModelShape
import kotlin.math.sqrt

/**
 * §0.4.469 — Phase H3a: the reference decode graph the exporter exports and
 * the certification runs. **Main source, not test source**, and that is the
 * point: the artifact a `main` writes and the artifact the cert checks are
 * produced by one function, so "the export path works" and "the exported
 * thing is right" are claims about the same code.
 *
 * It is H1c's end-to-end mini step, unchanged in structure:
 *
 * ```
 *   embed → q/k/v projections → KV_CACHE_WRITE ×2 → PAGED_ATTENTION → lm head
 * ```
 *
 * one layer, weights as in-body constants. That is deliberately **not** a
 * Llama: H3a is about the SEAM — export, load, compile, run, agree — and a
 * seam is certified by a graph small enough that a disagreement is
 * attributable. Exporting a real Llama decode step needs H2's HF
 * name-mapping and the un-embedded weights this slice defers; it is named
 * as the next slice (H3b) rather than half-done here.
 *
 * RoPE is still absent for the same reason H1c gave: `positions` is an
 * operand of the contract and is not consumed by this graph, because the
 * rotary tables are model-layer work.
 *
 * Weights come from a fixed LCG so two exports of the same model are
 * byte-identical artifacts — which is what makes the body hash a content
 * address rather than a timestamp.
 */
object ReferenceDecodeGraph {

    /** The small model the exporter demo and the cert share. */
    val MODEL: DecodeModelShape = DecodeModelShape(
        vocabSize = 11,
        hiddenSize = 8,
        numHeads = 4,
        numKvHeads = 2,
        headDim = 2,
        numLayers = 1,
        numBlocks = 6,
        blockSize = 2,
        dtype = F32,
    )

    /**
     * Batches {1,2,4} × contexts {2,4} at blockSize 2 — six compiled
     * entries. `maxBatch = 4` is chosen so a batch of THREE has to be
     * padded: a ladder every request lands on exactly would certify the
     * export path without ever exercising the padding convention the
     * whole bucketing story rests on.
     */
    val POLICY: DecodeBucketPolicy =
        DecodeBucketPolicy(maxBatch = 4, maxContext = 4, blockSize = 2, minContext = 2)

    const val MODEL_NAME: String = "tlaloc-reference-decode"

    /** Content address of the weights+architecture. Fixed, because the
     *  weights are fixed; a real model hashes its checkpoint here. */
    const val MODEL_HASH: String = "reference-decode-lcg-v1"

    private fun weights(seed: Int, n: Int): FloatArray {
        var s = seed.toLong() and 0xFFFFFFFFL
        return FloatArray(n) {
            s = (s * 1664525L + 1013904223L) and 0xFFFFFFFFL
            ((s ushr 8).toDouble() / (1 shl 24).toDouble()).toFloat() - 0.5f
        }
    }

    private val m = MODEL
    private val qWidth = m.numHeads * m.headDim
    private val kvWidth = m.numKvHeads * m.headDim

    private val embedTable = weights(1, m.vocabSize * m.hiddenSize)
    private val qProj = weights(2, m.hiddenSize * qWidth)
    private val kProj = weights(3, m.hiddenSize * kvWidth)
    private val vProj = weights(4, m.hiddenSize * kvWidth)
    private val lmHead = weights(5, qWidth * m.vocabSize)
    private val scale = 1.0f / sqrt(m.headDim.toFloat())

    /**
     * Build the decode graph for [spec]. Named [ServingArtifactWriter.ENTRY_POINT]
     * because that is the symbol XLA compiles and the loader invokes.
     */
    fun build(spec: DecodeGraphSpec): DxirFunction {
        require(spec.model == m) {
            "ReferenceDecodeGraph.build: spec is over ${spec.model}, not the reference $m"
        }
        val b = spec.bucket.batch
        val d = m.hiddenSize
        val hd = m.headDim

        return DxirBuilder.function(ServingArtifactWriter.ENTRY_POINT) {
            val tokenIds = param("tokenIds", spec.tokenIdsType)
            @Suppress("UNUSED_VARIABLE")
            val positions = param("positions", spec.positionsType)
            val blockTables = param("blockTables", spec.blockTablesType)
            val seqLens = param("seqLens", spec.seqLensType)
            val slotMapping = param("slotMapping", spec.slotMappingType)
            val keyCache = param("keyCache0", spec.poolType)
            val valueCache = param("valueCache0", spec.poolType)

            val tbl = const(embedTable, DxirType(F32, listOf(m.vocabSize, d)))
            val emb = op(OpKind.EMBEDDING, listOf(tbl, tokenIds), DxirType(F32, listOf(b, 1, d)))
            val h = op(OpKind.RESHAPE, listOf(emb), DxirType(F32, listOf(b, d)))

            fun project(w: FloatArray, cols: Int) = op(
                OpKind.MATMUL,
                listOf(h, const(w, DxirType(F32, listOf(d, cols)))),
                DxirType(F32, listOf(b, cols)),
            )

            val q = op(
                OpKind.RESHAPE, listOf(project(qProj, qWidth)),
                DxirType(F32, listOf(b, m.numHeads, hd)),
            )
            val newK = op(
                OpKind.RESHAPE, listOf(project(kProj, kvWidth)),
                DxirType(F32, listOf(b, m.numKvHeads, hd)),
            )
            val newV = op(
                OpKind.RESHAPE, listOf(project(vProj, kvWidth)),
                DxirType(F32, listOf(b, m.numKvHeads, hd)),
            )

            val kc = op(OpKind.KV_CACHE_WRITE, listOf(keyCache, newK, slotMapping), spec.poolType)
            val vc = op(OpKind.KV_CACHE_WRITE, listOf(valueCache, newV, slotMapping), spec.poolType)

            val att = op(
                OpKind.PAGED_ATTENTION,
                listOf(q, kc, vc, blockTables, seqLens),
                DxirType(F32, listOf(b, m.numHeads, hd)),
                mapOf("scale" to scale),
            )
            val attFlat = op(OpKind.RESHAPE, listOf(att), DxirType(F32, listOf(b, qWidth)))
            val logits2 = op(
                OpKind.MATMUL,
                listOf(attFlat, const(lmHead, DxirType(F32, listOf(qWidth, m.vocabSize)))),
                DxirType(F32, listOf(b, m.vocabSize)),
            )
            val logits = op(OpKind.RESHAPE, listOf(logits2), spec.logitsType)
            listOf(logits, kc, vc)
        }
    }

    /** Export the whole decode ladder into [dir]. */
    fun exportTo(dir: java.nio.file.Path): ServingManifest = ServingArtifactWriter.export(
        dir = dir,
        modelName = MODEL_NAME,
        modelHash = MODEL_HASH,
        model = MODEL,
        ladder = ServingArtifactWriter.ladderOf(POLICY),
        specs = ServingArtifactWriter.decodeSpecs(MODEL, POLICY),
        build = ::build,
    )
}
