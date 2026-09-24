package io.tlaloc.ir.inference

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.OpKind
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

// §0.4.479 (Phase H3c-2) — the REAL Llama decode step, built from an
// [HfLlamaConfig]. §0.4.478 gave the repo a way to ask a checkpoint for a
// tensor by role; this file is where the roles become a graph, and where the
// transposed-`[out, in]` layout that slice VERIFIED stops being a documented
// fact and becomes a matmul.
//
// PLACEMENT: `:ir` commonMain, beside [DecodeGraphSpec] — it is pure
// [DxirBuilder] arithmetic over an [HfLlamaConfig] and touches no file. The
// bytes are [HfLlamaCheckpoint]'s business (jvmMain) and staging them in slot
// order is [HfLlamaStagedWeights]'s.
//
// SCOPE, stated up front so nothing here implies more than it does:
//
//   * DECODE ONLY — [DecodeGraphKind.DECODE], one token per sequence.
//     [OpKind.PAGED_ATTENTION]'s query is `[numSeqs, numHeads, headDim]` by
//     design (§0.4.465), and the ragged `[numTokens, ...]` chunked-prefill
//     form with its `queryStartLoc` operand and intra-chunk causal mask is
//     that slice's NAMED DEFERRAL, still open. A prefill of N tokens is
//     therefore run here as N single-token steps with `seqLens` growing — the
//     same thing §0.4.470's runner lane already does, and arithmetically the
//     same answer, at N kernel launches instead of one. It is a PERFORMANCE
//     deferral, not a correctness one, and the parity lane exercises it.
//   * FLOAT32 THROUGHOUT. A bf16 checkpoint is decoded to f32 at ingestion.
//     What that costs and what it buys is in [HfLlamaStagedWeights].
//   * NO SAMPLING, NO BIAS, NO ROPE SCALING. The first is H1c's decision
//     (logits out, sampler host-side); the last two are refused BY NAME in
//     [HfLlamaConfig.toDecodeModelShape] and never reach here.

/**
 * Builds the Llama decode graph the serving contract calls.
 *
 * ```
 *   embed
 *   ├─ per layer ─ rms_norm → q/k/v → RoPE → KV_CACHE_WRITE ×2
 *   │              → PAGED_ATTENTION → o_proj → +residual
 *   │              → rms_norm → SwiGLU → down_proj → +residual
 *   └─ final rms_norm → lm_head → logits
 * ```
 *
 * The signature is [DecodeGraphSpec]'s, with [weightSlots] appended; the
 * built function is checked against it by [DecodeGraphSpec.verifySignature],
 * so "the builder produced the contract" is a gate and not a comment.
 */
object HfLlamaDecodeGraph {

    /** Default entry symbol, matching `ServingArtifactWriter.ENTRY_POINT`. */
    const val ENTRY_POINT: String = "main"

    // ---------------------------------------------------------------- slots

    /**
     * The staged-weight signature, in the ONE canonical order a loader binds
     * by index against. Embedding table, then each layer's tensors in the
     * order the layer consumes them, then the final norm and the head.
     *
     * **The dims here are MATH layout `[in, out]`, not the file's `[out,
     * in]`.** That is the load-bearing line of this file. HF stores every
     * `nn.Linear` weight transposed because `F.linear(x, W)` is `x @ W.T`
     * (verified against a real checkpoint's rectangular k/v and
     * gate/up); Tlaloc's [OpKind.MATMUL] contracts `last(A) × first(B)`, so
     * somebody has to transpose.
     *
     * The transpose happens ONCE, HOST-SIDE, at ingestion
     * ([HfLlamaStagedWeights]). REJECTED: a [OpKind.TRANSPOSE] node per
     * projection in the graph — the weight is a PARAMETER, not a constant, so
     * there is nothing for a constant-folder to fold, and the cost would be a
     * full re-layout of every projection matrix on every decode step. The
     * norms and the embedding table are NOT Linears and are staged verbatim
     * (`HfLlamaNames.isTransposedLinear` is the predicate, and it is the same
     * one this order is derived from).
     */
    fun weightSlots(config: HfLlamaConfig): List<DecodeSlot> = buildList {
        fun w(name: String, vararg dims: Int) =
            add(DecodeSlot(name, DxirType(F32, dims.toList()), DecodeSlotRole.WEIGHT))

        val d = config.hiddenSize
        w("embedTokens", config.vocabSize, d)
        for (l in 0 until config.numLayers) {
            w("inputNorm$l", d)
            w("qProj$l", d, config.qProjOut)
            w("kProj$l", d, config.kvProjOut)
            w("vProj$l", d, config.kvProjOut)
            w("oProj$l", config.qProjOut, d)
            w("postAttnNorm$l", d)
            w("gateProj$l", d, config.intermediateSize)
            w("upProj$l", d, config.intermediateSize)
            w("downProj$l", config.intermediateSize, d)
        }
        w("finalNorm", d)
        w("lmHead", d, config.vocabSize)
    }

    /** The [LlamaWeightRole] each slot of [weightSlots] carries, same order. */
    fun weightRoles(config: HfLlamaConfig): List<LlamaWeightRole> = buildList {
        add(LlamaWeightRole.EmbedTokens)
        for (l in 0 until config.numLayers) {
            add(LlamaWeightRole.Layer(l, LlamaLayerPart.INPUT_LAYERNORM))
            add(LlamaWeightRole.Layer(l, LlamaLayerPart.Q_PROJ))
            add(LlamaWeightRole.Layer(l, LlamaLayerPart.K_PROJ))
            add(LlamaWeightRole.Layer(l, LlamaLayerPart.V_PROJ))
            add(LlamaWeightRole.Layer(l, LlamaLayerPart.O_PROJ))
            add(LlamaWeightRole.Layer(l, LlamaLayerPart.POST_ATTENTION_LAYERNORM))
            add(LlamaWeightRole.Layer(l, LlamaLayerPart.GATE_PROJ))
            add(LlamaWeightRole.Layer(l, LlamaLayerPart.UP_PROJ))
            add(LlamaWeightRole.Layer(l, LlamaLayerPart.DOWN_PROJ))
        }
        add(LlamaWeightRole.FinalNorm)
        add(LlamaWeightRole.LmHead)
    }

    /** The full decode spec for [config] at a pool geometry and a bucket. */
    fun spec(config: HfLlamaConfig, model: DecodeModelShape, bucket: DecodeBucket): DecodeGraphSpec {
        require(model.numLayers == config.numLayers && model.hiddenSize == config.hiddenSize) {
            "HfLlamaDecodeGraph.spec: model shape $model does not describe this config " +
                "(layers=${config.numLayers}, hidden=${config.hiddenSize})"
        }
        return DecodeGraphSpec(
            model = model,
            bucket = bucket,
            kind = DecodeGraphKind.DECODE,
            weightSlots = weightSlots(config),
        )
    }

    // ----------------------------------------------------------- RoPE table

    /**
     * The RoPE cos/sin tables, `[positions, headDim]` each, in HF's own
     * layout: `inv_freq[i] = theta^(-2i/headDim)` for `i in [0, headDim/2)`,
     * `freqs = pos ⊗ inv_freq`, and then `cat(freqs, freqs)` along the last
     * axis — the DUPLICATED form, which is what makes `rotate_half` the right
     * partner rather than an interleave.
     *
     * These are the ONLY body constants in the graph, and they are derived
     * entirely from `config.json` (`rope_theta`, `head_dim`,
     * `max_position_embeddings`) — no checkpoint byte reaches them. At
     * TinyLlama's 2048 × 64 that is 512 KB each, which is a literal a
     * StableHLO file can carry; a weight matrix is not (see
     * [DecodeGraphSpec.weightSlots]).
     *
     * They are computed in DOUBLE and narrowed, because `theta^(-2i/d)` at
     * `theta = 500000` (Llama-3) loses bits in f32 exactly where the
     * high-frequency lanes live.
     */
    fun ropeTables(config: HfLlamaConfig, positions: Int): Pair<FloatArray, FloatArray> {
        require(positions >= 1) { "HfLlamaDecodeGraph.ropeTables: positions must be >= 1" }
        val hd = config.headDim
        require(hd % 2 == 0) {
            "HfLlamaDecodeGraph.ropeTables: head_dim $hd is odd — RoPE rotates PAIRS of " +
                "channels and HF's rotate_half splits the axis in half; an odd head_dim has " +
                "no such split and this builder refuses it BY NAME rather than dropping a lane"
        }
        val half = hd / 2
        val cosT = FloatArray(positions * hd)
        val sinT = FloatArray(positions * hd)
        for (p in 0 until positions) {
            for (i in 0 until half) {
                val invFreq = config.ropeTheta.pow(-2.0 * i / hd)
                val angle = p.toDouble() * invFreq
                val c = cos(angle).toFloat()
                val s = sin(angle).toFloat()
                cosT[p * hd + i] = c
                cosT[p * hd + half + i] = c
                sinT[p * hd + i] = s
                sinT[p * hd + half + i] = s
            }
        }
        return cosT to sinT
    }

    // --------------------------------------------------------------- build

    /**
     * Build the decode graph for [spec] over [config].
     *
     * A REDUCED slice — the first N layers of a real checkpoint — is not a
     * mode of this function: it is this same function over
     * `config.copy(numLayers = N)`, and the weights are the real file's
     * layers `0 until N`. That is what the parity lane certifies, because a
     * 22-layer 1.1-billion-parameter forward through a JVM reference
     * interpreter is minutes per decode step, and a parity claim nobody can
     * afford to re-run is a parity claim that rots. The oracle is reduced the
     * same way and by the same arithmetic — see `hf_llama_reference.py`.
     */
    fun build(
        spec: DecodeGraphSpec,
        config: HfLlamaConfig,
        entryName: String = ENTRY_POINT,
    ): DxirFunction {
        require(spec.kind == DecodeGraphKind.DECODE) {
            "HfLlamaDecodeGraph.build: ${spec.kind} is not implemented — PAGED_ATTENTION's " +
                "query is one token per sequence, and the ragged chunked-prefill " +
                "form is not implemented. Run a prefill as N decode steps"
        }
        require(spec.weightSlots == weightSlots(config)) {
            "HfLlamaDecodeGraph.build: the spec's weight signature is not this config's — " +
                "build the spec with HfLlamaDecodeGraph.spec(), which derives it"
        }
        val m = spec.model
        require(m.numLayers == config.numLayers) {
            "HfLlamaDecodeGraph.build: spec has ${m.numLayers} layers, config has " +
                "${config.numLayers}"
        }
        val b = spec.bucket.batch
        val d = config.hiddenSize
        val hd = config.headDim
        val half = hd / 2
        val qOut = config.qProjOut
        val kvOut = config.kvProjOut
        val eps = config.rmsNormEps.toFloat()
        val scale = 1.0f / sqrt(hd.toFloat())
        val ropePositions = config.maxPositionEmbeddings
        val (cosTable, sinTable) = ropeTables(config, ropePositions)

        val tH = DxirType(F32, listOf(b, d))
        val tRow = DxirType(F32, listOf(b, 1))
        val tFf = DxirType(F32, listOf(b, config.intermediateSize))

        val fn = DxirBuilder.function(entryName) {
            val tokenIds = param("tokenIds", spec.tokenIdsType)
            val positions = param("positions", spec.positionsType)
            val blockTables = param("blockTables", spec.blockTablesType)
            val seqLens = param("seqLens", spec.seqLensType)
            val slotMapping = param("slotMapping", spec.slotMappingType)
            val pools = (0 until m.numLayers).map { l ->
                param("keyCache$l", spec.poolType) to param("valueCache$l", spec.poolType)
            }
            val w = spec.weightSlots.map { param(it.name, it.type) }
            fun weight(i: Int): DxirNode = w[i]

            // ---- helpers ------------------------------------------------
            val epsConst = const(FloatArray(b) { eps }, tRow)

            /** HF `LlamaRMSNorm`: `x * rsqrt(mean(x^2) + eps) * gain`. */
            fun rmsNorm(x: DxirNode, gain: DxirNode): DxirNode {
                val sq = op(OpKind.MUL, listOf(x, x), tH)
                val mean = op(
                    OpKind.MEAN, listOf(sq), tRow,
                    attrs = mapOf("reduction_dims" to listOf(1)),
                )
                val rsq = op(OpKind.RSQRT, listOf(op(OpKind.ADD, listOf(mean, epsConst), tRow)), tRow)
                val scaled = op(OpKind.MUL, listOf(x, rsq), tH)
                val gainRow = op(OpKind.RESHAPE, listOf(gain), DxirType(F32, listOf(1, d)))
                val gainB = op(
                    OpKind.BROADCAST, listOf(gainRow), tH,
                    attrs = mapOf("broadcast_dimensions" to listOf(0, 1)),
                )
                return op(OpKind.MUL, listOf(scaled, gainB), tH)
            }

            /** HF `rotate_half`: `cat(-x[..., d/2:], x[..., :d/2])`. */
            fun rotateHalf(x: DxirNode, heads: Int): DxirNode {
                val t3 = DxirType(F32, listOf(b, heads, hd))
                val tHalf = DxirType(F32, listOf(b, heads, half))
                val x2 = op(
                    OpKind.SLICE, listOf(x), tHalf,
                    attrs = mapOf(
                        "start_indices" to listOf(0, 0, half),
                        "limit_indices" to listOf(b, heads, hd),
                        "strides" to listOf(1, 1, 1),
                    ),
                )
                val x1 = op(
                    OpKind.SLICE, listOf(x), tHalf,
                    attrs = mapOf(
                        "start_indices" to listOf(0, 0, 0),
                        "limit_indices" to listOf(b, heads, half),
                        "strides" to listOf(1, 1, 1),
                    ),
                )
                val negX2 = op(OpKind.NEG, listOf(x2), tHalf)
                return op(
                    OpKind.CONCAT, listOf(negX2, x1), t3,
                    attrs = mapOf("dimension" to 2),
                )
            }

            // ---- the RoPE angles for THIS step's positions ---------------
            // `positions` is [B, 1] I32 and is an OPERAND, never derived from
            // seqLens (H1c's decision — a padded row has no correct position
            // to derive, so the convention STATES one). Gathering cos/sin out
            // of a constant table by that operand is an EMBEDDING: the same
            // gather the token lookup is, over a different table.
            val posFlat = op(
                OpKind.RESHAPE, listOf(positions),
                DxirType(spec.positionsType.dtype, listOf(b)),
            )
            val cosT = const(cosTable, DxirType(F32, listOf(ropePositions, hd)))
            val sinT = const(sinTable, DxirType(F32, listOf(ropePositions, hd)))
            val cosRow = op(OpKind.EMBEDDING, listOf(cosT, posFlat), DxirType(F32, listOf(b, hd)))
            val sinRow = op(OpKind.EMBEDDING, listOf(sinT, posFlat), DxirType(F32, listOf(b, hd)))

            fun bcastToHeads(row: DxirNode, heads: Int): DxirNode {
                val unsq = op(OpKind.RESHAPE, listOf(row), DxirType(F32, listOf(b, 1, hd)))
                return op(
                    OpKind.BROADCAST, listOf(unsq), DxirType(F32, listOf(b, heads, hd)),
                    attrs = mapOf("broadcast_dimensions" to listOf(0, 1, 2)),
                )
            }
            val cosQ = bcastToHeads(cosRow, config.numHeads)
            val sinQ = bcastToHeads(sinRow, config.numHeads)
            val cosK = bcastToHeads(cosRow, config.numKvHeads)
            val sinK = bcastToHeads(sinRow, config.numKvHeads)

            /** `x*cos + rotate_half(x)*sin`, HF's `apply_rotary_pos_emb`. */
            fun rope(x: DxirNode, heads: Int, c: DxirNode, s: DxirNode): DxirNode {
                val t3 = DxirType(F32, listOf(b, heads, hd))
                return op(
                    OpKind.ADD,
                    listOf(
                        op(OpKind.MUL, listOf(x, c), t3),
                        op(OpKind.MUL, listOf(rotateHalf(x, heads), s), t3),
                    ),
                    t3,
                )
            }

            // ---- embed ---------------------------------------------------
            val embed3 = op(
                OpKind.EMBEDDING, listOf(weight(0), tokenIds),
                DxirType(F32, listOf(b, 1, d)),
            )
            var h = op(OpKind.RESHAPE, listOf(embed3), tH)

            // ---- decoder layers -----------------------------------------
            val poolOuts = ArrayList<DxirNode>(2 * m.numLayers)
            for (l in 0 until m.numLayers) {
                val base = 1 + l * PARTS_PER_LAYER
                val hn = rmsNorm(h, weight(base))

                val q = op(OpKind.MATMUL, listOf(hn, weight(base + 1)), DxirType(F32, listOf(b, qOut)))
                val k = op(OpKind.MATMUL, listOf(hn, weight(base + 2)), DxirType(F32, listOf(b, kvOut)))
                val v = op(OpKind.MATMUL, listOf(hn, weight(base + 3)), DxirType(F32, listOf(b, kvOut)))

                val q3 = op(OpKind.RESHAPE, listOf(q), DxirType(F32, listOf(b, config.numHeads, hd)))
                val k3 = op(OpKind.RESHAPE, listOf(k), DxirType(F32, listOf(b, config.numKvHeads, hd)))
                val v3 = op(OpKind.RESHAPE, listOf(v), DxirType(F32, listOf(b, config.numKvHeads, hd)))

                val qRot = rope(q3, config.numHeads, cosQ, sinQ)
                val kRot = rope(k3, config.numKvHeads, cosK, sinK)

                val (keyIn, valIn) = pools[l]
                val kc = op(OpKind.KV_CACHE_WRITE, listOf(keyIn, kRot, slotMapping), spec.poolType)
                val vc = op(OpKind.KV_CACHE_WRITE, listOf(valIn, v3, slotMapping), spec.poolType)
                poolOuts += kc
                poolOuts += vc

                val att = op(
                    OpKind.PAGED_ATTENTION,
                    listOf(qRot, kc, vc, blockTables, seqLens),
                    DxirType(F32, listOf(b, config.numHeads, hd)),
                    mapOf("scale" to scale.toDouble()),
                )
                val attFlat = op(OpKind.RESHAPE, listOf(att), DxirType(F32, listOf(b, qOut)))
                val attProj = op(OpKind.MATMUL, listOf(attFlat, weight(base + 4)), tH)
                val hAttn = op(OpKind.ADD, listOf(h, attProj), tH)

                val hn2 = rmsNorm(hAttn, weight(base + 5))
                val gate = op(OpKind.MATMUL, listOf(hn2, weight(base + 6)), tFf)
                val up = op(OpKind.MATMUL, listOf(hn2, weight(base + 7)), tFf)
                val swiglu = op(
                    OpKind.MUL,
                    listOf(op(OpKind.SILU, listOf(gate), tFf), up),
                    tFf,
                )
                val down = op(OpKind.MATMUL, listOf(swiglu, weight(base + 8)), tH)
                h = op(OpKind.ADD, listOf(hAttn, down), tH)
            }

            // ---- final norm + head ---------------------------------------
            val finalNormIdx = 1 + m.numLayers * PARTS_PER_LAYER
            val hf = rmsNorm(h, weight(finalNormIdx))
            val logits2 = op(
                OpKind.MATMUL, listOf(hf, weight(finalNormIdx + 1)),
                DxirType(F32, listOf(b, config.vocabSize)),
            )
            listOf(op(OpKind.RESHAPE, listOf(logits2), spec.logitsType)) + poolOuts
        }
        spec.verifySignature(fn, "HfLlamaDecodeGraph")
        return fn
    }

    /** Weight slots per decoder layer — the stride of the staged order. */
    const val PARTS_PER_LAYER: Int = 9
}
