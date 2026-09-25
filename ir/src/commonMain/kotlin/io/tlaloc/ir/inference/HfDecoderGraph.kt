package io.tlaloc.ir.inference

import io.tlaloc.core.BF16
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

// The decode and prefill graphs of an HF decoder-only model, built from an
// [HfDecoderConfig]. Pure [DxirBuilder] arithmetic in :ir commonMain: the
// bytes are HfCheckpoint's business (jvmMain) and staging them in slot order
// is HfStagedWeights'.
//
// Scope:
//
//   * DECODE AND PREFILL. Decode runs one token per sequence. Prefill runs a
//     chunk of T tokens per sequence in one call by treating every token as
//     its own attention row: the chunk's K/V are written to the pool first,
//     and row i then attends over the sequence's pages with a causal length of
//     positions[i] + 1. That is the arithmetic of the decode loop, row for
//     row, with no new op kind. The chunk is right-aligned (padding rows
//     first, slot -1), so the last real token is always row T - 1 and the
//     graph returns that row's logits.
//   * PER-LAYER SPECS. Each layer is built from its [DecoderLayerSpec]: full
//     or sliding-window causal attention, RoPE or none, per-head q/k RMSNorm
//     with or without a gain, an attention output gate, and norms on the
//     attention and MLP outputs. What a config asks for and the graph does not
//     implement is refused by name in [HfDecoderConfig.toDecodeModelShape].
//   * ACTIVATIONS IN FLOAT32. With F32 weights (Llama, Qwen3) a bf16
//     checkpoint is widened at ingestion (exactly: bf16 -> f32 is a 16-bit
//     shift) and everything is f32. With BF16 weights (Muse Glimmer) the
//     weights stay bf16 on the device; each projection rounds its f32 input
//     to bf16 and multiplies bf16 by bf16 into an f32 result, so the products
//     are exact and the sums are f32. Norms, RoPE, attention, the residual
//     stream and the logits stay f32.
//   * NO SAMPLING. Logits out; the sampler is host-side.

/**
 * Builds the decode graph the serving contract calls.
 *
 * ```
 *   embed → [rms_norm]
 *   ├─ per layer ─ rms_norm → q/k/v → [q/k rms_norm] → [q scale] → [RoPE] → KV_CACHE_WRITE ×2
 *   │              → PAGED_ATTENTION (full or sliding window) → [× sigmoid(gate)]
 *   │              → o_proj → [rms_norm] → +residual
 *   │              → rms_norm → SwiGLU → down_proj → [rms_norm] → +residual
 *   └─ final rms_norm → lm_head → [× multiplier] → [cap · tanh(· / cap)] → logits
 * ```
 *
 * The bracketed steps are the ones a [DecoderLayerSpec] or an
 * [HfDecoderConfig] setting turns on.
 *
 * The signature is [DecodeGraphSpec]'s, with [weightSlots] appended; the
 * built function is checked against it by [DecodeGraphSpec.verifySignature].
 */
object HfDecoderGraph {

    /** Default entry symbol, matching `ServingArtifactWriter.ENTRY_POINT`. */
    const val ENTRY_POINT: String = "main"

    // ---------------------------------------------------------------- slots

    /**
     * The staged-weight signature, in the one canonical order a loader binds
     * by index: the embedding table, each layer's [DecoderLayerSpec.parts],
     * the final norm and the head ([weightRoles] gives the role of each).
     *
     * The dims are math layout `[in, out]`, not the file's `[out, in]`. HF
     * stores every `nn.Linear` weight transposed (`F.linear(x, W)` is
     * `x @ W.T`), and [OpKind.MATMUL] contracts `last(A) × first(B)`, so the
     * Linears are transposed once, host-side, at ingestion (HfStagedWeights),
     * not by a [OpKind.TRANSPOSE] per step in the graph. The norms and the
     * embedding table are not Linears and are staged verbatim
     * ([HfDecoderNames.isTransposedLinear] is the predicate for both).
     */
    fun weightSlots(config: HfDecoderConfig): List<DecodeSlot> =
        weightRoles(config).map { role ->
            val fileDims = HfDecoderNames.expectedDims(role, config).toList()
            val dims = if (HfDecoderNames.isTransposedLinear(role)) fileDims.reversed() else fileDims
            DecodeSlot(slotName(role), DxirType(config.weightDType, dims), DecodeSlotRole.WEIGHT)
        }

    /** The [DecoderWeightRole] each slot of [weightSlots] carries, same order. */
    fun weightRoles(config: HfDecoderConfig): List<DecoderWeightRole> = HfDecoderNames.roles(config)

    /** The slot name of a role: `embedTokens`, `qProj3`, `kNorm0`, `lmHead`, ... */
    fun slotName(role: DecoderWeightRole): String = when (role) {
        DecoderWeightRole.EmbedTokens -> "embedTokens"
        DecoderWeightRole.FinalNorm -> "finalNorm"
        DecoderWeightRole.LmHead -> "lmHead"
        is DecoderWeightRole.Layer -> when (role.part) {
            DecoderLayerPart.INPUT_LAYERNORM -> "inputNorm"
            DecoderLayerPart.Q_PROJ -> "qProj"
            DecoderLayerPart.K_PROJ -> "kProj"
            DecoderLayerPart.V_PROJ -> "vProj"
            DecoderLayerPart.Q_NORM -> "qNorm"
            DecoderLayerPart.K_NORM -> "kNorm"
            DecoderLayerPart.O_PROJ -> "oProj"
            DecoderLayerPart.POST_ATTENTION_LAYERNORM -> "postAttnNorm"
            DecoderLayerPart.GATE_PROJ -> "gateProj"
            DecoderLayerPart.UP_PROJ -> "upProj"
            DecoderLayerPart.DOWN_PROJ -> "downProj"
            DecoderLayerPart.ATTN_GATE_PROJ -> "attnGate"
            DecoderLayerPart.ATTENTION_OUTPUT_NORM -> "attnOutNorm"
            DecoderLayerPart.FEEDFORWARD_OUTPUT_NORM -> "ffnOutNorm"
        } + role.layer
    }

    /**
     * The full spec for [config] at a pool geometry and a bucket: a decode
     * step by default, or a prefill chunk of `bucket.maxContext` tokens.
     */
    fun spec(
        config: HfDecoderConfig,
        model: DecodeModelShape,
        bucket: DecodeBucket,
        kind: DecodeGraphKind = DecodeGraphKind.DECODE,
    ): DecodeGraphSpec {
        require(model.numLayers == config.numLayers && model.hiddenSize == config.hiddenSize) {
            "HfDecoderGraph.spec: model shape $model does not describe this config " +
                "(layers=${config.numLayers}, hidden=${config.hiddenSize})"
        }
        return DecodeGraphSpec(
            model = model,
            bucket = bucket,
            kind = kind,
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
     * These are the only large body constants in the graph, and they are
     * derived entirely from `config.json` (`rope_theta`, `head_dim`) and the
     * entry's context; no checkpoint byte reaches them. [build] asks for the
     * positions its entry can reach (its context, capped by
     * `max_position_embeddings`), so a 64-token entry of Qwen3 carries
     * 64 × 128 floats per table, not 40960 × 128.
     *
     * They are computed in DOUBLE and narrowed, because `theta^(-2i/d)` at
     * `theta = 500000` (Llama-3) loses bits in f32 exactly where the
     * high-frequency lanes live.
     */
    fun ropeTables(config: HfDecoderConfig, positions: Int): Pair<FloatArray, FloatArray> {
        require(positions >= 1) { "HfDecoderGraph.ropeTables: positions must be >= 1" }
        val hd = config.headDim
        require(hd % 2 == 0) {
            "HfDecoderGraph.ropeTables: head_dim $hd is odd — RoPE rotates PAIRS of " +
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
     * Build the graph for [spec] over [config]: a decode step when
     * `spec.kind` is [DecodeGraphKind.DECODE], a prefill chunk when it is
     * [DecodeGraphKind.PREFILL].
     *
     * ## Prefill
     *
     * A prefill chunk is `T = spec.tokensPerSeq` tokens per sequence. The
     * graph flattens the batch and token axes into `B * T` token rows and runs
     * every per-token op (embedding, norms, projections, RoPE, SwiGLU) on the
     * rows exactly as a decode step runs them on its sequences. Attention is
     * the decode step's own [OpKind.PAGED_ATTENTION], with one query row per
     * token:
     *
     * - both [OpKind.KV_CACHE_WRITE]s of a layer run before its attention, so
     *   every token of the chunk is in the pool when any row reads it;
     * - row `i` reads its sequence's block table (the `blockTables` row,
     *   broadcast over the token axis) with a context length of
     *   `positions[i] + 1`, which is the causal mask: a token sees itself and
     *   everything before it, and nothing written after it.
     *
     * So each row computes what the decode loop computes at that position.
     * The prefill graph does not read `seqLens`; the operand stays in the
     * signature so that decode and prefill entries share one signature.
     *
     * The chunk is RIGHT-ALIGNED: padding rows come first
     * ([DecodePadding.PADDING_SLOT] -1, so their K/V writes are dropped;
     * position 0; any in-vocab token), then the real tokens, so the last real
     * token is always row `T - 1`. The graph applies the final norm and the
     * head to that row only and returns `[B, 1, vocab]` logits: what a caller
     * samples the next token from, without moving `T` rows of logits to the
     * host. A chunk may start at any position (`positions` need not start at
     * 0), which is how a prompt longer than one chunk is prefilled in pieces.
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
        config: HfDecoderConfig,
        entryName: String = ENTRY_POINT,
    ): DxirFunction {
        require(spec.weightSlots == weightSlots(config)) {
            "HfDecoderGraph.build: the spec's weight signature is not this config's — " +
                "build the spec with HfDecoderGraph.spec(), which derives it"
        }
        val m = spec.model
        require(m.numLayers == config.numLayers) {
            "HfDecoderGraph.build: spec has ${m.numLayers} layers, config has " +
                "${config.numLayers}"
        }
        val b = spec.bucket.batch
        val t = spec.tokensPerSeq
        // Token rows: every per-token op runs on B * T rows.
        val r = b * t
        val maxBlocks = spec.maxBlocksPerSeq
        val d = config.hiddenSize
        val hd = config.headDim
        val half = hd / 2
        val qOut = config.qProjOut
        val kvOut = config.kvProjOut
        val scale = 1.0f / sqrt(hd.toFloat())
        val wdt = config.weightDType
        // The table covers the positions this entry can reach: its context,
        // capped by the model's own limit. Qwen3's 40960 positions at
        // head_dim 128 would be 42 MB of constants in every body.
        val ropePositions = minOf(config.maxPositionEmbeddings, maxBlocks * m.blockSize)
        val (cosTable, sinTable) = ropeTables(config, ropePositions)
        val idx = spec.positionsType.dtype

        val tH = DxirType(F32, listOf(r, d))
        val tFf = DxirType(F32, listOf(r, config.intermediateSize))

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
            val byRole: Map<DecoderWeightRole, DxirNode> =
                weightRoles(config).withIndex().associate { (i, role) -> role to w[i] }
            fun weight(role: DecoderWeightRole): DxirNode = byRole.getValue(role)
            fun layerWeight(l: Int, part: DecoderLayerPart): DxirNode =
                weight(DecoderWeightRole.Layer(l, part))

            // ---- helpers ------------------------------------------------
            val epsConsts = HashMap<Pair<List<Int>, Float>, DxirNode>()

            /** A weight as f32: the slot itself, or its exact widening from bf16. */
            fun f32(x: DxirNode): DxirNode =
                if (x.type.dtype == F32) x else op(OpKind.CAST, listOf(x), DxirType(F32, x.type.dims))

            /**
             * `x @ W` for a staged weight `W` (`[in, out]`). With BF16 weights
             * the f32 input is rounded to bf16 and the product is taken into
             * an f32 result: exact products, f32 sums.
             */
            fun proj(x: DxirNode, wt: DxirNode, out: Int): DxirNode {
                val rows = x.type.dims[0]
                val lhs = if (wt.type.dtype == F32) {
                    x
                } else {
                    op(OpKind.CAST, listOf(x), DxirType(wt.type.dtype, x.type.dims))
                }
                return op(OpKind.MATMUL, listOf(lhs, wt), DxirType(F32, listOf(rows, out)))
            }

            /**
             * RMSNorm over the last axis of [dims]:
             * `x * rsqrt(mean(x^2) + eps)`, times `gain` (or `1 + gain` when
             * [plusOne]) when there is one. Rank 2 for the hidden-state norms,
             * rank 3 (`[rows, heads, headDim]`) for the per-head q/k norms.
             */
            fun rmsNorm(
                x: DxirNode,
                gain: DxirNode?,
                dims: List<Int>,
                eps: Double = config.rmsNormEps,
                plusOne: Boolean = false,
            ): DxirNode {
                val rank = dims.size
                val tX = DxirType(F32, dims)
                val rowDims = dims.dropLast(1) + 1
                val tRow = DxirType(F32, rowDims)
                val epsF = eps.toFloat()
                val epsConst = epsConsts.getOrPut(rowDims to epsF) {
                    const(FloatArray(rowDims.fold(1) { a, n -> a * n }) { epsF }, tRow)
                }
                val sq = op(OpKind.MUL, listOf(x, x), tX)
                val mean = op(
                    OpKind.MEAN, listOf(sq), tRow,
                    attrs = mapOf("reduction_dims" to listOf(rank - 1)),
                )
                val rsq = op(OpKind.RSQRT, listOf(op(OpKind.ADD, listOf(mean, epsConst), tRow)), tRow)
                val scaled = op(OpKind.MUL, listOf(x, rsq), tX)
                if (gain == null) return scaled
                val n = dims.last()
                var g = f32(gain)
                if (plusOne) {
                    g = op(OpKind.ADD, listOf(g, const(1f, DxirType(F32, listOf(n)))), g.type)
                }
                val gainRow = op(
                    OpKind.RESHAPE, listOf(g), DxirType(F32, List(rank - 1) { 1 } + n),
                )
                val gainB = op(
                    OpKind.BROADCAST, listOf(gainRow), tX,
                    attrs = mapOf("broadcast_dimensions" to (0 until rank).toList()),
                )
                return op(OpKind.MUL, listOf(scaled, gainB), tX)
            }

            /** `x * c` for a scalar constant [c] (a splat). */
            fun times(x: DxirNode, c: Double): DxirNode =
                op(OpKind.MUL, listOf(x, const(c.toFloat(), x.type)), x.type)

            /** HF `rotate_half`: `cat(-x[..., d/2:], x[..., :d/2])`. */
            fun rotateHalf(x: DxirNode, heads: Int): DxirNode {
                val t3 = DxirType(F32, listOf(r, heads, hd))
                val tHalf = DxirType(F32, listOf(r, heads, half))
                val x2 = op(
                    OpKind.SLICE, listOf(x), tHalf,
                    attrs = mapOf(
                        "start_indices" to listOf(0, 0, half),
                        "limit_indices" to listOf(r, heads, hd),
                        "strides" to listOf(1, 1, 1),
                    ),
                )
                val x1 = op(
                    OpKind.SLICE, listOf(x), tHalf,
                    attrs = mapOf(
                        "start_indices" to listOf(0, 0, 0),
                        "limit_indices" to listOf(r, heads, half),
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
            // `positions` is [B, T] I32 and is an OPERAND, never derived from
            // seqLens (H1c's decision — a padded row has no correct position
            // to derive, so the convention STATES one). Gathering cos/sin out
            // of a constant table by that operand is an EMBEDDING: the same
            // gather the token lookup is, over a different table.
            val posFlat = op(OpKind.RESHAPE, listOf(positions), DxirType(idx, listOf(r)))
            val anyRope = (0 until m.numLayers).any { config.layer(it).rope }
            val ropeRows: Pair<DxirNode, DxirNode>? = if (!anyRope) null else {
                val cosT = const(cosTable, DxirType(F32, listOf(ropePositions, hd)))
                val sinT = const(sinTable, DxirType(F32, listOf(ropePositions, hd)))
                op(OpKind.EMBEDDING, listOf(cosT, posFlat), DxirType(F32, listOf(r, hd))) to
                    op(OpKind.EMBEDDING, listOf(sinT, posFlat), DxirType(F32, listOf(r, hd)))
            }

            fun bcastToHeads(row: DxirNode, heads: Int): DxirNode {
                val unsq = op(OpKind.RESHAPE, listOf(row), DxirType(F32, listOf(r, 1, hd)))
                return op(
                    OpKind.BROADCAST, listOf(unsq), DxirType(F32, listOf(r, heads, hd)),
                    attrs = mapOf("broadcast_dimensions" to listOf(0, 1, 2)),
                )
            }
            val ropeQ = ropeRows?.let { (c, s) -> bcastToHeads(c, config.numHeads) to bcastToHeads(s, config.numHeads) }
            val ropeK = ropeRows?.let { (c, s) -> bcastToHeads(c, config.numKvHeads) to bcastToHeads(s, config.numKvHeads) }

            /** `x*cos + rotate_half(x)*sin`, HF's `apply_rotary_pos_emb`. */
            fun rope(x: DxirNode, heads: Int, cs: Pair<DxirNode, DxirNode>): DxirNode {
                val t3 = DxirType(F32, listOf(r, heads, hd))
                return op(
                    OpKind.ADD,
                    listOf(
                        op(OpKind.MUL, listOf(x, cs.first), t3),
                        op(OpKind.MUL, listOf(rotateHalf(x, heads), cs.second), t3),
                    ),
                    t3,
                )
            }

            // ---- attention rows -------------------------------------------
            // Decode: one row per sequence, the operands as given. Prefill:
            // one row per token, reading its sequence's block table with a
            // causal length of position + 1.
            val rowTables: DxirNode
            val rowLens: DxirNode
            if (spec.kind == DecodeGraphKind.DECODE) {
                rowTables = blockTables
                rowLens = seqLens
            } else {
                val tables3 = op(
                    OpKind.RESHAPE, listOf(blockTables),
                    DxirType(spec.blockTablesType.dtype, listOf(b, 1, maxBlocks)),
                )
                val perToken = op(
                    OpKind.BROADCAST, listOf(tables3),
                    DxirType(spec.blockTablesType.dtype, listOf(b, t, maxBlocks)),
                    attrs = mapOf("broadcast_dimensions" to listOf(0, 1, 2)),
                )
                rowTables = op(
                    OpKind.RESHAPE, listOf(perToken),
                    DxirType(spec.blockTablesType.dtype, listOf(r, maxBlocks)),
                )
                rowLens = op(
                    OpKind.ADD,
                    listOf(posFlat, const(1, DxirType(idx, listOf(r)))),
                    DxirType(spec.seqLensType.dtype, listOf(r)),
                )
            }

            // ---- embed ---------------------------------------------------
            val embed3 = op(
                OpKind.EMBEDDING, listOf(weight(DecoderWeightRole.EmbedTokens), tokenIds),
                DxirType(wdt, listOf(b, t, d)),
            )
            var h = f32(op(OpKind.RESHAPE, listOf(embed3), DxirType(wdt, listOf(r, d))))
            if (config.embeddingNorm) h = rmsNorm(h, null, listOf(r, d))
            val tQ3 = DxirType(F32, listOf(r, config.numHeads, hd))
            val tKv3 = DxirType(F32, listOf(r, config.numKvHeads, hd))
            val plusOne = config.layerNormGainPlusOne

            // ---- decoder layers -----------------------------------------
            val poolOuts = ArrayList<DxirNode>(2 * m.numLayers)
            for (l in 0 until m.numLayers) {
                val layerSpec = config.layer(l)
                val hn = rmsNorm(
                    h, layerWeight(l, DecoderLayerPart.INPUT_LAYERNORM), listOf(r, d), plusOne = plusOne,
                )

                val q = proj(hn, layerWeight(l, DecoderLayerPart.Q_PROJ), qOut)
                val k = proj(hn, layerWeight(l, DecoderLayerPart.K_PROJ), kvOut)
                val v = proj(hn, layerWeight(l, DecoderLayerPart.V_PROJ), kvOut)

                var q3: DxirNode = op(OpKind.RESHAPE, listOf(q), tQ3)
                var k3: DxirNode = op(OpKind.RESHAPE, listOf(k), tKv3)
                val v3 = op(OpKind.RESHAPE, listOf(v), tKv3)
                if (layerSpec.qkNorm) {
                    // RMSNorm over each head's head_dim, before RoPE.
                    val g = layerSpec.qkNormGain
                    q3 = rmsNorm(q3, if (g) layerWeight(l, DecoderLayerPart.Q_NORM) else null, tQ3.dims)
                    k3 = rmsNorm(k3, if (g) layerWeight(l, DecoderLayerPart.K_NORM) else null, tKv3.dims)
                }
                if (config.queryScale != 1.0) q3 = times(q3, config.queryScale)

                val qRot = if (layerSpec.rope) rope(q3, config.numHeads, ropeQ!!) else q3
                val kRot = if (layerSpec.rope) rope(k3, config.numKvHeads, ropeK!!) else k3

                val (keyIn, valIn) = pools[l]
                val kc = op(OpKind.KV_CACHE_WRITE, listOf(keyIn, kRot, slotMapping), spec.poolType)
                val vc = op(OpKind.KV_CACHE_WRITE, listOf(valIn, v3, slotMapping), spec.poolType)
                poolOuts += kc
                poolOuts += vc

                val attAttrs = buildMap<String, Any> {
                    put("scale", scale.toDouble())
                    if (layerSpec.attention == AttentionKind.SLIDING) {
                        put(io.tlaloc.ir.PagedAttentionAttrs.SLIDING_WINDOW, layerSpec.slidingWindow!!)
                    }
                }
                val att = op(
                    OpKind.PAGED_ATTENTION,
                    listOf(qRot, kc, vc, rowTables, rowLens),
                    DxirType(F32, listOf(r, config.numHeads, hd)),
                    attAttrs,
                )
                var attFlat = op(OpKind.RESHAPE, listOf(att), DxirType(F32, listOf(r, qOut)))
                if (layerSpec.attentionOutputGate) {
                    val gate = proj(hn, layerWeight(l, DecoderLayerPart.ATTN_GATE_PROJ), qOut)
                    attFlat = op(
                        OpKind.MUL, listOf(attFlat, op(OpKind.SIGMOID, listOf(gate), gate.type)), attFlat.type,
                    )
                }
                var attProj = proj(attFlat, layerWeight(l, DecoderLayerPart.O_PROJ), d)
                if (layerSpec.postAttentionOutputNorm) {
                    attProj = rmsNorm(
                        attProj, layerWeight(l, DecoderLayerPart.ATTENTION_OUTPUT_NORM), listOf(r, d),
                        eps = config.outputNormEps, plusOne = plusOne,
                    )
                }
                val hAttn = op(OpKind.ADD, listOf(h, attProj), tH)

                val hn2 = rmsNorm(
                    hAttn, layerWeight(l, DecoderLayerPart.POST_ATTENTION_LAYERNORM), listOf(r, d),
                    plusOne = plusOne,
                )
                val gate = proj(hn2, layerWeight(l, DecoderLayerPart.GATE_PROJ), config.intermediateSize)
                val up = proj(hn2, layerWeight(l, DecoderLayerPart.UP_PROJ), config.intermediateSize)
                val swiglu = op(
                    OpKind.MUL,
                    listOf(op(OpKind.SILU, listOf(gate), tFf), up),
                    tFf,
                )
                var down = proj(swiglu, layerWeight(l, DecoderLayerPart.DOWN_PROJ), d)
                if (layerSpec.postFeedforwardNorm) {
                    down = rmsNorm(
                        down, layerWeight(l, DecoderLayerPart.FEEDFORWARD_OUTPUT_NORM), listOf(r, d),
                        eps = config.outputNormEps, plusOne = plusOne,
                    )
                }
                h = op(OpKind.ADD, listOf(hAttn, down), tH)
            }

            // ---- the last row of each sequence ---------------------------
            val last = if (t == 1) {
                h
            } else {
                val h3 = op(OpKind.RESHAPE, listOf(h), DxirType(F32, listOf(b, t, d)))
                val lastRow = op(
                    OpKind.SLICE, listOf(h3), DxirType(F32, listOf(b, 1, d)),
                    attrs = mapOf(
                        "start_indices" to listOf(0, t - 1, 0),
                        "limit_indices" to listOf(b, t, d),
                        "strides" to listOf(1, 1, 1),
                    ),
                )
                op(OpKind.RESHAPE, listOf(lastRow), DxirType(F32, listOf(b, d)))
            }

            // ---- final norm + head ---------------------------------------
            val hf = rmsNorm(last, weight(DecoderWeightRole.FinalNorm), listOf(b, d))
            var logits2 = proj(hf, weight(DecoderWeightRole.LmHead), config.vocabSize)
            if (config.logitMultiplier != 1.0) logits2 = times(logits2, config.logitMultiplier)
            val cap = config.finalLogitSoftcap
            if (cap != null) {
                // cap * tanh(z / cap), in the reference's order of operations.
                val capConst = const(cap.toFloat(), logits2.type)
                val z = op(OpKind.DIV, listOf(logits2, capConst), logits2.type)
                logits2 = op(OpKind.MUL, listOf(op(OpKind.TANH, listOf(z), z.type), capConst), z.type)
            }
            listOf(op(OpKind.RESHAPE, listOf(logits2), spec.logitsType)) + poolOuts
        }
        spec.verifySignature(fn, "HfDecoderGraph")
        return fn
    }
}
