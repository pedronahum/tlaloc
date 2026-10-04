package io.tlaloc.ir.inference

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.I8
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
     * the final norm and the head ([weightSlotSources] gives the source of
     * each). Under [HfDecoderConfig.weightQuant] a quantized weight is int8
     * and is followed by its f32 scales.
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
        weightSlotSources(config).map { src ->
            val role = src.role
            val fileDims = HfDecoderNames.expectedDims(role, config).toList()
            val part = role.layerPart
            if (src.fused.isNotEmpty()) {
                // [hidden, sum of the fused roles' outputs], in role order; quantized: codes, then [out] scales.
                val out = (listOf(role) + src.fused).sumOf { HfDecoderNames.expectedDims(it, config)[0] }
                val name = fusedSlotName(role)
                return@map when {
                    src.scale -> DecodeSlot(name + "Scale", DxirType(F32, listOf(out)), DecodeSlotRole.WEIGHT)
                    HfDecoderNames.isQuantized(role, config) ->
                        DecodeSlot(name, DxirType(config.weightQuant.codeDType, listOf(fileDims[1], out)), DecodeSlotRole.WEIGHT)
                    else -> DecodeSlot(name, DxirType(config.weightDType, listOf(fileDims[1], out)), DecodeSlotRole.WEIGHT)
                }
            }
            val dims = when {
                HfDecoderNames.isTransposedLinear(role) -> fileDims.reversed()
                // The conv kernel [C, 1, K] is staged [K, C], as CAUSAL_CONV1D reads it.
                part == DecoderLayerPart.CONV1D -> listOf(fileDims[2], fileDims[0])
                else -> fileDims
            }
            when {
                // A Linear's scales are per output channel [out]; stacked experts' per expert and channel [E, out].
                src.scale && part?.isExperts == true ->
                    DecodeSlot(slotName(role) + "Scale", DxirType(F32, listOf(dims[0], dims[1])), DecodeSlotRole.WEIGHT)
                src.scale ->
                    DecodeSlot(slotName(role) + "Scale", DxirType(F32, listOf(dims.last())), DecodeSlotRole.WEIGHT)
                HfDecoderNames.isQuantized(role, config) ->
                    DecodeSlot(slotName(role), DxirType(config.weightQuant.codeDType, dims), DecodeSlotRole.WEIGHT)
                part?.alwaysF32 == true -> DecodeSlot(slotName(role), DxirType(F32, dims), DecodeSlotRole.WEIGHT)
                part?.f32BesideQuantized == true && config.weightQuant != WeightQuant.NONE ->
                    DecodeSlot(slotName(role), DxirType(F32, dims), DecodeSlotRole.WEIGHT)
                else -> DecodeSlot(slotName(role), DxirType(config.weightDType, dims), DecodeSlotRole.WEIGHT)
            }
        }

    /**
     * What one weight slot holds: the weight of [role], or, when [scale], the
     * per-output-channel scales of that role's quantized weight.
     */
    data class WeightSlotSource(
        val role: DecoderWeightRole,
        val scale: Boolean = false,
        /**
         * Roles staged in the same slot after [role], concatenated along the
         * output axis: a Gated DeltaNet layer's four input projections are
         * one `[hidden, qkv + z + b + a]` weight ([fusesLinearInputs]).
         */
        val fused: List<DecoderWeightRole> = emptyList(),
    )

    /**
     * The projections staged as one weight and computed as one matmul, the
     * product then sliced: each group's first part keys the parts staged after
     * it. Staged apart, XLA merges dots on a shared input by concatenating
     * their weights on every call (a copy of the layer's projections per
     * step). Unquantized: a Gated DeltaNet's q/k/v, z, b and a, and an MoE
     * layer's router with the shared expert's gate, up and output gate.
     * Quantized, a group holds only quantized parts: q/k/v with z, and the
     * shared expert's gate with its up; the small parts beside them are f32
     * ([DecoderLayerPart.f32BesideQuantized]).
     */
    fun fusedGroups(config: HfDecoderConfig): Map<DecoderLayerPart, List<DecoderLayerPart>> =
        if (config.weightQuant == WeightQuant.NONE) {
            mapOf(DecoderLayerPart.IN_PROJ_QKV to FUSED_AFTER_QKV, DecoderLayerPart.ROUTER to FUSED_AFTER_ROUTER)
        } else {
            mapOf(
                DecoderLayerPart.IN_PROJ_QKV to listOf(DecoderLayerPart.IN_PROJ_Z),
                DecoderLayerPart.SHARED_GATE_PROJ to listOf(DecoderLayerPart.SHARED_UP_PROJ),
            )
        }

    private val FUSED_AFTER_QKV = listOf(DecoderLayerPart.IN_PROJ_Z, DecoderLayerPart.IN_PROJ_B, DecoderLayerPart.IN_PROJ_A)

    /** A mixture-of-experts layer's router, the shared expert's gate and up, and its output gate: one `[hidden, E + 2 Is + 1]` weight. */
    private val FUSED_AFTER_ROUTER =
        listOf(DecoderLayerPart.SHARED_GATE_PROJ, DecoderLayerPart.SHARED_UP_PROJ, DecoderLayerPart.SHARED_EXPERT_GATE)

    private fun fusedSlotName(role: DecoderWeightRole): String =
        when (role.layerPart) {
            DecoderLayerPart.ROUTER -> "moeIn"
            DecoderLayerPart.SHARED_GATE_PROJ -> "sharedGateUp"
            else -> "inProj"
        } + if (role is DecoderWeightRole.Layer) role.layer.toString() else "Mtp"

    /**
     * The source of each slot of [weightSlots], same order: one per role of
     * [weightRoles], and under [HfDecoderConfig.weightQuant] a scale slot
     * (`qProj0Scale`, f32 `[out]`) right after each quantized weight
     * ([HfDecoderNames.isQuantized]), whose slot is then int8. Without
     * quantization the slots are exactly the roles.
     */
    fun weightSlotSources(config: HfDecoderConfig): List<WeightSlotSource> =
        weightRoles(config).flatMap { role ->
            val group = role.layerPart?.let { fusedGroups(config)[it] }
            if (group != null) {
                val fused = group.map { role.withPart(it) }
                return@flatMap if (HfDecoderNames.isQuantized(role, config)) {
                    listOf(WeightSlotSource(role, fused = fused), WeightSlotSource(role, scale = true, fused = fused))
                } else {
                    listOf(WeightSlotSource(role, fused = fused))
                }
            }
            if (HfDecoderNames.isQuantized(role, config)) {
                listOf(WeightSlotSource(role), WeightSlotSource(role, scale = true))
            } else {
                listOf(WeightSlotSource(role))
            }
        }

    /**
     * The [DecoderWeightRole]s whose weights are staged, in slot order
     * ([weightSlotSources] adds the scale slots of quantized weights).
     *
     * A tied head ([HfDecoderConfig.tieWordEmbeddings]) has no slot of its
     * own unless [HfDecoderConfig.tiedHeadCopy] asks for one: the head
     * contracts the final hidden state against the embedding table's hidden
     * axis ([tiedHead]), so the table is on the device once.
     */
    fun weightRoles(config: HfDecoderConfig): List<DecoderWeightRole> {
        var all = HfDecoderNames.roles(config)
        if (headReadsEmbedding(config)) all = all - DecoderWeightRole.LmHead
        // Staged inside their group's slot (see WeightSlotSource.fused).
        val tails = fusedGroups(config).values.flatten().toSet()
        all = all.filter { it.layerPart !in tails }
        return all
    }

    /** True when the head reads the embedding table directly (a tied head without a copy). */
    fun headReadsEmbedding(config: HfDecoderConfig): Boolean =
        config.tieWordEmbeddings && !config.tiedHeadCopy

    /** The slot name of a role: `embedTokens`, `qProj3`, `kNorm0`, `lmHead`, ... */
    fun slotName(role: DecoderWeightRole): String = when (role) {
        DecoderWeightRole.EmbedTokens -> "embedTokens"
        DecoderWeightRole.FinalNorm -> "finalNorm"
        DecoderWeightRole.LmHead -> "lmHead"
        is DecoderWeightRole.Mtp -> when (role.part) {
            MtpPart.FC -> "mtpFc"
            MtpPart.PRE_FC_NORM_EMBEDDING -> "mtpPreFcNormEmbedding"
            MtpPart.PRE_FC_NORM_HIDDEN -> "mtpPreFcNormHidden"
            MtpPart.NORM -> "mtpNorm"
        }
        is DecoderWeightRole.MtpLayer -> partSlotName(role.part) + "Mtp"
        is DecoderWeightRole.Layer -> partSlotName(role.part) + role.layer
    }

    private fun partSlotName(part: DecoderLayerPart): String = when (part) {
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
            DecoderLayerPart.IN_PROJ_QKV -> "inProjQkv"
            DecoderLayerPart.IN_PROJ_Z -> "inProjZ"
            DecoderLayerPart.IN_PROJ_B -> "inProjB"
            DecoderLayerPart.IN_PROJ_A -> "inProjA"
            DecoderLayerPart.CONV1D -> "conv1d"
            DecoderLayerPart.DT_BIAS -> "dtBias"
            DecoderLayerPart.A_LOG -> "aLog"
            DecoderLayerPart.LINEAR_NORM -> "linearNorm"
            DecoderLayerPart.OUT_PROJ -> "outProj"
            DecoderLayerPart.ROUTER -> "router"
            DecoderLayerPart.EXPERTS_GATE_UP -> "expertsGateUp"
            DecoderLayerPart.EXPERTS_DOWN -> "expertsDown"
            DecoderLayerPart.SHARED_GATE_PROJ -> "sharedGateProj"
            DecoderLayerPart.SHARED_UP_PROJ -> "sharedUpProj"
            DecoderLayerPart.SHARED_DOWN_PROJ -> "sharedDownProj"
            DecoderLayerPart.SHARED_EXPERT_GATE -> "sharedExpertGate"
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
        prefillChunk: Int? = null,
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
            prefillChunk = prefillChunk,
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
        // Over the rotated channels only: a partial rotary embedding computes
        // its frequencies against the rotary width, not head_dim.
        val hd = config.rotaryDim
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
     * - row `i` reads its sequence's block table (the T rows of a sequence
     *   share its `blockTables` row, see [io.tlaloc.ir.PagedAttentionAttrs])
     *   with a context length of `positions[i] + 1`, which is the causal
     *   mask: a token sees itself and everything before it, and nothing
     *   written after it.
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
        m.windowedKv?.let { w ->
            for (l in w.layers) {
                val ls = config.layer(l)
                require(ls.attention == AttentionKind.SLIDING && ls.slidingWindow!! <= w.window) {
                    "HfDecoderGraph.build: layer $l keeps its KV in the windowed pool (window " +
                        "${w.window}), but it is ${ls.attention} attention" +
                        (ls.slidingWindow?.let { " with window $it" } ?: "") +
                        "; a recycled page would drop positions it still reads"
                }
            }
        }
        val b = spec.bucket.batch
        val t = spec.tokensPerSeq
        // Token rows: every per-token op runs on B * T rows.
        val r = b * t
        val maxBlocks = spec.maxBlocksPerSeq
        val d = config.hiddenSize
        val hd = config.headDim
        val rd = config.rotaryDim
        val half = rd / 2
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
            val windowTables = if (m.windowedKv == null) null else param("windowBlockTables", spec.blockTablesType)
            val windowSlots = if (m.windowedKv == null) null else param("windowSlotMapping", spec.slotMappingType)
            val stateSlots = if (m.linearState == null) null else param("stateSlots", spec.stateSlotsType)
            val stateWriteSlots = if (spec.speculative) param("stateWriteSlots", spec.stateWriteSlotsType) else null
            val pools = (0 until m.numLayers).map { l ->
                val (a, bn) = spec.poolNames(l)
                val (ta, tb) = spec.poolTypesOf(l)
                param(a, ta) to param(bn, tb)
            }
            val mtpPools = spec.mtpPools.map { (name, ty) -> param(name, ty) }
            val w = spec.weightSlots.map { param(it.name, it.type) }
            val sources = weightSlotSources(config)
            val byRole: Map<DecoderWeightRole, DxirNode> =
                sources.withIndex().filter { !it.value.scale }.associate { (i, src) -> src.role to w[i] }
            // The per-output-channel scales of a quantized weight, keyed by the weight's node.
            val scaleOf: Map<DxirNode, DxirNode> =
                sources.withIndex().filter { it.value.scale }.associate { (i, src) -> byRole.getValue(src.role) to w[i] }
            fun weight(role: DecoderWeightRole): DxirNode = byRole.getValue(role)
            // The MTP head's layer is layer index numLayers.
            val mtpLayer = m.numLayers
            fun layerWeight(l: Int, part: DecoderLayerPart): DxirNode =
                weight(if (l == mtpLayer) DecoderWeightRole.MtpLayer(part) else DecoderWeightRole.Layer(l, part))

            // ---- helpers ------------------------------------------------
            val epsConsts = HashMap<Pair<List<Int>, Float>, DxirNode>()

            /** A weight as f32: the slot itself, or its exact widening from bf16. */
            fun f32(x: DxirNode): DxirNode =
                if (x.type.dtype == F32) x else op(OpKind.CAST, listOf(x), DxirType(F32, x.type.dims))

            /**
             * `x @ W` for a staged weight `W` (`[in, out]`). With BF16 weights
             * the f32 input is rounded to bf16 and the product is taken into
             * an f32 result: exact products, f32 sums.
             *
             * An int8 weight (see [WeightQuant.INT8]) is widened to the
             * config's weight dtype (exact), multiplied the same way, and each
             * output column of the f32 result is multiplied by its scale.
             */
            fun proj(x: DxirNode, wt: DxirNode, out: Int): DxirNode {
                val rows = x.type.dims[0]
                val scales = scaleOf[wt]
                val rhs = if (scales == null) wt else op(OpKind.CAST, listOf(wt), DxirType(wdt, wt.type.dims))
                val lhs = if (rhs.type.dtype == F32) {
                    x
                } else {
                    op(OpKind.CAST, listOf(x), DxirType(rhs.type.dtype, x.type.dims))
                }
                val y = op(OpKind.MATMUL, listOf(lhs, rhs), DxirType(F32, listOf(rows, out)))
                if (scales == null) return y
                val scaleRow = op(OpKind.RESHAPE, listOf(scales), DxirType(F32, listOf(1, out)))
                val scaleB = op(
                    OpKind.BROADCAST, listOf(scaleRow), y.type,
                    attrs = mapOf("broadcast_dimensions" to listOf(0, 1)),
                )
                return op(OpKind.MUL, listOf(y, scaleB), y.type)
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

            /** HF `rotate_half` over the rotary channels: `cat(-x[..., d/2:], x[..., :d/2])`. */
            fun rotateHalf(x: DxirNode, heads: Int): DxirNode {
                val r = x.type.dims[0]
                val t3 = DxirType(F32, listOf(r, heads, rd))
                val tHalf = DxirType(F32, listOf(r, heads, half))
                val x2 = op(
                    OpKind.SLICE, listOf(x), tHalf,
                    attrs = mapOf(
                        "start_indices" to listOf(0, 0, half),
                        "limit_indices" to listOf(r, heads, rd),
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
            val cosT by lazy { const(cosTable, DxirType(F32, listOf(ropePositions, rd))) }
            val sinT by lazy { const(sinTable, DxirType(F32, listOf(ropePositions, rd))) }

            fun bcastToHeads(row: DxirNode, heads: Int): DxirNode {
                val r = row.type.dims[0]
                val unsq = op(OpKind.RESHAPE, listOf(row), DxirType(F32, listOf(r, 1, rd)))
                return op(
                    OpKind.BROADCAST, listOf(unsq), DxirType(F32, listOf(r, heads, rd)),
                    attrs = mapOf("broadcast_dimensions" to listOf(0, 1, 2)),
                )
            }

            /** The (cos, sin) rows of the query heads and of the KV heads at [pos] (`[rows]`). */
            fun ropeAt(pos: DxirNode): Pair<Pair<DxirNode, DxirNode>, Pair<DxirNode, DxirNode>> {
                val n = pos.type.dims[0]
                val c = op(OpKind.EMBEDDING, listOf(cosT, pos), DxirType(F32, listOf(n, rd)))
                val sn = op(OpKind.EMBEDDING, listOf(sinT, pos), DxirType(F32, listOf(n, rd)))
                return (bcastToHeads(c, config.numHeads) to bcastToHeads(sn, config.numHeads)) to
                    (bcastToHeads(c, config.numKvHeads) to bcastToHeads(sn, config.numKvHeads))
            }
            val ropeQK = if (anyRope) ropeAt(posFlat) else null
            val ropeQ = ropeQK?.first
            val ropeK = ropeQK?.second

            /**
             * `x*cos + rotate_half(x)*sin`, HF's `apply_rotary_pos_emb`, on
             * the first [rd] channels of each head; the rest pass through.
             */
            fun rope(x: DxirNode, heads: Int, cs: Pair<DxirNode, DxirNode>): DxirNode {
                val r = x.type.dims[0]
                val tRot = DxirType(F32, listOf(r, heads, rd))
                fun channels(from: Int, until: Int) = op(
                    OpKind.SLICE, listOf(x), DxirType(F32, listOf(r, heads, until - from)),
                    attrs = mapOf(
                        "start_indices" to listOf(0, 0, from),
                        "limit_indices" to listOf(r, heads, until),
                        "strides" to listOf(1, 1, 1),
                    ),
                )
                val xr = if (rd == hd) x else channels(0, rd)
                val rotated = op(
                    OpKind.ADD,
                    listOf(
                        op(OpKind.MUL, listOf(xr, cs.first), tRot),
                        op(OpKind.MUL, listOf(rotateHalf(xr, heads), cs.second), tRot),
                    ),
                    tRot,
                )
                if (rd == hd) return rotated
                return op(
                    OpKind.CONCAT, listOf(rotated, channels(rd, hd)), DxirType(F32, listOf(r, heads, hd)),
                    attrs = mapOf("dimension" to 2),
                )
            }

            // ---- attention rows -------------------------------------------
            // Decode: one row per sequence, the operands as given. Prefill:
            // one row per token with a causal length of position + 1; the T
            // rows of a sequence share its block table (PAGED_ATTENTION reads
            // table r / T for row r), so the pages are gathered once per
            // sequence and not once per token. The windowed layers read their
            // own tables the same way.
            val rowTables = blockTables
            // A windowed layer reads only its ring: column j of its table is
            // ring page j (logical block b is on column b % ringPages), so the
            // first ringPages columns are the whole ring and the op reads them
            // as one (PagedAttentionAttrs.RING). A sliding layer then scores
            // at most ringPages * blockSize positions, not the bucket's
            // context. When the bucket is no longer than the ring, the table
            // never wraps and is read as it is.
            val ringPages = m.windowedKv?.ringPages
            val ringRead = ringPages != null && ringPages < maxBlocks
            val windowRowTables = if (windowTables == null || !ringRead) {
                windowTables
            } else {
                op(
                    OpKind.SLICE, listOf(windowTables), DxirType(spec.blockTablesType.dtype, listOf(b, ringPages!!)),
                    attrs = mapOf(
                        "start_indices" to listOf(0, 0),
                        "limit_indices" to listOf(b, ringPages),
                        "strides" to listOf(1, 1),
                    ),
                )
            }
            val rowLens: DxirNode
            // A verify step (speculative decode) is causal over its tokens, as a prefill chunk is.
            if (spec.kind == DecodeGraphKind.DECODE && !spec.speculative) {
                rowLens = seqLens
            } else {
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

            // ---- linear-attention token slots ---------------------------
            // Each token's state slot: its row's stateSlots entry, or -1 where
            // slotMapping marks it padding (the row rules of LinearStateRows).
            val tokenSlots: DxirNode? = stateSlots?.let { ss ->
                val tIdx = DxirType(idx, listOf(b, t))
                val rowSlots = op(
                    OpKind.BROADCAST, listOf(op(OpKind.RESHAPE, listOf(ss), DxirType(idx, listOf(b, 1)))), tIdx,
                    attrs = mapOf("broadcast_dimensions" to listOf(0, 1)),
                )
                val slots2 = op(OpKind.RESHAPE, listOf(slotMapping), tIdx)
                val live = op(
                    OpKind.COMPARE, listOf(slots2, const(0, tIdx)), DxirType(io.tlaloc.core.Bool, listOf(b, t)),
                    attrs = mapOf("direction" to "GE"),
                )
                op(OpKind.WHERE, listOf(live, rowSlots, const(-1, tIdx)), tIdx)
            }
            val positions2 = stateSlots?.let { op(OpKind.RESHAPE, listOf(positions), DxirType(idx, listOf(b, t))) }

            /** The norm before the MLP, SwiGLU, down_proj, the optional output norm, and the residual add. */
            fun mlp(hAttn: DxirNode, l: Int, layerSpec: DecoderLayerSpec): DxirNode {
                val r = hAttn.type.dims[0]
                val tH = hAttn.type
                val tFf = DxirType(F32, listOf(r, config.intermediateSize))
                val hn2 = rmsNorm(
                    hAttn, layerWeight(l, DecoderLayerPart.POST_ATTENTION_LAYERNORM), listOf(r, d),
                    plusOne = plusOne,
                )
                if (layerSpec.mlp == MlpKind.MOE) {
                    return op(OpKind.ADD, listOf(hAttn, moeMlp(this, config, l, hn2, r, ::layerWeight, ::proj) { scaleOf[it] }), tH)
                }
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
                return op(OpKind.ADD, listOf(hAttn, down), tH)
            }

            /**
             * The attention half of attention layer [l] over the rows of [hn]
             * (its input norm), with RoPE rows [rq] and [rk], writing K/V at
             * [slots] and reading [tables] up to [lens]: returns `h + attention`
             * and the two updated pools.
             */
            fun attention(
                l: Int,
                layerSpec: DecoderLayerSpec,
                h: DxirNode,
                hn: DxirNode,
                keyIn: DxirNode,
                valIn: DxirNode,
                slots: DxirNode,
                tables: DxirNode,
                lens: DxirNode,
                rq: Pair<DxirNode, DxirNode>?,
                rk: Pair<DxirNode, DxirNode>?,
                ring: Boolean,
            ): Triple<DxirNode, DxirNode, DxirNode> {
                val r = hn.type.dims[0]
                val tQ3 = DxirType(F32, listOf(r, config.numHeads, hd))
                val tKv3 = DxirType(F32, listOf(r, config.numKvHeads, hd))
                // With a query gate, q_proj gives [query | gate] per head.
                val qWide = proj(
                    hn, layerWeight(l, DecoderLayerPart.Q_PROJ), if (layerSpec.queryGate) 2 * qOut else qOut,
                )
                val (q, queryGate) = if (!layerSpec.queryGate) {
                    qWide to null
                } else {
                    val w3 = op(OpKind.RESHAPE, listOf(qWide), DxirType(F32, listOf(r, config.numHeads, 2 * hd)))
                    fun half(from: Int) = op(
                        OpKind.RESHAPE,
                        listOf(
                            op(
                                OpKind.SLICE, listOf(w3), tQ3,
                                attrs = mapOf(
                                    "start_indices" to listOf(0, 0, from),
                                    "limit_indices" to listOf(r, config.numHeads, from + hd),
                                    "strides" to listOf(1, 1, 1),
                                ),
                            ),
                        ),
                        DxirType(F32, listOf(r, qOut)),
                    )
                    half(0) to half(hd)
                }
                val k = proj(hn, layerWeight(l, DecoderLayerPart.K_PROJ), kvOut)
                val v = proj(hn, layerWeight(l, DecoderLayerPart.V_PROJ), kvOut)

                var q3: DxirNode = op(OpKind.RESHAPE, listOf(q), tQ3)
                var k3: DxirNode = op(OpKind.RESHAPE, listOf(k), tKv3)
                val v3 = op(OpKind.RESHAPE, listOf(v), tKv3)
                if (layerSpec.qkNorm) {
                    // RMSNorm over each head's head_dim, before RoPE.
                    val g = layerSpec.qkNormGain
                    val po = config.qkNormGainPlusOne
                    q3 = rmsNorm(q3, if (g) layerWeight(l, DecoderLayerPart.Q_NORM) else null, tQ3.dims, plusOne = po)
                    k3 = rmsNorm(k3, if (g) layerWeight(l, DecoderLayerPart.K_NORM) else null, tKv3.dims, plusOne = po)
                }
                if (config.queryScale != 1.0) q3 = times(q3, config.queryScale)

                val qRot = if (layerSpec.rope) rope(q3, config.numHeads, rq!!) else q3
                val kRot = if (layerSpec.rope) rope(k3, config.numKvHeads, rk!!) else k3

                val kc = op(OpKind.KV_CACHE_WRITE, listOf(keyIn, kRot, slots), keyIn.type)
                val vc = op(OpKind.KV_CACHE_WRITE, listOf(valIn, v3, slots), valIn.type)

                val attAttrs = buildMap<String, Any> {
                    put("scale", scale.toDouble())
                    if (layerSpec.attention == AttentionKind.SLIDING) {
                        put(io.tlaloc.ir.PagedAttentionAttrs.SLIDING_WINDOW, layerSpec.slidingWindow!!)
                    }
                    if (ring) put(io.tlaloc.ir.PagedAttentionAttrs.RING, true)
                }
                val att = op(
                    OpKind.PAGED_ATTENTION,
                    listOf(qRot, kc, vc, tables, lens),
                    DxirType(F32, listOf(r, config.numHeads, hd)),
                    attAttrs,
                )
                var attFlat = op(OpKind.RESHAPE, listOf(att), DxirType(F32, listOf(r, qOut)))
                if (queryGate != null) {
                    attFlat = op(
                        OpKind.MUL, listOf(attFlat, op(OpKind.SIGMOID, listOf(queryGate), queryGate.type)), attFlat.type,
                    )
                }
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
                return Triple(op(OpKind.ADD, listOf(h, attProj), h.type), kc, vc)
            }

            // ---- decoder layers -----------------------------------------
            val poolOuts = ArrayList<DxirNode>(2 * m.numLayers)
            for (l in 0 until m.numLayers) {
                val layerSpec = config.layer(l)
                val hn = rmsNorm(
                    h, layerWeight(l, DecoderLayerPart.INPUT_LAYERNORM), listOf(r, d), plusOne = plusOne,
                )

                if (layerSpec.mixer == TokenMixer.GATED_DELTA_NET) {
                    val (convIn, stateIn) = pools[l]
                    val (mixed, convOut, stateOut) = gatedDeltaNet(
                        this, config, l, hn, b, t, ::layerWeight, ::proj,
                        convIn, stateIn, tokenSlots!!, positions2!!,
                        { x, gain, dims -> rmsNorm(x, gain, dims) },
                        if (spec.speculative && spec.kind == DecodeGraphKind.DECODE) stateWriteSlots else null,
                    )
                    poolOuts += convOut
                    poolOuts += stateOut
                    h = mlp(op(OpKind.ADD, listOf(h, mixed), tH), l, layerSpec)
                    continue
                }

                val (keyIn, valIn) = pools[l]
                val windowed = spec.isWindowed(l)
                val (hAttn, kc, vc) = attention(
                    l, layerSpec, h, hn, keyIn, valIn,
                    if (windowed) windowSlots!! else slotMapping,
                    if (windowed) windowRowTables!! else rowTables,
                    rowLens, ropeQ, ropeK, windowed && ringRead,
                )
                poolOuts += kc
                poolOuts += vc
                h = mlp(hAttn, l, layerSpec)
            }

            /**
             * The rest of a speculative entry after the target's layers (see
             * docs/work-log/mtp-plan.md): the greedy tokens, the accepted
             * drafts, the carried hidden states, the MTP head over the step's
             * positions and its k drafts.
             */
            fun speculativeTail(): List<DxirNode> {
                val k = spec.draftTokens
                val verify = spec.kind == DecodeGraphKind.DECODE
                val bs = m.blockSize
                val nSlots = m.linearState!!.numSlots
                val boolType = io.tlaloc.core.Bool
                fun iT(vararg dims: Int) = DxirType(idx, dims.toList())
                fun ints(vals: IntArray, vararg dims: Int) = const(FloatArray(vals.size) { vals[it].toFloat() }, iT(*dims))
                fun reshape(x: DxirNode, vararg dims: Int) = op(OpKind.RESHAPE, listOf(x), DxirType(x.type.dtype, dims.toList()))
                fun slice(x: DxirNode, start: List<Int>, limit: List<Int>) = op(
                    OpKind.SLICE, listOf(x), DxirType(x.type.dtype, limit.zip(start).map { (a, z) -> a - z }),
                    attrs = mapOf("start_indices" to start, "limit_indices" to limit, "strides" to List(start.size) { 1 }),
                )
                fun concat(xs: List<DxirNode>, dim: Int): DxirNode {
                    if (xs.size == 1) return xs[0]
                    val dims = xs[0].type.dims.toMutableList()
                    dims[dim] = xs.sumOf { it.type.dims[dim] }
                    return op(OpKind.CONCAT, xs, DxirType(xs[0].type.dtype, dims), attrs = mapOf("dimension" to dim))
                }
                fun cmp(x: DxirNode, y: DxirNode, dir: String) =
                    op(OpKind.COMPARE, listOf(x, y), DxirType(boolType, x.type.dims), attrs = mapOf("direction" to dir))
                fun where(c: DxirNode, x: DxirNode, y: DxirNode) = op(OpKind.WHERE, listOf(c, x, y), x.type)
                fun add(x: DxirNode, y: DxirNode) = op(OpKind.ADD, listOf(x, y), x.type)
                fun mul(x: DxirNode, y: DxirNode) = op(OpKind.MUL, listOf(x, y), x.type)
                fun splat(v: Int, x: DxirNode) = const(v, DxirType(idx, x.type.dims))
                /** 1 where [c] holds, else 0. */
                fun flag(c: DxirNode) = where(c, splat(1, c), splat(0, c))
                fun isOne(x: DxirNode) = cmp(x, splat(1, x), "EQ")
                fun head(x: DxirNode): DxirNode = if (headReadsEmbedding(config)) {
                    tiedHead(this, x, weight(DecoderWeightRole.EmbedTokens))
                } else {
                    proj(x, weight(DecoderWeightRole.LmHead), config.vocabSize)
                }
                fun argmax(logits: DxirNode) = op(
                    OpKind.ARGMAX, listOf(logits), DxirType(idx, logits.type.dims.dropLast(1)),
                    attrs = mapOf("axis" to logits.type.rank - 1),
                )
                fun mtpWeight(part: MtpPart) = weight(DecoderWeightRole.Mtp(part))
                val plusOne = config.finalNormGainPlusOne
                /** The KV slot of position [pos] in batch row [row] (both `[n]`), -1 where [live] is 0. */
                fun kvSlot(pos: DxirNode, row: DxirNode, live: DxirNode): DxirNode {
                    val n = pos.type.dims[0]
                    val page = op(OpKind.DIV, listOf(pos, splat(bs, pos)), pos.type)
                    val col = where(cmp(page, splat(maxBlocks - 1, page), "GT"), splat(maxBlocks - 1, page), page)
                    val table = reshape(blockTables, b * maxBlocks, 1)
                    val at = add(mul(row, splat(maxBlocks, row)), col)
                    val pageId = reshape(op(OpKind.EMBEDDING, listOf(table, at), iT(n, 1)), n)
                    val slot = add(mul(pageId, splat(bs, pageId)), op(OpKind.SUB, listOf(pos, mul(page, splat(bs, page))), pos.type))
                    return where(isOne(live), slot, splat(-1, slot))
                }
                /** The MTP head over rows of [tokens] and target hidden states [hidden], RoPE at [pos]: (normed output, pools). */
                fun mtpHead(
                    tokens: DxirNode, hidden: DxirNode, pos: DxirNode, slots: DxirNode, keyIn: DxirNode, valIn: DxirNode,
                ): Triple<DxirNode, DxirNode, DxirNode> {
                    val n = tokens.type.dims[0]
                    val e = f32(op(OpKind.EMBEDDING, listOf(weight(DecoderWeightRole.EmbedTokens), tokens), DxirType(wdt, listOf(n, d))))
                    val en = rmsNorm(e, mtpWeight(MtpPart.PRE_FC_NORM_EMBEDDING), listOf(n, d), plusOne = plusOne)
                    val hn2 = rmsNorm(hidden, mtpWeight(MtpPart.PRE_FC_NORM_HIDDEN), listOf(n, d), plusOne = plusOne)
                    val x = proj(concat(listOf(en, hn2), 1), mtpWeight(MtpPart.FC), d)
                    val xn = rmsNorm(x, layerWeight(mtpLayer, DecoderLayerPart.INPUT_LAYERNORM), listOf(n, d), plusOne = config.layerNormGainPlusOne)
                    val (rq, rk) = ropeAt(pos)
                    val lens = op(OpKind.ADD, listOf(pos, splat(1, pos)), DxirType(spec.seqLensType.dtype, listOf(n)))
                    val (hA, kc, vc) = attention(mtpLayer, config.mtpLayerSpec, x, xn, keyIn, valIn, slots, blockTables, lens, rq, rk, false)
                    val o = rmsNorm(mlp(hA, mtpLayer, config.mtpLayerSpec), mtpWeight(MtpPart.NORM), listOf(n, d), plusOne = plusOne)
                    return Triple(o, kc, vc)
                }

                // 1. The target's final norm on every row, and its greedy tokens.
                val hn = rmsNorm(h, weight(DecoderWeightRole.FinalNorm), listOf(r, d), plusOne = plusOne)
                val hn3 = reshape(hn, b, t, d)
                val g = if (verify) {
                    reshape(argmax(head(hn)), b, t)
                } else {
                    reshape(argmax(head(reshape(slice(hn3, listOf(0, t - 1, 0), listOf(b, t, d)), b, d))), b, 1)
                }
                val liveTok = flag(cmp(reshape(slotMapping, b, t), const(0, iT(b, t)), "GE"))

                // 2. Accepted drafts: the leading run of tokenIds[:, i + 1] == g[:, i].
                val accepted = if (verify && k > 0) {
                    var run: DxirNode = const(1, iT(b, 1))
                    var acc: DxirNode = const(0, iT(b, 1))
                    for (i in 0 until k) {
                        val eq = flag(cmp(slice(tokenIds, listOf(0, i + 1), listOf(b, i + 2)), slice(g, listOf(0, i), listOf(b, i + 1)), "EQ"))
                        run = mul(run, eq)
                        acc = add(acc, run)
                    }
                    reshape(acc, b)
                } else {
                    const(0, iT(b))
                }

                // 3. The target's hidden state after each token, carried per state slot.
                val (mtpKeyIn, mtpValIn, mtpHiddenIn) = mtpPools
                val hiddenWrites = if (verify) {
                    reshape(stateWriteSlots!!, r)
                } else {
                    // A prefill chunk keeps the state of its last token at the row's slot.
                    val lastCol = ints(IntArray(b * t) { if (it % t == t - 1) 1 else 0 }, b, t)
                    val rowSlot = op(
                        OpKind.BROADCAST, listOf(reshape(stateSlots!!, b, 1)), iT(b, t),
                        attrs = mapOf("broadcast_dimensions" to listOf(0, 1)),
                    )
                    reshape(where(isOne(mul(lastCol, liveTok)), rowSlot, const(-1, iT(b, t))), r)
                }
                val hiddenOut = op(OpKind.KV_CACHE_WRITE, listOf(mtpHiddenIn, reshape(hn, r, 1, d), hiddenWrites), mtpHiddenIn.type)
                val carried = op(
                    OpKind.EMBEDDING, listOf(reshape(mtpHiddenIn, nSlots, d), stateSlots!!), DxirType(F32, listOf(b, d)),
                )

                // 4. The head over T + 1 rows per sequence: row j < T at the position before
                // target row j (token: that row's token; hidden: the row before, or the carried
                // state), and row T at the last position (token: the greedy one). A verify
                // step's rows after its first take the target's greedy tokens, which equal the
                // accepted drafts and replace the first rejected one.
                val t1 = t + 1
                val n2 = b * t1
                val tokensM = if (verify) {
                    concat(listOf(slice(tokenIds, listOf(0, 0), listOf(b, 1)), g), 1)
                } else {
                    concat(listOf(tokenIds, g), 1)
                }
                val prevDead = concat(listOf(const(1, iT(b, 1)), op(OpKind.SUB, listOf(const(1, iT(b, t)), liveTok), iT(b, t))), 1)
                val carriedB = op(
                    OpKind.BROADCAST, listOf(reshape(carried, b, 1, d)), DxirType(F32, listOf(b, t1, d)),
                    attrs = mapOf("broadcast_dimensions" to listOf(0, 1, 2)),
                )
                val prevDeadB = op(
                    OpKind.BROADCAST, listOf(reshape(isOne(prevDead), b, t1, 1)), DxirType(boolType, listOf(b, t1, d)),
                    attrs = mapOf("broadcast_dimensions" to listOf(0, 1, 2)),
                )
                val hidM = where(prevDeadB, carriedB, concat(listOf(reshape(carried, b, 1, d), hn3), 1))
                val posM = concat(
                    listOf(op(OpKind.SUB, listOf(positions, const(1, iT(b, t))), iT(b, t)), slice(positions, listOf(0, t - 1), listOf(b, t))), 1,
                )
                val posPositive = flag(cmp(slice(posM, listOf(0, 0), listOf(b, t)), const(0, iT(b, t)), "GE"))
                val liveM = concat(listOf(mul(liveTok, posPositive), slice(liveTok, listOf(0, t - 1), listOf(b, t))), 1)
                val posMc = where(isOne(liveM), posM, const(0, iT(b, t1)))
                val rowsM = ints(IntArray(n2) { it / t1 }, n2)
                val slotsM = kvSlot(reshape(posMc, n2), rowsM, reshape(liveM, n2))
                var (o: DxirNode, kc: DxirNode, vc: DxirNode) = mtpHead(reshape(tokensM, n2), reshape(hidM, n2, d), reshape(posMc, n2), slotsM, mtpKeyIn, mtpValIn)

                // 5. Drafts: the first from row (accepted + 1) of a verify step or row T of a
                // prefill chunk, then k - 1 single steps feeding the head its own output.
                val sel = if (verify) add(accepted, const(1, iT(b))) else const(t, iT(b))
                val selFlat = add(ints(IntArray(b) { it * t1 }, b), sel)
                var oPrev: DxirNode = op(OpKind.EMBEDDING, listOf(o, selFlat), DxirType(F32, listOf(b, d)))
                var pos: DxirNode = reshape(op(OpKind.EMBEDDING, listOf(reshape(posMc, n2, 1), selFlat), iT(b, 1)), b)
                val liveB = reshape(slice(liveTok, listOf(0, t - 1), listOf(b, t)), b)
                val rowsB = ints(IntArray(b) { it }, b)
                val drafts = ArrayList<DxirNode>(k)
                drafts += argmax(head(oPrev))
                for (j in 1 until k) {
                    pos = add(pos, const(1, iT(b)))
                    val (oj, kcj, vcj) = mtpHead(drafts.last(), oPrev, pos, kvSlot(pos, rowsB, liveB), kc, vc)
                    kc = kcj
                    vc = vcj
                    oPrev = oj
                    drafts += argmax(head(oj))
                }
                val draftsOut = concat(drafts.map { reshape(it, b, 1) }, 1)
                val nextTokens = if (verify) g else concat(listOf(g, const(0, iT(b, k))), 1)
                return listOf(nextTokens, accepted, draftsOut) + poolOuts + listOf(kc, vc, hiddenOut)
            }

            if (spec.speculative) return@function speculativeTail()

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
            val hf = rmsNorm(last, weight(DecoderWeightRole.FinalNorm), listOf(b, d), plusOne = config.finalNormGainPlusOne)
            var logits2 = if (headReadsEmbedding(config)) {
                tiedHead(this, hf, weight(DecoderWeightRole.EmbedTokens))
            } else {
                proj(hf, weight(DecoderWeightRole.LmHead), config.vocabSize)
            }
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

    /**
     * A mixture-of-experts MLP over [hn] (`[rows, hidden]`), transformers'
     * `Qwen3_5MoeSparseMoeBlock`: the routed experts ([OpKind.MOE_EXPERTS])
     * plus the shared expert scaled by `sigmoid(hn · g)`. The router logits,
     * the shared expert's gate and up and its output gate are one matmul
     * against the fused `moeIn` weight, or, quantized, the shared expert's
     * gate and up are ([fusedGroups]).
     */
    private fun moeMlp(
        bld: DxirBuilder,
        config: HfDecoderConfig,
        l: Int,
        hn: DxirNode,
        r: Int,
        layerWeight: (Int, DecoderLayerPart) -> DxirNode,
        proj: (DxirNode, DxirNode, Int) -> DxirNode,
        scaleOf: (DxirNode) -> DxirNode?,
    ): DxirNode = with(bld) {
        val m = config.moe!!
        val d = config.hiddenSize
        val e = m.numExperts
        val si = m.sharedIntermediate
        fun w(part: DecoderLayerPart) = layerWeight(l, part)
        fun cols(x: DxirNode, from: Int, width: Int) = op(
            OpKind.SLICE, listOf(x), DxirType(F32, listOf(r, width)),
            attrs = mapOf("start_indices" to listOf(0, from), "limit_indices" to listOf(r, from + width), "strides" to listOf(1, 1)),
        )
        val groups = fusedGroups(config)
        val (logits, sGate, sUp, sOut) = if (groups.containsKey(DecoderLayerPart.ROUTER)) {
            val all = proj(hn, w(DecoderLayerPart.ROUTER), e + 2 * si + 1)
            listOf(cols(all, 0, e), cols(all, e, si), cols(all, e + si, si), cols(all, e + 2 * si, 1))
        } else if (groups.containsKey(DecoderLayerPart.SHARED_GATE_PROJ)) {
            val gu = proj(hn, w(DecoderLayerPart.SHARED_GATE_PROJ), 2 * si)
            listOf(
                proj(hn, w(DecoderLayerPart.ROUTER), e), cols(gu, 0, si), cols(gu, si, si),
                proj(hn, w(DecoderLayerPart.SHARED_EXPERT_GATE), 1),
            )
        } else {
            listOf(
                proj(hn, w(DecoderLayerPart.ROUTER), e),
                proj(hn, w(DecoderLayerPart.SHARED_GATE_PROJ), si),
                proj(hn, w(DecoderLayerPart.SHARED_UP_PROJ), si),
                proj(hn, w(DecoderLayerPart.SHARED_EXPERT_GATE), 1),
            )
        }
        val tS = DxirType(F32, listOf(r, si))
        val shared = proj(op(OpKind.MUL, listOf(op(OpKind.SILU, listOf(sGate), tS), sUp), tS), w(DecoderLayerPart.SHARED_DOWN_PROJ), d)
        val tH = DxirType(F32, listOf(r, d))
        val gate = op(
            OpKind.BROADCAST, listOf(op(OpKind.SIGMOID, listOf(sOut), sOut.type)), tH,
            attrs = mapOf("broadcast_dimensions" to listOf(0, 1)),
        )
        val gu = w(DecoderLayerPart.EXPERTS_GATE_UP)
        val dn = w(DecoderLayerPart.EXPERTS_DOWN)
        // The rows go in the compute dtype: the weights' own, or, for codes, the config's weight dtype.
        val wdt = scaleOf(gu)?.let { config.weightDType } ?: gu.type.dtype
        val xw = if (wdt == F32) hn else op(OpKind.CAST, listOf(hn), DxirType(wdt, hn.type.dims))
        val experts = scaleOf(gu)?.let { listOf(gu, it, dn, scaleOf(dn)!!) } ?: listOf(gu, dn)
        val routed = op(
            OpKind.MOE_EXPERTS, listOf(xw, logits) + experts, tH,
            attrs = mapOf("top_k" to m.topK),
        )
        op(OpKind.ADD, listOf(routed, op(OpKind.MUL, listOf(shared, gate), tH)), tH)
    }

    /**
     * A Gated DeltaNet layer's token mixer over the normalized input [hn]
     * (`[B*T, hidden]`), transformers' `Qwen3_5GatedDeltaNet.forward`:
     *
     * ```
     *   qkv = hn Wqkv; z = hn Wz; b = hn Wb; a = hn Wa
     *   y   = silu(CAUSAL_CONV1D(qkv))            -> q | k | v
     *   q, k = l2norm(q), l2norm(k); q = q / sqrt(Dk)
     *   beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias)
     *   o   = GATED_DELTA_RULE(q, k, v, g, beta)
     *   out = (w * rmsnorm(o) * silu(z)) Wout      per value head
     * ```
     *
     * Returns the mixer's output `[B*T, hidden]` and the two updated pools.
     */
    private fun gatedDeltaNet(
        bld: DxirBuilder,
        config: HfDecoderConfig,
        l: Int,
        hn: DxirNode,
        b: Int,
        t: Int,
        layerWeight: (Int, DecoderLayerPart) -> DxirNode,
        proj: (DxirNode, DxirNode, Int) -> DxirNode,
        convIn: DxirNode,
        stateIn: DxirNode,
        tokenSlots: DxirNode,
        positions: DxirNode,
        rmsNorm: (DxirNode, DxirNode?, List<Int>) -> DxirNode,
        writeSlots: DxirNode? = null,
    ): Triple<DxirNode, DxirNode, DxirNode> = with(bld) {
        val la = config.linearAttention!!
        val r = b * t
        val c = la.convChannels
        val hk = la.numKeyHeads; val hv = la.numValueHeads
        val dk = la.keyHeadDim; val dv = la.valueHeadDim
        fun w(part: DecoderLayerPart) = layerWeight(l, part)
        fun f32(x: DxirNode): DxirNode =
            if (x.type.dtype == F32) x else op(OpKind.CAST, listOf(x), DxirType(F32, x.type.dims))
        fun t3(vararg dims: Int) = DxirType(F32, listOf(b, t) + dims.toList())

        val group = fusedGroups(config)[DecoderLayerPart.IN_PROJ_QKV].orEmpty()
        val (qkv, z, bLogit, aIn) = if (group.isNotEmpty()) {
            // One matmul against [qkv | z (| b | a)], then the column ranges.
            val widths = listOf(c, la.valueWidth, hv, hv).take(1 + group.size)
            val all = proj(hn, w(DecoderLayerPart.IN_PROJ_QKV), widths.sum())
            var from = 0
            widths.map { width ->
                op(
                    OpKind.SLICE, listOf(all), DxirType(F32, listOf(r, width)),
                    attrs = mapOf(
                        "start_indices" to listOf(0, from),
                        "limit_indices" to listOf(r, from + width),
                        "strides" to listOf(1, 1),
                    ),
                ).also { from += width }
            } + listOf(DecoderLayerPart.IN_PROJ_B to hv, DecoderLayerPart.IN_PROJ_A to hv).drop(group.size - 1).map { (p, n) -> proj(hn, w(p), n) }
        } else {
            listOf(
                proj(hn, w(DecoderLayerPart.IN_PROJ_QKV), c),
                proj(hn, w(DecoderLayerPart.IN_PROJ_Z), la.valueWidth),
                proj(hn, w(DecoderLayerPart.IN_PROJ_B), hv),
                proj(hn, w(DecoderLayerPart.IN_PROJ_A), hv),
            )
        }

        val x3 = op(OpKind.RESHAPE, listOf(qkv), t3(c))
        val conv = opMulti(
            OpKind.CAUSAL_CONV1D,
            listOf(x3, f32(w(DecoderLayerPart.CONV1D)), convIn, tokenSlots, positions) + listOfNotNull(writeSlots),
            listOf(x3.type, convIn.type),
        )
        val act = op(OpKind.SILU, listOf(conv.result(0)), x3.type)
        fun channels(from: Int, width: Int) = op(
            OpKind.SLICE, listOf(act), t3(width),
            attrs = mapOf(
                "start_indices" to listOf(0, 0, from),
                "limit_indices" to listOf(b, t, from + width),
                "strides" to listOf(1, 1, 1),
            ),
        )
        val kw = la.keyWidth
        val q4 = op(OpKind.RESHAPE, listOf(channels(0, kw)), t3(hk, dk))
        val k4 = op(OpKind.RESHAPE, listOf(channels(kw, kw)), t3(hk, dk))
        val v4 = op(OpKind.RESHAPE, listOf(channels(2 * kw, la.valueWidth)), t3(hv, dv))

        // transformers' l2norm: x * rsqrt(sum(x^2) + 1e-6), over each head.
        fun l2norm(x: DxirNode): DxirNode {
            val rowT = t3(hk, 1)
            val ss = op(
                OpKind.SUM, listOf(op(OpKind.MUL, listOf(x, x), x.type)), rowT,
                attrs = mapOf("reduction_dims" to listOf(3)),
            )
            val inv = op(OpKind.RSQRT, listOf(op(OpKind.ADD, listOf(ss, const(1e-6f, rowT)), rowT)), rowT)
            val invB = op(OpKind.BROADCAST, listOf(inv), x.type, attrs = mapOf("broadcast_dimensions" to listOf(0, 1, 2, 3)))
            return op(OpKind.MUL, listOf(x, invB), x.type)
        }
        val qn = op(OpKind.MUL, listOf(l2norm(q4), const((1.0 / sqrt(dk.toDouble())).toFloat(), q4.type)), q4.type)
        val kn = l2norm(k4)

        val tH = t3(hv)
        val beta = op(OpKind.SIGMOID, listOf(op(OpKind.RESHAPE, listOf(bLogit), tH)), tH)
        // g = -exp(A_log) * softplus(a + dt_bias), softplus(x) = relu(x) + log(1 + exp(-|x|)).
        fun headRow(x: DxirNode): DxirNode = op(
            OpKind.BROADCAST, listOf(op(OpKind.RESHAPE, listOf(f32(x)), DxirType(F32, listOf(1, 1, hv)))), tH,
            attrs = mapOf("broadcast_dimensions" to listOf(0, 1, 2)),
        )
        val pre = op(OpKind.ADD, listOf(op(OpKind.RESHAPE, listOf(aIn), tH), headRow(w(DecoderLayerPart.DT_BIAS))), tH)
        val softplus = op(
            OpKind.ADD,
            listOf(
                op(OpKind.RELU, listOf(pre), tH),
                op(
                    OpKind.LOG,
                    listOf(
                        op(
                            OpKind.ADD,
                            listOf(const(1f, tH), op(OpKind.EXP, listOf(op(OpKind.NEG, listOf(op(OpKind.ABS, listOf(pre), tH)), tH)), tH)),
                            tH,
                        ),
                    ),
                    tH,
                ),
            ),
            tH,
        )
        val rate = op(OpKind.EXP, listOf(headRow(w(DecoderLayerPart.A_LOG))), tH)
        val g = op(OpKind.NEG, listOf(op(OpKind.MUL, listOf(rate, softplus), tH)), tH)

        val rule = opMulti(
            OpKind.GATED_DELTA_RULE,
            listOf(qn, kn, v4, g, beta, stateIn, tokenSlots, positions) + listOfNotNull(writeSlots),
            listOf(v4.type, stateIn.type),
        )
        // The gated RMSNorm per value head: w * norm(o) * silu(z).
        val o2 = op(OpKind.RESHAPE, listOf(rule.result(0)), DxirType(F32, listOf(r * hv, dv)))
        val normed = rmsNorm(o2, w(DecoderLayerPart.LINEAR_NORM), listOf(r * hv, dv))
        val zRows = op(OpKind.RESHAPE, listOf(z), DxirType(F32, listOf(r * hv, dv)))
        val gated = op(OpKind.MUL, listOf(normed, op(OpKind.SILU, listOf(zRows), zRows.type)), zRows.type)
        val out = proj(op(OpKind.RESHAPE, listOf(gated), DxirType(F32, listOf(r, la.valueWidth))), w(DecoderLayerPart.OUT_PROJ), config.hiddenSize)
        Triple(out, conv.result(1), rule.result(1))
    }

    /**
     * The logits of a tied head: `h @ E^T` for the final hidden state `h`
     * (`[rows, hidden]`, f32) and the embedding table `E` (`[vocab, hidden]`),
     * as one [OpKind.MATMUL] that contracts the hidden axis of both
     * (`lhs_contracting_dims = [1]`, `rhs_contracting_dims = [1]`, a
     * `stablehlo.dot_general` with no transpose and no second copy of the
     * table). With BF16 weights `h` is rounded to bf16 first and the product
     * is taken into f32, as every other projection is.
     */
    private fun tiedHead(b: DxirBuilder, h: DxirNode, table: DxirNode): DxirNode = with(b) {
        val rows = h.type.dims[0]
        val vocab = table.type.dims[0]
        val lhs = if (table.type.dtype == F32) h else op(OpKind.CAST, listOf(h), DxirType(table.type.dtype, h.type.dims))
        op(
            OpKind.MATMUL, listOf(lhs, table), DxirType(F32, listOf(rows, vocab)),
            attrs = mapOf("lhs_contracting_dims" to listOf(1), "rhs_contracting_dims" to listOf(1)),
        )
    }
}
