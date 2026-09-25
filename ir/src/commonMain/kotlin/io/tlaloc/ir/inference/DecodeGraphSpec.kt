package io.tlaloc.ir.inference

import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.core.I64
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.recognizer.quant.KvQuantConfig

/**
 * The DECODE-GRAPH SHAPE CONTRACT.
 *
 * [io.tlaloc.ir.OpKind.PAGED_ATTENTION] and [io.tlaloc.ir.OpKind.KV_CACHE_WRITE]
 * supply the attention and the cache write; this contract is the *signature*
 * a serving loop calls: which tensors cross the host/device boundary, in what
 * order, at what static shapes, and how a batch of 3 runs in a graph compiled
 * for 4.
 *
 * ## Placement
 *
 * `:ir`, not `:runtime-pjrt`. The contract is expressed in [DxirType], it is
 * consumed by graph construction (`:nn`), by the emitter's shape checks, by
 * the `:maestro` `ProgramManifest` that IS the deployment artifact, and by
 * `:runtime-pjrt`'s executable cache — all of which already depend on `:ir`,
 * and only the last of which is JVM-only. REJECTED: a new `:inference` module
 * (three files and a circular pull on `:ir` for no isolation), and `:core`
 * (no [DxirType] there).
 *
 * ## The signature
 *
 * ```
 *   inputs
 *     0            tokenIds     [B, T]                                   I32
 *     1            positions    [B, T]                                   I32
 *     2            blockTables  [B, maxBlocksPerSeq]                     I32
 *     3            seqLens      [B]                                      I32
 *     4            slotMapping  [B * T]                                  I32
 *     5 .. 5+2L-1  kv pools     [numBlocks, blockSize, numKvHeads, headDim] × 2L
 *   outputs
 *     0            logits       [B, 1, vocabSize]      the last token of each sequence
 *     1 .. 2L      kv pools     the SAME shapes, updated
 * ```
 *
 * A model with a [WindowedKvPool] (sliding-window layers whose KV pages are
 * recycled) has two more scheduler inputs after `slotMapping`:
 * `windowBlockTables` `[B, maxBlocksPerSeq]` and `windowSlotMapping`
 * `[B * T]`, which the windowed layers read instead of `blockTables` and
 * `slotMapping`. Its windowed layers' pools are
 * `[windowedKv.numBlocks, blockSize, numKvHeads, headDim]`, with the roles
 * [DecodeSlotRole.WINDOW_KV_POOL_IN] and [DecodeSlotRole.WINDOW_KV_POOL_OUT];
 * the pools stay in layer order, so the pools start at input 7
 * ([kvPoolInputBase]).
 *
 * `T` is the token axis: **1 for decode**, the bucket's context width for
 * **prefill**. That is the whole difference — which is why this is one
 * contract with a [DecodeGraphKind] and not two.
 *
 * A prefill chunk is RIGHT-ALIGNED on the token axis: padding tokens first
 * (slot -1, position 0), then the real tokens, so the last real token of
 * every sequence is at `T - 1`, and the logits are that position's. A caller
 * samples from them; the other positions' logits are never needed, so they
 * are never computed or copied. The prefill graph derives each token's
 * causal context length from its position (`position + 1`) and does not
 * read `seqLens`, which stays in the signature so both kinds share one.
 *
 * Every shape is static per bucket. Nothing in the signature is derived from a
 * *value*: `seqLens` and `blockTables` are runtime tensors and the graph never
 * reads them at compile time, which is the sentinel-dims rule stated
 * from the serving side.
 *
 * ### Decisions
 *
 * - **Token IDs, not embeddings.** REJECTED: taking `[B, T, hiddenSize]`
 *   embeddings and leaving the embedding lookup host-side — it moves
 *   `B*T*hiddenSize` floats across PCIe every step instead of `B*T` ints, and
 *   the embedding table is already a device weight. Speculative decoding and
 *   multimodal prefixes do want an embedding entry point; that is not
 *   supported (it would be a second, sibling spec — not a mode flag on this one).
 * - **The KV pools cross the boundary as ordinary in/out operands**, two per
 *   layer, in `(key, value)` order per layer, layers ascending. This is what
 *   makes the graph FUNCTIONAL end to end (as [io.tlaloc.ir.OpKind.KV_CACHE_WRITE] is) and it is what
 *   XLA's buffer donation then makes free. REJECTED: hiding the pools in a
 *   side-channel resource the way a stateful runtime would — it buys nothing
 *   here and costs every pass a memory model.
 * - **Sampling is NOT in the graph.** The graph returns logits; the plugin
 *   samples host-side.
 *   REJECTED: fusing top-k/top-p/temperature in, because each sampling
 *   config would fork the executable cache on an axis that has nothing to do
 *   with shape.
 * - **`positions` is an operand, not derived from `seqLens`.** RoPE needs the
 *   absolute position of each token, and for a padded row there is no correct
 *   one to derive; making it an operand lets the padding convention state a
 *   value ([DecodePadding.PADDING_POSITION]) instead of computing a lie.
 *
 * ## The padding invariant
 *
 * See [DecodePadding]. In short: **a batch of 3 run in a bucket-4 graph
 * produces bit-identical logits for rows 0–2 against the same batch run in a
 * bucket-3 graph** — in the reference interpreter, pinned by
 * `DecodeGraphPaddingInvariantTest`. On device the same claim holds only to a
 * float tolerance, and [DecodePadding] says exactly why.
 */
data class DecodeGraphSpec(
    val model: DecodeModelShape,
    val bucket: DecodeBucket,
    val kind: DecodeGraphKind = DecodeGraphKind.DECODE,
    /**
     * STAGED WEIGHTS, appended to the signature after
     * the KV pools. Empty (the default) means the graph carries its weights as
     * body constants, as the small reference graphs do.
     *
     * A real checkpoint cannot do that. TinyLlama-1.1B is 1.1e9 parameters;
     * rendered as StableHLO `dense<[...]>` literals that is **tens of
     * gigabytes of TEXT** in a file whose whole premise is that it is
     * the deployment. So a real model's weights cross the boundary the same
     * way its KV pools do — as ordinary operands, staged once and reused
     * across every decode step.
     *
     * Three further reasons this is the right shape and not merely the
     * feasible one:
     *
     * - **One copy per model, not per bucket.** The ladder compiles six or
     *   more executables; constants would put a private copy of every weight
     *   in each one. Staged, the six executables share one set of device
     *   buffers.
     * - **It is what the checkpoint already is.** The weights are bytes in a
     *   safetensors file; staging is a read and an upload, and there is no
     *   step where they become program text at all.
     * - **`modelHash` keeps its meaning.** The executable stops depending on
     *   the weight VALUES, so two checkpoints of one architecture share a
     *   compile — and the cache key keeps the hash anyway, because the
     *   *artifact* is still per-checkpoint.
     *
     * The ORDER is fixed by whoever builds the graph (for Llama:
     * [HfDecoderGraph.weightSlots]) and is part of the contract, because a
     * loader binds by index.
     *
     * The serving artifact writes one file per slot, in this order, and
     * names them in its manifest's weight table.
     */
    val weightSlots: List<DecodeSlot> = emptyList(),
) {
    /** The token axis: 1 for decode, the bucket's context width for prefill. */
    val tokensPerSeq: Int = when (kind) {
        DecodeGraphKind.DECODE -> 1
        DecodeGraphKind.PREFILL -> bucket.maxContext
    }

    /** Block-table width, exact because context buckets are page-aligned. */
    val maxBlocksPerSeq: Int = bucket.maxBlocksPerSeq(model.blockSize)

    /** Tokens written to the pools this step: one slot each. */
    val totalTokens: Int = bucket.batch * tokensPerSeq

    val poolType: DxirType =
        DxirType(model.kvDtype, listOf(model.numBlocks, model.blockSize, model.numKvHeads, model.headDim))

    /** The type of a windowed layer's pool, or null without a [WindowedKvPool]. */
    val windowPoolType: DxirType? = model.windowedKv?.let {
        DxirType(model.kvDtype, listOf(it.numBlocks, model.blockSize, model.numKvHeads, model.headDim))
    }

    /** Whether layer [l] keeps its KV in the windowed pool class. */
    fun isWindowed(l: Int): Boolean = model.windowedKv?.layers?.contains(l) == true

    /** The pool type of layer [l]: [windowPoolType] for a windowed layer, else [poolType]. */
    fun poolTypeOf(l: Int): DxirType = if (isWindowed(l)) windowPoolType!! else poolType

    /** Index of the first KV pool input: after the five scheduler tensors, or seven with a windowed pool. */
    val kvPoolInputBase: Int = if (model.windowedKv == null) KV_POOL_INPUT_BASE else KV_POOL_INPUT_BASE + 2

    init {
        val w = model.windowedKv
        if (w != null) {
            val need = minOf(WindowedKvPool.minRingPages(w.window, model.blockSize), maxBlocksPerSeq)
            require(w.ringPages >= need) {
                "DecodeGraphSpec: a windowed ring of ${w.ringPages} pages cannot hold a window of " +
                    "${w.window} positions in pages of ${model.blockSize} at context " +
                    "${bucket.maxContext}; a decode step needs $need"
            }
        }
    }

    val tokenIdsType: DxirType = DxirType(I32, listOf(bucket.batch, tokensPerSeq))
    val positionsType: DxirType = DxirType(I32, listOf(bucket.batch, tokensPerSeq))
    val blockTablesType: DxirType = DxirType(I32, listOf(bucket.batch, maxBlocksPerSeq))
    val seqLensType: DxirType = DxirType(I32, listOf(bucket.batch))
    val slotMappingType: DxirType = DxirType(I32, listOf(totalTokens))
    /** `[B, 1, vocab]` for both kinds: the logits of each sequence's last token. */
    val logitsType: DxirType = DxirType(model.dtype, listOf(bucket.batch, 1, model.vocabSize))

    /** The full input signature, in call order. */
    val inputs: List<DecodeSlot> = buildList {
        add(DecodeSlot("tokenIds", tokenIdsType, DecodeSlotRole.TOKEN_IDS))
        add(DecodeSlot("positions", positionsType, DecodeSlotRole.POSITIONS))
        add(DecodeSlot("blockTables", blockTablesType, DecodeSlotRole.BLOCK_TABLES))
        add(DecodeSlot("seqLens", seqLensType, DecodeSlotRole.SEQ_LENS))
        add(DecodeSlot("slotMapping", slotMappingType, DecodeSlotRole.SLOT_MAPPING))
        if (model.windowedKv != null) {
            add(DecodeSlot("windowBlockTables", blockTablesType, DecodeSlotRole.WINDOW_BLOCK_TABLES))
            add(DecodeSlot("windowSlotMapping", slotMappingType, DecodeSlotRole.WINDOW_SLOT_MAPPING))
        }
        for (l in 0 until model.numLayers) {
            val role = if (isWindowed(l)) DecodeSlotRole.WINDOW_KV_POOL_IN else DecodeSlotRole.KV_POOL_IN
            add(DecodeSlot("keyCache$l", poolTypeOf(l), role))
            add(DecodeSlot("valueCache$l", poolTypeOf(l), role))
        }
        addAll(weightSlots)
    }

    /** The full result signature, in return order. */
    val outputs: List<DecodeSlot> = buildList {
        add(DecodeSlot("logits", logitsType, DecodeSlotRole.LOGITS))
        for (l in 0 until model.numLayers) {
            val role = if (isWindowed(l)) DecodeSlotRole.WINDOW_KV_POOL_OUT else DecodeSlotRole.KV_POOL_OUT
            add(DecodeSlot("keyCache${l}Out", poolTypeOf(l), role))
            add(DecodeSlot("valueCache${l}Out", poolTypeOf(l), role))
        }
    }

    /**
     * Input index ↔ output index for every pool, the pairs a runtime hands to
     * XLA as donated buffers. Input `i`
     * aliases output `o`; the pool is written in place and nothing is copied.
     */
    val donationPairs: List<Pair<Int, Int>>
        get() = (0 until 2 * model.numLayers).map { p -> Pair(kvPoolInputBase + p, 1 + p) }

    /**
     * The executable-cache key: what a serving process looks a compiled decode
     * step up by. Everything that changes the *program* is in here and nothing
     * that does not.
     *
     * [modelHash] is the content address of the weights+graph the manifest
     * already carries (`:maestro`'s `ProgramManifest` is content-addressed, so
     * this is a field it has, not one we invent). Two deployments of the same
     * architecture with different weights must not share a compiled step.
     *
     * REJECTED: keying on the emitted StableHLO text, which is what
     * `PjrtSession` does today. It is CORRECT — structurally identical graphs
     * hash together — but it costs a full re-emission of the program on every
     * lookup, which at decode rates is the one thing a cache is supposed to
     * avoid. This key is the cheap front door; the MLIR text stays the
     * back-stop.
     *
     * REJECTED: object identity of the [DxirFunction]. A plugin that rebuilds
     * the graph per request (the obvious thing to write) would miss every time
     * while holding a cache full of structurally identical entries.
     */
    fun executableCacheKey(modelHash: String): String {
        require(modelHash.isNotBlank()) {
            "DecodeGraphSpec.executableCacheKey: modelHash must not be blank — two deployments " +
                "of the same architecture with different weights must not share an executable"
        }
        return listOf(
            "tlaloc-decode-v1",
            modelHash,
            kind.name.lowercase(),
            "b${bucket.batch}",
            "c${bucket.maxContext}",
            "t$tokensPerSeq",
            "dt${model.dtype.name}",
            "kv${model.kvDtype.name}",
        ).plus(
            // A windowed pool changes the signature; artifacts without one keep their keys.
            listOfNotNull(model.windowedKv?.let { "w${it.window}r${it.ringPages}n${it.numBlocks}" }),
        ).joinToString("/")
    }

    /**
     * Check that a built [fn] actually matches this contract. The graph
     * builder is free code; this is the gate that says it produced the
     * signature the plugin will call. Refuses by name, naming the slot.
     */
    fun verifySignature(fn: DxirFunction, layer: String) {
        require(fn.params.size == inputs.size) {
            "$layer: decode graph '${fn.name}' has ${fn.params.size} params but the contract " +
                "for $this declares ${inputs.size} (${inputs.joinToString { it.name }})"
        }
        for ((i, slot) in inputs.withIndex()) {
            val p = fn.params[i]
            require(p.type.dtype == slot.type.dtype && p.type.dims == slot.type.dims) {
                "$layer: decode graph '${fn.name}' param $i ('${p.name}') has type ${p.type}, " +
                    "but the contract's slot '${slot.name}' is ${slot.type}"
            }
        }
        require(fn.returns.size == outputs.size) {
            "$layer: decode graph '${fn.name}' returns ${fn.returns.size} values but the " +
                "contract declares ${outputs.size} (logits + ${2 * model.numLayers} pools)"
        }
        for ((i, slot) in outputs.withIndex()) {
            val r = fn.returns[i]
            require(r.type.dtype == slot.type.dtype && r.type.dims == slot.type.dims) {
                "$layer: decode graph '${fn.name}' return $i has type ${r.type}, but the " +
                    "contract's slot '${slot.name}' is ${slot.type}"
            }
        }
    }

    override fun toString(): String =
        "DecodeGraphSpec(${kind.name.lowercase()}, $bucket, T=$tokensPerSeq, " +
            "layers=${model.numLayers}, dtype=${model.dtype.name})"

    companion object {
        /** First input index of the KV pools without a windowed pool — the five
         *  scheduler tensors come first, then `2 * numLayers` pools. With one,
         *  see [kvPoolInputBase]. */
        const val KV_POOL_INPUT_BASE: Int = 5
    }
}

/** DECODE runs one token per sequence; PREFILL runs the bucket's full token
 *  axis. Same contract, longer `T`. */
enum class DecodeGraphKind { DECODE, PREFILL }

/** One typed, named position in the decode graph's signature. */
data class DecodeSlot(val name: String, val type: DxirType, val role: DecodeSlotRole)

/** What a slot MEANS, so a plugin can bind by role rather than by index. */
enum class DecodeSlotRole {
    TOKEN_IDS, POSITIONS, BLOCK_TABLES, SEQ_LENS, SLOT_MAPPING,
    KV_POOL_IN, KV_POOL_OUT, LOGITS,

    /** The block tables of the [WindowedKvPool] layers: entry `b` is the ring page of logical block `b`. */
    WINDOW_BLOCK_TABLES,

    /** The slot mapping of the [WindowedKvPool] layers: each token's slot on its ring page. */
    WINDOW_SLOT_MAPPING,

    /** A [WindowedKvPool] layer's pool, as [KV_POOL_IN] is a full-history layer's. */
    WINDOW_KV_POOL_IN,

    /** A [WindowedKvPool] layer's updated pool, aliased to its [WINDOW_KV_POOL_IN]. */
    WINDOW_KV_POOL_OUT,

    /**
     * A model weight staged as an operand rather than baked in as a
     * body constant. See [DecodeGraphSpec.weightSlots] for why a real
     * checkpoint has no other option.
     */
    WEIGHT,
}

/**
 * The model-side constants of a decode step: everything that does NOT vary
 * with the bucket. Two specs that differ only in [DecodeBucket] describe the
 * same model at two compiled shapes.
 */
data class DecodeModelShape(
    val vocabSize: Int,
    val hiddenSize: Int,
    val numHeads: Int,
    val numKvHeads: Int,
    val headDim: Int,
    val numLayers: Int,
    val numBlocks: Int,
    val blockSize: Int,
    val dtype: DType = F32,
    /** The KV pools' dtype as the decode graph's boundary carries it. Equal to
     *  [dtype] for an unquantized pool; under [kvQuant] it is the CODES'
     *  integer dtype (I32 in v1 — see [kvQuant]). */
    val kvDtype: DType = dtype,
    /**
     * The KV-quant format, or null for a float pool.
     *
     * Non-null means the pools this model's graphs read are
     * [io.tlaloc.ir.inference.KvQuantPool]-quantized: integer codes in
     * [kvDtype] plus a per-head (or per-tensor) scale vector, read back
     * through [io.tlaloc.ir.OpKind.DEQUANTIZE_KV]. This is the field the
     * serving manifest's `kvQuantDtype` slot is fed
     * from, and the one an artifact consumer reads to know that a pool buffer
     * is codes and not values — a distinction no tensor type carries.
     *
     * Only the INTEGER-CODED formats are admitted (int8/int4); fp8 is refused
     * by name here for the same reason it is refused in the codec and the op.
     */
    val kvQuant: KvQuantConfig? = null,
    /**
     * The sliding-window layers whose KV pages are recycled, or null when
     * every layer keeps full-history pages. See [WindowedKvPool].
     */
    val windowedKv: WindowedKvPool? = null,
) {
    init {
        require(vocabSize >= 1 && hiddenSize >= 1 && headDim >= 1) {
            "DecodeModelShape: vocabSize/hiddenSize/headDim must be >= 1, got " +
                "$vocabSize/$hiddenSize/$headDim"
        }
        require(numKvHeads >= 1 && numHeads >= 1 && numHeads % numKvHeads == 0) {
            "DecodeModelShape: numHeads $numHeads must be a positive multiple of numKvHeads " +
                "$numKvHeads (the GQA grouping PAGED_ATTENTION derives from operand shapes)"
        }
        require(numLayers >= 1) { "DecodeModelShape: numLayers must be >= 1, got $numLayers" }
        require(numBlocks >= 1 && blockSize >= 1) {
            "DecodeModelShape: numBlocks/blockSize must be >= 1, got $numBlocks/$blockSize"
        }
        windowedKv?.let { w ->
            require(w.layers.last() < numLayers) {
                "DecodeModelShape: windowed layers ${w.layers} name a layer outside 0..${numLayers - 1}"
            }
        }
        val q = kvQuant
        if (q != null) {
            require(q.dtype.isIntegerCoded) {
                "DecodeModelShape: kvQuant ${q.dtype.nameTag} is refused BY NAME — the KV-quant " +
                    "contract (KvQuantPool, DEQUANTIZE_KV) is integer-coded " +
                    "(value = code * scale), and a float format's code is a bit pattern; fp8 " +
                    "pools need a narrow DType of their own, as bf16 has"
            }
            require(kvDtype == I32 || kvDtype == I64) {
                "DecodeModelShape: a ${q.dtype.nameTag}-quantized pool carries integer CODES, so " +
                    "kvDtype must be an integer dtype, got $kvDtype (v1 rides I32 — there is no " +
                    "I8 DType yet, which is why the artifact states the code dtype explicitly " +
                    "instead of letting a reader assume the pool is byte-narrow)"
            }
        } else {
            require(kvDtype != I32 && kvDtype != I64) {
                "DecodeModelShape: kvDtype $kvDtype is an integer dtype but kvQuant is null — an " +
                    "integer KV pool with no quantization format is a pool nobody can read " +
                    "(the scales and the code range are exactly what kvQuant carries)"
            }
        }
    }

    /** The format tag the serving manifest publishes, or null for a float pool. */
    val kvQuantDtypeTag: String? get() = kvQuant?.dtype?.nameTag

    /** GQA grouping: query heads per kv head. */
    val group: Int get() = numHeads / numKvHeads
}

/**
 * The PADDING AND MASKING CONVENTION, and the invariant it buys.
 *
 * A bucket-4 graph running a batch of 3 has one slack row. The convention says
 * what goes in it, and the invariant says what that costs the three real rows:
 * **nothing at all.**
 *
 * | tensor | padding row value | why |
 * |---|---|---|
 * | `tokenIds` | [PADDING_TOKEN_ID] = 0 | any in-vocab id; the row is discarded |
 * | `positions` | [PADDING_POSITION] = 0 | a legal RoPE position; no NaN from a rotation |
 * | `blockTables` | [PADDING_BLOCK] = 0 | must be an ALLOCATED page: the gather is in-bounds or XLA's behaviour is on us |
 * | `seqLens` | [PADDING_SEQ_LEN] = 1 | **not 0** — see below |
 * | `slotMapping` | [PADDING_SLOT] = -1 | the KV_CACHE_WRITE convention: the write is dropped |
 *
 * ### Why `seqLens = 1` and never 0
 *
 * This is the one genuinely load-bearing choice. `seqLens[s] = 0` masks every
 * context lane of sequence `s` to −Inf, and a softmax whose every term is
 * `exp(-Inf - (-Inf))` is `0/0` — **NaN**. The reference interpreter takes a
 * `len == 0` early-out and leaves that row zero; the gather-composed emission
 * has no such arm and produces NaN. The PAGED_ATTENTION emitter documents this as the
 * single degenerate case where the two disagree. A padding convention that
 * routed through it would make every padded batch differ between the oracle
 * and the device, and would put NaN in a device buffer that any later fused
 * cross-row reduction (a batched layernorm, a top-k over the batch) would
 * spread into the real rows.
 *
 * `seqLens = 1` instead: the padded row attends to exactly one KV position,
 * reads whatever page 0 holds, and produces finite garbage that is discarded.
 * REJECTED: making the emission match the interpreter's zero arm with a
 * `select` on `seqLens == 0` — it puts a branch in the hot path of every
 * sequence to serve a row whose output is thrown away.
 *
 * Page 0 is the conventional scratch page, and vLLM reserves a null block for
 * the same reason. Correctness here does not DEPEND on the reservation (the
 * row is discarded either way) — but a deployment that reserves it gets a
 * padded row that reads nothing live, which is worth having when profiling.
 *
 * ### The invariant
 *
 * Every op in the decode contract is **row-independent along the batch axis**:
 * the embedding lookup is a per-position gather, RoPE is per-position,
 * `PAGED_ATTENTION` walks each sequence with its own `seqLens`/`blockTables`
 * row, `KV_CACHE_WRITE` writes disjoint slots (and none at all for `-1`), and
 * the logits projection contracts the hidden axis, never the batch. So row `r`
 * of the output is a function of row `r` of the input and of the weights —
 * and of nothing else, in particular not of `B`.
 *
 * Therefore, in the reference interpreter, rows 0..n-1 of a batch of `n` run
 * in a bucket-`B` graph are **BIT-IDENTICAL** to the same rows run in a
 * bucket-`n` graph, and the updated pools are bit-identical too. This is
 * pinned elementwise with `==` in `DecodeGraphPaddingInvariantTest`.
 *
 * ### What is NOT claimed on device
 *
 * Bit-identity is an *interpreter* claim. Two buckets are two compiled
 * executables, and XLA is free to tile a `[4, H] × [H, V]` matmul differently
 * from a `[3, H] × [H, V]` one — a different tiling is a different
 * accumulation order, and floating-point addition is not associative. The
 * device claim is agreement to the GPU-vs-host floor, **~4e-5,
 * never pinned tighter than 1e-4**.
 */
object DecodePadding {
    /** `slotMapping` for a padding token: KV_CACHE_WRITE drops the write. */
    const val PADDING_SLOT: Int = -1

    /** `seqLens` for a padding row. NOT 0 — see the object doc. */
    const val PADDING_SEQ_LEN: Int = 1

    /** `blockTables` for a padding row: page 0, the conventional scratch page.
     *  Must be an allocated page so the gather stays in bounds. */
    const val PADDING_BLOCK: Int = 0

    /** `tokenIds` for a padding token: any in-vocab id. */
    const val PADDING_TOKEN_ID: Int = 0

    /** `positions` for a padding token: a legal RoPE position. */
    const val PADDING_POSITION: Int = 0

    /**
     * Pad [real] out to [width] with [pad]. Refuses BY NAME when the request
     * does not fit — a batch larger than its bucket is a bucket-selection bug
     * upstream, and silently truncating it would drop a user's sequence.
     */
    fun padTo(real: IntArray, width: Int, pad: Int, what: String): IntArray {
        require(real.size <= width) {
            "DecodePadding.padTo: $what has ${real.size} entries but the bucket is $width wide — " +
                "the bucket must be chosen to COVER the request (DecodeBucketPolicy.bucketFor)"
        }
        return IntArray(width) { if (it < real.size) real[it] else pad }
    }

    /** Float twin of [padTo]; the pad value is 0f because every float tensor
     *  in the contract is a discarded row, never a masked one. */
    fun padTo(real: FloatArray, width: Int, what: String): FloatArray {
        require(real.size <= width) {
            "DecodePadding.padTo: $what has ${real.size} entries but the bucket is $width wide"
        }
        return FloatArray(width) { if (it < real.size) real[it] else 0f }
    }
}
