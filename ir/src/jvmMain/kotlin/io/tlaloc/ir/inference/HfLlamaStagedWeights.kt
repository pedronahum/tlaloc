package io.tlaloc.ir.inference

import io.tlaloc.core.io.JsonException

// §0.4.479 (Phase H3c-2) — the bridge from §0.4.478's "a tensor by role" to
// "the operand list a decode graph is called with". One responsibility, and it
// is the one the whole slice turns on: THE TRANSPOSE.

/**
 * Stage a checkpoint's weights into [HfLlamaDecodeGraph.weightSlots] order.
 *
 * ## The transpose, performed
 *
 * §0.4.478 established, against the real file, that HF stores every
 * `nn.Linear` weight as `[out_features, in_features]` — because
 * `F.linear(x, W)` is `x @ W.T`. Tlaloc's [io.tlaloc.ir.OpKind.MATMUL]
 * contracts `last(A) × first(B)`, so a projection wants `[in, out]`. This
 * function is where the two meet, and it is deliberately the ONLY place: the
 * graph has no transpose node, and every downstream reader of a staged buffer
 * can assume math layout.
 *
 * WHY HOST-SIDE AND ONCE. The weight is a graph PARAMETER (see
 * [DecodeGraphSpec.weightSlots]), so an in-graph
 * [io.tlaloc.ir.OpKind.TRANSPOSE] has no constant to fold into and would
 * re-lay-out every projection matrix on every decode step — at TinyLlama's
 * shapes, ~1.1e9 element moves per token, to save a one-off pass at load.
 *
 * WHICH TENSORS. Exactly the ones [HfLlamaNames.isTransposedLinear] says: the
 * seven projections per layer and `lm_head`. The two RMSNorm gains are rank-1
 * and the embedding table is a LOOKUP, not a Linear — transposing either would
 * be a bug that a square model could not detect, which is why the inventory
 * check and this function both go through the same predicate rather than a
 * second list.
 *
 * ## Dtype
 *
 * Everything is staged as **f32**. TinyLlama's bytes are bf16, and
 * [io.tlaloc.core.io.WeightSource] already widens them exactly (§0.4.468:
 * bf16→f32 is lossless, it is a 16-bit left shift). The graph is f32 because
 * [DecodeGraphSpec]'s default `dtype` is and because the reference
 * interpreter is; a bf16 decode graph is the G1 path and a named tail, not
 * this slice. The consequence is honest and stated in the parity lane: we
 * compute in f32 from bf16-exact inputs, and the oracle is asked to do the
 * same, so a disagreement is arithmetic ORDER and not width.
 */
object HfLlamaStagedWeights {

    /**
     * Read every slot of [HfLlamaDecodeGraph.weightSlots] for [config] out of
     * [ckpt], transposing the Linears, and return them in call order.
     *
     * [config] may be a REDUCED copy of the checkpoint's own config (fewer
     * layers); the roles it implies are then a prefix-by-layer of the file's,
     * which is exactly what a reduced-layer certification wants. Everything
     * else — hidden size, head counts, vocab — must still match, and [load]
     * refuses per tensor if it does not because §0.4.478's dim check is
     * against the config it is handed.
     */
    fun stage(ckpt: HfLlamaCheckpoint, config: HfLlamaConfig = ckpt.config): List<FloatArray> {
        require(config.numLayers <= ckpt.config.numLayers) {
            "HfLlamaStagedWeights.stage: asked for ${config.numLayers} layers but the " +
                "checkpoint has ${ckpt.config.numLayers}"
        }
        val slots = HfLlamaDecodeGraph.weightSlots(config)
        val roles = HfLlamaDecodeGraph.weightRoles(config)
        check(slots.size == roles.size) {
            "HfLlamaStagedWeights: ${slots.size} slots vs ${roles.size} roles — " +
                "HfLlamaDecodeGraph.weightSlots and .weightRoles must stay in lockstep"
        }
        return slots.indices.map { i ->
            val role = roles[i]
            val slot = slots[i]
            val t = loadFor(ckpt, role, config)
            val data = t.toF32Array()
            val fileDims = t.dims
            val staged = if (HfLlamaNames.isTransposedLinear(role)) {
                require(fileDims.size == 2) {
                    "HfLlamaStagedWeights: $role is a Linear but its file tensor is " +
                        "${fileDims.toList()} — expected rank 2"
                }
                transpose(data, fileDims[0], fileDims[1])
            } else {
                data
            }
            val want = slot.type.dims.fold(1) { a, b -> a * b }
            check(staged.size == want) {
                "HfLlamaStagedWeights: slot '${slot.name}' wants $want elements " +
                    "(${slot.type.dims}) but $role staged ${staged.size} from " +
                    "${fileDims.toList()}"
            }
            staged
        }
    }

    /**
     * Load one role, verifying against [config] rather than the checkpoint's
     * own — so a reduced-layer config still gets §0.4.478's dim check, and a
     * reduced config that quietly disagrees about hidden size or head count
     * is caught at the tensor, by name, instead of inside a matmul.
     */
    private fun loadFor(
        ckpt: HfLlamaCheckpoint,
        role: LlamaWeightRole,
        config: HfLlamaConfig,
    ): io.tlaloc.core.io.LoadedTensor {
        val t = ckpt.load(role)
        val want = HfLlamaNames.expectedDims(role, config)
        if (!t.dims.contentEquals(want)) {
            throw JsonException(
                "HfLlamaStagedWeights: ${ckpt.resolveName(role)} is ${t.dims.toList()} but the " +
                    "config this graph is being built for says ${want.toList()} — the reduced " +
                    "config disagrees with the checkpoint about something other than layer count",
            )
        }
        return t
    }

    /** Row-major `[rows, cols]` → `[cols, rows]`. */
    fun transpose(src: FloatArray, rows: Int, cols: Int): FloatArray {
        require(src.size == rows * cols) {
            "HfLlamaStagedWeights.transpose: ${src.size} elements is not $rows x $cols"
        }
        val out = FloatArray(src.size)
        for (r in 0 until rows) {
            val base = r * cols
            for (c in 0 until cols) out[c * rows + r] = src[base + c]
        }
        return out
    }
}
