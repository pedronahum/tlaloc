package io.tlaloc.ir.inference

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirType

/**
 * The per-sequence state of a model's linear-attention (Gated DeltaNet)
 * layers: which layers they are, and the two pools each keeps in place of
 * a KV pool pair, both f32 and both indexed by a sequence's state slot.
 *
 * - `convState$l`      `[numSlots, convKernel - 1, convChannels]`: the last
 *   K-1 inputs of the layer's causal conv ([io.tlaloc.ir.OpKind.CAUSAL_CONV1D]);
 * - `recurrentState$l` `[numSlots, valueHeads, keyDim, valueDim]`: the delta
 *   rule's state ([io.tlaloc.ir.OpKind.GATED_DELTA_RULE]).
 *
 * A sequence holds one slot from its first prefill call to its end, whatever
 * its length; [numSlots] is how many sequences can be live at once. The
 * serving loop hands each row's slot in the `stateSlots` input. A sequence
 * whose first token is at position 0 starts from zero state, so a freed slot
 * needs no clearing.
 */
data class LinearStatePool(
    /** The linear-attention layers, ascending. */
    val layers: List<Int>,
    val numSlots: Int,
    val convChannels: Int,
    val convKernel: Int,
    val valueHeads: Int,
    val keyDim: Int,
    val valueDim: Int,
) {
    init {
        require(layers.isNotEmpty() && layers == layers.sorted().distinct() && layers.first() >= 0) {
            "LinearStatePool: layers must be distinct, ascending and non-negative, got $layers"
        }
        require(numSlots >= 1 && convChannels >= 1 && convKernel >= 2) {
            "LinearStatePool: numSlots >= 1, convChannels >= 1 and convKernel >= 2, got " +
                "$numSlots/$convChannels/$convKernel"
        }
        require(valueHeads >= 1 && keyDim >= 1 && valueDim >= 1) {
            "LinearStatePool: valueHeads/keyDim/valueDim must be >= 1, got $valueHeads/$keyDim/$valueDim"
        }
    }

    val convStateType: DxirType get() = DxirType(F32, listOf(numSlots, convKernel - 1, convChannels))

    val recurrentStateType: DxirType get() = DxirType(F32, listOf(numSlots, valueHeads, keyDim, valueDim))

    /** Bytes one sequence's state takes, over every linear layer. */
    val bytesPerSequence: Long
        get() = layers.size.toLong() * 4L * ((convKernel - 1L) * convChannels + valueHeads.toLong() * keyDim * valueDim)
}
