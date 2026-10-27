package io.tlaloc.autograd

import io.tlaloc.core.Shape
import io.tlaloc.ir.passes.DxirRemat

/**
 * Marks this value as a checkpoint: a gradient rematerialized with
 * [DxirRemat.SAVE_MARKED] keeps it from the forward pass and recomputes what lies between
 * checkpoints (JAX's `checkpoint_name`). Without rematerialization the mark changes nothing.
 * Only an op's result can be marked, not an input.
 */
fun <S : Shape> Tracer<S>.checkpoint(): Tracer<S> = Tracer(tape, tape.mark(id, DxirRemat.SAVE_ATTR))

/**
 * Marks this value as recomputed under every rematerialization policy ([DxirRemat.RECOMPUTE_ATTR]):
 * for a large result that is cheap to compute again, such as a chunk of an LM head's logits.
 * Without rematerialization the mark changes nothing.
 */
fun <S : Shape> Tracer<S>.recompute(): Tracer<S> = Tracer(tape, tape.mark(id, DxirRemat.RECOMPUTE_ATTR))
