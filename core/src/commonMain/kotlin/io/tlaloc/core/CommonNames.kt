package io.tlaloc.core

/**
 * Pre-defined [IndexName] singletons for common ML tensor axes.
 *
 * Importing this package makes the standard names usable without per-project
 * boilerplate:
 *
 *     typealias Activations =
 *         DTensor<Rank2<Named<Batch, Sym>, Named<SeqLen, Sym>>, F32>
 *
 * User code is free to define additional [IndexName] singletons elsewhere; the
 * K2 plugin keys off the singleton's class FQN, not membership in this file.
 * Names here are deliberately conservative — broadly recognisable in ML
 * literature, no domain-specific naming.
 */

/** Batch axis (training-data parallelism). */
object Batch : IndexName { override val name = "batch" }

/** Sequence-length axis (transformers, RNNs). */
object SeqLen : IndexName { override val name = "seq" }

/** Hidden-representation axis. */
object Hidden : IndexName { override val name = "hidden" }

/** Attention-heads axis (multi-head attention). */
object Heads : IndexName { override val name = "heads" }

/** Per-head feature axis (often `d_head` in the literature). */
object Dim : IndexName { override val name = "dim" }

/** Vocabulary axis (logits over tokens). */
object Vocab : IndexName { override val name = "vocab" }

/** Channel axis (CNNs). */
object Channel : IndexName { override val name = "channel" }

/** Spatial-height axis (CNNs / image data). */
object Height : IndexName { override val name = "height" }

/** Spatial-width axis (CNNs / image data). */
object Width : IndexName { override val name = "width" }
