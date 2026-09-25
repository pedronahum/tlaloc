package io.tlaloc.ir.inference

// The names these types had before a second model family (Qwen3) was added.
// Kept so code written against 0.1.0-alpha02 still compiles.

/** Former name of [HfCheckpoint]. */
@Deprecated("Renamed when Qwen3 was added", ReplaceWith("HfCheckpoint"))
typealias HfLlamaCheckpoint = HfCheckpoint

/** Former name of [HfStagedWeights]. */
@Deprecated("Renamed when Qwen3 was added", ReplaceWith("HfStagedWeights"))
typealias HfLlamaStagedWeights = HfStagedWeights
