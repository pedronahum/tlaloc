package io.tlaloc.ir.inference

// The names these types had before a second model family (Qwen3) was added.
// Kept so code written against 0.1.0-alpha02 still compiles.

/** Former name of [HfDecoderConfig]. */
@Deprecated("Renamed when Qwen3 was added", ReplaceWith("HfDecoderConfig"))
typealias HfLlamaConfig = HfDecoderConfig

/** Former name of [HfDecoderNames]. */
@Deprecated("Renamed when Qwen3 was added", ReplaceWith("HfDecoderNames"))
typealias HfLlamaNames = HfDecoderNames

/** Former name of [DecoderWeightRole]. */
@Deprecated("Renamed when Qwen3 was added", ReplaceWith("DecoderWeightRole"))
typealias LlamaWeightRole = DecoderWeightRole

/** Former name of [DecoderLayerPart]. */
@Deprecated("Renamed when Qwen3 was added", ReplaceWith("DecoderLayerPart"))
typealias LlamaLayerPart = DecoderLayerPart

/** Former name of [HfDecoderGraph]. */
@Deprecated("Renamed when Qwen3 was added", ReplaceWith("HfDecoderGraph"))
typealias HfLlamaDecodeGraph = HfDecoderGraph
