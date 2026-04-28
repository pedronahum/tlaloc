package io.tlaloc.core

/**
 * Pre-defined [MeshDim] singletons for common parallelism axes (Layer 2
 * §0.4.243+). Mirrors the `CommonNames.kt` convention from Layer 1 for
 * tensor-axis names. User code may declare additional axes elsewhere.
 */

/** Data-parallel axis: replicate model, partition the batch. */
object DataAxis : MeshDim { override val name = "data" }

/** Model-parallel axis: partition model parameters across devices. */
object ModelAxis : MeshDim { override val name = "model" }

/** Pipeline-parallel axis: partition layer stages across devices. */
object PipelineAxis : MeshDim { override val name = "pipeline" }

/** Tensor-parallel axis: partition tensor dims across devices (Megatron-style). */
object TensorAxis : MeshDim { override val name = "tensor" }

/** Expert-parallel axis: partition MoE experts across devices. */
object ExpertAxis : MeshDim { override val name = "expert" }

/** Replica axis: redundant placement for failure-tolerance. */
object ReplicaAxis : MeshDim { override val name = "replica" }
