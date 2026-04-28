package io.tlaloc.core

/**
 * Layer 2 §0.4.243+ — phantom-typed device-mesh markers for compile-time
 * placement tracking on [BufferHandle].
 *
 * These types are entirely separate from [MeshSpec] (the runtime device-
 * topology value class). [MeshSpec] describes a mesh's name + axes + sizes
 * at runtime; [Mesh] is a *type-level* placement marker that flows through
 * step-boundary signatures so a `BufferHandle<T, Mesh1<DataAxis>>` can't
 * accidentally be passed to a step expecting `BufferHandle<T, Mesh1<ModelAxis>>`.
 *
 * The pattern mirrors Layer 1's [IndexName] / [Named]:
 *
 * - [MeshDim] is a non-sealed marker interface. User-defined subtypes are
 *   one-line `object MyAxis : MeshDim { override val name = "my-axis" }`
 *   declarations.
 * - [Mesh] is a sealed interface with one concrete subtype per rank
 *   ([Mesh0], [Mesh1], [Mesh2], [Mesh3], [Mesh4]) — same shape as the
 *   `RankN` family for tensor shapes.
 *
 * A workflow that reshards across two distinct phantom meshes inserts an
 * explicit reshard step (Layer 2.2 / §0.4.243's workflow builder); the
 * type system catches mesh mismatches at compose time.
 */

/**
 * Type-level marker for a named mesh axis (e.g. "data parallel", "model
 * parallel", "pipeline parallel"). Mirrors [IndexName] from Layer 1 — the
 * interface is non-sealed so user code can declare additional axes.
 */
interface MeshDim {
    val name: String
}

/** Phantom-typed device mesh. Layer 2's compile-time placement marker. */
sealed interface Mesh

/** Zero-dimensional ("singleton") mesh — a single device, no parallelism. */
data object Mesh0 : Mesh

/** One-axis mesh, e.g. `Mesh1<DataAxis>` for pure data-parallel placement. */
class Mesh1<A0 : MeshDim> : Mesh

/** Two-axis mesh, e.g. `Mesh2<DataAxis, ModelAxis>`. */
class Mesh2<A0 : MeshDim, A1 : MeshDim> : Mesh

/** Three-axis mesh, e.g. `Mesh3<DataAxis, ModelAxis, PipelineAxis>`. */
class Mesh3<A0 : MeshDim, A1 : MeshDim, A2 : MeshDim> : Mesh

/**
 * Four-axis mesh. Higher arities can be added as the need arises; v1.x
 * caps at 4 to mirror the rank-4 tensor surface from Layer 1.5.
 */
class Mesh4<A0 : MeshDim, A1 : MeshDim, A2 : MeshDim, A3 : MeshDim> : Mesh
