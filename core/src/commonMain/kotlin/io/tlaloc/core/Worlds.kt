package io.tlaloc.core

/**
 * **Four-worlds taxonomy**, the structural backbone
 * separating pure tensor compute from buffer management from multi-step
 * orchestration from cluster lifecycle.
 *
 * # The four worlds
 *
 * Each world is a Kotlin scope ([KernelScope], [OrchestrationScope],
 * [ProgramScope], [ClusterScope]). Ops legal in one world are illegal in
 * others; cross-world calls fail to compile because the target function's
 * receiver type doesn't match the implicit receiver in scope. This is
 * enforced by Kotlin's native overload resolution + receiver-only DSL —
 * no plugin extension required.
 *
 * | World | Concerns | Examples |
 * |-------|----------|----------|
 * | [KernelScope] | Pure tensor compute, no I/O, no side effects. | `valueAndGrad`, named-index `contract`, activations, reductions. Lowers to a StableHLO body. |
 * | [OrchestrationScope] | Buffer management, dispatch, futures. | `compile`, `dispatch`, `await`, `program { }`. One `program {}` artifact lives here. |
 * | [ProgramScope] | Multi-step DAGs, pipeline stages, cross-island composition. | `step()` composition inside a `workflow { }`, mesh transfers, `@Pipeline` annotations. |
 * | [ClusterScope] | Multi-node lifecycle. | `connect`, `discoverTopology`, `barrier`, fault-handler hooks. Initializes the device sets the other worlds use. |
 *
 * # Why DslMarker over context parameters
 *
 * Tlaloc uses **DslMarker-based receiver-only scoping**
 * over Kotlin 2.2's Beta `context` parameters because:
 *
 * - Each Tlaloc op has exactly one valid scope (no multi-scope composition
 *   identified). Receiver-only is sufficient; multi-receiver context-
 *   parameter machinery isn't needed.
 * - DslMarker is a mature, no-flag idiom — no `-Xcontext-parameters` opt-in.
 * - Cross-scope calls produce native Kotlin "Unresolved reference" errors
 *   pointed at the call site, no plugin diagnostic required.
 * - Readers of the IDE error don't need to know about scope semantics —
 *   they see "function X is not callable here" with normal IDE highlighting.
 *
 * The trade-off: a function legal in *both* Kernel and Orchestration
 * scopes would require either two overloads or one declared on a
 * supertype. v1 keeps each op single-scope; if multi-scope ops surface
 * later, we revisit (could migrate to context parameters then if the Beta
 * stabilises around Kotlin 2.3).
 *
 * # Scope nesting
 *
 * Builders open scopes inside their bodies:
 *
 * ```kotlin
 * fun OrchestrationScope.program(body: KernelScope.() -> Unit): MaestroStep<...>
 * fun ProgramScope.workflow(body: ProgramScope.() -> Unit): MaestroWorkflow
 * ```
 *
 * Inside `program { }` the receiver is [KernelScope], so kernel-only ops
 * (`contract`, `relu`, etc.) compile and orchestration-only ops
 * (`dispatch`) do not. The `@DslMarker` annotation prevents the outer
 * `OrchestrationScope` receiver from leaking implicit-call access into
 * the inner Kernel block — a `@DslMarker`-tagged receiver shadows other
 * `@DslMarker`-tagged receivers when they're both in scope.
 */

/**
 * DslMarker for the four-world scoping. Tagged on every [KernelScope] /
 * [OrchestrationScope] / [ProgramScope] / [ClusterScope] interface so
 * Kotlin's compiler enforces that one DslMarker-tagged receiver shadows
 * another in nested builders.
 */
@DslMarker
@Target(AnnotationTarget.CLASS, AnnotationTarget.TYPE)
annotation class WorldScope

/**
 * Pure-functional tensor-compute world. The `program { }` builder opens
 * this scope inside its body lambda; all kernel ops (`valueAndGrad`,
 * named-index `contract`, activations, reductions) are declared as
 * extension functions on this interface.
 *
 * No I/O, no side effects. Anything that touches buffers, dispatch, or
 * futures lives in [OrchestrationScope]; anything that touches a cluster
 * lives in [ClusterScope].
 */
@ExperimentalTlalocApi
@WorldScope
interface KernelScope

/**
 * Buffer-management and dispatch world. Owns a `program { }` artifact's
 * lifecycle: compilation, dispatch to a runtime, future awaiting,
 * BufferHandle lifecycle (release).
 *
 * `program { }` is callable from this scope and opens a [KernelScope]
 * inside its body. `dispatch`, `compile`, `await` live here.
 */
@ExperimentalTlalocApi
@WorldScope
interface OrchestrationScope

/**
 * Multi-step DAG / workflow world. `workflow { }` is callable from this
 * scope; inside the workflow body, [step] composition produces a
 * [io.tlaloc.maestro.MaestroWorkflow] artifact. Cross-mesh transfers and
 * pipeline stages also live here.
 */
@ExperimentalTlalocApi
@WorldScope
interface ProgramScope

/**
 * Multi-node lifecycle world. Initializes the device sets that the other
 * worlds use — cluster connection, topology discovery, barriers, fault-
 * handler hook registration.
 *
 * ClusterScope is currently a marker only; it declares no operations yet.
 */
@ExperimentalTlalocApi
@WorldScope
interface ClusterScope

// --------------------------------------------------------------------------
// Default scope instances
// --------------------------------------------------------------------------

/**
 * Default [OrchestrationScope] singleton. Top-level entry point: a user
 * who wants to build a `program { }` artifact starts from `with(Tlaloc) { … }`
 * (or a function that takes [OrchestrationScope] as receiver). The
 * default singleton has no state — concrete orchestration runtimes
 * (e.g. an IREE-backed dispatcher) subtype this.
 */
@ExperimentalTlalocApi
object Tlaloc : OrchestrationScope, ProgramScope

// --------------------------------------------------------------------------
// Smoke ops on each scope (Layer 2.0 substrate placeholders).
//
// These exist primarily so the compile-fail tests in :compiler-plugin have
// concrete targets to assert against — calling [kernelMarker] outside a
// [KernelScope] receiver doesn't compile, and the test verifies that.
//
// Real ops attach to scopes in subsequent phases:
// - Layer 2.1 wires `program { }` on [OrchestrationScope].
// - Layer 2.2 wires `workflow { }` and `step()` on [ProgramScope].
// - Layer 3+ attaches concrete dispatch / barrier ops on [OrchestrationScope]
//   and [ClusterScope].
// --------------------------------------------------------------------------

/** Smoke marker, callable only from [KernelScope]. */
@ExperimentalTlalocApi
fun KernelScope.kernelMarker(): String = "kernel"

/** Smoke marker, callable only from [OrchestrationScope]. */
@ExperimentalTlalocApi
fun OrchestrationScope.orchestrationMarker(): String = "orchestration"

/** Smoke marker, callable only from [ProgramScope]. */
@ExperimentalTlalocApi
fun ProgramScope.programMarker(): String = "program"

/** Smoke marker, callable only from [ClusterScope]. */
@ExperimentalTlalocApi
fun ClusterScope.clusterMarker(): String = "cluster"
