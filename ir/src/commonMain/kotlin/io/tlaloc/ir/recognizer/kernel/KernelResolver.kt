package io.tlaloc.ir.recognizer.kernel

import kotlin.concurrent.Volatile
import kotlin.jvm.Synchronized
import io.tlaloc.core.ExperimentalTlalocApi

/**
 * Backend-agnostic resolution of `kernel_descriptor`
 * names to concrete implementations.
 *
 * # What this is
 *
 * The shared substrate that
 * the IREE and PJRT-XLA backends both register against — without
 * this, each backend would either bake its own kernel-name → dispatch
 * map into the interpreter (creating two divergent code paths to drift)
 * or share a JNI-libtorch assumption.
 *
 * # The contract
 *
 * A [KernelResolver] knows how to satisfy *some* kernel-name + target
 * pair on its own backend. The recognized COARSENED ops produced by
 * `coarsenRecognizedPatterns` carry a `kernel_descriptor` whose
 * `kernel_name` field (e.g. `"flash_attn_v3"`, `"rms_norm_v1"`,
 * `"swiglu_fused"`) is the lookup key.
 *
 * The shared [KernelResolverRegistry] holds zero or more registered
 * resolvers. At lowering time:
 *
 * - Single-backend deployment: one resolver registered; `preferred(...)`
 *   returns its resolution or null.
 * - Dual-backend deployment: both IREE + PJRT-XLA
 *   registered; `resolveAll(...)` returns both options for a caller to
 *   choose between.
 *
 * # Why not ServiceLoader
 *
 * Tlaloc's `:ir` module is Kotlin Multiplatform (commonMain). JVM's
 * `ServiceLoader` is JVM-specific and would require an `expect/actual`
 * dance for what is — at the API level — a single-line register call.
 * Explicit registration also keeps test isolation simple ([clear] at
 * the top of each test) and avoids classpath-resolution surprises in
 * single-jar deployments.
 *
 * A future JVM-side helper module can wrap `ServiceLoader` over this
 * interface for production deployments that want auto-discovery.
 */
@ExperimentalTlalocApi
interface KernelResolver {
    /**
     * Stable identifier for the backend (e.g. `"iree"`, `"pjrt-xla"`).
     * Multiple resolvers can share the same backend id only if they
     * cover disjoint kernel-name + target sets — the registry doesn't
     * dedupe.
     */
    val backendId: String

    /**
     * Quick-check: does this resolver claim to satisfy [kernelName] on
     * [target] at all? Used by [resolve] internally and by tests.
     */
    fun supports(kernelName: String, target: KernelTarget): Boolean

    /**
     * Build a concrete resolution for [kernelName] on [target] with the
     * given [attrs] (the COARSENED's `customCallAttrs` — e.g. the
     * `supported_kv_dtypes` list a FlashAttention v3 variant declares).
     * Returns null if this resolver cannot satisfy the request — either
     * because it doesn't support the kernel/target, or because the
     * attrs are incompatible with what this backend exposes.
     */
    fun resolve(
        kernelName: String,
        target: KernelTarget,
        attrs: Map<String, Any> = emptyMap(),
    ): KernelResolution?
}

/**
 * The output of a [KernelResolver.resolve] call: one backend's plan for
 * how a given `kernel_descriptor` will be dispatched at runtime.
 *
 * @property backendId the id of the resolver that produced this
 *   resolution (== [KernelResolver.backendId]).
 * @property implementation backend-specific dispatch identifier
 *   (e.g. `"iree.cudnn_flash_attention"` for an IREE custom dispatch
 *   wrapping cuDNN, or `"xla.custom_call.flash_attention_v3"` for an
 *   XLA custom-call name).
 * @property notes free-form metadata the cost model / debug printer can
 *   consume — e.g. `"requires_sm" to "100"` to flag a Blackwell-only
 *   path, or `"fp8_supported" to true`.
 */
@ExperimentalTlalocApi
data class KernelResolution(
    val backendId: String,
    val implementation: String,
    val notes: Map<String, Any> = emptyMap(),
)

/**
 * Process-global registry of [KernelResolver]s. Safe to use from several
 * threads. Mutable; call [clear] in tests to ensure isolation.
 *
 * Resolution order is registration order — first registered wins on
 * [preferred]. A caller wanting cost-aware selection can score each
 * resolution from [resolveAll] against the active [DeviceDescriptor].
 */
@ExperimentalTlalocApi
object KernelResolverRegistry {
    // Copy-on-write: writers replace the list under the object's monitor,
    // readers take the current snapshot without locking, so a resolver is
    // never called while the registry is locked.
    @Volatile
    private var resolvers: List<KernelResolver> = emptyList()

    /** Register [resolver] at the end of the registration order. */
    @Synchronized
    fun register(resolver: KernelResolver) {
        resolvers = resolvers + resolver
    }

    /**
     * Remove every registered resolver whose [KernelResolver.backendId]
     * matches [backendId]. Returns true if at least one was removed.
     */
    @Synchronized
    fun unregister(backendId: String): Boolean {
        val kept = resolvers.filterNot { it.backendId == backendId }
        val removed = kept.size != resolvers.size
        resolvers = kept
        return removed
    }

    /**
     * All [KernelResolution]s every registered resolver can produce for
     * [kernelName] on [target]. Empty list if no resolver supports the
     * request.
     */
    fun resolveAll(
        kernelName: String,
        target: KernelTarget,
        attrs: Map<String, Any> = emptyMap(),
    ): List<KernelResolution> {
        return resolvers.mapNotNull { it.resolve(kernelName, target, attrs) }
    }

    /**
     * The first matching resolution in registration order, or null if
     * none.
     */
    fun preferred(
        kernelName: String,
        target: KernelTarget,
        attrs: Map<String, Any> = emptyMap(),
    ): KernelResolution? {
        return resolveAll(kernelName, target, attrs).firstOrNull()
    }

    /** Backend ids of every currently-registered resolver, in order. */
    fun registered(): List<String> = resolvers.map { it.backendId }

    /** Drop every registered resolver. For test isolation. */
    @Synchronized
    fun clear() {
        resolvers = emptyList()
    }
}
