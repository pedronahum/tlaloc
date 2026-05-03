package io.tlaloc.ir.recognizer.kernel

/**
 * Layer 4 §0.4.270 — backend-agnostic resolution of `kernel_descriptor`
 * names to concrete implementations.
 *
 * # What this is
 *
 * Phase 1 step 6 of the dual-track Llama-decoder benchmark plan
 * (`memory/llama_benchmark_dual_track.md`). The shared substrate that
 * Track 1 (IREE) and Track 2 (PJRT-XLA) both register against — without
 * this, each backend would either bake its own kernel-name → dispatch
 * map into the interpreter (creating two divergent code paths to drift)
 * or share a JNI-libtorch assumption that contradicts the spec.
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
 * - Dual-backend deployment (the v0.5 story): both IREE + PJRT-XLA
 *   registered; `resolveAll(...)` returns both options. The eventual
 *   L4.3 cost-driven scheduler picks per kernel based on cost-model
 *   scoring of each resolution.
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
data class KernelResolution(
    val backendId: String,
    val implementation: String,
    val notes: Map<String, Any> = emptyMap(),
)

/**
 * Process-global registry of [KernelResolver]s. Mutable; call [clear]
 * in tests to ensure isolation.
 *
 * Resolution order is registration order — first registered wins on
 * [preferred]. The eventual L4.3 cost-driven scheduler (post-Phase-3)
 * will replace `preferred` with a cost-aware picker that scores each
 * resolution from [resolveAll] against the active [DeviceDescriptor].
 */
object KernelResolverRegistry {
    private val resolvers = mutableListOf<KernelResolver>()

    /** Register [resolver] at the end of the registration order. */
    fun register(resolver: KernelResolver) {
        resolvers += resolver
    }

    /**
     * Remove every registered resolver whose [KernelResolver.backendId]
     * matches [backendId]. Returns true if at least one was removed.
     */
    fun unregister(backendId: String): Boolean {
        return resolvers.removeAll { it.backendId == backendId }
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
     * none. v1 picks first-registered; the cost-aware picker lands at
     * L4.3 once both backends are wired up.
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
    fun clear() {
        resolvers.clear()
    }
}
