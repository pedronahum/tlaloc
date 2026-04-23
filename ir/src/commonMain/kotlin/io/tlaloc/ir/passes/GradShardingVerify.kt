package io.tlaloc.ir.passes

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirSharding
import io.tlaloc.ir.OpKind

/**
 * Spec §9.6 step 3 — `tlaloc-grad-sharding-pass`. Runs between Shardy's propagation and
 * export stages; catches the class of bugs where a gradient function's shardings don't
 * correspond to the forward function's.
 *
 * ### What "dual" means here
 *
 * In reverse-mode AD, the tangent space mirrors the primal:
 * - For every forward parameter `x` with some sharding `S_x`, the gradient `∂L/∂x` is
 *   logically a co-tangent with the same layout — so `S_{∂L/∂x} == S_x`.
 * - Where the forward function crosses a sharding boundary (e.g. an `all_reduce`), the
 *   backward function emits the dual collective (identity / reduce-scatter / etc.).
 *
 * This v0 of the pass checks the **identity dual** case — no collectives crossed. The
 * ops-with-dual-collectives case belongs in a later revision, after the K2 plugin's `grad`
 * transform actually produces such pairs.
 *
 * ### Usage
 *
 * ```kotlin
 * val forward = ...
 * val grad = ...  // produced by grad(forward), user-written, or a test fixture
 * val violations = GradShardingVerify.verify(forward, grad)
 * if (violations.isNotEmpty()) { /* report */ }
 * ```
 *
 * ### Intentional v0 limits
 * - Correspondence is by position: `grad.returns[i]` is the gradient of `forward.params[i]`.
 *   Future revisions can support an explicit mapping (e.g. via op attrs) for
 *   `valueAndGrad`-style functions that return `(value, gradients)`.
 * - Only forward-input / backward-output shardings are checked. Activations/intermediates
 *   need op-by-op adjoint rules, which are a follow-up.
 * - No collective-aware duality yet (all_reduce ↔ identity).
 */
object GradShardingVerify {

    /**
     * A single sharding-duality problem. Callers can format these as they see fit; the pass
     * itself is non-throwing so it can be composed with other verifiers.
     */
    data class Violation(
        val site: String,
        val forwardSharding: DxirSharding?,
        val gradSharding: DxirSharding?,
        val message: String,
    ) {
        override fun toString(): String = "[$site] $message"
    }

    /**
     * Verifies `grad` is a valid reverse-mode dual of `forward` w.r.t. shardings.
     * Returns an empty list iff the pair is valid.
     */
    fun verify(forward: DxirFunction, grad: DxirFunction): List<Violation> {
        val violations = mutableListOf<Violation>()

        if (grad.returns.size != forward.params.size) {
            violations += Violation(
                site = "function arity",
                forwardSharding = null,
                gradSharding = null,
                message = "grad function has ${grad.returns.size} returns but forward has " +
                    "${forward.params.size} parameters; expected one gradient per param",
            )
            // Stop here — further checks are meaningless without a valid pairing.
            return violations
        }

        for ((i, fp) in forward.params.withIndex()) {
            val gr = grad.returns[i]
            val site = "param[$i] (${fp.name})"

            // Shape/dtype: the gradient of a tensor shares its shape and dtype.
            if (fp.type.dims != gr.type.dims || fp.type.dtype != gr.type.dtype) {
                violations += Violation(
                    site = site,
                    forwardSharding = fp.sharding,
                    gradSharding = gr.sharding,
                    message = "gradient type ${gr.type} does not match forward param type ${fp.type}",
                )
            }

            // Sharding identity-dual: tangent layout mirrors primal for simple cases.
            val fShard = fp.sharding
            val gShard = gr.sharding
            if (!shardingEquivalent(fShard, gShard)) {
                violations += Violation(
                    site = site,
                    forwardSharding = fShard,
                    gradSharding = gShard,
                    message = "gradient sharding does not match forward sharding; " +
                        "expected the same (identity dual)",
                )
            }
        }

        return violations
    }

    /**
     * Convenience: assert-style wrapper for tests or pipeline integration that prefers
     * failure-fast over cumulative reports.
     */
    fun verifyOrThrow(forward: DxirFunction, grad: DxirFunction) {
        val v = verify(forward, grad)
        if (v.isNotEmpty()) {
            error(
                "grad-sharding verification failed:\n" +
                    v.joinToString("\n") { "  - $it" },
            )
        }
    }

    /**
     * Two shardings are equivalent if they reference the same mesh and have structurally
     * identical per-dim + replicated axis lists. v0 is conservative: exact equality. Future
     * revisions can relax this for semantically-equivalent-but-syntactically-different
     * forms (e.g. open-axis refinements).
     */
    private fun shardingEquivalent(a: DxirSharding?, b: DxirSharding?): Boolean = a == b

    // --- v1: op-by-op collective adjoint rules ---

    /**
     * Verifies that the collectives emitted in the forward and backward functions form
     * a valid reverse-mode dual pairing.
     *
     * Reverse-mode AD reverses the topological order of the forward function, so if the
     * forward body has collectives `[C1, C2, ..., Cn]` (in program order), the backward body
     * should have the dual sequence `[adjoint(Cn), adjoint(Cn-1), ..., adjoint(C1)]` in
     * program order. Any `adjoint(Ci)` that is identity (no-op) is omitted from the grad
     * sequence — hence "identity ↔ all_reduce" only surfaces as an *expected* collective
     * when a forward identity-boundary is known from context (parameter shardings); we
     * handle that implicitly by treating the fwd/grad lists independently and only
     * enforcing the non-identity duals match.
     *
     * ### Rules encoded
     *
     * | forward        | backward          |
     * |----------------|-------------------|
     * | `ALL_REDUCE`   | identity (omit)   |
     * | identity (omit)| `ALL_REDUCE`      |
     * | `ALL_GATHER`   | `REDUCE_SCATTER`  |
     * | `REDUCE_SCATTER`| `ALL_GATHER`     |
     *
     * The "identity ↔ ALL_REDUCE" rule is checked at the *parameter-sharding* boundary by
     * [verify]; this pass handles in-body collectives. Concretely that means a forward
     * function with no collectives and a gradient with an ALL_REDUCE on a sharded-param
     * gradient will pass [verify]'s identity-dual check (shardings match) and can
     * optionally be extended later.
     *
     * Returns an empty list iff the collective sequences are valid duals.
     */
    fun verifyCollectiveDuality(forward: DxirFunction, grad: DxirFunction): List<Violation> {
        val violations = mutableListOf<Violation>()

        val fwdCollectives = collectivesOf(forward)
        val gradCollectives = collectivesOf(grad)

        // Expected grad sequence: fwd's collectives reversed then mapped through adjointOf,
        // dropping identity (null) duals.
        val expected = fwdCollectives.asReversed().mapNotNull(::adjointOf)

        if (expected.size != gradCollectives.size) {
            violations += Violation(
                site = "collective count",
                forwardSharding = null,
                gradSharding = null,
                message = "forward body has ${fwdCollectives.size} collective(s) " +
                    "${fwdCollectives.map { it.name }}; grad body has ${gradCollectives.size} " +
                    "${gradCollectives.map { it.name }}; expected dual sequence " +
                    "${expected.map { it.name }} (reverse-dual of forward)",
            )
            return violations
        }

        for (i in expected.indices) {
            val want = expected[i]
            val got = gradCollectives[i]
            if (want != got) {
                violations += Violation(
                    site = "collective[$i]",
                    forwardSharding = null,
                    gradSharding = null,
                    message = "grad collective #$i is ${got.name}; expected ${want.name} " +
                        "(reverse-dual of forward collective #${fwdCollectives.size - 1 - i})",
                )
            }
        }

        return violations
    }

    /**
     * The reverse-mode adjoint of a collective op. Returns `null` for ops whose dual is the
     * identity (no collective needed).
     *
     * Exposed for test fixtures and future passes that need to reason about duality
     * independently of the verification logic.
     */
    fun adjointOf(k: OpKind): OpKind? = when (k) {
        OpKind.ALL_REDUCE -> null // dual is identity
        OpKind.ALL_GATHER -> OpKind.REDUCE_SCATTER
        OpKind.REDUCE_SCATTER -> OpKind.ALL_GATHER
        else -> error("adjointOf: $k is not a collective OpKind")
    }

    /** Collective ops in a function body, in program order. */
    private fun collectivesOf(fn: DxirFunction): List<OpKind> =
        fn.body.asSequence()
            .filterIsInstance<DxirOp>()
            .filter { it.op in COLLECTIVE_KINDS }
            .map { it.op }
            .toList()

    private val COLLECTIVE_KINDS: Set<OpKind> = setOf(
        OpKind.ALL_REDUCE,
        OpKind.ALL_GATHER,
        OpKind.REDUCE_SCATTER,
    )
}
