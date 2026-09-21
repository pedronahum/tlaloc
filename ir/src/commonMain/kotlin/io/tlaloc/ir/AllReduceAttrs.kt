package io.tlaloc.ir

/**
 * §0.4.460 — Phase G3a: the ALL_REDUCE attribute convention, shared by every
 * layer that touches the kind (interpreter, both AD transforms, the StableHLO
 * emitter). One parser so the layers cannot disagree about what a legal
 * ALL_REDUCE looks like.
 *
 * Attributes:
 * - `attrs["replica_groups"]: List<List<Int>>` — the replica partition, in
 *   StableHLO's `replica_groups` shape. ABSENT means the single-replica
 *   program `[[0]]` (the local, pre-G2b/G4 world). Groups must be non-empty,
 *   UNIFORM in size (the dense<NxMxi64> emission cannot spell ragged groups —
 *   StableHLO's -1 padding is a NAMED DEFERRAL), pairwise disjoint, with
 *   non-negative ids, and replica 0 must belong to some group (the
 *   single-process interpreter models replica 0's view).
 * - `attrs["reduction"]: String` — the reduction kind; ABSENT means `"sum"`.
 *   v1 supports `"sum"` ONLY. The general-op story, recorded here so every
 *   refusal can cite it: `"mean"` is (1/|group|)·sum, so its adjoint SCALES
 *   the upstream by 1/|group| instead of being self-adjoint; `"max"`/`"min"`
 *   need subgradient routing (a WHERE mask against the reduced result, the
 *   MaxRule shape) — both DEFER BY NAME until a consumer exists.
 *
 * Single-process semantics (the SPMD replicated-value view): the interpreter
 * models ONE replica whose value every group member also holds, so
 * all-reduce-sum evaluates to `|group(0)| × value`. With one replica that is
 * exactly identity — the case certified end-to-end today. The multi-replica
 * arm (`|group| > 1`) is exercised only via these unit semantics until
 * G2b/G4 put real devices behind the groups.
 */
object AllReduceAttrs {

    /** The parsed, validated form. [groups] is never empty and never ragged. */
    data class Parsed(val groups: List<List<Int>>, val reduction: String) {
        /** Size of the group containing replica 0 — the single-process scale factor. */
        val groupSizeOfReplicaZero: Int = groups.first { 0 in it }.size
    }

    /** The v1-supported reduction set. */
    private val SUPPORTED_REDUCTIONS = setOf("sum")

    /** Reductions we know the story for but deliberately refuse, by name. */
    private val DEFERRED_REDUCTIONS = mapOf(
        "mean" to "its adjoint scales the upstream by 1/|group| (not self-adjoint)",
        "max" to "it needs subgradient routing (a WHERE mask, the MaxRule shape)",
        "min" to "it needs subgradient routing (a WHERE mask, the MaxRule shape)",
    )

    /**
     * Parse and validate [op]'s ALL_REDUCE attributes, refusing loudly with
     * [layer] in the message. Every refusal names the kind and the rule.
     */
    fun parse(op: DxirOp, layer: String): Parsed {
        require(op.op == OpKind.ALL_REDUCE) {
            "$layer: AllReduceAttrs.parse called on ${op.op} (a compiler bug)"
        }
        val reduction = (op.attrs["reduction"] as? String) ?: "sum"
        if (reduction !in SUPPORTED_REDUCTIONS) {
            val why = DEFERRED_REDUCTIONS[reduction]
                ?: "it is not a known reduction (supported: $SUPPORTED_REDUCTIONS)"
            error(
                "$layer: ALL_REDUCE reduction '$reduction' is refused by name — $why; " +
                    "v1 supports 'sum' only (§0.4.460 G3a)",
            )
        }
        val raw = op.attrs["replica_groups"] ?: return Parsed(listOf(listOf(0)), reduction)
        val groups = (raw as? List<*>)?.map { g ->
            (g as? List<*>)?.map { (it as Number).toInt() }
                ?: error("$layer: ALL_REDUCE replica_groups rows must be List<Int>; got $g")
        } ?: error("$layer: ALL_REDUCE replica_groups must be List<List<Int>>; got $raw")
        require(groups.isNotEmpty() && groups.all { it.isNotEmpty() }) {
            "$layer: ALL_REDUCE replica_groups must be non-empty groups of replica ids; got $groups"
        }
        val size = groups[0].size
        require(groups.all { it.size == size }) {
            "$layer: ALL_REDUCE ragged replica_groups $groups are refused by name — the " +
                "dense<NxMxi64> emission needs uniform group sizes (StableHLO's -1 padding " +
                "is a named deferral, §0.4.460 G3a)"
        }
        val flat = groups.flatten()
        require(flat.all { it >= 0 }) {
            "$layer: ALL_REDUCE replica ids must be non-negative; got $groups"
        }
        require(flat.size == flat.toSet().size) {
            "$layer: ALL_REDUCE replica_groups must be pairwise disjoint; got $groups"
        }
        require(groups.any { 0 in it }) {
            "$layer: ALL_REDUCE replica_groups must place replica 0 in some group (the " +
                "single-process interpreter models replica 0's view); got $groups"
        }
        return Parsed(groups, reduction)
    }
}
