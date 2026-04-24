package io.tlaloc.autograd

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.VjpRegistry

/*
 * Runtime-tape reverse-mode AD. Every arm whose op has a registered [VjpRule] delegates
 * per-op math to `io.tlaloc.ir.passes.VjpRegistry` — the IR-side registry is the single
 * source of truth. Each reverse-walk step builds a transient primal [DxirOp] wrapping
 * the tape entry (with [DxirType] dims matching the tape entry's actual shape), invokes
 * `rule.apply` to materialise the adjoint as a dxir expression tree, and evaluates it
 * against the tape's cached values via [DxirInterpreter].
 *
 * As of §0.4.9 every op the tape currently traces (ADD/SUB/MUL/DIV/NEG/RELU/SUM/MEAN/
 * MATMUL) has a registered rule and routes through [applyRegistryRule]. STEP stays as
 * an inline no-op arm (its derivative is identically zero, so upstream gradients never
 * propagate past it).
 */

internal class Gradients(tape: Tape) {
    private val grads: Array<FloatArray?> = arrayOfNulls(tape.size())

    fun seed(id: Int, g: FloatArray) {
        val existing = grads[id]
        if (existing == null) {
            grads[id] = g.copyOf()
        } else {
            require(existing.size == g.size)
            for (i in existing.indices) existing[i] += g[i]
        }
    }

    fun get(id: Int): FloatArray? = grads[id]
}

internal fun backward(tape: Tape, outputId: Int, seedGrad: FloatArray): Gradients {
    val grads = Gradients(tape)
    grads.seed(outputId, seedGrad)

    val entries = tape.entries
    for (i in entries.indices.reversed()) {
        val entry = entries[i]
        val g = grads.get(entry.id) ?: continue
        when (entry.op) {
            null -> {} // leaf; nothing upstream to propagate
            OpKind.ADD, OpKind.SUB, OpKind.MUL, OpKind.DIV, OpKind.NEG, OpKind.RELU,
            OpKind.SUM, OpKind.MEAN, OpKind.MATMUL,
            // §0.4.63 — routed alongside the existing arms now that the Tracer
            // surface exposes sqrt/exp/log/tanh/sigmoid. Rules were already in
            // VjpRegistry since §0.4.22; prior to §0.4.63 nothing on the tape
            // produced these op kinds, so the dispatch arm was dead.
            OpKind.SQRT, OpKind.EXP, OpKind.LOG, OpKind.TANH, OpKind.SIGMOID,
            // §0.4.64 — POW joins the registry-dispatch arm now that
            // `Tracer<S>.pow(Tracer<S>)` can emit it. Rule has been in VjpRegistry
            // since §0.4.22 and was already internally used by C6's closed-form
            // output (§0.4.52); this is the new surface-producer.
            OpKind.POW,
            // §0.4.77 — BROADCAST produced by the scalar-op-rank-N operators
            // (e.g. `Tracer<Rank1<A>>.plus(Tracer<ScalarShape>)`). The
            // BroadcastRule's reverse is SUM-to-scalar; the rule is registered
            // in VjpRegistry for MVP scalar-input case.
            OpKind.BROADCAST ->
                applyRegistryRule(entry.op, entry, entries, g, grads)
            OpKind.STEP -> {
                // step(x) = 1 if x > 0 else 0.  Its derivative is identically zero
                // (a Dirac impulse at x = 0, treated as 0 in practice), so upstream
                // gradients flowing into STEP never propagate past it.  Leaving the
                // arm empty is equivalent to seeding the input with zeros.
            }
            else -> error("VJP not implemented: ${entry.op}")
        }
    }
    return grads
}

/**
 * Bridges one reverse-walk step through [VjpRegistry]. Builds a transient primal
 * [DxirOp] with a dedicated [DxirParam] per operand position (so repeat-input aliasing
 * like `x * x` is disambiguated via `indexOf`), invokes the rule, and evaluates each
 * returned contribution against an env populated from the tape's cached operand values.
 * Each evaluated contribution is seeded back into [grads] at the corresponding tape-id.
 *
 * The transient [DxirOp] and its [DxirParam] operands are typed with the tape entry's
 * actual `dims` (not a uniform scalar type), so rules that inspect operand shape —
 * [io.tlaloc.ir.passes.VjpRegistry.SumRule] / [io.tlaloc.ir.passes.VjpRegistry.MeanRule]
 * emitting BROADCAST with the primal operand's shape — produce output nodes whose
 * typed length matches what the interpreter needs to materialise.
 */
private fun applyRegistryRule(
    kind: OpKind,
    entry: TapeEntry,
    entries: List<TapeEntry>,
    upstreamGrad: FloatArray,
    grads: Gradients,
) {
    val rule = VjpRegistry[kind]
        ?: error("VjpRegistry has no rule for $kind but Backward tried to route through it")
    val outputType = DxirType(F32, entry.dims.toList())
    // Scratch DxirBuilder — all emitted ops are thrown away once the env-walk completes.
    // We still pay a per-entry allocation; acceptable for the runtime-tape path.
    DxirBuilder.function("bridge_${kind.name}_${entry.id}") {
        val upstream = param("upstream", outputType)
        // One dedicated param per operand position — NOT per unique tape id — so the
        // rule's returned (operandKey, contribution) pairs resolve to distinct positions
        // via `indexOf`, even when two operand slots reference the same tape entry.
        // Each param carries the tape entry's actual dims so rules emitting shape-
        // dependent ops (BROADCAST, etc.) see the right target shapes.
        val operandParams = entry.inputs.indices.map { i ->
            val opType = DxirType(F32, entries[entry.inputs[i]].dims.toList())
            param("op$i", opType)
        }
        // §0.4.85 — carry the tape entry's attrs onto the transient primal.
        // BroadcastRule reads `broadcast_dimensions` from here; without this
        // propagation any non-scalar-input BROADCAST would hit the rule's
        // "empty broadcast_dimensions requires scalar input" guard.
        val primal = op(kind, operandParams, outputType, attrs = entry.attrs)

        val contributions = rule.apply(primal, upstream, this)

        val env = HashMap<Int, FloatArray>()
        env[upstream.id] = upstreamGrad
        for ((i, p) in operandParams.withIndex()) {
            env[p.id] = entries[entry.inputs[i]].value
        }
        for ((operandKey, contribution) in contributions) {
            val idx = primal.operands.indexOf(operandKey)
            require(idx >= 0) {
                "VjpRule for $kind returned an operand key not present in primal.operands"
            }
            // §0.4.65 — skip gradient materialisation for operands flagged as
            // opaque constants. The VJP rule already built its contribution tree;
            // we cut off the evaluate-and-seed step, which is where the cost
            // lives (DxirInterpreter walks the tree and allocates a FloatArray
            // per node). The builder walk above is unavoidable with the current
            // rule protocol; cheap next to the interpreter step we save here.
            val targetEntry = entries[entry.inputs[idx]]
            if (targetEntry.isConstant) continue
            val evaluated = DxirInterpreter.evalNode(contribution, env)
            grads.seed(entry.inputs[idx], evaluated)
        }
        emptyList()
    }
}
