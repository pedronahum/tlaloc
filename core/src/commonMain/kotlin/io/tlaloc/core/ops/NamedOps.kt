package io.tlaloc.core.ops

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.IndexName
import io.tlaloc.core.Named
import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.ShapeAtom
import io.tlaloc.core.hostF32

/**
 * Layer 1 §0.4.241+ — **named-index contraction** (Refined Option A).
 *
 * `contract` is the named-aware tensor contraction operator: when both
 * operands carry a [Named] axis with the same [IndexName] singleton at the
 * appropriate position, the contraction is over that axis. The result type
 * has both contracted axes removed.
 *
 * # Design — Refined Option A: native Kotlin type inference
 *
 * Per the Layer 1 design audit (`docs/audits/named_indices_audit.md`), this
 * file uses *Kotlin's native type-inference machinery* — not a K2 plugin
 * extension — to enforce contraction correctness. Each overload below
 * declares the contraction structure as a relationship between bound type
 * variables ([NameK], [K] are shared between the two operands; the result
 * type drops them). Kotlin's type checker does the unification:
 *
 * - **Disjoint named axes** ⇒ no overload matches the call, the compiler
 *   reports a native type-mismatch error at the call site (e.g. "expected
 *   `Named<SeqLen, Sym>` but got `Named<Vocab, Sym>`"). No plugin
 *   diagnostic is needed — Kotlin's own error rendering is already crisp.
 * - **Shared name + matching symbolic dim** ⇒ the overload resolves and
 *   the result type is correctly computed.
 * - **Mixed named / positional** ⇒ does not unify under these overloads.
 *   v1 deliberately requires both operands to carry named axes for
 *   `contract`. Mixed-mode use should fall back to the existing positional
 *   `matmul` operator.
 *
 * The trade-off vs. an `FirExpressionResolutionExtension`: Refined Option A
 * cannot synthesise *new* result types at FIR resolution time, so the
 * surface is a finite set of declared overloads (one per rank pair). The
 * benefits are: zero new plugin extension surface, native Kotlin error
 * messages, and the type checker enforces correctness even outside
 * `grad { }` lambdas. A v2 enhancement could add an
 * `FirExpressionResolutionExtension` for arbitrary-rank contraction; v1
 * scopes to the rank pairs declared below.
 *
 * # Plugin interaction
 *
 * The K2 plugin's `FirLambdaToDxirLowering` recognises `contract` calls
 * inside `grad { … }` lambdas and lowers them to `OpKind.MATMUL` (or
 * `OpKind.DOT` for the rank-1 × rank-1 case) with `axisNames` populated
 * on the result `DxirType` and `contracted_names` / `preserved_names`
 * attrs on the op. Outside `grad { }`, the runtime bodies below execute.
 *
 * # Supported shapes (v1)
 *
 * | LHS shape | RHS shape | Result shape | Contracted axis |
 * |-----------|-----------|--------------|-----------------|
 * | `Rank1<Named<K, _>>` | `Rank1<Named<K, _>>` | `ScalarShape` | K |
 * | `Rank2<Named<M, _>, Named<K, _>>` | `Rank2<Named<K, _>, Named<N, _>>` | `Rank2<Named<M, _>, Named<N, _>>` | K |
 *
 * Rank-3 batched matmul (`Rank3<B, M, K> contract Rank2<K, N> → Rank3<B, M, N>`)
 * is a deliberate v1 follow-up — listed in the audit's open-issues section.
 */

/**
 * Rank-1 × rank-1 contraction over the shared named axis. Produces a scalar.
 *
 * @param other the right-hand operand, typed to share [NameK] with `this`.
 * @return the inner product as a `DTensor<ScalarShape, F32>`.
 */
@JvmName("contractRank1")
infix fun <NameK : IndexName, K : ShapeAtom>
    DTensor<Rank1<Named<NameK, K>>, F32>.contract(
        other: DTensor<Rank1<Named<NameK, K>>, F32>,
    ): DTensor<ScalarShape, F32> {
    require(rank == 1 && other.rank == 1) { "contract requires rank-1 tensors (got ${rank}, ${other.rank})" }
    val n = dims[0]
    require(n == other.dims[0]) {
        "contract dim mismatch: $n vs ${other.dims[0]} on shared axis"
    }
    val a = hostF32()
    val b = other.hostF32()
    var sum = 0.0f
    for (i in 0 until n) sum += a[i] * b[i]
    return DTensor(HostF32Storage(floatArrayOf(sum)), intArrayOf(), F32)
}

/**
 * Rank-2 × rank-2 contraction over the shared named axis (matmul shape).
 *
 * The shared name [NameK] anchors at axis 1 of `this` and axis 0 of [other].
 * Result is `Rank2<Named<NameM, M>, Named<NameN, N>>` — the contracted axis
 * is removed from both operands and the remaining named axes line up
 * positionally in the result.
 *
 * Runtime delegates to the existing [matmul] implementation; `Named<*, *>`
 * is a [ShapeAtom] by construction so type-variable unification with
 * matmul's `<R, K, C>` is automatic.
 */
@JvmName("contractRank2")
infix fun <NameM : IndexName, NameK : IndexName, NameN : IndexName,
           M : ShapeAtom, K : ShapeAtom, N : ShapeAtom>
    DTensor<Rank2<Named<NameM, M>, Named<NameK, K>>, F32>.contract(
        other: DTensor<Rank2<Named<NameK, K>, Named<NameN, N>>, F32>,
    ): DTensor<Rank2<Named<NameM, M>, Named<NameN, N>>, F32> = matmul(other)
