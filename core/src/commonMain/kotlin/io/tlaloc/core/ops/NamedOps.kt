package io.tlaloc.core.ops

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.IndexName
import io.tlaloc.core.Named
import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.Rank3
import io.tlaloc.core.Rank4
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
 * # Supported shapes (Layer 1.5 §0.4.242+)
 *
 * | LHS shape | RHS shape | Result shape | Batching | Contracting |
 * |-----------|-----------|--------------|----------|-------------|
 * | `Rank1<Named<K, _>>` | `Rank1<Named<K, _>>` | `ScalarShape` | — | K |
 * | `Rank2<Named<M, _>, Named<K, _>>` | `Rank2<Named<K, _>, Named<N, _>>` | `Rank2<Named<M, _>, Named<N, _>>` | — | K |
 * | `Rank3<Named<B, _>, Named<M, _>, Named<K, _>>` | `Rank3<Named<B, _>, Named<K, _>, Named<N, _>>` | `Rank3<Named<B, _>, Named<M, _>, Named<N, _>>` | B | K |
 * | `Rank4<Named<NB, _>, Named<NH, _>, Named<NT, _>, Named<ND, _>>` | `Rank4<Named<NB, _>, Named<NH, _>, Named<ND, _>, Named<NT2, _>>` | `Rank4<Named<NB, _>, Named<NH, _>, Named<NT, _>, Named<NT2, _>>` | NB, NH | ND |
 *
 * The position-based partition follows from each overload's type-variable
 * structure: a name appearing at the *same dim position* in both operands
 * (declared via the same type variable at that position) is a batching
 * axis; a name shared at *different positions* is the contracting axis.
 * The K2 plugin's `emitContract` lowering re-derives this partition from
 * the operands' [io.tlaloc.ir.DxirType.axisNames] and emits MATMUL with
 * the corresponding `lhs_batching_dims` / `lhs_contracting_dims` attrs.
 *
 * The rank-4 row models the QK^T attention forward pass exactly:
 * `(Batch, Heads, Time, Dim) × (Batch, Heads, Dim, Time') → (Batch, Heads,
 * Time, Time')`. v2 enhancements (rank-5+, multi-axis contraction) are
 * tracked in `docs/audits/named_indices_audit.md`.
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

/**
 * Layer 1.5 §0.4.242+ — rank-3 × rank-3 batched contraction (closes audit
 * OQ-5 partway). Both operands carry a leading batching axis [NameB] (same
 * type variable at axis 0) and contract over the inner axis [NameK]. The
 * output preserves the batch axis and the two non-shared inner axes.
 *
 * Implementation: per-batch loop over rank-2 matmul. The runtime is
 * straightforward but JIT-unfriendly; the K2-plugin path inside
 * `grad { }` lambdas emits MATMUL with `lhs_batching_dims=[0]` so the
 * StableHLO emitter produces a single batched `dot_general` instead.
 */
@JvmName("contractRank3Batched")
infix fun <NameB : IndexName, NameM : IndexName, NameK : IndexName, NameN : IndexName,
           B : ShapeAtom, M : ShapeAtom, K : ShapeAtom, N : ShapeAtom>
    DTensor<Rank3<Named<NameB, B>, Named<NameM, M>, Named<NameK, K>>, F32>.contract(
        other: DTensor<Rank3<Named<NameB, B>, Named<NameK, K>, Named<NameN, N>>, F32>,
    ): DTensor<Rank3<Named<NameB, B>, Named<NameM, M>, Named<NameN, N>>, F32> {
    require(rank == 3 && other.rank == 3) {
        "rank-3 batched contract requires rank-3 tensors (got $rank, ${other.rank})"
    }
    val nb = dims[0]
    val m = dims[1]
    val k = dims[2]
    val nbB = other.dims[0]
    val kB = other.dims[1]
    val n = other.dims[2]
    require(nb == nbB) { "rank-3 contract batch mismatch: $nb vs $nbB" }
    require(k == kB) { "rank-3 contract inner-dim mismatch: $k vs $kB" }
    val a = hostF32()
    val b = other.hostF32()
    val out = FloatArray(nb * m * n)
    for (batch in 0 until nb) {
        val aOff = batch * m * k
        val bOff = batch * k * n
        val outOff = batch * m * n
        for (i in 0 until m) {
            for (p in 0 until k) {
                val aip = a[aOff + i * k + p]
                if (aip == 0f) continue
                val rowOff = outOff + i * n
                val bRowOff = bOff + p * n
                for (j in 0 until n) {
                    out[rowOff + j] += aip * b[bRowOff + j]
                }
            }
        }
    }
    return DTensor(HostF32Storage(out), intArrayOf(nb, m, n), F32)
}

/**
 * Layer 1.5 §0.4.242+ — rank-4 × rank-4 batched contraction (closes audit
 * OQ-5). Models the QK^T attention forward pass: two leading batching axes
 * ([NameB] = batch, [NameH] = heads) shared at the same positions in both
 * operands, plus a contracting axis [NameD] at LHS axis 3 / RHS axis 2.
 *
 * Type-level structure verbatim:
 *
 *     (NameB, NameH, NameT,  NameD) ×
 *     (NameB, NameH, NameD,  NameT2) →
 *     (NameB, NameH, NameT,  NameT2)
 *
 * This is the single use case the user flagged as load-bearing for Layer 2:
 * the typed-step-boundary work would otherwise have to walk around a hole
 * in the type system. With this overload the type checker enforces every
 * axis identity at the call site.
 *
 * Runtime: nested loops over `batch` × `heads`, calling the rank-2 matmul
 * core on each `(time × dim) · (dim × time')` slice. Plugin path inside
 * `grad { }` lambdas emits a single batched `dot_general` with
 * `lhs_batching_dims=[0,1]` / `rhs_batching_dims=[0,1]`.
 */
@JvmName("contractRank4Attention")
infix fun <NameB : IndexName, NameH : IndexName, NameT : IndexName,
           NameD : IndexName, NameT2 : IndexName,
           B : ShapeAtom, H : ShapeAtom, T : ShapeAtom,
           D : ShapeAtom, T2 : ShapeAtom>
    DTensor<Rank4<Named<NameB, B>, Named<NameH, H>, Named<NameT, T>, Named<NameD, D>>, F32>.contract(
        other: DTensor<Rank4<Named<NameB, B>, Named<NameH, H>, Named<NameD, D>, Named<NameT2, T2>>, F32>,
    ): DTensor<Rank4<Named<NameB, B>, Named<NameH, H>, Named<NameT, T>, Named<NameT2, T2>>, F32> {
    require(rank == 4 && other.rank == 4) {
        "rank-4 attention contract requires rank-4 tensors (got $rank, ${other.rank})"
    }
    val nb = dims[0]
    val nh = dims[1]
    val tq = dims[2]
    val d = dims[3]
    val nbR = other.dims[0]
    val nhR = other.dims[1]
    val dR = other.dims[2]
    val tk = other.dims[3]
    require(nb == nbR) { "rank-4 contract batch mismatch: $nb vs $nbR" }
    require(nh == nhR) { "rank-4 contract heads mismatch: $nh vs $nhR" }
    require(d == dR) { "rank-4 contract inner-dim mismatch: $d vs $dR" }
    val a = hostF32()
    val b = other.hostF32()
    val out = FloatArray(nb * nh * tq * tk)
    val sliceA = tq * d
    val sliceB = d * tk
    val sliceOut = tq * tk
    for (batch in 0 until nb) {
        val batchOffA = batch * nh * sliceA
        val batchOffB = batch * nh * sliceB
        val batchOffOut = batch * nh * sliceOut
        for (head in 0 until nh) {
            val aOff = batchOffA + head * sliceA
            val bOff = batchOffB + head * sliceB
            val outOff = batchOffOut + head * sliceOut
            for (i in 0 until tq) {
                for (p in 0 until d) {
                    val aip = a[aOff + i * d + p]
                    if (aip == 0f) continue
                    val rowOff = outOff + i * tk
                    val bRowOff = bOff + p * tk
                    for (j in 0 until tk) {
                        out[rowOff + j] += aip * b[bRowOff + j]
                    }
                }
            }
        }
    }
    return DTensor(HostF32Storage(out), intArrayOf(nb, nh, tq, tk), F32)
}
