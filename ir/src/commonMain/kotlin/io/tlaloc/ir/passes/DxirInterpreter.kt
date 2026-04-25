package io.tlaloc.ir.passes

import io.tlaloc.ir.DxirBlockArg
import io.tlaloc.ir.DxirCall
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirOpResult
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.pow

/**
 * Minimal `FloatArray`-valued interpreter for the subset of [DxirNode] emitted by
 * [VjpRegistry]'s currently-registered rules (ADD, SUB, MUL, DIV, NEG, STEP, BROADCAST,
 * TRANSPOSE, MATMUL) over [DxirParam] / [DxirConst] operands, plus structured control
 * flow (IF, WHILE — added in B.0a, see docs/STAGE_B_PLAN.md §7.1.a). Callers supply an
 * `env` mapping SSA ids to concrete `FloatArray`s; the interpreter walks a node's
 * operand tree, materialising intermediate values in the env as it goes.
 *
 * ### Role in the tape↔registry bridge (§11.8.1 step 1)
 *
 * [io.tlaloc.autograd.backward]'s elementwise arms used to inline the per-op adjoint
 * math. They now build a transient primal [DxirOp] whose [DxirParam] operands and
 * result are typed with the tape entry's actual `dims`, hand it to
 * `VjpRegistry[op]!!.apply(...)`, and evaluate the returned contribution nodes via
 * [evalNode]. The registry is the canonical math source-of-truth; this interpreter is
 * how runtime-tape code consumes it.
 *
 * ### Shape-awareness
 *
 * Each node produces a `FloatArray` whose length is derived from its [DxirType.dims]:
 * `product(dims)` for rank-N values, `1` for scalars. No single "size" parameter is
 * threaded through the walker — constants broadcast to their own typed length, ops
 * materialise to the length implied by their result type. Elementwise ops still require
 * their operands to share a length (the interpreter does not broadcast silently).
 *
 * [OpKind.BROADCAST] is supported only in the narrow shape this session's [VjpRegistry]
 * emits (scalar → rank-N uniform, `broadcast_dimensions = []`); general rank-K → rank-N
 * broadcasting is [error]'d pending a rule that demands it. This keeps the interpreter
 * honest: if a future rule produces a shape we can't evaluate, the bridge fails loudly
 * instead of silently producing wrong gradients.
 *
 * ### Scope limits
 *
 * - **No type coercion.** F32/F64 bits are both fit through `FloatArray` here — dtype
 *   is used only for const-literal conversion. Callers working in F64 still get F32
 *   numeric precision through this path; that's fine because the runtime tape is F32
 *   today anyway. The moment a non-F32 tape surface lands, this needs to widen.
 * - **Single-block, straight-line bodies only.** Multi-result ops, block-arg
 *   references, and nested regions all error.
 *
 * ### Equivalence testing
 *
 * The companion [evalFunction] evaluates an entire [DxirFunction] end-to-end: it
 * populates `env` with each param's concrete input (whose size must match the param's
 * typed length), walks body nodes in program order, and returns the evaluated returns.
 * Used by the dxir-eval equivalence tests to drive gradient functions produced by
 * [DxirReverseTransform] against concrete inputs and compare the numerical result with
 * the runtime tape's `Backward.kt` path.
 */
object DxirInterpreter {

    /**
     * Hard cap on WHILE iteration count. Catches infinite loops in malformed test
     * primals before they hang the test suite; chosen to be large enough that any
     * realistic Stage B test (`iterate5(x)`, BGDHyperOpt with K=10..100) terminates
     * well below it. Increase only if a test legitimately needs more iterations.
     */
    const val WHILE_ITERATION_CAP: Int = 1_000_000

    private fun sizeOf(type: DxirType): Int =
        if (type.dims.isEmpty()) 1 else type.dims.fold(1) { acc, d -> acc * d }

    /**
     * Evaluates [node] against [env], producing a `FloatArray` whose length is derived
     * from the node's [DxirType] (params take their length from the env entry).
     *
     * @param env mutable map from SSA id to its materialised `FloatArray`. The
     *   interpreter memoizes every op result it computes back into [env] so repeated
     *   sub-trees (e.g. `x` appearing twice in `MUL(x, x)`) don't re-evaluate.
     * @param multiResults side-map for ops with N > 1 results (IF, WHILE). Keyed by
     *   `(sourceId, resultIndex)`; result index 0 also lands in [env] for the common
     *   single-result lookup path. Defaults to a fresh map; callers that build their
     *   own subtrees can ignore it. [DxirBlockArg] reads its bound value from [env]
     *   too — region-arg binding happens in the WHILE / IF arms of [evalOp].
     */
    fun evalNode(
        node: DxirNode,
        env: MutableMap<Int, FloatArray>,
        multiResults: MutableMap<Long, FloatArray> = HashMap(),
    ): FloatArray {
        // DxirOpResult shares its source op's SSA id (see DxirNode.kt — `override val id:
        // Int get() = source.id`), so a cache check keyed on `node.id` would wrongly
        // return the source's result[0] for every index. Route DxirOpResult through its
        // own branch first, which looks up the correct slot in `multiResults`.
        if (node is DxirOpResult) {
            val src = node.source
            evalNode(src, env, multiResults)
            val result = if (node.index == 0) {
                env[src.id]
                    ?: error("DxirInterpreter: source op id=${src.id} missing from env after eval")
            } else {
                multiResults[multiResultKey(src.id, node.index)]
                    ?: error(
                        "DxirInterpreter: multi-result slot (id=${src.id}, index=${node.index}) " +
                            "not materialised — source op ${src.op} did not populate multiResults",
                    )
            }
            // NB: do NOT cache into env[node.id] here — node.id == src.id, which would
            // stomp env[src.id] with the `index`-th result instead of result[0].
            return result
        }
        env[node.id]?.let { return it }
        val result = when (node) {
            is DxirParam -> error(
                "DxirInterpreter: param '${node.name}' (id=${node.id}) missing from env",
            )
            is DxirConst -> {
                val size = sizeOf(node.type)
                when (val v = node.value) {
                    // Scalar-valued const: splat the single value over the declared
                    // output shape. Canonical form for `const(1.0f, f32)` etc.
                    is Number -> FloatArray(size) { v.toFloat() }
                    // Bool const encoded as Kotlin Boolean (the form
                    // [BreakBearingWhile.classifyBreakCond] recognises for
                    // [BreakBearingWhile.BreakCondClass.Constant]). Materialised as
                    // 1f / 0f to match the interpreter's internal Bool encoding —
                    // see `if (pred[0] == 0f) break` in [evalWhile].
                    is Boolean -> FloatArray(size) { if (v) 1f else 0f }
                    // §0.4.72 — rank-N const whose value is a full FloatArray.
                    // Capture.kt (§0.4.71 fix) uses this form to carry the tape's
                    // cached rank-N leaf value through to the captured function.
                    // The array's size must match the declared type's size — a
                    // mismatch here is a compile-pipeline bug, not a user error.
                    is FloatArray -> {
                        require(v.size == size) {
                            "DxirInterpreter: FloatArray const has size ${v.size} but type ${node.type} " +
                                "requires $size"
                        }
                        v.copyOf()
                    }
                    else -> error("DxirInterpreter: non-numeric const value ${node.value}")
                }
            }
            is DxirOp -> evalOp(node, env, multiResults)
            is DxirOpResult -> error("unreachable — handled above")
            is DxirCall -> error(
                "DxirInterpreter: DxirCall not supported (id=${node.id})",
            )
            is DxirBlockArg -> error(
                "DxirInterpreter: DxirBlockArg id=${node.id} not bound — region-arg " +
                    "binding must happen via the enclosing IF / WHILE arm of evalOp",
            )
        }
        env[node.id] = result
        return result
    }

    /** Pack `(sourceId, resultIndex)` into a single Long key for [multiResults] lookup. */
    private fun multiResultKey(sourceId: Int, index: Int): Long =
        (sourceId.toLong() shl 32) or (index.toLong() and 0xFFFFFFFFL)

    private fun evalOp(
        op: DxirOp,
        env: MutableMap<Int, FloatArray>,
        multiResults: MutableMap<Long, FloatArray>,
    ): FloatArray {
        return when (op.op) {
            OpKind.ADD -> {
                val a = evalNode(op.operands[0], env, multiResults)
                val b = evalNode(op.operands[1], env, multiResults)
                require(a.size == b.size) {
                    "DxirInterpreter: ADD operands have different sizes ${a.size} vs ${b.size}"
                }
                FloatArray(a.size) { a[it] + b[it] }
            }
            OpKind.SUB -> {
                val a = evalNode(op.operands[0], env, multiResults)
                val b = evalNode(op.operands[1], env, multiResults)
                require(a.size == b.size) {
                    "DxirInterpreter: SUB operands have different sizes ${a.size} vs ${b.size}"
                }
                FloatArray(a.size) { a[it] - b[it] }
            }
            OpKind.MUL -> {
                val a = evalNode(op.operands[0], env, multiResults)
                val b = evalNode(op.operands[1], env, multiResults)
                require(a.size == b.size) {
                    "DxirInterpreter: MUL operands have different sizes ${a.size} vs ${b.size}"
                }
                FloatArray(a.size) { a[it] * b[it] }
            }
            OpKind.DIV -> {
                val a = evalNode(op.operands[0], env, multiResults)
                val b = evalNode(op.operands[1], env, multiResults)
                require(a.size == b.size) {
                    "DxirInterpreter: DIV operands have different sizes ${a.size} vs ${b.size}"
                }
                FloatArray(a.size) { a[it] / b[it] }
            }
            OpKind.NEG -> {
                val a = evalNode(op.operands[0], env, multiResults)
                FloatArray(a.size) { -a[it] }
            }
            OpKind.STEP -> {
                // step(x) = 1 if x > 0 else 0 — matches XLA's GT+select semantics at x=0.
                val a = evalNode(op.operands[0], env, multiResults)
                FloatArray(a.size) { if (a[it] > 0f) 1f else 0f }
            }
            OpKind.NOT -> {
                // Boolean negation; same 0f/1f encoding as STEP. NOT is the dxir
                // primitive F3 emits when canonicalising an IF's branch order
                // (docs/STAGE_B_PLAN.md §4.4).
                val a = evalNode(op.operands[0], env, multiResults)
                FloatArray(a.size) { if (a[it] != 0f) 0f else 1f }
            }
            OpKind.LAND -> {
                // §0.4.50 — logical AND on Bool values encoded as 0f/1f. Emitted by
                // the FIR break-hoist path: `LAND(original_cond, NOT(break_cond))`.
                val a = evalNode(op.operands[0], env, multiResults)
                val b = evalNode(op.operands[1], env, multiResults)
                require(a.size == b.size) {
                    "DxirInterpreter: LAND operands different sizes ${a.size} vs ${b.size}"
                }
                FloatArray(a.size) { if (a[it] != 0f && b[it] != 0f) 1f else 0f }
            }
            OpKind.POW -> {
                // Element-wise power: result[i] = base[i] ^ exp[i]. Stage B.3 added this
                // arm because C6's closed form for `Σ_{i=0}^{n-1} a^i` resolves to a
                // POW-bearing dxir expression when the trip count is symbolic. Math is
                // routed through `kotlin.math.pow` (Double precision then truncated to
                // Float) — both negative-base + non-integer-exp and zero^zero edge
                // cases follow `kotlin.math.pow`'s definitions; document if a benchmark
                // exposes a deviation.
                val a = evalNode(op.operands[0], env, multiResults)
                val b = evalNode(op.operands[1], env, multiResults)
                require(a.size == b.size) {
                    "DxirInterpreter: POW operands different sizes ${a.size} vs ${b.size}"
                }
                FloatArray(a.size) { a[it].toDouble().pow(b[it].toDouble()).toFloat() }
            }
            OpKind.LOG -> {
                // Element-wise natural logarithm. Stage B.3 PowRule needs LOG for the
                // exp-gradient adjoint `d/dexp(base^exp) = base^exp · ln(base)`; without
                // LOG in the interpreter, grad-through-POW with a variable exponent
                // can't evaluate. `kotlin.math.ln` follows IEEE semantics: ln(0) = -∞,
                // ln(negative) = NaN.
                val a = evalNode(op.operands[0], env, multiResults)
                FloatArray(a.size) { kotlin.math.ln(a[it].toDouble()).toFloat() }
            }
            OpKind.EXP -> {
                // Element-wise natural exponential. Stage A §0.4.22 ExpRule evaluates
                // through this; also produces the correct `exp(x)` as a forward value.
                val a = evalNode(op.operands[0], env, multiResults)
                FloatArray(a.size) { kotlin.math.exp(a[it].toDouble()).toFloat() }
            }
            OpKind.SQRT -> {
                // Element-wise square root. `kotlin.math.sqrt` returns NaN for negative
                // operands, matching IEEE semantics.
                val a = evalNode(op.operands[0], env, multiResults)
                FloatArray(a.size) { kotlin.math.sqrt(a[it].toDouble()).toFloat() }
            }
            OpKind.TANH -> {
                val a = evalNode(op.operands[0], env, multiResults)
                FloatArray(a.size) { kotlin.math.tanh(a[it].toDouble()).toFloat() }
            }
            OpKind.SIGMOID -> {
                // σ(x) = 1 / (1 + exp(-x)). Evaluated in Double precision for stability
                // at large |x| before narrowing to Float.
                val a = evalNode(op.operands[0], env, multiResults)
                FloatArray(a.size) { (1.0 / (1.0 + kotlin.math.exp(-a[it].toDouble()))).toFloat() }
            }
            OpKind.TRANSPOSE -> {
                // §0.4.136 — rank-N stride-based transpose. The original rank-2-only
                // path covered all paths VjpRules emitted (MatmulRule's `[1, 0]` swap),
                // but §0.4.135's batched-MATMUL substrate implies an eventual batched
                // MatmulRule which would emit `[0, 2, 1]`-style permutations. The
                // implementation below handles any valid permutation of any rank.
                //
                // Algorithm: the output element at multi-index (i_0, …, i_{N-1})
                // corresponds to the input element at multi-index
                // (j_0, …, j_{N-1}) where `j[perm[k]] = i_k`. Equivalently, walking
                // the output in row-major order, the input flat offset accumulates
                // `i_k * inputStrides[perm[k]]` per output axis `k`.
                val a = evalNode(op.operands[0], env, multiResults)
                val inputType = op.operands[0].type
                val perm = (op.attrs["permutation"] as? List<*>)
                    ?.map { (it as Number).toInt() }
                    ?: emptyList()
                val rank = inputType.rank
                require(perm.size == rank) {
                    "DxirInterpreter: TRANSPOSE permutation length ${perm.size} ≠ rank $rank"
                }
                require(perm.toSet() == (0 until rank).toSet()) {
                    "DxirInterpreter: TRANSPOSE permutation $perm must be a permutation of [0..${rank - 1}]"
                }
                val totalSize = if (rank == 0) 1 else inputType.dims.reduce(Int::times)
                require(a.size == totalSize) {
                    "DxirInterpreter: TRANSPOSE input size ${a.size} does not match shape ${inputType.dims}"
                }
                if (rank <= 1) {
                    // Rank-0 / rank-1 transpose is the identity (only valid permutation
                    // is `[0]` for rank-1, `[]` for rank-0). Return a copy to preserve
                    // the "fresh array per node" invariant.
                    a.copyOf()
                } else {
                    val outputDims = perm.map { inputType.dims[it] }
                    // Row-major strides for input and output. `inputStrides[a]` is the
                    // flat offset increment per unit step along input axis `a`.
                    val inputStrides = IntArray(rank)
                    inputStrides[rank - 1] = 1
                    for (i in rank - 2 downTo 0) inputStrides[i] = inputStrides[i + 1] * inputType.dims[i + 1]
                    val outputStrides = IntArray(rank)
                    outputStrides[rank - 1] = 1
                    for (i in rank - 2 downTo 0) outputStrides[i] = outputStrides[i + 1] * outputDims[i + 1]
                    FloatArray(totalSize) { outFlat ->
                        var rem = outFlat
                        var inFlat = 0
                        for (k in 0 until rank) {
                            val idxK = rem / outputStrides[k]
                            rem -= idxK * outputStrides[k]
                            inFlat += idxK * inputStrides[perm[k]]
                        }
                        a[inFlat]
                    }
                }
            }
            OpKind.MATMUL -> {
                // §0.4.135 — rank-2 (`(M,K) @ (K,N) → (M,N)`) plus rank-3+ batched
                // (`(B0..Bk, M, K) @ (B0..Bk, K, N) → (B0..Bk, M, N)`). The batched
                // path uses the canonical convention: all leading axes are batching
                // dims; the last two are M/K (lhs) and K/N (rhs). Equivalent to MLIR
                // `stablehlo.dot_general` with `batching_dims = [0..k]`,
                // `contracting_dims = [k+2] x [k+1]`.
                val aType = op.operands[0].type
                val bType = op.operands[1].type
                require(aType.rank >= 2 && bType.rank >= 2) {
                    "DxirInterpreter: MATMUL requires rank ≥ 2 operands, got ${aType.dims} x ${bType.dims}"
                }
                require(aType.rank == bType.rank) {
                    "DxirInterpreter: MATMUL operands must have matching ranks for canonical batched " +
                        "shape; got ${aType.dims} x ${bType.dims}"
                }
                val r = aType.rank
                val m = aType.dims[r - 2]
                val k = aType.dims[r - 1]
                val kB = bType.dims[r - 2]
                val n = bType.dims[r - 1]
                require(k == kB) {
                    "DxirInterpreter: MATMUL inner dim mismatch: ${aType.dims} x ${bType.dims}"
                }
                // Batch dims (axes 0..r-3) must agree elementwise.
                for (axis in 0 until r - 2) {
                    require(aType.dims[axis] == bType.dims[axis]) {
                        "DxirInterpreter: MATMUL batch axis $axis mismatch: ${aType.dims} x ${bType.dims}"
                    }
                }
                val a = evalNode(op.operands[0], env, multiResults)
                val b = evalNode(op.operands[1], env, multiResults)
                val batchSize = if (r == 2) 1 else aType.dims.subList(0, r - 2).reduce(Int::times)
                require(a.size == batchSize * m * k) {
                    "DxirInterpreter: MATMUL lhs size ${a.size} does not match batch*m*k=${batchSize * m * k}"
                }
                require(b.size == batchSize * k * n) {
                    "DxirInterpreter: MATMUL rhs size ${b.size} does not match batch*k*n=${batchSize * k * n}"
                }
                val out = FloatArray(batchSize * m * n)
                for (batch in 0 until batchSize) {
                    val aBase = batch * m * k
                    val bBase = batch * k * n
                    val outBase = batch * m * n
                    for (i in 0 until m) {
                        for (p in 0 until k) {
                            val aip = a[aBase + i * k + p]
                            if (aip == 0f) continue
                            val rowOff = outBase + i * n
                            val bOff = bBase + p * n
                            for (j in 0 until n) {
                                out[rowOff + j] += aip * b[bOff + j]
                            }
                        }
                    }
                }
                out
            }
            OpKind.BROADCAST -> {
                val a = evalNode(op.operands[0], env, multiResults)
                val outSize = sizeOf(op.type)
                val bcastDims = (op.attrs["broadcast_dimensions"] as? List<*>)
                    ?.map { (it as Number).toInt() }
                    ?: emptyList()
                // Only the narrow scalar → rank-N uniform case is needed by SumRule /
                // MeanRule today. General rank-K → rank-N broadcasting (with non-empty
                // broadcast_dimensions or a non-scalar input) needs rank-aware indexing
                // and is deferred until a rule actually demands it.
                require(bcastDims.isEmpty()) {
                    "DxirInterpreter: BROADCAST with non-empty broadcast_dimensions=" +
                        "$bcastDims not yet supported (only scalar → rank-N uniform)"
                }
                require(a.size == 1) {
                    "DxirInterpreter: scalar BROADCAST requires a size-1 input, got ${a.size}"
                }
                FloatArray(outSize) { a[0] }
            }
            OpKind.GATHER -> {
                // §0.4.41 — original S2 shape: `GATHER(arr: rank-1, idx: scalar I32) → scalar`.
                // §0.4.111 — extended to `GATHER(arr: rank-2, idx: scalar I32) → rank-1`,
                // selecting the row at row-major offset `idx`. §0.4.132 — generalised
                // to any rank ≥ 1: `GATHER(arr: rank-r, idx: scalar I32) → rank-(r-1)`,
                // selecting the slice at axis-0 position `idx` (row-major contiguous
                // block of size `prod(arr.dims[1..])`). `idx` out-of-bounds is a
                // fail-loud runtime error; benchmark porters are expected to keep
                // indices within `0 until arr.dims[0]`.
                require(op.operands.size == 2) {
                    "DxirInterpreter: GATHER requires 2 operands (arr, idx), got ${op.operands.size}"
                }
                val arrType = op.operands[0].type
                val idxType = op.operands[1].type
                require(arrType.rank >= 1) {
                    "DxirInterpreter: GATHER arr must be rank ≥ 1, got rank=${arrType.rank}"
                }
                require(idxType.isScalar && idxType.dtype == io.tlaloc.core.I32) {
                    "DxirInterpreter: GATHER idx must be scalar I32, got $idxType"
                }
                val arr = evalNode(op.operands[0], env, multiResults)
                val idxArr = evalNode(op.operands[1], env, multiResults)
                val i = idxArr[0].toInt()
                val outer = arrType.dims[0]
                val sliceSize = if (arrType.rank == 1) 1 else arrType.dims.drop(1).reduce(Int::times)
                require(i in 0 until outer) {
                    "DxirInterpreter: GATHER idx=$i out of bounds for rank-${arrType.rank} " +
                        "array of shape ${arrType.dims}"
                }
                require(arr.size == outer * sliceSize) {
                    "DxirInterpreter: GATHER rank-${arrType.rank} arr backing size ${arr.size} " +
                        "doesn't match shape ${arrType.dims}"
                }
                FloatArray(sliceSize) { off -> arr[i * sliceSize + off] }
            }
            OpKind.SCATTER_ADD -> {
                // §0.4.45 — original shape: `base[idx] += value` with rank-1 base + scalar value.
                // §0.4.111 — extended to `base[idx, :] += value` with rank-2 base + rank-1 value
                // (the gradient shape produced by GatherRule against a rank-2 GATHER).
                // §0.4.132 — generalised to any rank ≥ 1: `base[idx, ...] += value` where
                // `value` is rank-(base.rank - 1) and matches the trailing slice shape.
                require(op.operands.size == 3) {
                    "DxirInterpreter: SCATTER_ADD requires 3 operands (base, idx, value), got ${op.operands.size}"
                }
                val baseType = op.operands[0].type
                val idxType = op.operands[1].type
                val valueType = op.operands[2].type
                require(baseType.rank >= 1) {
                    "DxirInterpreter: SCATTER_ADD base must be rank ≥ 1, got rank=${baseType.rank}"
                }
                require(idxType.isScalar && idxType.dtype == io.tlaloc.core.I32) {
                    "DxirInterpreter: SCATTER_ADD idx must be scalar I32, got $idxType"
                }
                val expectedValueRank = baseType.rank - 1
                require(valueType.rank == expectedValueRank) {
                    "DxirInterpreter: SCATTER_ADD value must be rank-${expectedValueRank} for " +
                        "rank-${baseType.rank} base, got rank=${valueType.rank}"
                }
                val base = evalNode(op.operands[0], env, multiResults)
                val idxArr = evalNode(op.operands[1], env, multiResults)
                val value = evalNode(op.operands[2], env, multiResults)
                val i = idxArr[0].toInt()
                val outer = baseType.dims[0]
                val sliceSize = if (baseType.rank == 1) 1 else baseType.dims.drop(1).reduce(Int::times)
                require(i in 0 until outer) {
                    "DxirInterpreter: SCATTER_ADD idx=$i out of bounds for rank-${baseType.rank} " +
                        "array of shape ${baseType.dims}"
                }
                require(value.size == sliceSize) {
                    "DxirInterpreter: SCATTER_ADD value size ${value.size} doesn't match slice " +
                        "size $sliceSize for rank-${baseType.rank} base of shape ${baseType.dims}"
                }
                val out = base.copyOf()
                for (off in 0 until sliceSize) out[i * sliceSize + off] += value[off]
                out
            }
            OpKind.SCATTER -> {
                // §0.4.41 — original S2 shape: `SCATTER(base: rank-1, idx: scalar I32, value: scalar) → rank-1`.
                // §0.4.114 — extended to `SCATTER(base: rank-2, idx: scalar I32, value: rank-1) → rank-2`,
                // replacing row [idx] of `base` with `value`. §0.4.132 — generalised to
                // any rank ≥ 1: `SCATTER(base: rank-r, idx: scalar I32, value: rank-(r-1)) → rank-r`,
                // replacing the slice at axis-0 position `idx`. Non-destructive: returns
                // a fresh copy. The "into zeros" variant used by GatherRule's adjoint
                // is expressed as `SCATTER(BROADCAST(const(0), arr.type), idx, value)` —
                // one op kind, composed with BROADCAST at the emit site.
                require(op.operands.size == 3) {
                    "DxirInterpreter: SCATTER requires 3 operands (base, idx, value), got ${op.operands.size}"
                }
                val baseType = op.operands[0].type
                val idxType = op.operands[1].type
                val valueType = op.operands[2].type
                require(baseType.rank >= 1) {
                    "DxirInterpreter: SCATTER base must be rank ≥ 1, got rank=${baseType.rank}"
                }
                require(idxType.isScalar && idxType.dtype == io.tlaloc.core.I32) {
                    "DxirInterpreter: SCATTER idx must be scalar I32, got $idxType"
                }
                val expectedValueRank = baseType.rank - 1
                require(valueType.rank == expectedValueRank) {
                    "DxirInterpreter: SCATTER value must be rank-${expectedValueRank} for " +
                        "rank-${baseType.rank} base, got rank=${valueType.rank}"
                }
                val base = evalNode(op.operands[0], env, multiResults)
                val idxArr = evalNode(op.operands[1], env, multiResults)
                val value = evalNode(op.operands[2], env, multiResults)
                val i = idxArr[0].toInt()
                val outer = baseType.dims[0]
                val sliceSize = if (baseType.rank == 1) 1 else baseType.dims.drop(1).reduce(Int::times)
                require(i in 0 until outer) {
                    "DxirInterpreter: SCATTER idx=$i out of bounds for rank-${baseType.rank} " +
                        "array of shape ${baseType.dims}"
                }
                require(value.size == sliceSize) {
                    "DxirInterpreter: SCATTER value size ${value.size} doesn't match slice " +
                        "size $sliceSize for rank-${baseType.rank} base of shape ${baseType.dims}"
                }
                val out = base.copyOf()
                for (off in 0 until sliceSize) out[i * sliceSize + off] = value[off]
                out
            }
            OpKind.IF -> evalIf(op, env, multiResults)
            OpKind.WHILE -> evalWhile(op, env, multiResults)
            OpKind.COARSENED -> evalCoarsened(op, env, multiResults)
            OpKind.CAST -> {
                // §0.4.40 — dtype conversion at evaluation time. The FloatArray
                // storage already stores every dtype as a float-width view (see
                // this file's top comment on single-buffer storage), so the only
                // arithmetic that needs adjusting is truncation for Float→Int
                // and rounding for Double→Int. For Int→Float / Long→Double / etc.
                // widening, the stored float already represents the correct
                // integer value.
                val a = evalNode(op.operands[0], env, multiResults)
                val srcDtype = op.operands[0].type.dtype
                val dstDtype = op.type.dtype
                when {
                    // Narrowing-to-integer paths truncate toward zero (Kotlin's
                    // `toInt()` / `toLong()` semantics on Float/Double).
                    (srcDtype == io.tlaloc.core.F32 || srcDtype == io.tlaloc.core.F64) &&
                        (dstDtype == io.tlaloc.core.I32 || dstDtype == io.tlaloc.core.I64) ->
                        FloatArray(a.size) { a[it].toInt().toFloat() }
                    // All other scalar-to-scalar conversions are no-ops at the
                    // FloatArray storage level — the integer value 0, 1, 2, ...
                    // stored as 0f, 1f, 2f already reads correctly as Float/Double.
                    else -> a.copyOf()
                }
            }
            // §0.4.77 — SUM is the reverse of BROADCAST (scalar-input MVP),
            // widened in §0.4.84 to handle axis-aware partial SUM via a
            // `reduction_dims` attr.
            //   * No attrs (or empty list) → sum over all axes → scalar.
            //     Matches Tracer.sum()'s all-axis primal and §0.4.77 MVP.
            //   * `reduction_dims = [a, b, ...]` → sum over those axes in
            //     the input, preserving the remaining axes. Used by
            //     BroadcastRule's axis-aware reverse (§0.4.84).
            OpKind.SUM -> {
                val a = evalNode(op.operands[0], env, multiResults)
                @Suppress("UNCHECKED_CAST")
                val reduceDims = (op.attrs["reduction_dims"] as? List<Int>) ?: emptyList()
                val inputType = op.operands[0].type
                if (reduceDims.isEmpty()) {
                    var acc = 0f
                    for (x in a) acc += x
                    floatArrayOf(acc)
                } else {
                    // Generic axis-aware sum: iterate input linearly, project each
                    // element to an output index (by dropping the reduced axes'
                    // coordinates), accumulate.
                    val inputDims = inputType.dims
                    val keepDims = (0 until inputDims.size).filter { it !in reduceDims }
                    val outShape = keepDims.map { inputDims[it] }
                    val outSize = if (outShape.isEmpty()) 1 else outShape.fold(1) { acc, d -> acc * d }
                    val out = FloatArray(outSize)
                    // Row-major strides for input dims.
                    val inStrides = IntArray(inputDims.size)
                    if (inputDims.isNotEmpty()) {
                        inStrides[inputDims.size - 1] = 1
                        for (i in inputDims.size - 2 downTo 0) {
                            inStrides[i] = inStrides[i + 1] * inputDims[i + 1]
                        }
                    }
                    val outStrides = IntArray(keepDims.size)
                    if (keepDims.isNotEmpty()) {
                        outStrides[keepDims.size - 1] = 1
                        for (i in keepDims.size - 2 downTo 0) {
                            outStrides[i] = outStrides[i + 1] * outShape[i + 1]
                        }
                    }
                    for (flat in 0 until a.size) {
                        // Decompose `flat` into input coords, then project to out-coord.
                        var rem = flat
                        var outIdx = 0
                        for (d in inputDims.indices) {
                            val coord = rem / inStrides[d]
                            rem -= coord * inStrides[d]
                            val keepPos = keepDims.indexOf(d)
                            if (keepPos >= 0) outIdx += coord * outStrides[keepPos]
                        }
                        out[outIdx] += a[flat]
                    }
                    out
                }
            }
            else -> error("DxirInterpreter: op ${op.op} not in the bridge's supported set")
        }
    }

    /**
     * Evaluate an IF op (Stage B.0a). The cond operand is evaluated against [env];
     * the chosen region's block body is then evaluated in program order, and its
     * terminator value(s) are returned. Multi-result IFs cache extra results in
     * [multiResults] keyed by `(op.id, i)` for `i > 0` so [DxirOpResult] lookups work.
     *
     * Uses the value `0f` for false and any non-zero for true, matching the
     * convention `OpKind.STEP` uses to encode booleans in `FloatArray`.
     */
    private fun evalIf(
        op: DxirOp,
        env: MutableMap<Int, FloatArray>,
        multiResults: MutableMap<Long, FloatArray>,
    ): FloatArray {
        val cond = evalNode(op.operands[0], env, multiResults)
        require(cond.size == 1) {
            "DxirInterpreter: IF op id=${op.id} predicate must be size-1 scalar; got ${cond.size}"
        }
        val takeThen = cond[0] != 0f
        val region = if (takeThen) op.regions[0] else op.regions[1]
        val block = region.blocks.single()
        for (n in block.body) evalNode(n, env, multiResults)
        val terms = block.terminator.map { evalNode(it, env, multiResults) }
        for (i in 1 until terms.size) {
            multiResults[multiResultKey(op.id, i)] = terms[i]
        }
        return terms[0]
    }

    /**
     * Evaluate a WHILE op (Stage B.0a). Iterates the cond + body regions until the
     * cond region's predicate evaluates to `0f` (false), or until [WHILE_ITERATION_CAP]
     * is exceeded (in which case [error] fires loudly so an infinite-loop test primal
     * doesn't hang the suite).
     *
     * Per-iteration env hygiene: every SSA id declared inside either region (block args
     * + body op ids + nested-region declarations) is removed from [env] at the start of
     * each iteration, so memoised values from iteration N do not leak into iteration N+1
     * and corrupt downstream evaluations. Outer-scope ids are never touched.
     *
     * Multi-result WHILE: result 0 is returned (per the [evalNode] convention); results
     * 1..N-1 are stashed in [multiResults] keyed by `(op.id, i)` for [DxirOpResult]
     * extraction.
     *
     * Single back-edge only (Stage B.0a scope per plan §3.1.2). Multi-back-edge WHILEs
     * (the `break` form, with `m > 1` loop-exit φ arguments per paper F5) are deferred
     * post-Stage-B.
     */
    private fun evalWhile(
        op: DxirOp,
        env: MutableMap<Int, FloatArray>,
        multiResults: MutableMap<Long, FloatArray>,
    ): FloatArray {
        val n = op.operands.size
        val condBlock = op.regions[0].blocks.single()
        val bodyBlock = op.regions[1].blocks.single()
        val condArgs = condBlock.args
        val bodyArgs = bodyBlock.args
        // Initialise loop-carried values from operands. Operand evaluation happens in
        // outer-scope env; their ids are NOT in `scratch` so we keep them across iters.
        var carried: List<FloatArray> = op.operands.map { evalNode(it, env, multiResults) }
        // Scratch ids = everything declared inside either region. Cleared each iteration.
        val scratch = HashSet<Int>().apply {
            addAll(condBlock.declaredIds())
            addAll(bodyBlock.declaredIds())
        }
        var iter = 0
        while (iter < WHILE_ITERATION_CAP) {
            // Clear stale per-iteration entries before re-binding args.
            for (id in scratch) env.remove(id)
            // Bind carried values to cond-region args, then evaluate cond body + predicate.
            for (i in 0 until n) env[condArgs[i].id] = carried[i]
            for (cn in condBlock.body) evalNode(cn, env, multiResults)
            val pred = evalNode(condBlock.terminator.single(), env, multiResults)
            require(pred.size == 1) {
                "DxirInterpreter: WHILE op id=${op.id} cond terminator must be size-1 scalar; got ${pred.size}"
            }
            if (pred[0] == 0f) break
            // Bind carried values to body-region args, evaluate body, capture new carried.
            for (i in 0 until n) env[bodyArgs[i].id] = carried[i]
            for (bn in bodyBlock.body) evalNode(bn, env, multiResults)
            carried = bodyBlock.terminator.map { evalNode(it, env, multiResults) }
            iter++
        }
        if (iter >= WHILE_ITERATION_CAP) {
            error(
                "DxirInterpreter: WHILE op id=${op.id} exceeded iteration cap of " +
                    "$WHILE_ITERATION_CAP — likely an infinite loop in a test primal",
            )
        }
        // Final clear so the WHILE's outer-scope post-loop ops don't see stale region entries.
        for (id in scratch) env.remove(id)
        // Materialise N results: result[0] returned (also lands in env via evalNode caller),
        // results[i>0] stashed for DxirOpResult lookups.
        for (i in 1 until n) {
            multiResults[multiResultKey(op.id, i)] = carried[i]
        }
        return carried[0]
    }

    /**
     * §0.4.31 — evaluate [OpKind.COARSENED] by nesting [evalFunction] on the stored
     * `primal_body`. Operand values (from the outer env) map positionally to the
     * primal_body's params. Multi-result COARSENED stashes results[i > 0] in
     * [multiResults] via [multiResultKey] for downstream [DxirOpResult] lookups —
     * same convention as the multi-result IF/WHILE arms.
     */
    private fun evalCoarsened(
        op: DxirOp,
        env: MutableMap<Int, FloatArray>,
        multiResults: MutableMap<Long, FloatArray>,
    ): FloatArray {
        val primal = op.attrs["primal_body"] as? DxirFunction
            ?: error("DxirInterpreter: COARSENED op id=${op.id} missing `primal_body` attr")
        val inputs = op.operands.map { evalNode(it, env, multiResults) }
        val outputs = evalFunction(primal, inputs)
        require(outputs.size == op.types.size) {
            "DxirInterpreter: COARSENED op id=${op.id} primal_body returned ${outputs.size} values " +
                "but op.types has ${op.types.size}"
        }
        for (i in 1 until outputs.size) {
            multiResults[multiResultKey(op.id, i)] = outputs[i]
        }
        return outputs[0]
    }

    /**
     * Evaluate an entire [DxirFunction]. [inputs] are aligned to [DxirFunction.params]
     * positionally; each input's size must match its param's typed length
     * (`product(param.type.dims)`). Body nodes are evaluated in program order; the
     * returned list pairs with [DxirFunction.returns].
     */
    fun evalFunction(fn: DxirFunction, inputs: List<FloatArray>): List<FloatArray> {
        require(fn.params.size == inputs.size) {
            "evalFunction: param count ${fn.params.size} != input count ${inputs.size}"
        }
        for ((i, p) in fn.params.withIndex()) {
            val expected = sizeOf(p.type)
            require(inputs[i].size == expected) {
                "evalFunction: param '${p.name}' expects size $expected (type ${p.type}) " +
                    "but received input of size ${inputs[i].size}"
            }
        }
        val env: MutableMap<Int, FloatArray> = HashMap()
        val multiResults: MutableMap<Long, FloatArray> = HashMap()
        for ((i, p) in fn.params.withIndex()) env[p.id] = inputs[i]
        for (node in fn.body) evalNode(node, env, multiResults)
        return fn.returns.map { evalNode(it, env, multiResults) }
    }
}
