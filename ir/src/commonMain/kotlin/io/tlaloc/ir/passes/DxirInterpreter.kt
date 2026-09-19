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
     * Phase A5c — elementwise binary evaluation with NumPy implicit broadcasting.
     *
     * The result shape is the op's own type; each operand right-aligns against it
     * (operand axis `j` maps to result axis `outRank − inRank + j`), an operand axis
     * of extent 1 stretches over the result axis, and axes the operand lacks are
     * replicated. Every aligned pair must be equal, or 1 on the operand side — the
     * same contract [validateDxirShapes] checks statically for equal-rank operands
     * and the emitter's `broadcast_in_dim` injection enforces for XLA.
     *
     * EQUAL-shape operands take the flat zip the interpreter has always used: no
     * stride arithmetic and no per-element index remap, so programs that evaluated
     * before A5c are bit-identical (and just as fast). Only genuinely mixed shapes
     * pay for the broadcast walk.
     */
    private fun binaryBroadcast(
        op: DxirOp,
        env: MutableMap<Int, FloatArray>,
        multiResults: MutableMap<Long, FloatArray>,
        f: (Float, Float) -> Float,
    ): FloatArray {
        val aNode = op.operands[0]
        val bNode = op.operands[1]
        val a = evalNode(aNode, env, multiResults)
        val b = evalNode(bNode, env, multiResults)
        val aDims = aNode.type.dims
        val bDims = bNode.type.dims
        if (aDims == bDims) {
            require(a.size == b.size) {
                "DxirInterpreter: ${op.op} operands have different sizes ${a.size} vs ${b.size}"
            }
            return FloatArray(a.size) { f(a[it], b[it]) }
        }
        val outDims = op.type.dims
        val aStrides = broadcastStrides(op, aDims, outDims, a.size, 0)
        val bStrides = broadcastStrides(op, bDims, outDims, b.size, 1)
        val outStrides = IntArray(outDims.size)
        run {
            var s = 1
            for (k in outDims.indices.reversed()) { outStrides[k] = s; s *= outDims[k] }
        }
        return FloatArray(sizeOf(op.type)) { flat ->
            var rem = flat
            var ia = 0
            var ib = 0
            for (k in outDims.indices) {
                val coord = rem / outStrides[k]
                rem -= coord * outStrides[k]
                ia += coord * aStrides[k]
                ib += coord * bStrides[k]
            }
            f(a[ia], b[ib])
        }
    }

    /**
     * Per-result-axis strides for one operand of a broadcasting binary op: the
     * operand's own row-major strides right-aligned into the result rank, with 0 on
     * the axes it lacks and on its size-1 axes (both replicate, so they never
     * advance the flat index).
     */
    private fun broadcastStrides(
        op: DxirOp,
        inDims: List<Int>,
        outDims: List<Int>,
        inSize: Int,
        operandIndex: Int,
    ): IntArray {
        val r = outDims.size
        require(inDims.size <= r) {
            "DxirInterpreter: ${op.op} operand $operandIndex rank ${inDims.size} exceeds result rank $r"
        }
        var expected = 1
        for (d in inDims) expected *= d
        require(expected == inSize) {
            "DxirInterpreter: ${op.op} operand $operandIndex holds $inSize elements but its shape is $inDims"
        }
        val offset = r - inDims.size
        val own = IntArray(inDims.size)
        run {
            var s = 1
            for (k in inDims.indices.reversed()) { own[k] = s; s *= inDims[k] }
        }
        val strides = IntArray(r)
        for (k in 0 until r) {
            val ik = k - offset
            if (ik < 0) continue
            require(inDims[ik] == outDims[k] || inDims[ik] == 1) {
                "DxirInterpreter: ${op.op} operand $operandIndex axis $ik = ${inDims[ik]} cannot " +
                    "broadcast to result axis $k = ${outDims[k]} (must be equal or 1)"
            }
            if (inDims[ik] != 1) strides[k] = own[ik]
        }
        return strides
    }

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
            // Phase A5c — the elementwise binaries evaluate with NumPy implicit
            // broadcasting when their operand shapes DIFFER: operands right-align
            // against the result shape, a size-1 axis stretches, a rank-deficient
            // operand gains replicated leading axes. Equal-shape operands keep the
            // flat zip they have always had (see [binaryBroadcast]), so every
            // pre-A5c program evaluates bit-identically.
            OpKind.ADD -> binaryBroadcast(op, env, multiResults) { x, y -> x + y }
            OpKind.SUB -> binaryBroadcast(op, env, multiResults) { x, y -> x - y }
            OpKind.MUL -> binaryBroadcast(op, env, multiResults) { x, y -> x * y }
            OpKind.DIV -> binaryBroadcast(op, env, multiResults) { x, y -> x / y }
            OpKind.NEG -> {
                val a = evalNode(op.operands[0], env, multiResults)
                FloatArray(a.size) { -a[it] }
            }
            OpKind.ABS -> {
                // §0.4.167 — element-wise absolute value. AbsRule's adjoint emits
                // STEP(x) - STEP(-x) (= sign(x)); both STEPs route through this
                // interpreter via the existing arm.
                val a = evalNode(op.operands[0], env, multiResults)
                FloatArray(a.size) { kotlin.math.abs(a[it]) }
            }
            OpKind.STEP -> {
                // step(x) = 1 if x > 0 else 0 — matches XLA's GT+select semantics at x=0.
                val a = evalNode(op.operands[0], env, multiResults)
                FloatArray(a.size) { if (a[it] > 0f) 1f else 0f }
            }
            OpKind.SIGN -> {
                // §0.4.204 — sign(x) = +1 / -1 / 0 for x>0 / x<0 / x=0. Used by
                // CartPole's policy `a = sign(tanh(...) - ε)`. SignRule emits a
                // zero gradient for it (sign is non-differentiable at the origin
                // and constant elsewhere); the runtime forward path is what this
                // interpreter arm covers.
                val a = evalNode(op.operands[0], env, multiResults)
                FloatArray(a.size) {
                    when {
                        a[it] > 0f -> 1f
                        a[it] < 0f -> -1f
                        else -> 0f
                    }
                }
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
                // exposes a deviation. Phase A5c routes it through the same
                // broadcasting evaluator as ADD/SUB/MUL/DIV.
                binaryBroadcast(op, env, multiResults) { x, y ->
                    x.toDouble().pow(y.toDouble()).toFloat()
                }
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
            OpKind.SIN -> {
                // §0.4.166 — element-wise sine. SinRule's adjoint emits a fresh COS;
                // both ops route through this interpreter.
                val a = evalNode(op.operands[0], env, multiResults)
                FloatArray(a.size) { kotlin.math.sin(a[it].toDouble()).toFloat() }
            }
            OpKind.COS -> {
                // §0.4.166 — element-wise cosine. CosRule's adjoint emits SIN.
                val a = evalNode(op.operands[0], env, multiResults)
                FloatArray(a.size) { kotlin.math.cos(a[it].toDouble()).toFloat() }
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
                evalTranspose(inputType, perm, a)
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
            OpKind.CONV2D, OpKind.CONV_TRANSPOSE2D -> {
                val lhs = evalNode(op.operands[0], env, multiResults)
                val rhs = evalNode(op.operands[1], env, multiResults)
                evalConv2d(op, lhs, rhs)
            }
            // §0.4.385 — the fused conv adjoints (runtime-solved padding).
            OpKind.CONV2D_DATA_ADJOINT, OpKind.CONV2D_KERNEL_ADJOINT ->
                evalConvAdjoint(op, env, multiResults)
            OpKind.MAXPOOL2D, OpKind.AVGPOOL2D -> {
                evalPool2d(op, evalNode(op.operands[0], env, multiResults))
            }
            // §0.4.386 — the fused avgpool adjoint (runtime-extent).
            OpKind.AVGPOOL2D_GRAD ->
                evalAvgPoolGrad(op, evalNode(op.operands[0], env, multiResults))
            OpKind.BROADCAST -> {
                val a = evalNode(op.operands[0], env, multiResults)
                val outSize = sizeOf(op.type)
                val bcastDims = (op.attrs["broadcast_dimensions"] as? List<*>)
                    ?.map { (it as Number).toInt() }
                    ?: emptyList()
                val inDims = op.operands[0].type.dims
                val outDims = op.type.dims
                // §0.4.359 — the EMPTY broadcast_dimensions form is polymorphic:
                // a scalar seed splats to rank-N (SumRule/MeanRule), and a
                // keepdims-shaped equal-rank input stretches by shape (many
                // reduction VJP rules — Tanh/Max/Min/Softmax adjoints — emit the
                // un-reduce with empty dims). §0.4.371 — the NON-empty form is
                // full `stablehlo.broadcast_in_dim`: input axis j maps to output
                // axis broadcast_dimensions[j]; a size-1 input axis stretches, an
                // unlisted output axis is a NEW replicated axis (the
                // rank-increasing broadcastTo). Matches the emitter's
                // [emitBroadcast]. Empty+equal-rank folds into the general path
                // via a synthesized identity mapping.
                val effectiveDims = if (bcastDims.isEmpty() && a.size != 1 && inDims.size == outDims.size) {
                    inDims.indices.toList()
                } else {
                    bcastDims
                }
                if (effectiveDims.isEmpty()) {
                    require(a.size == 1) {
                        "DxirInterpreter: BROADCAST with empty broadcast_dimensions requires a scalar " +
                            "(size-1) input or an equal-rank keepdims input; got shape $inDims -> $outDims"
                    }
                    FloatArray(outSize) { a[0] }
                } else {
                    val bcastDims = effectiveDims
                    require(bcastDims.size == inDims.size) {
                        "DxirInterpreter: BROADCAST broadcast_dimensions length ${bcastDims.size} " +
                            "must equal input rank ${inDims.size}"
                    }
                    require(bcastDims.all { it in outDims.indices } && bcastDims.toSet().size == bcastDims.size) {
                        "DxirInterpreter: BROADCAST broadcast_dimensions=$bcastDims out of range / " +
                            "not unique for output rank ${outDims.size}"
                    }
                    for (j in inDims.indices) {
                        val od = outDims[bcastDims[j]]
                        require(inDims[j] == od || inDims[j] == 1) {
                            "DxirInterpreter: BROADCAST input dim $j = ${inDims[j]} cannot map to " +
                                "output axis ${bcastDims[j]} = $od (must be equal or 1)"
                        }
                    }
                    val inStrides = IntArray(inDims.size)
                    var st = 1
                    for (k in inDims.indices.reversed()) { inStrides[k] = st; st *= inDims[k] }
                    val outStrides = IntArray(outDims.size)
                    st = 1
                    for (k in outDims.indices.reversed()) { outStrides[k] = st; st *= outDims[k] }
                    FloatArray(outSize) { flat ->
                        var rem = flat
                        var src = 0
                        for (k in outDims.indices) {
                            val coord = rem / outStrides[k]
                            rem %= outStrides[k]
                            val j = bcastDims.indexOf(k)
                            if (j >= 0 && inDims[j] != 1) src += coord * inStrides[j]
                        }
                        a[src]
                    }
                }
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
            OpKind.EMBEDDING -> {
                // §0.4.370 — EMBEDDING(table: rank-2 [V, D], indices: int rank-r)
                // → [indices.dims ++ [D]]. Each output row out[p, :] = table[idx[p], :]
                // for every flat index position p (a gather along the vocab axis).
                require(op.operands.size == 2) {
                    "DxirInterpreter: EMBEDDING requires 2 operands (table, indices), got ${op.operands.size}"
                }
                val tableType = op.operands[0].type
                val idxType = op.operands[1].type
                require(tableType.rank == 2) {
                    "DxirInterpreter: EMBEDDING table must be rank-2 (V, D), got ${tableType.dims}"
                }
                require(idxType.dtype == io.tlaloc.core.I32 || idxType.dtype == io.tlaloc.core.I64) {
                    "DxirInterpreter: EMBEDDING indices must be integer, got ${idxType.dtype}"
                }
                val table = evalNode(op.operands[0], env, multiResults)
                val idx = evalNode(op.operands[1], env, multiResults)
                val vocab = tableType.dims[0]
                val embedDim = tableType.dims[1]
                val positions = idx.size
                val out = FloatArray(positions * embedDim)
                for (p in 0 until positions) {
                    val v = idx[p].toInt()
                    require(v in 0 until vocab) {
                        "DxirInterpreter: EMBEDDING index $v out of bounds for vocab $vocab"
                    }
                    for (d in 0 until embedDim) out[p * embedDim + d] = table[v * embedDim + d]
                }
                out
            }
            OpKind.EMBEDDING_GRAD -> {
                // §0.4.370 — reverse of EMBEDDING w.r.t. the table.
                // EMBEDDING_GRAD(indices: int rank-r, upstream: [indices.dims ++ [D]])
                // → dTable [V, D]. Scatter-ADD each upstream row back to the vocab slot
                // its index selected: dTable[idx[p], :] += upstream[p, :]. The result
                // type [V, D] carries V and D (indices carry no gradient).
                require(op.operands.size == 2) {
                    "DxirInterpreter: EMBEDDING_GRAD requires 2 operands (indices, upstream), got ${op.operands.size}"
                }
                val vocab = op.type.dims[0]
                val embedDim = op.type.dims[1]
                val idx = evalNode(op.operands[0], env, multiResults)
                val upstream = evalNode(op.operands[1], env, multiResults)
                val positions = idx.size
                require(upstream.size == positions * embedDim) {
                    "DxirInterpreter: EMBEDDING_GRAD upstream size ${upstream.size} != positions " +
                        "$positions * embedDim $embedDim"
                }
                val out = FloatArray(vocab * embedDim)
                for (p in 0 until positions) {
                    val v = idx[p].toInt()
                    require(v in 0 until vocab) {
                        "DxirInterpreter: EMBEDDING_GRAD index $v out of bounds for vocab $vocab"
                    }
                    for (d in 0 until embedDim) out[v * embedDim + d] += upstream[p * embedDim + d]
                }
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
            // §0.4.360 — shape-plumbing evals.
            OpKind.CONCAT -> {
                val parts = op.operands.map { evalNode(it, env, multiResults) }
                val dim = (op.attrs["dimension"] as? Number)?.toInt() ?: 0
                val dims0 = op.operands[0].type.dims
                var outer = 1
                for (k in 0 until dim) outer *= dims0[k]
                var inner = 1
                for (k in dim + 1 until dims0.size) inner *= dims0[k]
                val axisLens = op.operands.map { it.type.dims[dim] }
                val out = FloatArray(sizeOf(op.type))
                var dst = 0
                for (o in 0 until outer) {
                    for ((pi, part) in parts.withIndex()) {
                        val len = axisLens[pi] * inner
                        val src = o * len
                        part.copyInto(out, dst, src, src + len)
                        dst += len
                    }
                }
                out
            }
            OpKind.SLICE -> {
                val a = evalNode(op.operands[0], env, multiResults)
                val inDims = op.operands[0].type.dims
                @Suppress("UNCHECKED_CAST")
                val starts = op.attrs["start_indices"] as List<Int>
                @Suppress("UNCHECKED_CAST")
                val limits = op.attrs["limit_indices"] as List<Int>
                @Suppress("UNCHECKED_CAST")
                val strides = op.attrs["strides"] as List<Int>
                val outDims = op.type.dims
                val inStrides = IntArray(inDims.size)
                var st = 1
                for (k in inDims.indices.reversed()) { inStrides[k] = st; st *= inDims[k] }
                val outStrides = IntArray(outDims.size)
                st = 1
                for (k in outDims.indices.reversed()) { outStrides[k] = st; st *= outDims[k] }
                FloatArray(sizeOf(op.type)) { flat ->
                    var rem = flat
                    var src = 0
                    for (k in outDims.indices) {
                        val coord = rem / outStrides[k]
                        rem %= outStrides[k]
                        src += (starts[k] + coord * strides[k]) * inStrides[k]
                    }
                    a[src]
                }
            }
            OpKind.WHERE -> {
                val pred = evalNode(op.operands[0], env, multiResults)
                val a = evalNode(op.operands[1], env, multiResults)
                val b = evalNode(op.operands[2], env, multiResults)
                FloatArray(a.size) { if (pred[it] != 0f) a[it] else b[it] }
            }
            OpKind.COMPARE -> {
                val a = evalNode(op.operands[0], env, multiResults)
                val b = evalNode(op.operands[1], env, multiResults)
                val dir = op.attrs["direction"] as? String ?: error("COMPARE missing `direction` attr")
                FloatArray(a.size) {
                    val hit = when (dir) {
                        "EQ" -> a[it] == b[it]
                        "NE" -> a[it] != b[it]
                        "LT" -> a[it] < b[it]
                        "LE" -> a[it] <= b[it]
                        "GT" -> a[it] > b[it]
                        "GE" -> a[it] >= b[it]
                        else -> error("COMPARE: unknown direction `$dir`")
                    }
                    if (hit) 1f else 0f
                }
            }
            OpKind.PAD -> {
                val a = evalNode(op.operands[0], env, multiResults)
                val inDims = op.operands[0].type.dims
                @Suppress("UNCHECKED_CAST")
                val low = op.attrs["low"] as List<Int>
                val outDims = op.type.dims
                val inStrides = IntArray(inDims.size)
                var st = 1
                for (k in inDims.indices.reversed()) { inStrides[k] = st; st *= inDims[k] }
                val outStrides = IntArray(outDims.size)
                st = 1
                for (k in outDims.indices.reversed()) { outStrides[k] = st; st *= outDims[k] }
                val out = FloatArray(sizeOf(op.type))
                for (flat in a.indices) {
                    var rem = flat
                    var dst = 0
                    for (k in inDims.indices) {
                        val coord = rem / inStrides[k]
                        rem %= inStrides[k]
                        dst += (coord + low[k]) * outStrides[k]
                    }
                    out[dst] = a[flat]
                }
                out
            }
            // §0.4.359 — element-count-preserving relayout: row-major copy.
            OpKind.RESHAPE -> {
                val a = evalNode(op.operands[0], env, multiResults)
                require(a.size == sizeOf(op.type)) {
                    "DxirInterpreter: RESHAPE element count ${a.size} != ${sizeOf(op.type)}"
                }
                a.copyOf()
            }
            // §0.4.359 — axis-aware max/min reduction (the SUM skeleton with a
            // different accumulator; keepdims and dropped-dims outputs have the
            // same flat size, so op.type disambiguates for free).
            OpKind.MAX, OpKind.MIN -> {
                val a = evalNode(op.operands[0], env, multiResults)
                @Suppress("UNCHECKED_CAST")
                val reduceDims = (op.attrs["reduction_dims"] as? List<Int>)
                    ?: (0 until op.operands[0].type.rank).toList()
                val inputDims = op.operands[0].type.dims
                val isMax = op.op == OpKind.MAX
                if (reduceDims.size == inputDims.size || inputDims.isEmpty()) {
                    var acc = if (isMax) Float.NEGATIVE_INFINITY else Float.POSITIVE_INFINITY
                    for (x in a) acc = if (isMax) maxOf(acc, x) else minOf(acc, x)
                    floatArrayOf(acc)
                } else {
                    val keepDims = (0 until inputDims.size).filter { it !in reduceDims }
                    val outShape = keepDims.map { inputDims[it] }
                    val outSize = if (outShape.isEmpty()) 1 else outShape.fold(1) { acc, d -> acc * d }
                    val out = FloatArray(outSize) { if (isMax) Float.NEGATIVE_INFINITY else Float.POSITIVE_INFINITY }
                    val inStrides = IntArray(inputDims.size)
                    var st = 1
                    for (k in inputDims.indices.reversed()) { inStrides[k] = st; st *= inputDims[k] }
                    for (flat in a.indices) {
                        var rem = flat
                        var outIdx = 0
                        var outStride = 1
                        // project by dropping reduced axes (matches SUM's projection).
                        val coords = IntArray(inputDims.size)
                        for (k in inputDims.indices) { coords[k] = rem / inStrides[k]; rem %= inStrides[k] }
                        for (k in keepDims.indices.reversed()) {
                            outIdx += coords[keepDims[k]] * outStride
                            outStride *= inputDims[keepDims[k]]
                        }
                        out[outIdx] = if (isMax) maxOf(out[outIdx], a[flat]) else minOf(out[outIdx], a[flat])
                    }
                    out
                }
            }
            // §0.4.359 — numerically-stable softmax along the `axis` attr
            // (default last), matching the emitter's max-subtracting lowering.
            OpKind.SOFTMAX -> {
                val a = evalNode(op.operands[0], env, multiResults)
                val dims = op.operands[0].type.dims
                val rank = dims.size
                val axisRaw = (op.attrs["axis"] as? Number)?.toInt() ?: (rank - 1)
                val axis = if (axisRaw < 0) axisRaw + rank else axisRaw
                val axisLen = dims[axis]
                var inner = 1
                for (k in axis + 1 until rank) inner *= dims[k]
                var outer = 1
                for (k in 0 until axis) outer *= dims[k]
                val out = FloatArray(a.size)
                for (o in 0 until outer) {
                    for (i in 0 until inner) {
                        val base = o * axisLen * inner + i
                        var mx = Float.NEGATIVE_INFINITY
                        for (j in 0 until axisLen) mx = maxOf(mx, a[base + j * inner])
                        var sum = 0f
                        for (j in 0 until axisLen) {
                            val e = kotlin.math.exp((a[base + j * inner] - mx).toDouble()).toFloat()
                            out[base + j * inner] = e
                            sum += e
                        }
                        for (j in 0 until axisLen) out[base + j * inner] /= sum
                    }
                }
                out
            }
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
            // §0.4.373 — SUM_TO (numpy unbroadcast): reduce operand[0] (value,
            // shape U) down to operand[1] (template, shape T) — the reverse
            // mirror of BROADCAST's in-place size-1 stretch. Sum over the
            // leading (U.rank − T.rank) axes AND over aligned axes where T == 1
            // but U > 1, keeping those axes size-1. Template contributes SHAPE
            // ONLY (its values are never evaluated — we read `.type.dims`).
            OpKind.SUM_TO -> {
                val value = evalNode(op.operands[0], env, multiResults)
                val uDims = op.operands[0].type.dims
                val tDims = op.operands[1].type.dims
                val ru = uDims.size
                val rt = tDims.size
                require(rt <= ru) {
                    "DxirInterpreter: SUM_TO template rank $rt exceeds value rank $ru"
                }
                val offset = ru - rt
                for (i in 0 until rt) {
                    require(tDims[i] == uDims[offset + i] || tDims[i] == 1) {
                        "DxirInterpreter: SUM_TO template dim $i = ${tDims[i]} incompatible with " +
                            "value axis ${offset + i} = ${uDims[offset + i]} (must be equal or 1)"
                    }
                }
                var outSize = 1
                for (d in tDims) outSize *= d
                val out = FloatArray(outSize)
                // Row-major strides.
                val inStrides = IntArray(ru)
                run { var s = 1; for (i in ru - 1 downTo 0) { inStrides[i] = s; s *= uDims[i] } }
                val outStrides = IntArray(rt)
                run { var s = 1; for (i in rt - 1 downTo 0) { outStrides[i] = s; s *= tDims[i] } }
                for (flat in value.indices) {
                    var rem = flat
                    var outIdx = 0
                    for (k in 0 until ru) {
                        val coord = rem / inStrides[k]
                        rem -= coord * inStrides[k]
                        val tAxis = k - offset
                        // Leading axes (tAxis < 0) and size-1-stretched aligned
                        // axes (tDims == 1) fold into index 0 → summed.
                        if (tAxis >= 0 && tDims[tAxis] != 1) outIdx += coord * outStrides[tAxis]
                    }
                    out[outIdx] += value[flat]
                }
                out
            }
            // §0.4.374 — PAD_TO (zero-pad to template): place operand[0] (value,
            // shape U) into a zero tensor of operand[1] (template, shape T = op.type)
            // at offset `low` per axis — the reverse mirror of SLICE. The trailing
            // pad is derived from the template's shape (`high[i] = T[i] − low[i] −
            // U[i]`), never an attr. Template contributes SHAPE ONLY (`.type.dims`).
            OpKind.PAD_TO -> {
                val value = evalNode(op.operands[0], env, multiResults)
                val vDims = op.operands[0].type.dims
                val tDims = op.type.dims
                val r = tDims.size
                require(vDims.size == r) {
                    "DxirInterpreter: PAD_TO value rank ${vDims.size} != template rank $r"
                }
                @Suppress("UNCHECKED_CAST")
                val low = op.attrs["low"] as List<Int>
                require(low.size == r) { "DxirInterpreter: PAD_TO `low` size ${low.size} != rank $r" }
                for (i in 0 until r) {
                    require(low[i] >= 0 && low[i] + vDims[i] <= tDims[i]) {
                        "DxirInterpreter: PAD_TO axis $i: low ${low[i]} + value ${vDims[i]} exceeds template ${tDims[i]}"
                    }
                }
                val inStrides = IntArray(r)
                run { var s = 1; for (i in r - 1 downTo 0) { inStrides[i] = s; s *= vDims[i] } }
                val outStrides = IntArray(r)
                run { var s = 1; for (i in r - 1 downTo 0) { outStrides[i] = s; s *= tDims[i] } }
                val out = FloatArray(sizeOf(op.type))
                for (flat in value.indices) {
                    var rem = flat
                    var dst = 0
                    for (k in 0 until r) {
                        val coord = rem / inStrides[k]
                        rem -= coord * inStrides[k]
                        dst += (coord + low[k]) * outStrides[k]
                    }
                    out[dst] = value[flat]
                }
                out
            }
            // Phase A2b — SLICE_LIKE (CONCAT's adjoint): the window along `axis`
            // starts at the sum of the PRIOR templates' axis extents and runs for
            // `thisTemplate`'s, every other axis taken whole. Mirrors the CONCAT
            // arm's outer/inner block copy: `outer` whole rows before the axis,
            // `inner` contiguous elements after it, so each of the `outer` blocks
            // contributes one contiguous run.
            OpKind.SLICE_LIKE -> {
                val value = evalNode(op.operands[0], env, multiResults)
                val vDims = op.operands[0].type.dims
                val tDims = op.operands[1].type.dims
                val r = vDims.size
                require(tDims.size == r) {
                    "DxirInterpreter: SLICE_LIKE template rank ${tDims.size} != value rank $r"
                }
                val axis = (op.attrs["axis"] as? Number)?.toInt()
                    ?: error("DxirInterpreter: SLICE_LIKE missing `axis` attr")
                require(axis in 0 until r) { "DxirInterpreter: SLICE_LIKE axis $axis outside rank $r" }
                val start = op.operands.drop(2).sumOf { it.type.dims[axis] }
                val len = tDims[axis]
                require(start >= 0 && start + len <= vDims[axis]) {
                    "DxirInterpreter: SLICE_LIKE window [$start, ${start + len}) exceeds the value's " +
                        "axis-$axis extent ${vDims[axis]}"
                }
                for (i in 0 until r) {
                    require(i == axis || tDims[i] == vDims[i]) {
                        "DxirInterpreter: SLICE_LIKE non-axis $i template ${tDims[i]} != value ${vDims[i]}"
                    }
                }
                var outer = 1
                for (k in 0 until axis) outer *= vDims[k]
                var inner = 1
                for (k in axis + 1 until r) inner *= vDims[k]
                val out = FloatArray(outer * len * inner)
                var dst = 0
                for (o in 0 until outer) {
                    val src = o * (vDims[axis] * inner) + start * inner
                    value.copyInto(out, dst, src, src + len * inner)
                    dst += len * inner
                }
                out
            }
            // §0.4.366 — MEAN (Phase A1): the SUM arm divided by the reduced
            // element count. Until now MEAN had no interpreter arm at all — it
            // was unreachable from the user surface (no UNARY_OP_MAP entry) and
            // MeanRule's adjoint never re-emits MEAN. The axis form reuses the
            // SUM projection via a recursive eval of a synthetic SUM op with
            // the same operands/attrs; divisor = product of reduced extents.
            OpKind.MEAN -> {
                val inputType = op.operands[0].type
                @Suppress("UNCHECKED_CAST")
                val reduceDims = (op.attrs["reduction_dims"] as? List<Int>) ?: emptyList()
                val summed = evalOp(
                    DxirOp(op.id, OpKind.SUM, op.operands, op.attrs, op.type),
                    env,
                    multiResults,
                )
                val n = if (reduceDims.isEmpty()) {
                    if (inputType.dims.isEmpty()) 1 else inputType.dims.fold(1) { acc, d -> acc * d }
                } else {
                    reduceDims.fold(1) { acc, d -> acc * inputType.dims[d] }
                }
                FloatArray(summed.size) { summed[it] / n }
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
     * §0.4.136 — rank-N stride-based transpose. The original rank-2-only path
     * covered all paths VjpRules emitted (MatmulRule's `[1, 0]` swap), but
     * §0.4.135's batched-MATMUL substrate implies an eventual batched MatmulRule
     * which would emit `[0, 2, 1]`-style permutations. Handles any valid
     * permutation of any rank.
     *
     * Algorithm: the output element at multi-index (i_0, …, i_{N-1}) corresponds
     * to the input element at multi-index (j_0, …, j_{N-1}) where
     * `j[perm[k]] = i_k`. Equivalently, walking the output in row-major order,
     * the input flat offset accumulates `i_k * inputStrides[perm[k]]` per output
     * axis `k`.
     *
     * §0.4.385 — lifted out of the TRANSPOSE arm so [evalConvAdjoint] can run the
     * batch↔feature swap its kernel gradient needs without materialising
     * throwaway TRANSPOSE nodes.
     */
    private fun evalTranspose(inputType: DxirType, perm: List<Int>, a: FloatArray): FloatArray {
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
            return a.copyOf()
        }
        val outputDims = perm.map { inputType.dims[it] }
        // Row-major strides for input and output. `inputStrides[a]` is the
        // flat offset increment per unit step along input axis `a`.
        val inputStrides = IntArray(rank)
        inputStrides[rank - 1] = 1
        for (i in rank - 2 downTo 0) inputStrides[i] = inputStrides[i + 1] * inputType.dims[i + 1]
        val outputStrides = IntArray(rank)
        outputStrides[rank - 1] = 1
        for (i in rank - 2 downTo 0) outputStrides[i] = outputStrides[i + 1] * outputDims[i + 1]
        return FloatArray(totalSize) { outFlat ->
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

    /**
     * §0.4.362 — general 2-D convolution matching `stablehlo.convolution`
     * semantics for the layouts [StablehloEmitter] fixes: lhs NCHW
     * `[b, f, 0, 1]`, kernel OIHW `[o, i, 0, 1]` for [OpKind.CONV2D] /
     * IOHW `[i, o, 0, 1]` for [OpKind.CONV_TRANSPOSE2D], output NCHW.
     *
     * Honoured attrs: `window_strides` [sH, sW], `padding`
     * [[top, bottom], [left, right]] (negative values crop, as in
     * StableHLO), `lhs_dilation` (interior-dilates the input — the
     * transposed-conv mechanism), `rhs_dilation` (à-trous kernel), and
     * `window_reversal` [Bool, Bool] (spatially flips the kernel taps —
     * what the conv adjoint needs). `feature_group_count` /
     * `batch_group_count` must be 1 in v1.
     *
     * Reference implementation, deliberately direct: for every output
     * element walk the kernel window, map each tap back through stride,
     * dilation, and padding to an input coordinate, and skip taps that
     * land in padding or between lhs-dilation holes.
     */
    private fun evalConv2d(op: DxirOp, lhs: FloatArray, rhs: FloatArray): FloatArray =
        conv2dCore(
            op.op, op.operands[0].type, op.operands[1].type, op.type, op.attrs, lhs, rhs,
        )

    /**
     * §0.4.385 — [evalConv2d]'s body, lifted so a caller can supply the operand and
     * result types EXPLICITLY instead of reading them off a [DxirOp]. The fused conv
     * adjoints need that: they run the transposed/strided conv whose padding they have
     * just solved at runtime, and whose transposed operand types (the batch↔feature
     * swap) exist nowhere in the graph as nodes of their own.
     */
    private fun conv2dCore(
        kind: OpKind,
        lhsT: DxirType,
        rhsT: DxirType,
        outT: DxirType,
        attrs: Map<String, Any>,
        lhs: FloatArray,
        rhs: FloatArray,
    ): FloatArray {
        require(lhsT.rank == 4 && rhsT.rank == 4 && outT.rank == 4) {
            "DxirInterpreter: $kind requires rank-4 lhs/kernel/output; got " +
                "${lhsT.dims} / ${rhsT.dims} / ${outT.dims}"
        }
        fun intPair(key: String, def: List<Int>): List<Int> =
            (attrs[key] as? List<*>)?.map { (it as Number).toInt() } ?: def
        val strides = intPair("window_strides", listOf(1, 1))
        val lhsDil = intPair("lhs_dilation", listOf(1, 1))
        val rhsDil = intPair("rhs_dilation", listOf(1, 1))
        val reversal = (attrs["window_reversal"] as? List<*>)?.map { it as Boolean }
            ?: listOf(false, false)
        val padding = (attrs["padding"] as? List<*>)
            ?.map { row -> (row as List<*>).map { (it as Number).toInt() } }
            ?: listOf(listOf(0, 0), listOf(0, 0))
        val fgc = (attrs["feature_group_count"] as? Int) ?: 1
        val bgc = (attrs["batch_group_count"] as? Int) ?: 1
        require(fgc == 1 && bgc == 1) {
            "DxirInterpreter: $kind feature/batch groups unsupported in v1 (got $fgc/$bgc)"
        }

        val (nB, cIn, h, w) = lhsT.dims
        // Kernel dims under the op's fixed layout.
        val cOut: Int
        val cKIn: Int
        val kh = rhsT.dims[2]
        val kw = rhsT.dims[3]
        if (kind == OpKind.CONV2D) {
            cOut = rhsT.dims[0]; cKIn = rhsT.dims[1]
        } else {
            cKIn = rhsT.dims[0]; cOut = rhsT.dims[1]
        }
        require(cKIn == cIn) {
            "DxirInterpreter: $kind kernel input channels $cKIn ≠ lhs channels $cIn"
        }

        val hDil = (h - 1) * lhsDil[0] + 1
        val wDil = (w - 1) * lhsDil[1] + 1
        val kEffH = (kh - 1) * rhsDil[0] + 1
        val kEffW = (kw - 1) * rhsDil[1] + 1
        val hOut = (hDil + padding[0][0] + padding[0][1] - kEffH) / strides[0] + 1
        val wOut = (wDil + padding[1][0] + padding[1][1] - kEffW) / strides[1] + 1
        require(outT.dims == listOf(nB, cOut, hOut, wOut)) {
            "DxirInterpreter: $kind output type ${outT.dims} ≠ derived " +
                "[$nB, $cOut, $hOut, $wOut]"
        }

        val out = FloatArray(nB * cOut * hOut * wOut)
        var outIdx = 0
        for (n in 0 until nB) {
            for (o in 0 until cOut) {
                for (y in 0 until hOut) {
                    for (x in 0 until wOut) {
                        var acc = 0.0
                        for (i in 0 until cIn) {
                            for (ky in 0 until kh) {
                                // Tap position in the dilated+padded input space.
                                val yDil = y * strides[0] + ky * rhsDil[0] - padding[0][0]
                                if (yDil < 0 || yDil % lhsDil[0] != 0) continue
                                val inY = yDil / lhsDil[0]
                                if (inY >= h) continue
                                val wKy = if (reversal[0]) kh - 1 - ky else ky
                                for (kx in 0 until kw) {
                                    val xDil = x * strides[1] + kx * rhsDil[1] - padding[1][0]
                                    if (xDil < 0 || xDil % lhsDil[1] != 0) continue
                                    val inX = xDil / lhsDil[1]
                                    if (inX >= w) continue
                                    val wKx = if (reversal[1]) kw - 1 - kx else kx
                                    val wIdx = if (kind == OpKind.CONV2D) {
                                        ((o * cIn + i) * kh + wKy) * kw + wKx
                                    } else {
                                        ((i * cOut + o) * kh + wKy) * kw + wKx
                                    }
                                    acc += lhs[((n * cIn + i) * h + inY) * w + inX].toDouble() *
                                        rhs[wIdx]
                                }
                            }
                        }
                        out[outIdx++] = acc.toFloat()
                    }
                }
            }
        }
        return out
    }

    /**
     * §0.4.385 — the fused conv adjoints (see [OpKind.CONV2D_DATA_ADJOINT]). Both
     * solve the classical adjoint padding from the RUNTIME extents of their
     * operands and their shape-only template, then delegate to [conv2dCore] — so
     * there is still exactly one conv evaluator, and the sentinel-safety lives
     * entirely in where the numbers come from.
     *
     * Per spatial axis, with `k` the kernel's spatial extent, `hOut` the
     * upstream's, `H` the target's, `s` the primal `window_strides`, `d` the
     * primal `rhs_dilation` and `p_low` the primal padding:
     *
     *   dX: kEff = (k−1)·d + 1,  dilSize = (hOut−1)·s + 1,
     *       low = kEff − 1 − p_low,   high = p_low + H − dilSize
     *   dW: low = p_low,   high = (k−1)·d + dilSize − H − p_low
     *
     * Algebraically these are the values Conv2dRule used to bake at TRANSFORM
     * time; the difference is that `H` (and `k`, `hOut`) are read at execution,
     * when they exist, instead of out of a type carrying -1 sentinels.
     *
     * The kernel gradient additionally fuses the batch↔feature transpose trick:
     * `dW = (Xᵀ ⋆ dYᵀ)ᵀ` with stride and rhs_dilation swapping roles, so the
     * result type is simply the kernel's and no rank-4 TRANSPOSE nodes are left
     * in the gradient body for the synthesis to type.
     */
    private fun evalConvAdjoint(
        op: DxirOp,
        env: MutableMap<Int, FloatArray>,
        multiResults: MutableMap<Long, FloatArray>,
    ): FloatArray {
        val dataAdj = op.op == OpKind.CONV2D_DATA_ADJOINT
        require(op.operands.size == 3) {
            "DxirInterpreter: ${op.op} takes (upstream, kernel, xTemplate) or " +
                "(x, upstream, wTemplate); got ${op.operands.size} operands"
        }
        // Operand order differs between the two: dX convolves the upstream with
        // the kernel, dW convolves the input with the upstream.
        val upNode = if (dataAdj) op.operands[0] else op.operands[1]
        val otherNode = if (dataAdj) op.operands[1] else op.operands[0]
        val upT = upNode.type
        val otherT = otherNode.type
        val outT = op.type
        require(upT.rank == 4 && otherT.rank == 4 && outT.rank == 4) {
            "DxirInterpreter: ${op.op} requires rank-4 operands and result; got " +
                "${upT.dims} / ${otherT.dims} / ${outT.dims}"
        }
        val up = evalNode(upNode, env, multiResults)
        val other = evalNode(otherNode, env, multiResults)
        // operands[2] is the shape template: its VALUES are never read, and its
        // extents are the result's, which `outT` already carries.

        fun intPair(key: String, def: List<Int>): List<Int> =
            (op.attrs[key] as? List<*>)?.map { (it as Number).toInt() } ?: def
        val s = intPair("window_strides", listOf(1, 1))
        val d = intPair("rhs_dilation", listOf(1, 1))
        val primalPad = (op.attrs["padding"] as? List<*>)
            ?.map { row -> (row as List<*>).map { (it as Number).toInt() } }
            ?: listOf(listOf(0, 0), listOf(0, 0))

        val padding = (0..1).map { a ->
            val dilSize = (upT.dims[2 + a] - 1) * s[a] + 1
            if (dataAdj) {
                // otherNode IS the kernel; the target is the primal input's extent.
                val kEff = (otherT.dims[2 + a] - 1) * d[a] + 1
                val low = kEff - 1 - primalPad[a][0]
                listOf(low, primalPad[a][0] + outT.dims[2 + a] - dilSize)
            } else {
                // otherNode IS the primal input; the target is the kernel's extent.
                val low = primalPad[a][0]
                listOf(
                    low,
                    (outT.dims[2 + a] - 1) * d[a] + dilSize - otherT.dims[2 + a] - low,
                )
            }
        }

        return if (dataAdj) {
            conv2dCore(
                OpKind.CONV_TRANSPOSE2D, upT, otherT, outT,
                mapOf(
                    "window_strides" to listOf(1, 1),
                    "padding" to padding,
                    "lhs_dilation" to s,
                    "rhs_dilation" to d,
                    "window_reversal" to listOf(true, true),
                ),
                up, other,
            )
        } else {
            val swap = listOf(1, 0, 2, 3)
            val xT = evalTranspose(otherT, swap, other)
            val upSwapped = evalTranspose(upT, swap, up)
            val xTT = DxirType(outT.dtype, swap.map { otherT.dims[it] })
            val upTT = DxirType(outT.dtype, swap.map { upT.dims[it] })
            // [Ci, Co, kh, kw] — the kernel's shape with its channel axes swapped.
            val dwtT = DxirType(outT.dtype, swap.map { outT.dims[it] })
            val dwt = conv2dCore(
                OpKind.CONV2D, xTT, upTT, dwtT,
                mapOf(
                    "window_strides" to d,
                    "padding" to padding,
                    "rhs_dilation" to s,
                ),
                xT, upSwapped,
            )
            evalTranspose(dwtT, swap, dwt)
        }
    }

    /**
     * §0.4.363 — 2-D window pooling over NCHW, matching
     * `stablehlo.reduce_window` semantics: MAXPOOL2D reduces each window
     * with max (padding taps contribute −∞, i.e. are skipped); AVGPOOL2D
     * sums each window (padding taps contribute 0) and divides by the
     * **full** window size kh·kw — the count_include_pad convention
     * (PyTorch's AvgPool2d default), which is also what
     * `reduce_window(add) × 1/(kh·kw)` produces on XLA.
     *
     * Attrs: `window` [kh, kw], `window_strides` (default = window — the
     * classic non-overlapping pool), `padding` [[top, bottom], [left,
     * right]] (default 0).
     */
    private fun evalPool2d(op: DxirOp, lhs: FloatArray): FloatArray {
        val lhsT = op.operands[0].type
        require(lhsT.rank == 4 && op.type.rank == 4) {
            "DxirInterpreter: ${op.op} requires rank-4 NCHW input/output; got " +
                "${lhsT.dims} / ${op.type.dims}"
        }
        fun intPair(key: String, def: List<Int>): List<Int> =
            (op.attrs[key] as? List<*>)?.map { (it as Number).toInt() } ?: def
        val window = intPair("window", emptyList())
        require(window.size == 2) { "DxirInterpreter: ${op.op} needs `window` [kh, kw]; got $window" }
        val strides = intPair("window_strides", window)
        val padding = (op.attrs["padding"] as? List<*>)
            ?.map { row -> (row as List<*>).map { (it as Number).toInt() } }
            ?: listOf(listOf(0, 0), listOf(0, 0))

        val (nB, c, h, w) = lhsT.dims
        val hOut = (h + padding[0][0] + padding[0][1] - window[0]) / strides[0] + 1
        val wOut = (w + padding[1][0] + padding[1][1] - window[1]) / strides[1] + 1
        require(op.type.dims == listOf(nB, c, hOut, wOut)) {
            "DxirInterpreter: ${op.op} output type ${op.type.dims} ≠ derived [$nB, $c, $hOut, $wOut]"
        }

        val isMax = op.op == OpKind.MAXPOOL2D
        val windowSize = window[0] * window[1]
        val out = FloatArray(nB * c * hOut * wOut)
        var outIdx = 0
        for (n in 0 until nB) {
            for (ch in 0 until c) {
                val planeBase = (n * c + ch) * h * w
                for (y in 0 until hOut) {
                    for (x in 0 until wOut) {
                        var acc = if (isMax) Double.NEGATIVE_INFINITY else 0.0
                        for (ky in 0 until window[0]) {
                            val inY = y * strides[0] + ky - padding[0][0]
                            if (inY < 0 || inY >= h) continue
                            for (kx in 0 until window[1]) {
                                val inX = x * strides[1] + kx - padding[1][0]
                                if (inX < 0 || inX >= w) continue
                                val v = lhs[planeBase + inY * w + inX].toDouble()
                                acc = if (isMax) maxOf(acc, v) else acc + v
                            }
                        }
                        out[outIdx++] = if (isMax) acc.toFloat() else (acc / windowSize).toFloat()
                    }
                }
            }
        }
        return out
    }

    /**
     * §0.4.386 — [OpKind.AVGPOOL2D_GRAD]: the adjoint of [evalPool2d]'s AVGPOOL2D
     * branch, written as the mirror image of that gather. Each input element
     * collects the upstream of every output window covering it, divided by the FULL
     * window size `kh·kw` — the count_include_pad convention the primal uses, so a
     * padded tap contributes to the denominator exactly as it does on the way in.
     *
     * The window is inverted rather than solved: for a target input row `iy` and tap
     * `ky`, the covering output row is `(iy + padTop − ky) / strideH` when that
     * divides evenly and lands in range. So nothing here needs the primal's extents
     * as attrs — the upstream's runtime shape bounds the loop and the template's
     * bounds the result — which is what makes the op safe under `grad {}`'s -1
     * sentinels. Only the LOW padding participates, for the same reason as the conv
     * adjoints: the high side is implied by the upstream's shape.
     *
     * Same accumulation discipline as the primal (Double, then one Float conversion)
     * and the same loop order, so the host twin can be bit-exact against it.
     */
    private fun evalAvgPoolGrad(op: DxirOp, up: FloatArray): FloatArray {
        val upT = op.operands[0].type
        val outT = op.type
        require(upT.rank == 4 && outT.rank == 4) {
            "DxirInterpreter: ${op.op} requires rank-4 NCHW upstream/result; got " +
                "${upT.dims} / ${outT.dims}"
        }
        fun intPair(key: String, def: List<Int>): List<Int> =
            (op.attrs[key] as? List<*>)?.map { (it as Number).toInt() } ?: def
        val window = intPair("window", emptyList())
        require(window.size == 2) { "DxirInterpreter: ${op.op} needs `window` [kh, kw]; got $window" }
        val strides = intPair("window_strides", window)
        val padding = (op.attrs["padding"] as? List<*>)
            ?.map { row -> (row as List<*>).map { (it as Number).toInt() } }
            ?: listOf(listOf(0, 0), listOf(0, 0))

        val (nB, c, h, w) = outT.dims
        val hOut = upT.dims[2]
        val wOut = upT.dims[3]
        require(upT.dims[0] == nB && upT.dims[1] == c) {
            "DxirInterpreter: ${op.op} upstream batch/channels ${upT.dims.take(2)} ≠ " +
                "target ${outT.dims.take(2)}"
        }
        val windowSize = window[0] * window[1]
        val out = FloatArray(nB * c * h * w)
        var outIdx = 0
        for (n in 0 until nB) {
            for (ch in 0 until c) {
                val upBase = (n * c + ch) * hOut * wOut
                for (iy in 0 until h) {
                    for (ix in 0 until w) {
                        var acc = 0.0
                        for (ky in 0 until window[0]) {
                            val dy = iy + padding[0][0] - ky
                            if (dy < 0 || dy % strides[0] != 0) continue
                            val y = dy / strides[0]
                            if (y >= hOut) continue
                            for (kx in 0 until window[1]) {
                                val dx = ix + padding[1][0] - kx
                                if (dx < 0 || dx % strides[1] != 0) continue
                                val x = dx / strides[1]
                                if (x >= wOut) continue
                                acc += up[upBase + y * wOut + x].toDouble()
                            }
                        }
                        out[outIdx++] = (acc / windowSize).toFloat()
                    }
                }
            }
        }
        return out
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
