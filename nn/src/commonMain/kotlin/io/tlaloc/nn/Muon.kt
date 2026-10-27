package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import kotlin.math.max
import kotlin.math.sqrt

/** [Muon]'s state: a momentum buffer per matrix it updates, and its [Muon.fallback]'s state. */
class MuonState internal constructor(
    val momentum: Map<String, DTensor<*, F32>>,
    val adam: AdamState,
)

/** How [Muon] scales its learning rate for a `[rows, cols]` matrix. */
enum class MuonLrAdjust {
    /** `√max(1, rows / cols)` (Keller Jordan's Muon; PyTorch's default, `"original"`). */
    ORIGINAL,

    /** `0.2·√max(rows, cols)`, so the update's RMS matches AdamW's and AdamW's learning rate and
     * weight decay carry over (Moonshot; PyTorch's `"match_rms_adamw"`). */
    MATCH_RMS_ADAMW,

    /** No scaling. */
    NONE,
}

/**
 * Muon: momentum, then the update orthogonalized by a Newton–Schulz iteration, for the 2-D
 * parameters [useMuon] selects; [fallback] (AdamW) updates the others (biases, norms, and by
 * the usual choice embeddings and the output head, which [useMuon] can exclude by key).
 *
 * As PyTorch's `torch.optim.Muon`, per matrix:
 *
 *   buf' = buf + (1 − μ)(g − buf);  u = g + μ(buf' − g) with [nesterov], buf' without
 *   O = NS(u): X = u (transposed when it has more rows than columns), X ← X / max(‖X‖_F, ε),
 *              [nsSteps] times: A = X·Xᵀ, X ← a·X + (b·A + c·A·A)·X
 *   θ' = θ·(1 − lr·weightDecay) − lr·s·O, s from [lrAdjust]
 *
 * PyTorch runs the iteration in bf16; this runs it in f32. [withDeviceMuon] compiles the same
 * update into the training step.
 */
class Muon(
    val learningRate: Float = 0.001f,
    val weightDecay: Float = 0.1f,
    val momentum: Float = 0.95f,
    val nesterov: Boolean = true,
    val nsSteps: Int = 5,
    val nsA: Float = 3.4445f,
    val nsB: Float = -4.7750f,
    val nsC: Float = 2.0315f,
    val eps: Float = 1e-7f,
    val lrAdjust: MuonLrAdjust = MuonLrAdjust.ORIGINAL,
    val fallback: AdamW = AdamW(),
    val useMuon: (key: String, dims: IntArray) -> Boolean = { _, dims -> dims.size == 2 },
) : CheckpointableOptimizer<MuonState> {

    init {
        require(momentum in 0f..1f) { "Muon: momentum must be in [0, 1] (got $momentum)" }
        require(weightDecay >= 0f) { "Muon: weightDecay must be >= 0 (got $weightDecay)" }
        require(nsSteps in 0..99) { "Muon: nsSteps must be in [0, 99] (got $nsSteps)" }
    }

    /** Whether Muon updates the parameter [key] of [dims]; only 2-D parameters qualify. */
    fun updatesWithMuon(key: String, dims: IntArray): Boolean = dims.size == 2 && useMuon(key, dims)

    /** The learning-rate scale for a `[rows, cols]` matrix. */
    fun lrScale(rows: Int, cols: Int): Float = when (lrAdjust) {
        MuonLrAdjust.ORIGINAL -> sqrt(max(1f, rows.toFloat() / cols))
        MuonLrAdjust.MATCH_RMS_ADAMW -> 0.2f * sqrt(max(rows, cols).toFloat())
        MuonLrAdjust.NONE -> 1f
    }

    override fun initialState() = MuonState(emptyMap(), fallback.initialState())

    override fun step(
        params: List<NamedParameter>,
        grads: Map<String, DTensor<*, F32>>,
        state: MuonState,
    ): OptimizerStep<MuonState> {
        val (matrices, others) = params.partition { updatesWithMuon(it.key, it.tensor.dims) }
        val known = params.mapTo(HashSet()) { it.key }
        require(known.containsAll(grads.keys)) { "Muon: gradients carry unknown keys ${grads.keys - known}" }
        val updated = LinkedHashMap<String, DTensor<*, F32>>(params.size)
        val buffers = HashMap(state.momentum)
        for (p in matrices) {
            val (rows, cols) = p.tensor.dims.toList()
            val g = requireNotNull(grads[p.key]) { "Muon: no gradient for '${p.key}'" }.hostF32()
            val prev = state.momentum[p.key]?.hostF32() ?: FloatArray(g.size)
            val buf = FloatArray(g.size) { prev[it] + (1f - momentum) * (g[it] - prev[it]) }
            buffers[p.key] = DTensor<Shape, F32>(HostF32Storage(buf), p.tensor.dims.copyOf(), F32)
            val u = if (nesterov) FloatArray(g.size) { g[it] + momentum * (buf[it] - g[it]) } else buf
            val o = orthogonalized(u, rows, cols)
            val theta = p.tensor.hostF32()
            val shrink = 1f - learningRate * weightDecay
            val stepLr = learningRate * lrScale(rows, cols)
            updated[p.key] = DTensor<Shape, F32>(
                HostF32Storage(FloatArray(theta.size) { theta[it] * shrink - stepLr * o[it] }),
                p.tensor.dims.copyOf(), F32,
            )
        }
        val otherKeys = others.mapTo(HashSet()) { it.key }
        val adam = fallback.step(others, grads.filterKeys { it in otherKeys }, state.adam)
        val out = LinkedHashMap<String, DTensor<*, F32>>(params.size)
        for (p in params) out[p.key] = updated[p.key] ?: adam.params.getValue(p.key)
        return OptimizerStep(out, MuonState(buffers, adam.state))
    }

    /** The Newton–Schulz iteration on the row-major `[rows, cols]` matrix [u]. */
    fun orthogonalized(u: FloatArray, rows: Int, cols: Int): FloatArray {
        val tall = rows > cols
        val r = if (tall) cols else rows
        val c = if (tall) rows else cols
        var x = if (tall) transpose(u, rows, cols) else u.copyOf()
        var ss = 0f
        for (v in x) ss += v * v
        val n = sqrt(ss)
        val norm = if (n > eps) n else eps
        for (i in x.indices) x[i] /= norm
        repeat(nsSteps) {
            val gram = matmul(x, transpose(x, r, c), r, c, r)
            val gg = matmul(gram, gram, r, r, r)
            val poly = FloatArray(r * r) { nsB * gram[it] + nsC * gg[it] }
            val px = matmul(poly, x, r, r, c)
            x = FloatArray(r * c) { nsA * x[it] + px[it] }
        }
        return if (tall) transpose(x, r, c) else x
    }

    private fun transpose(a: FloatArray, rows: Int, cols: Int) = FloatArray(a.size) { a[(it % rows) * cols + it / rows] }

    private fun matmul(a: FloatArray, b: FloatArray, m: Int, k: Int, n: Int): FloatArray {
        val out = FloatArray(m * n)
        for (i in 0 until m) for (j in 0 until n) {
            var s = 0f
            for (l in 0 until k) s += a[i * k + l] * b[l * n + j]
            out[i * n + j] = s
        }
        return out
    }

    override val checkpointKind: String get() = "Muon"

    override fun saveState(state: MuonState): OptimizerCheckpoint {
        val adam = fallback.saveState(state.adam)
        return OptimizerCheckpoint(
            checkpointKind,
            adam.scalars,
            OptimizerCheckpoint.grouped("momentum", state.momentum) + adam.tensors,
        )
    }

    override fun loadState(checkpoint: OptimizerCheckpoint): MuonState =
        MuonState(checkpoint.group("momentum"), fallback.loadState(checkpoint))
}
