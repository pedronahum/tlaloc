package io.tlaloc.autograd

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Rank2
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym

/**
 * §0.4.394 — Phase B2: the `jacobian` / `hessian` user intrinsics (DiffKT's
 * identity-seeded `reverseDerivative` on tensor→tensor `f`, and its
 * second-order analogue).
 *
 * Both curry like [grad]: `jacobian(f)` returns `(x) → J` where
 * `J[i, j] = ∂yᵢ/∂xⱼ` over the ROW-MAJOR FLATTENED input and output
 * (shape `[m, n]` for `f: ℝⁿ → ℝᵐ`); `hessian(f)` returns `(x) → H` with
 * `H[i, j] = ∂²f/∂xᵢ∂xⱼ` (shape `[n, n]`, symmetric for twice-differentiable
 * `f`). Neither shape is statically witnessed — under `grad {}`'s -1 sentinel
 * dims `n` and `m` are runtime quantities — so the result erases to
 * `DTensor<Rank2<Sym, Sym>, F32>`, the `concat`/`slice` convention.
 *
 * Mechanism (the plugin rewrite): the K2 plugin lowers the lambda once and
 * synthesises a SEEDED single-pass derivative —
 *  - `jacobian`: the §0.4.361 forward transform's `jvp_f(x, dx) → dy`
 *    (dual-number, one forward pass per column);
 *  - `hessian`: forward-over-reverse, `hvp_f(x, v) → H·v` — the composition
 *    pinned at IR level since §0.4.361 (the reverse transform's gradient body
 *    is straight-line, and every runtime-extent adjoint op — SUM_TO, PAD_TO,
 *    SLICE_LIKE — has carried a forward tangent from birth, which is why the
 *    Hessian could be forward-OVER-reverse long before those ops had VjpRules
 *    of their own; §0.4.399/§0.4.404 later closed the reverse-over-reverse
 *    route too) —
 * then wraps it in [assembleJacobianForward] / [assembleHessianForward],
 * which loop over the standard basis at RUNTIME (where the actual extents are
 * known) and stack the resulting columns/rows. Cost: n passes of the seeded
 * function — the classic dense-Jacobian trade, matching DiffKT's own
 * identity-seeding loop.
 *
 * These bodies are the no-plugin fallbacks (see [pluginMissing]). Unlike
 * `grad`, there is no runtime-tape path for a failed synthesis — a fallback
 * is a loud error at first call, the `concat` precedent.
 *
 * v1 scope: single-argument `f`, straight-line bodies, F32 host tensors.
 */
fun <A, R> jacobian(f: (A) -> R): (A) -> DTensor<Rank2<Sym, Sym>, F32> =
    { _ -> pluginMissing("jacobian") }

/** Dense Hessian of a scalar-valued `f`, assembled from Hessian-vector products. */
fun <A, R> hessian(f: (A) -> R): (A) -> DTensor<Rank2<Sym, Sym>, F32> =
    { _ -> pluginMissing("hessian") }

/**
 * §0.4.406 — the two-argument Jacobian, closing the "multi-arg
 * `jacobian2`" tail §0.4.394 recorded. `jacobian2(f)` returns
 * `(x, w) → Pair(J_x, J_w)` where `J_x[i, j] = ∂yᵢ/∂xⱼ` (shape `[m, nx]`)
 * and `J_w[i, j] = ∂yᵢ/∂wⱼ` (shape `[m, nw]`), each over the ROW-MAJOR
 * FLATTENED input/output — the same convention as [jacobian], returned as
 * separate per-argument blocks rather than one glued `[m, nx+nw]` matrix
 * (the caller usually wants exactly one of them; gluing is one `concat`).
 *
 * Mechanism: the plugin synthesises `jvp2`'s tangent-only
 * `(x, w, dx, dw) → dy` ONCE and [assembleJacobian2Forward] loops the
 * standard basis on EACH input with a ZERO tangent on the other —
 * `nx + nw` forward passes total, the same dense trade as [jacobian].
 */
fun <A, B, R> jacobian2(
    f: (A, B) -> R,
): (A, B) -> Pair<DTensor<Rank2<Sym, Sym>, F32>, DTensor<Rank2<Sym, Sym>, F32>> =
    { _, _ -> pluginMissing("jacobian2") }

/**
 * §0.4.406 — the two-argument Hessian of a scalar-valued `f(x, w)`,
 * closing the "multi-arg `hessian2`" tail §0.4.394 recorded. Returns the
 * FULL dense Hessian over the CONCATENATED row-major-flattened input
 * `z = (x; w)` — one `[(nx+nw), (nx+nw)]` matrix in the block layout
 *
 * ```
 *   H = [ H_xx  H_xw ]      H[i, j] = ∂²f/∂zᵢ∂zⱼ,   z = (flat x) ++ (flat w)
 *       [ H_wx  H_ww ]
 * ```
 *
 * i.e. rows/columns `0 until nx` index `x`'s flat elements and rows/columns
 * `nx until nx+nw` index `w`'s. One matrix rather than a Quadruple of
 * blocks because (a) it erases to the same `DTensor<Rank2<Sym, Sym>, F32>`
 * as [hessian] — `hessian2(f)(x, w)` EQUALS `hessian(g)(z)` for the
 * concatenated-input equivalent `g(z) = f(z[0:nx], z[nx:])`, so the two
 * surfaces agree by construction — and (b) the block extents are runtime
 * quantities under `grad {}`'s -1 sentinel dims, so no static witness
 * could name them anyway; slice the blocks out at runtime if needed.
 *
 * Mechanism: forward-over-reverse of the TWO-return `grad2` body —
 * `hvp2(x, w, dx, dw) → (H_xx·dx + H_xw·dw, H_wx·dx + H_ww·dw)` — and
 * [assembleHessian2Forward] runs it once per basis vector of each input
 * (zero tangent on the other), writing full columns of `H`.
 */
fun <A, B, R> hessian2(f: (A, B) -> R): (A, B) -> DTensor<Rank2<Sym, Sym>, F32> =
    { _, _ -> pluginMissing("hessian2") }

/**
 * Runtime column assembly for [jacobian] — the target of the plugin rewrite,
 * not user API. Given the synthesised `jvp(x, dx) → dy`, evaluates one
 * forward pass per input basis vector `eⱼ` and writes `dy` into column `j`
 * of the `[m, n]` result. Accepts a `Float` `dy` too (scalar-valued `f`
 * degenerates to the `[1, n]` gradient row).
 */
fun <A, R> assembleJacobianForward(jvp: (A, A) -> R): (A) -> DTensor<Rank2<Sym, Sym>, F32> =
    { x ->
        val (xData, xDims) = hostF32DataOf(x, "jacobian input")
        val n = xData.size
        require(n > 0) { "jacobian: input tensor has zero elements — the column count is undefined" }
        var out: FloatArray? = null
        var m = 0
        for (j in 0 until n) {
            val basis = FloatArray(n)
            basis[j] = 1.0f
            @Suppress("UNCHECKED_CAST")
            val dx = DTensor<Shape, F32>(HostF32Storage(basis), xDims.copyOf(), F32) as A
            val dy = jvp(x, dx)
            val dyData: FloatArray = when (dy) {
                is Float -> floatArrayOf(dy)
                else -> hostF32DataOf(dy, "jacobian output tangent").first
            }
            val acc = out ?: FloatArray(dyData.size * n).also { out = it; m = dyData.size }
            require(dyData.size == m) {
                "jacobian: output size changed between columns ($m vs ${dyData.size})"
            }
            for (i in 0 until m) acc[i * n + j] = dyData[i]
        }
        DTensor(HostF32Storage(out!!), intArrayOf(m, n), F32)
    }

/**
 * Runtime row assembly for [hessian] — the target of the plugin rewrite, not
 * user API. Given the synthesised forward-over-reverse `hvp(x, v) → H·v`,
 * evaluates one pass per basis vector `eᵢ` and writes `H·eᵢ` into row `i` of
 * the `[n, n]` result (for a symmetric Hessian rows and columns coincide).
 */
fun <A> assembleHessianForward(hvp: (A, A) -> A): (A) -> DTensor<Rank2<Sym, Sym>, F32> =
    { x ->
        val (xData, xDims) = hostF32DataOf(x, "hessian input")
        val n = xData.size
        require(n > 0) { "hessian: input tensor has zero elements — the row count is undefined" }
        val out = FloatArray(n * n)
        for (i in 0 until n) {
            val basis = FloatArray(n)
            basis[i] = 1.0f
            @Suppress("UNCHECKED_CAST")
            val v = DTensor<Shape, F32>(HostF32Storage(basis), xDims.copyOf(), F32) as A
            val hv = hvp(x, v)
            val row = hostF32DataOf(hv, "hessian-vector product").first
            require(row.size == n) {
                "hessian: H·v size ${row.size} does not match input size $n"
            }
            for (j in 0 until n) out[i * n + j] = row[j]
        }
        DTensor(HostF32Storage(out), intArrayOf(n, n), F32)
    }

/**
 * Runtime column assembly for [jacobian2] — the target of the plugin rewrite,
 * not user API. Given the synthesised `jvp2(x, w, dx, dw) → dy`, evaluates
 * one forward pass per basis vector of EACH input (the other input's tangent
 * held at zero) and writes `dy` into the matching column of `J_x` (`[m, nx]`)
 * or `J_w` (`[m, nw]`). Accepts a `Float` `dy` (scalar-valued `f` degenerates
 * to the two `[1, n]` gradient rows).
 */
fun <A, B, R> assembleJacobian2Forward(
    jvp: (A, B, A, B) -> R,
): (A, B) -> Pair<DTensor<Rank2<Sym, Sym>, F32>, DTensor<Rank2<Sym, Sym>, F32>> =
    { x, w ->
        val (xData, xDims) = hostF32DataOf(x, "jacobian2 input x")
        val (wData, wDims) = hostF32DataOf(w, "jacobian2 input w")
        val nx = xData.size
        val nw = wData.size
        require(nx > 0 && nw > 0) {
            "jacobian2: an input tensor has zero elements — the column count is undefined"
        }
        var m = -1
        var outX: FloatArray? = null
        var outW: FloatArray? = null
        for (j in 0 until nx + nw) {
            val dx = FloatArray(nx)
            val dw = FloatArray(nw)
            if (j < nx) dx[j] = 1.0f else dw[j - nx] = 1.0f
            @Suppress("UNCHECKED_CAST")
            val dxT = DTensor<Shape, F32>(HostF32Storage(dx), xDims.copyOf(), F32) as A
            @Suppress("UNCHECKED_CAST")
            val dwT = DTensor<Shape, F32>(HostF32Storage(dw), wDims.copyOf(), F32) as B
            val dy = jvp(x, w, dxT, dwT)
            val dyData: FloatArray = when (dy) {
                is Float -> floatArrayOf(dy)
                else -> hostF32DataOf(dy, "jacobian2 output tangent").first
            }
            if (m < 0) {
                m = dyData.size
                outX = FloatArray(m * nx)
                outW = FloatArray(m * nw)
            }
            require(dyData.size == m) {
                "jacobian2: output size changed between columns ($m vs ${dyData.size})"
            }
            if (j < nx) {
                for (i in 0 until m) outX!![i * nx + j] = dyData[i]
            } else {
                for (i in 0 until m) outW!![i * nw + (j - nx)] = dyData[i]
            }
        }
        Pair(
            DTensor(HostF32Storage(outX!!), intArrayOf(m, nx), F32),
            DTensor(HostF32Storage(outW!!), intArrayOf(m, nw), F32),
        )
    }

/**
 * Runtime column assembly for [hessian2] — the target of the plugin rewrite,
 * not user API. Given the synthesised forward-over-reverse
 * `hvp2(x, w, dx, dw) → (t_x̄, t_w̄)` (the directional derivatives of the two
 * gradients), each basis pass with `dx = eⱼ, dw = 0` yields column `j` of the
 * full Hessian's left block-column — `t_x̄ = H_xx·eⱼ` (rows `0..nx-1`) and
 * `t_w̄ = H_wx·eⱼ` (rows `nx..nx+nw-1`) — and each pass with `dx = 0,
 * dw = eⱼ` yields column `nx + j` from `(H_xw·eⱼ; H_ww·eⱼ)`. Columns are
 * exact (no symmetry assumption): `nx + nw` passes fill the whole
 * `[(nx+nw), (nx+nw)]` matrix.
 */
fun <A, B> assembleHessian2Forward(
    hvp: (A, B, A, B) -> Pair<A, B>,
): (A, B) -> DTensor<Rank2<Sym, Sym>, F32> =
    { x, w ->
        val (xData, xDims) = hostF32DataOf(x, "hessian2 input x")
        val (wData, wDims) = hostF32DataOf(w, "hessian2 input w")
        val nx = xData.size
        val nw = wData.size
        require(nx > 0 && nw > 0) {
            "hessian2: an input tensor has zero elements — the row count is undefined"
        }
        val n = nx + nw
        val out = FloatArray(n * n)
        for (j in 0 until n) {
            val dx = FloatArray(nx)
            val dw = FloatArray(nw)
            if (j < nx) dx[j] = 1.0f else dw[j - nx] = 1.0f
            @Suppress("UNCHECKED_CAST")
            val dxT = DTensor<Shape, F32>(HostF32Storage(dx), xDims.copyOf(), F32) as A
            @Suppress("UNCHECKED_CAST")
            val dwT = DTensor<Shape, F32>(HostF32Storage(dw), wDims.copyOf(), F32) as B
            val (tx, tw) = hvp(x, w, dxT, dwT)
            val txData = hostF32DataOf(tx, "hessian2 x-gradient tangent").first
            val twData = hostF32DataOf(tw, "hessian2 w-gradient tangent").first
            require(txData.size == nx && twData.size == nw) {
                "hessian2: H·v block sizes (${txData.size}, ${twData.size}) do not match " +
                    "input sizes ($nx, $nw)"
            }
            for (r in 0 until nx) out[r * n + j] = txData[r]
            for (r in 0 until nw) out[(nx + r) * n + j] = twData[r]
        }
        DTensor(HostF32Storage(out), intArrayOf(n, n), F32)
    }

private fun hostF32DataOf(t: Any?, what: String): Pair<FloatArray, IntArray> {
    val dt = t as? DTensor<*, *> ?: throw IllegalArgumentException(
        "Tlaloc: $what must be a DTensor (got ${t?.let { it::class.simpleName } ?: "null"})",
    )
    val st = dt.storage as? HostF32Storage ?: throw IllegalArgumentException(
        "Tlaloc: $what must be a host F32 tensor (got storage ${dt.storage::class.simpleName})",
    )
    return st.data to dt.dims
}
