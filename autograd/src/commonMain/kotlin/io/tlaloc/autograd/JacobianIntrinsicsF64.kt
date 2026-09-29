package io.tlaloc.autograd

import io.tlaloc.core.DTensor
import io.tlaloc.core.F64
import io.tlaloc.core.HostF64Storage
import io.tlaloc.core.Rank2
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym

// The F64 overloads of `jacobian`, `hessian`, `jacobianReverse` and their two-argument forms. Kotlin picks these
// over the generic ones in JacobianIntrinsics.kt for a lambda over `DTensor<S, F64>`
// (they are more specific), so the result is an F64 matrix assembled from F64 passes.
// They are in their own file because each erases to the JVM signature of its generic
// twin.

/** [jacobian] of a function of an F64 tensor: `J[i, j] = ∂yᵢ/∂xⱼ`, F64. */
fun <S : Shape, R> jacobian(f: (DTensor<S, F64>) -> R): (DTensor<S, F64>) -> DTensor<Rank2<Sym, Sym>, F64> =
    { _ -> pluginMissing("jacobian") }

/** [hessian] of a scalar-valued function of an F64 tensor, F64. */
fun <S : Shape, R> hessian(f: (DTensor<S, F64>) -> R): (DTensor<S, F64>) -> DTensor<Rank2<Sym, Sym>, F64> =
    { _ -> pluginMissing("hessian") }

/** [jacobianReverse] of a function of an F64 tensor, F64. */
fun <S : Shape, R> jacobianReverse(f: (DTensor<S, F64>) -> R): (DTensor<S, F64>) -> DTensor<Rank2<Sym, Sym>, F64> =
    { _ -> pluginMissing("jacobianReverse") }

/** [jacobian2] of a function of two F64 tensors: the two Jacobian blocks, F64. */
fun <S1 : Shape, S2 : Shape, R> jacobian2(
    f: (DTensor<S1, F64>, DTensor<S2, F64>) -> R,
): (DTensor<S1, F64>, DTensor<S2, F64>) -> Pair<DTensor<Rank2<Sym, Sym>, F64>, DTensor<Rank2<Sym, Sym>, F64>> =
    { _, _ -> pluginMissing("jacobian2") }

/** [hessian2] of a scalar-valued function of two F64 tensors, F64. */
fun <S1 : Shape, S2 : Shape, R> hessian2(
    f: (DTensor<S1, F64>, DTensor<S2, F64>) -> R,
): (DTensor<S1, F64>, DTensor<S2, F64>) -> DTensor<Rank2<Sym, Sym>, F64> =
    { _, _ -> pluginMissing("hessian2") }

/** [jacobianReverse2] of a function of two F64 tensors, F64. */
fun <S1 : Shape, S2 : Shape, R> jacobianReverse2(
    f: (DTensor<S1, F64>, DTensor<S2, F64>) -> R,
): (DTensor<S1, F64>, DTensor<S2, F64>) -> Pair<DTensor<Rank2<Sym, Sym>, F64>, DTensor<Rank2<Sym, Sym>, F64>> =
    { _, _ -> pluginMissing("jacobianReverse2") }

/** [assembleJacobianForward] for F64 inputs; a scalar-valued `f` returns a `Double` `dy`. */
fun <A, R> assembleJacobianForwardF64(jvp: (A, A) -> R): (A) -> DTensor<Rank2<Sym, Sym>, F64> =
    { x ->
        val (xData, xDims) = hostF64DataOf(x, "jacobian input")
        val n = xData.size
        require(n > 0) { "jacobian: input tensor has zero elements — the column count is undefined" }
        var out: DoubleArray? = null
        var m = 0
        for (j in 0 until n) {
            val basis = DoubleArray(n)
            basis[j] = 1.0
            @Suppress("UNCHECKED_CAST")
            val dx = DTensor<Shape, F64>(HostF64Storage(basis), xDims.copyOf(), F64) as A
            val dyData: DoubleArray = when (val dy = jvp(x, dx)) {
                is Double -> doubleArrayOf(dy)
                else -> hostF64DataOf(dy, "jacobian output tangent").first
            }
            val acc = out ?: DoubleArray(dyData.size * n).also { out = it; m = dyData.size }
            require(dyData.size == m) { "jacobian: output size changed between columns ($m vs ${dyData.size})" }
            for (i in 0 until m) acc[i * n + j] = dyData[i]
        }
        DTensor(HostF64Storage(out!!), intArrayOf(m, n), F64)
    }

/** [assembleHessianForward] for F64 inputs. */
fun <A> assembleHessianForwardF64(hvp: (A, A) -> A): (A) -> DTensor<Rank2<Sym, Sym>, F64> =
    { x ->
        val (xData, xDims) = hostF64DataOf(x, "hessian input")
        val n = xData.size
        require(n > 0) { "hessian: input tensor has zero elements — the row count is undefined" }
        val out = DoubleArray(n * n)
        for (i in 0 until n) {
            val basis = DoubleArray(n)
            basis[i] = 1.0
            @Suppress("UNCHECKED_CAST")
            val v = DTensor<Shape, F64>(HostF64Storage(basis), xDims.copyOf(), F64) as A
            val row = hostF64DataOf(hvp(x, v), "hessian-vector product").first
            require(row.size == n) { "hessian: H·v size ${row.size} does not match input size $n" }
            for (j in 0 until n) out[i * n + j] = row[j]
        }
        DTensor(HostF64Storage(out), intArrayOf(n, n), F64)
    }

/** [assembleJacobianReverse] for F64 inputs; a `Double` primal value is the scalar case. */
fun <A, R> assembleJacobianReverseF64(
    f: (A) -> R,
    vjp: (A, R) -> A,
): (A) -> DTensor<Rank2<Sym, Sym>, F64> =
    { x ->
        val n = hostF64DataOf(x, "jacobianReverse input").first.size
        require(n > 0) { "jacobianReverse: input tensor has zero elements — the column count is undefined" }
        val y = f(x)
        val scalarOut = y is Double
        val yDims: IntArray
        val m: Int
        if (scalarOut) {
            yDims = IntArray(0)
            m = 1
        } else {
            val (yData, dims) = hostF64DataOf(y, "jacobianReverse primal output")
            yDims = dims
            m = yData.size
            require(m > 0) { "jacobianReverse: output tensor has zero elements — the row count is undefined" }
        }
        val out = DoubleArray(m * n)
        for (i in 0 until m) {
            @Suppress("UNCHECKED_CAST")
            val cot: R = if (scalarOut) {
                1.0 as R
            } else {
                val basis = DoubleArray(m)
                basis[i] = 1.0
                DTensor<Shape, F64>(HostF64Storage(basis), yDims.copyOf(), F64) as R
            }
            val row = hostF64DataOf(vjp(x, cot), "jacobianReverse pullback").first
            require(row.size == n) { "jacobianReverse: pullback size ${row.size} does not match input size $n" }
            for (j in 0 until n) out[i * n + j] = row[j]
        }
        DTensor(HostF64Storage(out), intArrayOf(m, n), F64)
    }

/**
 * Runtime row assembly for [jacobianReverse2] — the target of the plugin
 * rewrite, not user API. The [assembleJacobianReverse] mechanism at arity 2:
 * takes the ORIGINAL user lambda `f` (its eager host execution is the primal,
 * run ONCE to learn the output extent `m` and dims) and the synthesised
 * seeded reverse pullback `vjp2(x, w, ȳ) → (x̄, w̄)`; each of the `m`
 * pullback passes writes `x̄` into row `i` of `J_x` (`[m, nx]`) and `w̄`
 * into row `i` of `J_w` (`[m, nw]`). A `Double` primal value means the
 * scalar degenerate: `m = 1`, the cotangent is the unit seed `1.0`.
 */
fun <A, B, R> assembleJacobianReverse2F64(
    f: (A, B) -> R,
    vjp: (A, B, R) -> Pair<A, B>,
): (A, B) -> Pair<DTensor<Rank2<Sym, Sym>, F64>, DTensor<Rank2<Sym, Sym>, F64>> =
    { x, w ->
        val (xData, _) = hostF64DataOf(x, "jacobianReverse2 input x")
        val (wData, _) = hostF64DataOf(w, "jacobianReverse2 input w")
        val nx = xData.size
        val nw = wData.size
        require(nx > 0 && nw > 0) {
            "jacobianReverse2: an input tensor has zero elements — the column count is undefined"
        }
        val y = f(x, w)
        val scalarOut = y is Double
        val yDims: IntArray
        val m: Int
        if (scalarOut) {
            yDims = IntArray(0)
            m = 1
        } else {
            val (yData, dims) = hostF64DataOf(y, "jacobianReverse2 primal output")
            yDims = dims
            m = yData.size
            require(m > 0) {
                "jacobianReverse2: output tensor has zero elements — the row count is undefined"
            }
        }
        val outX = DoubleArray(m * nx)
        val outW = DoubleArray(m * nw)
        for (i in 0 until m) {
            val cot: R = if (scalarOut) {
                @Suppress("UNCHECKED_CAST")
                (1.0 as R)
            } else {
                val basis = DoubleArray(m)
                basis[i] = 1.0
                @Suppress("UNCHECKED_CAST")
                (DTensor<Shape, F64>(HostF64Storage(basis), yDims.copyOf(), F64) as R)
            }
            val (xbar, wbar) = vjp(x, w, cot)
            val rowX = hostF64DataOf(xbar, "jacobianReverse2 x-pullback").first
            val rowW = hostF64DataOf(wbar, "jacobianReverse2 w-pullback").first
            require(rowX.size == nx && rowW.size == nw) {
                "jacobianReverse2: pullback sizes (${rowX.size}, ${rowW.size}) do not match " +
                    "input sizes ($nx, $nw)"
            }
            for (j in 0 until nx) outX[i * nx + j] = rowX[j]
            for (j in 0 until nw) outW[i * nw + j] = rowW[j]
        }
        Pair(
            DTensor(HostF64Storage(outX), intArrayOf(m, nx), F64),
            DTensor(HostF64Storage(outW), intArrayOf(m, nw), F64),
        )
    }

/**
 * Runtime column assembly for [jacobian2] — the target of the plugin rewrite,
 * not user API. Given the synthesised `jvp2(x, w, dx, dw) → dy`, evaluates
 * one forward pass per basis vector of EACH input (the other input's tangent
 * held at zero) and writes `dy` into the matching column of `J_x` (`[m, nx]`)
 * or `J_w` (`[m, nw]`). Accepts a `Double` `dy` (scalar-valued `f` degenerates
 * to the two `[1, n]` gradient rows).
 */
fun <A, B, R> assembleJacobian2ForwardF64(
    jvp: (A, B, A, B) -> R,
): (A, B) -> Pair<DTensor<Rank2<Sym, Sym>, F64>, DTensor<Rank2<Sym, Sym>, F64>> =
    { x, w ->
        val (xData, xDims) = hostF64DataOf(x, "jacobian2 input x")
        val (wData, wDims) = hostF64DataOf(w, "jacobian2 input w")
        val nx = xData.size
        val nw = wData.size
        require(nx > 0 && nw > 0) {
            "jacobian2: an input tensor has zero elements — the column count is undefined"
        }
        var m = -1
        var outX: DoubleArray? = null
        var outW: DoubleArray? = null
        for (j in 0 until nx + nw) {
            val dx = DoubleArray(nx)
            val dw = DoubleArray(nw)
            if (j < nx) dx[j] = 1.0 else dw[j - nx] = 1.0
            @Suppress("UNCHECKED_CAST")
            val dxT = DTensor<Shape, F64>(HostF64Storage(dx), xDims.copyOf(), F64) as A
            @Suppress("UNCHECKED_CAST")
            val dwT = DTensor<Shape, F64>(HostF64Storage(dw), wDims.copyOf(), F64) as B
            val dy = jvp(x, w, dxT, dwT)
            val dyData: DoubleArray = when (dy) {
                is Double -> doubleArrayOf(dy)
                else -> hostF64DataOf(dy, "jacobian2 output tangent").first
            }
            if (m < 0) {
                m = dyData.size
                outX = DoubleArray(m * nx)
                outW = DoubleArray(m * nw)
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
            DTensor(HostF64Storage(outX!!), intArrayOf(m, nx), F64),
            DTensor(HostF64Storage(outW!!), intArrayOf(m, nw), F64),
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
fun <A, B> assembleHessian2ForwardF64(
    hvp: (A, B, A, B) -> Pair<A, B>,
): (A, B) -> DTensor<Rank2<Sym, Sym>, F64> =
    { x, w ->
        val (xData, xDims) = hostF64DataOf(x, "hessian2 input x")
        val (wData, wDims) = hostF64DataOf(w, "hessian2 input w")
        val nx = xData.size
        val nw = wData.size
        require(nx > 0 && nw > 0) {
            "hessian2: an input tensor has zero elements — the row count is undefined"
        }
        val n = nx + nw
        val out = DoubleArray(n * n)
        for (j in 0 until n) {
            val dx = DoubleArray(nx)
            val dw = DoubleArray(nw)
            if (j < nx) dx[j] = 1.0 else dw[j - nx] = 1.0
            @Suppress("UNCHECKED_CAST")
            val dxT = DTensor<Shape, F64>(HostF64Storage(dx), xDims.copyOf(), F64) as A
            @Suppress("UNCHECKED_CAST")
            val dwT = DTensor<Shape, F64>(HostF64Storage(dw), wDims.copyOf(), F64) as B
            val (tx, tw) = hvp(x, w, dxT, dwT)
            val txData = hostF64DataOf(tx, "hessian2 x-gradient tangent").first
            val twData = hostF64DataOf(tw, "hessian2 w-gradient tangent").first
            require(txData.size == nx && twData.size == nw) {
                "hessian2: H·v block sizes (${txData.size}, ${twData.size}) do not match " +
                    "input sizes ($nx, $nw)"
            }
            for (r in 0 until nx) out[r * n + j] = txData[r]
            for (r in 0 until nw) out[(nx + r) * n + j] = twData[r]
        }
        DTensor(HostF64Storage(out), intArrayOf(n, n), F64)
    }

private fun hostF64DataOf(t: Any?, what: String): Pair<DoubleArray, IntArray> {
    val dt = t as? DTensor<*, *> ?: throw IllegalArgumentException(
        "Tlaloc: $what must be a DTensor (got ${t?.let { it::class.simpleName } ?: "null"})",
    )
    val st = dt.storage as? HostF64Storage ?: throw IllegalArgumentException(
        "Tlaloc: $what must be a host F64 tensor (got storage ${dt.storage::class.simpleName})",
    )
    return st.data to dt.dims
}
