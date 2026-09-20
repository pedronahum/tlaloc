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

private fun hostF32DataOf(t: Any?, what: String): Pair<FloatArray, IntArray> {
    val dt = t as? DTensor<*, *> ?: throw IllegalArgumentException(
        "Tlaloc: $what must be a DTensor (got ${t?.let { it::class.simpleName } ?: "null"})",
    )
    val st = dt.storage as? HostF32Storage ?: throw IllegalArgumentException(
        "Tlaloc: $what must be a host F32 tensor (got storage ${dt.storage::class.simpleName})",
    )
    return st.data to dt.dims
}
