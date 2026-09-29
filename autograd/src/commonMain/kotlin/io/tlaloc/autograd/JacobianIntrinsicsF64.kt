package io.tlaloc.autograd

import io.tlaloc.core.DTensor
import io.tlaloc.core.F64
import io.tlaloc.core.HostF64Storage
import io.tlaloc.core.Rank2
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym

// The F64 overloads of `jacobian`, `hessian` and `jacobianReverse`. Kotlin picks these
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

private fun hostF64DataOf(t: Any?, what: String): Pair<DoubleArray, IntArray> {
    val dt = t as? DTensor<*, *> ?: throw IllegalArgumentException(
        "Tlaloc: $what must be a DTensor (got ${t?.let { it::class.simpleName } ?: "null"})",
    )
    val st = dt.storage as? HostF64Storage ?: throw IllegalArgumentException(
        "Tlaloc: $what must be a host F64 tensor (got storage ${dt.storage::class.simpleName})",
    )
    return st.data to dt.dims
}
