package io.tlaloc.core

object Tensors {

    fun f32Scalar(v: Float): DTensor<ScalarShape, F32> =
        DTensor(HostF32Storage(floatArrayOf(v)), intArrayOf(), F32)

    fun <A : ShapeAtom> f32Vector(data: FloatArray): DTensor<Rank1<A>, F32> =
        DTensor(HostF32Storage(data.copyOf()), intArrayOf(data.size), F32)

    fun <R : ShapeAtom, C : ShapeAtom> f32Matrix(
        rows: Int,
        cols: Int,
        data: FloatArray,
    ): DTensor<Rank2<R, C>, F32> {
        require(data.size == rows * cols) {
            "data.size=${data.size} does not match rows*cols=${rows * cols}"
        }
        return DTensor(HostF32Storage(data.copyOf()), intArrayOf(rows, cols), F32)
    }

    fun <R : ShapeAtom, C : ShapeAtom> f32MatrixOf(
        rows: Int,
        cols: Int,
        vararg values: Float,
    ): DTensor<Rank2<R, C>, F32> = f32Matrix(rows, cols, values)

    fun <R : ShapeAtom, C : ShapeAtom> f32Zeros(
        rows: Int,
        cols: Int,
    ): DTensor<Rank2<R, C>, F32> = f32Matrix(rows, cols, FloatArray(rows * cols))

    /**
     * §0.4.97 — rank-3 tensor constructor. Mirrors [f32Matrix] for one axis higher.
     * Used wherever the user has a 3D tensor (a stack of matrices, or a sequence of
     * rank-2 frames, or a single rank-3 conv-input chunk).
     *
     * The phantom shape is `Rank3<A, B, C>`; like [f32Matrix] the caller picks the
     * branding axis types (typically all `Sym` when the dimensions aren't named).
     */
    fun <A : ShapeAtom, B : ShapeAtom, C : ShapeAtom> f32Tensor3(
        d0: Int,
        d1: Int,
        d2: Int,
        data: FloatArray,
    ): DTensor<Rank3<A, B, C>, F32> {
        require(data.size == d0 * d1 * d2) {
            "data.size=${data.size} does not match d0*d1*d2=${d0 * d1 * d2}"
        }
        return DTensor(HostF32Storage(data.copyOf()), intArrayOf(d0, d1, d2), F32)
    }

    /**
     * §0.4.384 — rank-4 tensor constructor, mirroring [f32Tensor3] for the NCHW
     * conv/pool tensors and their OIHW/IOHW kernels (Phase A3b's rank-4
     * substrate). Callers brand the axes as they do for the lower ranks —
     * typically all `Sym`, since the conv spatial extents are runtime facts.
     */
    fun <A : ShapeAtom, B : ShapeAtom, C : ShapeAtom, D : ShapeAtom> f32Tensor4(
        d0: Int,
        d1: Int,
        d2: Int,
        d3: Int,
        data: FloatArray,
    ): DTensor<Rank4<A, B, C, D>, F32> {
        require(data.size == d0 * d1 * d2 * d3) {
            "data.size=${data.size} does not match d0*d1*d2*d3=${d0 * d1 * d2 * d3}"
        }
        return DTensor(HostF32Storage(data.copyOf()), intArrayOf(d0, d1, d2, d3), F32)
    }
}

fun DTensor<*, F32>.hostF32(): FloatArray {
    val s = storage
    require(s is HostF32Storage) {
        "operation requires HostF32Storage, got ${s::class.simpleName}"
    }
    return s.data
}
