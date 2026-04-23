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
}

fun DTensor<*, F32>.hostF32(): FloatArray {
    val s = storage
    require(s is HostF32Storage) {
        "operation requires HostF32Storage, got ${s::class.simpleName}"
    }
    return s.data
}
