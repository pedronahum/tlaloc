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
    /**
     * §0.4.400 — rank-1 I32 tensor constructor: the index vectors `embedding`
     * takes. Mirrors [f32Vector] with [HostI32Storage] behind it.
     */
    fun <A : ShapeAtom> i32Vector(data: IntArray): DTensor<Rank1<A>, I32> =
        DTensor(HostI32Storage(data.copyOf()), intArrayOf(data.size), I32)

    /**
     * §0.4.409 — rank-2 I32 tensor constructor: the batched `[B, N]` index
     * matrices `embedding` takes. Mirrors [f32Matrix] with [HostI32Storage].
     */
    fun <A : ShapeAtom, B : ShapeAtom> i32Matrix(rows: Int, cols: Int, data: IntArray): DTensor<Rank2<A, B>, I32> {
        require(data.size == rows * cols) { "data.size=${data.size} does not match rows*cols=${rows * cols}" }
        return DTensor(HostI32Storage(data.copyOf()), intArrayOf(rows, cols), I32)
    }

    /**
     * §0.4.455 (Phase G1a) — bf16 scalar constructor. Takes an f32 value and
     * NARROWS it (round-to-nearest-even, [floatToBf16Bits]); callers hand us
     * f32 because no Kotlin bf16 literal exists. The stored pattern is the
     * rounded value — `bf16Scalar(v).toF32()` is v rounded to bf16, not v.
     */
    fun bf16Scalar(v: Float): DTensor<ScalarShape, BF16> =
        DTensor(HostBf16Storage(shortArrayOf(floatToBf16Bits(v))), intArrayOf(), BF16)

    /** §0.4.455 — [f32Vector]'s bf16 twin: narrows each element (RNE). */
    fun <A : ShapeAtom> bf16Vector(data: FloatArray): DTensor<Rank1<A>, BF16> =
        DTensor(HostBf16Storage(floatArrayToBf16Bits(data)), intArrayOf(data.size), BF16)

    /** §0.4.455 — [f32Matrix]'s bf16 twin: narrows each element (RNE). */
    fun <R : ShapeAtom, C : ShapeAtom> bf16Matrix(
        rows: Int,
        cols: Int,
        data: FloatArray,
    ): DTensor<Rank2<R, C>, BF16> {
        require(data.size == rows * cols) {
            "data.size=${data.size} does not match rows*cols=${rows * cols}"
        }
        return DTensor(HostBf16Storage(floatArrayToBf16Bits(data)), intArrayOf(rows, cols), BF16)
    }

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

/** §0.4.400 — [hostF32]'s I32 twin, for `embedding`'s index vectors. */
fun DTensor<*, I32>.hostI32(): IntArray {
    val s = storage
    require(s is HostI32Storage) {
        "operation requires HostI32Storage, got ${s::class.simpleName}"
    }
    return s.data
}
