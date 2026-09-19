package io.tlaloc.core.ops

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith

/**
 * Phase A5c-2 — implicit broadcasting on the elementwise binaries (DiffKT's
 * `broadcast(S1, S2)` under every binary op). Mirrors the dxir interpreter's
 * `binaryBroadcast` pins in `DxirBroadcastBinaryGradTest`, so the host path and
 * the IR path are held to the same NumPy rule.
 */
class BroadcastOpsTest {

    private val row = Tensors.f32Matrix<Sym, Sym>(1, 3, floatArrayOf(1f, 2f, 3f))
    private val col = Tensors.f32Matrix<Sym, Sym>(2, 1, floatArrayOf(10f, 20f))

    /**
     * Resolution pin: same-shape operands must keep the shape-PRESERVING overload
     * from HostOps.kt, whose result carries the precise `Rank2<Sym, Sym>` witness.
     * The broadcasting overload returns `DTensor<Shape, F32>`, so this assignment
     * only compiles if the more specific overload won.
     */
    @Test
    fun sameShapeOperandsKeepTheirStaticWitness() {
        val a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
        val b = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(10f, 20f, 30f, 40f))
        val c: DTensor<Rank2<Sym, Sym>, F32> = a + b
        assertContentEquals(floatArrayOf(11f, 22f, 33f, 44f), c.hostF32())
        assertContentEquals(intArrayOf(2, 2), c.dims)
    }

    /**
     * The case static resolution cannot see: `col` and `row` are BOTH
     * `DTensor<Rank2<Sym, Sym>, F32>`, so Kotlin picks the shape-preserving
     * overload — yet their runtime dims differ ([2,1] vs [1,3]). That overload
     * delegates to the broadcasting walk, so this broadcasts instead of throwing
     * the old "elementwise shape mismatch". It is also exactly what a `grad {}`
     * body sees under -1 sentinel dims.
     */
    @Test
    fun sameStaticTypeWithDifferentRuntimeDimsBroadcasts() {
        val c: DTensor<Rank2<Sym, Sym>, F32> = col + row
        assertContentEquals(intArrayOf(2, 3), c.dims)
        assertContentEquals(floatArrayOf(11f, 12f, 13f, 21f, 22f, 23f), c.hostF32())
    }

    /** [2,1] ⊙ [1,3] → [2,3]: both operands stretch, on different axes. */
    @Test
    fun twoAxisStretchBroadcasts() {
        val sum = col + row
        assertContentEquals(intArrayOf(2, 3), sum.dims)
        assertContentEquals(floatArrayOf(11f, 12f, 13f, 21f, 22f, 23f), sum.hostF32())

        val diff = col - row
        assertContentEquals(floatArrayOf(9f, 8f, 7f, 19f, 18f, 17f), diff.hostF32())

        val prod = col * row
        assertContentEquals(floatArrayOf(10f, 20f, 30f, 20f, 40f, 60f), prod.hostF32())

        val quot = col / row
        assertContentEquals(floatArrayOf(10f, 5f, 10f / 3f, 20f, 10f, 20f / 3f), quot.hostF32())
    }

    /** [3] ⊙ [2,3] → [2,3]: the rank-deficient operand gains a replicated leading axis. */
    @Test
    fun rankExtendingBroadcasts() {
        val v = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f))
        val m = Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(10f, 20f, 30f, 40f, 50f, 60f))
        val c = m * v
        assertContentEquals(intArrayOf(2, 3), c.dims)
        assertContentEquals(floatArrayOf(10f, 40f, 90f, 40f, 100f, 180f), c.hostF32())
        // Operand order must not matter for the commutative ops.
        assertContentEquals(c.hostF32(), (v * m).hostF32())
        // …and must matter for the non-commutative one: v − m, not m − v.
        assertContentEquals(floatArrayOf(-9f, -18f, -27f, -39f, -48f, -57f), (v - m).hostF32())
    }

    /** rank-0 ⊙ [2,2]: a scalar-shaped DTensor splats over every axis. */
    @Test
    fun scalarShapedOperandSplats() {
        val s = Tensors.f32Scalar(3f)
        val m = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
        val c: DTensor<Shape, F32> = s + m
        assertContentEquals(intArrayOf(2, 2), c.dims)
        assertContentEquals(floatArrayOf(4f, 5f, 6f, 7f), c.hostF32())
        assertContentEquals(floatArrayOf(3f, 6f, 9f, 12f), (s * m).hostF32())
    }

    /** Aligned extents that are neither equal nor 1 fail loudly rather than mis-zip. */
    @Test
    fun incompatibleShapesFailFast() {
        val three = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f))
        val four = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f, 4f))
        assertFailsWith<IllegalArgumentException> { three + four }
        val wide = Tensors.f32Matrix<Sym, Sym>(2, 3, FloatArray(6))
        assertFailsWith<IllegalArgumentException> { wide + four }
    }

    /** The named `*Broadcast` entry points the plugin synthesises with. */
    @Test
    fun explicitBroadcastHelpersMatchTheOperators() {
        val viaHelper: DTensor<Rank2<Sym, Sym>, F32> = plusBroadcast(col, row)
        assertContentEquals((col + row).hostF32(), viaHelper.hostF32())
        assertContentEquals(intArrayOf(2, 3), viaHelper.dims)

        val r1: DTensor<Rank1<Sym>, F32> = timesBroadcast(
            Tensors.f32Vector<Sym>(floatArrayOf(2f)),
            Tensors.f32Vector<Sym>(floatArrayOf(3f)),
        )
        assertContentEquals(floatArrayOf(6f), r1.hostF32())

        val scalar: DTensor<ScalarShape, F32> = divBroadcast(
            Tensors.f32Scalar(6f),
            Tensors.f32Scalar(3f),
        )
        assertContentEquals(intArrayOf(), scalar.dims)
        assertContentEquals(floatArrayOf(2f), scalar.hostF32())
    }
}
