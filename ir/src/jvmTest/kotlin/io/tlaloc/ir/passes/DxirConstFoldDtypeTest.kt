package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * §0.4.500 — `DxirReverseTransform`'s scalar constant folding must produce a
 * constant whose VALUE has the width its TYPE claims.
 *
 * It did not. `applyConstFold`'s `asFloatConst` projects every operand to `Float`,
 * and every fold built its replacement as `DxirConst(id, <Float arithmetic>,
 * n.type)` — so an f64 node folded to a node typed `f64` carrying a
 * `java.lang.Float`. Nothing in the IR noticed; the pretty printer did not notice;
 * `validateDxirShapes` does not look at value classes. What noticed was
 * `DxirToIrSynthesis`'s `IrConstImpl(..., IrConstKind.Double, v as Double)`, three
 * phases later and in another module, which threw a bare
 * `java.lang.ClassCastException` out of the K2 IR generation extension. The
 * consumer-visible symptom was that
 *
 *     grad { x: Double -> x * 1.5 }
 *
 * killed the compilation with no Tlaloc diagnostic at all — neither a gradient nor
 * a refusal, which is the one outcome this repository's house rules do not allow.
 *
 * This test is at the layer the defect is at: build an f64 scalar body, run the
 * reverse transform, and assert the invariant directly over the returned DXIR. It
 * is the pin that makes the plugin-level regression test in
 * `CapturedConstantGradientTest` a consequence rather than the only evidence.
 */
class DxirConstFoldDtypeTest {

    private val f32s = DxirType(F32, emptyList())
    private val f64s = DxirType(F64, emptyList())

    @Test
    fun `every folded f64 constant carries a Double, never a Float`() {
        // f(x) = (x + 0.25) * 1.5, in f64. The adjoint chain multiplies constants
        // together, which is exactly what `applyConstFold` folds.
        val fn = DxirBuilder.function("f64_body") {
            val x = param("x", f64s)
            val off = const(0.25, f64s)
            val k = const(1.5, f64s)
            val sum = op(OpKind.ADD, listOf(x, off), f64s)
            listOf(op(OpKind.MUL, listOf(sum, k), f64s))
        }

        val grad = DxirReverseTransform.apply(fn)

        val offenders = grad.body.filterIsInstance<DxirConst>()
            .filter { it.type.dtype == F64 && it.value !is Double }
        assertTrue(
            offenders.isEmpty(),
            "an f64-typed const must hold a Double; these hold something else: " +
                offenders.joinToString { "%${it.id} = ${it.value} (${it.value::class.simpleName})" },
        )
        // And the fold really did run — otherwise the assertion above is vacuous.
        val consts = grad.body.filterIsInstance<DxirConst>().filter { it.type.dtype == F64 }
        assertTrue(consts.isNotEmpty(), "expected the f64 adjoint to carry constants at all")
        // d/dx (x + 0.25)·1.5 = 1.5, which the fold collapses to a single constant.
        assertTrue(
            consts.any { (it.value as Double) == 1.5 },
            "expected the folded derivative constant 1.5; got ${consts.map { it.value }}",
        )
    }

    @Test
    fun `an f64 fold is done in double precision, not widened back from float`() {
        // 0.1 and 0.3 are the classic pair: their f64 product is
        // 0.030000000000000002, and their f32 product widened to f64 is
        // 0.030000001192092896. Folding through the Float projection produced the
        // second one and called it f64 — a single-precision answer wearing a
        // double-precision type, which is why the f64 arm of the fold does its
        // arithmetic in Double rather than just converting at the end.
        val fn = DxirBuilder.function("f64_precision") {
            val x = param("x", f64s)
            val a = const(0.1, f64s)
            val b = const(0.3, f64s)
            val ax = op(OpKind.MUL, listOf(x, a), f64s)
            listOf(op(OpKind.MUL, listOf(ax, b), f64s))
        }

        val grad = DxirReverseTransform.apply(fn)
        val values = grad.body.filterIsInstance<DxirConst>()
            .filter { it.type.dtype == F64 }
            .map { it.value }
        assertTrue(
            values.any { it == 0.1 * 0.3 },
            "the folded derivative must be the f64 product ${0.1 * 0.3}; got $values",
        )
        assertTrue(
            values.none { it == (0.1f * 0.3f).toDouble() },
            "the f32 product widened to f64 (${(0.1f * 0.3f).toDouble()}) must not appear; got $values",
        )
    }

    @Test
    fun `f32 folding is bit-for-bit what it was before the f64 arm existed`() {
        // The f32 arm is untouched on purpose, and this is what says so: the same
        // body in f32 must still fold to the f32 product, as a Float.
        val fn = DxirBuilder.function("f32_body") {
            val x = param("x", f32s)
            val a = const(0.1f, f32s)
            val b = const(0.3f, f32s)
            val ax = op(OpKind.MUL, listOf(x, a), f32s)
            listOf(op(OpKind.MUL, listOf(ax, b), f32s))
        }

        val grad = DxirReverseTransform.apply(fn)
        val consts = grad.body.filterIsInstance<DxirConst>().filter { it.type.dtype == F32 }
        assertTrue(consts.all { it.value is Float }, "an f32 const must hold a Float")
        assertTrue(
            consts.any { it.value == 0.1f * 0.3f },
            "expected the f32 product ${0.1f * 0.3f}; got ${consts.map { it.value }}",
        )
    }

    @Test
    fun `a folded f64 zero is a Double zero`() {
        // The `MUL(x, 0)` short-circuit had its own hard-coded `0.0f`, so it was a
        // second, independent way to put a Float in an f64 node.
        val fn = DxirBuilder.function("f64_zero") {
            val x = param("x", f64s)
            val zero = const(0.0, f64s)
            listOf(op(OpKind.MUL, listOf(x, zero), f64s))
        }

        val grad = DxirReverseTransform.apply(fn)
        val f64Consts = grad.body.filterIsInstance<DxirConst>().filter { it.type.dtype == F64 }
        assertTrue(f64Consts.all { it.value is Double }, "got ${f64Consts.map { it.value to it.value::class.simpleName }}")
        assertEquals(
            0.0, f64Consts.mapNotNull { it.value as? Double }.lastOrNull(),
            "d/dx (x · 0) = 0",
        )
    }

    /**
     * §0.4.508 — the fold DECISION, not the folded value.
     *
     * §0.4.500 made the folded constant carry a value of the node's own width and left
     * the identity predicates comparing the `Float` PROJECTION of the operand, on the
     * argument that the comparison targets (0 and 1) are exactly representable. They
     * are; the projection of the operand is not. `1.0 + 1e-10` is a distinct double
     * that `toFloat()` rounds to exactly `1.0f`, so `MUL`'s `a == 1.0f` arm fired and
     * returned the other operand unchanged — d/dx of `x · 1.0000000001` came back as
     * `1.0`. A single-precision answer inside an f64 program, arrived at by a
     * single-precision decision, with nothing anywhere reporting a loss.
     */
    @Test
    fun `an f64 constant a float projection would round to 1 does not fold as identity`() {
        val nearlyOne = 1.0 + 1e-10
        assertEquals(1.0f, nearlyOne.toFloat(), "premise: this double projects to exactly 1.0f")
        assertTrue(nearlyOne != 1.0, "premise: but it is not 1.0")

        // The shape matters, and getting it wrong makes this test vacuous: for
        // `x · k` the adjoint multiplies the SEED (exactly 1.0) by k, so the identity
        // arm fires on the seed and correctly hands back k — k survives either way.
        // The defect needs k itself in the identity position against a NON-constant
        // operand, which `(x · x) · k` produces: adj(x) = MUL(k, x).
        val fn = DxirBuilder.function("f64_near_one") {
            val x = param("x", f64s)
            val k = const(nearlyOne, f64s)
            val sq = op(OpKind.MUL, listOf(x, x), f64s)
            listOf(op(OpKind.MUL, listOf(sq, k), f64s))
        }

        val grad = DxirReverseTransform.apply(fn)
        val f64Consts = grad.body.filterIsInstance<DxirConst>()
            .filter { it.type.dtype == F64 }
            .mapNotNull { it.value as? Double }

        // d/dx (x · k) = k. The gradient must still carry k itself, at full width.
        assertTrue(
            f64Consts.contains(nearlyOne),
            "the derivative must be the constant itself, not its f32 projection; " +
                "f64 consts in the gradient were $f64Consts",
        )
        assertFalse(
            f64Consts.contains(1.0) && !f64Consts.contains(nearlyOne),
            "folding to exactly 1.0 is the pre-§0.4.508 defect: $f64Consts",
        )
    }

    /**
     * §0.4.508 — the same defect at the other end of the range. `1e-50` projects to
     * `0.0f`, so `MUL`'s zero arm collapsed the entire product to a constant zero.
     */
    @Test
    fun `an f64 constant a float projection would round to 0 does not fold to zero`() {
        val tiny = 1e-50
        assertEquals(0.0f, tiny.toFloat(), "premise: this double underflows to 0.0f")
        assertTrue(tiny != 0.0, "premise: but it is not 0.0")

        // `(x · x) · k`, for the reason spelled out in the near-one test above.
        val fn = DxirBuilder.function("f64_tiny") {
            val x = param("x", f64s)
            val k = const(tiny, f64s)
            val sq = op(OpKind.MUL, listOf(x, x), f64s)
            listOf(op(OpKind.MUL, listOf(sq, k), f64s))
        }

        val grad = DxirReverseTransform.apply(fn)
        val f64Consts = grad.body.filterIsInstance<DxirConst>()
            .filter { it.type.dtype == F64 }
            .mapNotNull { it.value as? Double }

        assertTrue(
            f64Consts.contains(tiny),
            "d/dx ((x·x)·1e-50) must still carry 1e-50, not collapse to 0.0; " +
                "f64 consts were $f64Consts",
        )
    }
}
