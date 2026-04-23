package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.pretty
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * §0.4.26 — tests for [DxirCanonical] serialiser + hash: round-trip equivalence,
 * SSA-renaming hash stability, hash sensitivity to structural changes, and coverage
 * for the surfaces Stage B.4 emits (IF with empty-region adjoint shape + WHILE with
 * counter/carried block args).
 */
class DxirCanonicalTest {

    private val f32s = DxirType(F32, emptyList())
    private val i32s = DxirType(I32, emptyList())
    private val boolS = DxirType(Bool, emptyList())

    // ------------------------------------------------------------------------
    // Round-trip tests — serialise → deserialise → serialise must be idempotent
    // ------------------------------------------------------------------------

    @Test
    fun roundTripScalarFunction() {
        // f(x) = 2x + 1. No regions, just arithmetic + const.
        val fn = DxirBuilder.function("simple") {
            val x = param("x", f32s)
            val two = const(2f, f32s)
            val mul = op(OpKind.MUL, listOf(x, two), f32s)
            val one = const(1f, f32s)
            val add = op(OpKind.ADD, listOf(mul, one), f32s)
            listOf(add)
        }
        assertCanonicalRoundTrip(fn)
    }

    @Test
    fun roundTripIfAdjointShape() {
        // Shape emitted by `DxirReverseTransform.handleIfAdjoint`: IF(pred, {yields thenAdj},
        // {yields elseAdj}) — empty-body regions yielding outer-scope values.
        val fn = DxirBuilder.function("abs_grad") {
            val x = param("x", f32s)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val negX = op(OpKind.NEG, listOf(x), f32s)
            val one = const(1f, f32s)
            val ifResult = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region { yields(one) },
                elseRegion = region { yields(negX) },
            )
            listOf(ifResult)
        }
        assertCanonicalRoundTrip(fn)
    }

    @Test
    fun roundTripWhilePrimalShape() {
        // Shape emitted by `FirLambdaToDxirLowering.lowerDesugaredForLoop`: WHILE with a
        // carried f32 + counter i32, body has nested ops inside its region.
        val fn = DxirBuilder.function("iterate5") {
            val x = param("x", f32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val n = const(5, i32s)
                    val diff = op(OpKind.SUB, listOf(n, args[1]), i32s)
                    val pred = op(OpKind.STEP, listOf(diff), boolS)
                    yields(pred)
                },
                body = { args ->
                    val two = const(2f, f32s)
                    val newX = op(OpKind.MUL, listOf(args[0], two), f32s)
                    val one = const(1, i32s)
                    val newI = op(OpKind.ADD, listOf(args[1], one), i32s)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        assertCanonicalRoundTrip(fn)
    }

    @Test
    fun roundTripPreservesFloatBitsExactly() {
        // π, e, MIN_VALUE, subnormals: values whose decimal rendering would lose precision
        // but bit-exact hex round-trips.
        val values = floatArrayOf(
            kotlin.math.PI.toFloat(),
            kotlin.math.E.toFloat(),
            Float.MIN_VALUE,
            Float.MAX_VALUE,
            Float.fromBits(0x12345678),
            -0.0f,
            0.0f,
        )
        for (v in values) {
            val fn = DxirBuilder.function("c") {
                val x = param("x", f32s)
                val c = const(v, f32s)
                val add = op(OpKind.ADD, listOf(x, c), f32s)
                listOf(add)
            }
            val deserialised = DxirCanonical.deserialise(DxirCanonical.serialise(fn))
            val cDeser = deserialised.body.filterIsInstance<io.tlaloc.ir.DxirConst>()
                .first { it.type == f32s && (it.value as Float).toRawBits() == v.toRawBits() }
            assertEquals(
                v.toRawBits(),
                (cDeser.value as Float).toRawBits(),
                "bit-exact mismatch for $v",
            )
        }
    }

    // ------------------------------------------------------------------------
    // Hash stability — structurally-equal functions with different SSA ids must hash same
    // ------------------------------------------------------------------------

    @Test
    fun hashIsStableUnderSsaRenumbering() {
        // Two structurally-identical functions. Since DxirBuilder assigns monotonic ids
        // internally, both emissions end up with the same absolute ids; but the POINT
        // of canonical hashing is that if they DID differ, the hash wouldn't. Enforce
        // this by serialising one, deserialising, re-serialising — canonicalisation is
        // idempotent, and the resulting hash must match the original.
        val build = {
            DxirBuilder.function("f") {
                val x = param("x", f32s)
                val c = const(2f, f32s)
                val m = op(OpKind.MUL, listOf(x, c), f32s)
                listOf(m)
            }
        }
        val fn1 = build()
        val fn2 = build()
        assertEquals(DxirCanonical.hash(fn1), DxirCanonical.hash(fn2))

        val fn3 = DxirCanonical.deserialise(DxirCanonical.serialise(fn1))
        assertEquals(DxirCanonical.hash(fn1), DxirCanonical.hash(fn3))
    }

    @Test
    fun hashChangesWhenOpKindChanges() {
        val add = DxirBuilder.function("f") {
            val x = param("x", f32s)
            val c = const(2f, f32s)
            listOf(op(OpKind.ADD, listOf(x, c), f32s))
        }
        val mul = DxirBuilder.function("f") {
            val x = param("x", f32s)
            val c = const(2f, f32s)
            listOf(op(OpKind.MUL, listOf(x, c), f32s))
        }
        assertNotEquals(DxirCanonical.hash(add), DxirCanonical.hash(mul))
    }

    @Test
    fun hashChangesWhenConstValueChanges() {
        val a = DxirBuilder.function("f") {
            val x = param("x", f32s)
            val c = const(2f, f32s)
            listOf(op(OpKind.MUL, listOf(x, c), f32s))
        }
        val b = DxirBuilder.function("f") {
            val x = param("x", f32s)
            val c = const(3f, f32s)
            listOf(op(OpKind.MUL, listOf(x, c), f32s))
        }
        assertNotEquals(DxirCanonical.hash(a), DxirCanonical.hash(b))
    }

    @Test
    fun hashChangesWhenOperandOrderChanges() {
        // SUB is non-commutative; swapping operands must produce a different hash.
        val ab = DxirBuilder.function("f") {
            val x = param("x", f32s)
            val c = const(2f, f32s)
            listOf(op(OpKind.SUB, listOf(x, c), f32s))
        }
        val ba = DxirBuilder.function("f") {
            val x = param("x", f32s)
            val c = const(2f, f32s)
            listOf(op(OpKind.SUB, listOf(c, x), f32s))
        }
        assertNotEquals(DxirCanonical.hash(ab), DxirCanonical.hash(ba))
    }

    @Test
    fun hashIsDeterministicAcrossCalls() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", f32s)
            val c = const(2f, f32s)
            listOf(op(OpKind.ADD, listOf(x, c), f32s))
        }
        val h1 = DxirCanonical.hash(fn)
        val h2 = DxirCanonical.hash(fn)
        val h3 = DxirCanonical.hash(fn)
        assertEquals(h1, h2)
        assertEquals(h2, h3)
        // SHA-256 hex length = 64.
        assertEquals(64, h1.length)
        assertTrue(h1.all { it in '0'..'9' || it in 'a'..'f' }, "hash must be lowercase hex")
    }

    // ------------------------------------------------------------------------
    // Helper
    // ------------------------------------------------------------------------

    /**
     * Round-trip invariant: `serialise(deserialise(serialise(fn))) == serialise(fn)`.
     * Also asserts the re-emitted function's `pretty()` output matches the original
     * (a cheap semantic check — pretty is stable under SSA renumbering when ids are
     * issued in the same order, which they are for builder-reemitted functions).
     */
    private fun assertCanonicalRoundTrip(fn: DxirFunction) {
        val s1 = DxirCanonical.serialise(fn)
        val fn2 = DxirCanonical.deserialise(s1)
        val s2 = DxirCanonical.serialise(fn2)
        assertEquals(s1, s2, "canonical form is not idempotent")
        assertEquals(fn.pretty().trim(), fn2.pretty().trim(), "pretty output diverged after round-trip")
    }
}
