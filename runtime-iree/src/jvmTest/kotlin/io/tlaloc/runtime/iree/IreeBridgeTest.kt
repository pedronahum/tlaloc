package io.tlaloc.runtime.iree

import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.287 — Phase 2 of the IREE port plan: numerical-agreement smoke between
 * `runOnIree(fn, inputs)` and `DxirInterpreter.evalFunction(fn, inputs)` on a
 * spread of hand-built primals (scalar / rank-1 / rank-2 / multi-input /
 * multi-return). LlamaDecoder-style end-to-end runs are §0.4.288's concern;
 * this test pins the bridge's marshalling correctness in isolation.
 */
class IreeBridgeTest {

    private fun requireIreeOrSkip() {
        assumeTrue(
            IreeBinaries.available,
            "iree-compile / iree-run-module not resolved — skipping. " +
                "Install via `pip install iree-base-compiler iree-base-runtime` into ~/.local/venvs/iree, " +
                "or set TLALOC_IREE_BIN to a directory containing the binaries.",
        )
    }

    private fun assertAgreesWithInterpreter(
        fn: io.tlaloc.ir.DxirFunction,
        inputs: List<FloatArray>,
        tol: Float = 1e-5f,
    ) {
        val interpOut = DxirInterpreter.evalFunction(fn, inputs)
        val ireeOut = runOnIree(fn, inputs)
        assertEquals(interpOut.size, ireeOut.size, "result count mismatch")
        for (i in interpOut.indices) {
            val a = interpOut[i]
            val b = ireeOut[i]
            assertEquals(a.size, b.size, "result[$i] size mismatch")
            for (j in a.indices) {
                val diff = kotlin.math.abs(a[j] - b[j])
                assertTrue(
                    diff <= tol,
                    "result[$i][$j] disagreement: interpreter=${a[j]} iree=${b[j]} diff=$diff (tol=$tol)",
                )
            }
        }
    }

    // --- Marshalling unit tests (no IREE binary needed) ---

    @Test
    fun formatInputScalar() {
        val t = DxirType(F32, emptyList())
        assertEquals("f32=2.5", formatInput(t, floatArrayOf(2.5f)))
    }

    @Test
    fun formatInputRank1() {
        val t = DxirType(F32, listOf(4))
        assertEquals("4xf32=1.0,2.0,3.0,4.0", formatInput(t, floatArrayOf(1f, 2f, 3f, 4f)))
    }

    @Test
    fun formatInputRank2() {
        val t = DxirType(F32, listOf(2, 3))
        assertEquals(
            "2x3xf32=1.0,2.0,3.0,4.0,5.0,6.0",
            formatInput(t, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)),
        )
    }

    @Test
    fun parseOutputScalar() {
        val t = DxirType(F32, emptyList())
        val parsed = parseOutput(t, "f32=3")
        assertEquals(1, parsed.size)
        assertEquals(3f, parsed[0])
    }

    @Test
    fun parseOutputRank1SpaceSeparated() {
        // `iree-run-module` emits rank-1 results as `4xf32=v0 v1 v2 v3`.
        val t = DxirType(F32, listOf(4))
        val parsed = parseOutput(t, "4xf32=11 22 33 44")
        assertTrue(parsed.contentEquals(floatArrayOf(11f, 22f, 33f, 44f)))
    }

    @Test
    fun parseOutputRank2BracketedRows() {
        // `iree-run-module` emits rank-2 results as `2x3xf32=[v00 v01 v02][v10 v11 v12]`.
        val t = DxirType(F32, listOf(2, 3))
        val parsed = parseOutput(t, "2x3xf32=[2 4 6][8 10 12]")
        assertTrue(parsed.contentEquals(floatArrayOf(2f, 4f, 6f, 8f, 10f, 12f)))
    }

    @Test
    fun parseOutputRank3NestedBrackets() {
        // Rank-3+ uses fully nested brackets: `[[v000 v001][v010 v011]][[v100 v101][v110 v111]]`.
        val t = DxirType(F32, listOf(2, 2, 2))
        val parsed = parseOutput(t, "2x2x2xf32=[[2 4][6 8]][[10 12][14 16]]")
        assertTrue(parsed.contentEquals(floatArrayOf(2f, 4f, 6f, 8f, 10f, 12f, 14f, 16f)))
    }

    @Test
    fun parseOutputScientificNotation() {
        // IREE emits very-large or very-small values in scientific form (e.g. -5.25E+10).
        val t = DxirType(F32, emptyList())
        val parsed = parseOutput(t, "f32=-5.25E+10")
        assertEquals(-5.25e10f, parsed[0])
    }

    @Test
    fun parseOutputCountMismatchSurfacesPreciseError() {
        val t = DxirType(F32, listOf(4))
        val ex = assertFailsWith<IllegalArgumentException> {
            parseOutput(t, "4xf32=1 2 3")
        }
        assertTrue("expected 4 values" in (ex.message ?: ""))
    }

    // --- Validation guards (no IREE binary needed) ---

    @Test
    fun runOnIreeRejectsArityMismatch() {
        val f32Scalar = DxirType(F32, emptyList())
        val fn = DxirBuilder.function("identity") {
            val x = param("x", f32Scalar)
            listOf(x)
        }
        val ex = assertFailsWith<IllegalArgumentException> {
            runOnIree(fn, listOf(floatArrayOf(1f), floatArrayOf(2f)))
        }
        assertTrue("param count" in (ex.message ?: ""))
    }

    @Test
    fun runOnIreeRejectsF64Param() {
        val f64Scalar = DxirType(F64, emptyList())
        val fn = DxirBuilder.function("identity") {
            val x = param("x", f64Scalar)
            listOf(x)
        }
        val ex = assertFailsWith<IllegalArgumentException> {
            runOnIree(fn, listOf(floatArrayOf(1f)))
        }
        assertTrue(
            "F32" in (ex.message ?: "") && "dtype" in (ex.message ?: ""),
            "expected F32-dtype guard message; got: ${ex.message}",
        )
    }

    @Test
    fun runOnIreeRejectsInputSizeMismatch() {
        val v4 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("identity") {
            val x = param("x", v4)
            listOf(x)
        }
        val ex = assertFailsWith<IllegalArgumentException> {
            runOnIree(fn, listOf(floatArrayOf(1f, 2f)))
        }
        assertTrue(
            "expects size 4" in (ex.message ?: "") && "got input of size 2" in (ex.message ?: "") ||
                "size 4" in (ex.message ?: ""),
            "expected param-size mismatch message; got: ${ex.message}",
        )
    }

    // --- Numerical agreement (subprocess-driven; self-skips without IREE) ---

    @Test
    fun scalarAffineAgreesWithInterpreter() {
        // f(x) = 2x + 1
        val f32Scalar = DxirType(F32, emptyList())
        val fn = DxirBuilder.function("scalar_affine") {
            val x = param("x", f32Scalar)
            val two = const(2.0f, f32Scalar)
            val one = const(1.0f, f32Scalar)
            val mul = op(OpKind.MUL, listOf(x, two), f32Scalar)
            val add = op(OpKind.ADD, listOf(mul, one), f32Scalar)
            listOf(add)
        }
        requireIreeOrSkip()
        assertAgreesWithInterpreter(fn, listOf(floatArrayOf(3.5f)))
    }

    @Test
    fun rank1ElementwiseAgreesWithInterpreter() {
        // f(v) = v + v + v
        val v4 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("rank1_triple") {
            val v = param("v", v4)
            val twoV = op(OpKind.ADD, listOf(v, v), v4)
            val threeV = op(OpKind.ADD, listOf(twoV, v), v4)
            listOf(threeV)
        }
        requireIreeOrSkip()
        assertAgreesWithInterpreter(fn, listOf(floatArrayOf(1f, 2f, -3f, 4f)))
    }

    @Test
    fun rank2ElementwiseAgreesWithInterpreter() {
        val mat = DxirType(F32, listOf(2, 3))
        val fn = DxirBuilder.function("rank2_squared_plus") {
            val a = param("a", mat)
            val sq = op(OpKind.MUL, listOf(a, a), mat)
            val plus = op(OpKind.ADD, listOf(sq, a), mat)
            listOf(plus)
        }
        requireIreeOrSkip()
        assertAgreesWithInterpreter(fn, listOf(floatArrayOf(1f, 2f, 3f, -1f, -2f, 0.5f)))
    }

    @Test
    fun multiInputMatmulAgreesWithInterpreter() {
        // Two rank-2 inputs, one rank-2 output via MATMUL.
        val tA = DxirType(F32, listOf(2, 3))
        val tB = DxirType(F32, listOf(3, 4))
        val tC = DxirType(F32, listOf(2, 4))
        val fn = DxirBuilder.function("matmul_2x3_3x4") {
            val a = param("a", tA)
            val b = param("b", tB)
            val c = op(OpKind.MATMUL, listOf(a, b), tC)
            listOf(c)
        }
        requireIreeOrSkip()
        val aIn = FloatArray(6) { (it + 1).toFloat() }                // 1..6
        val bIn = FloatArray(12) { ((it % 4) + 1).toFloat() * 0.5f }  // small repeating
        assertAgreesWithInterpreter(fn, listOf(aIn, bIn))
    }

    @Test
    fun multiReturnAgreesWithInterpreter() {
        // f(x) returns (x+1, x*2) — pins multi-output marshalling.
        val f32Scalar = DxirType(F32, emptyList())
        val fn = DxirBuilder.function("scalar_pair") {
            val x = param("x", f32Scalar)
            val one = const(1.0f, f32Scalar)
            val two = const(2.0f, f32Scalar)
            val plus = op(OpKind.ADD, listOf(x, one), f32Scalar)
            val times = op(OpKind.MUL, listOf(x, two), f32Scalar)
            listOf(plus, times)
        }
        requireIreeOrSkip()
        assertAgreesWithInterpreter(fn, listOf(floatArrayOf(7.0f)))
    }
}
