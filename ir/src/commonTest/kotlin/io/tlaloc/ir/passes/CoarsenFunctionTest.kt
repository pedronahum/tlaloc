package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * §0.4.33 — Stage C.3b.3a tests for [PhiCalculus.coarsenFunction]. Engine-free:
 * verifies the splice pass produces a COARSENED-wrapped function for straight-line
 * primals and leaves region-bearing / multi-return primals unchanged. Engine-backed
 * tests (which exercise the Symja-path in `apply`) live in `:ir/jvmTest` to keep the
 * commonTest surface JVM-independent.
 */
class CoarsenFunctionTest {

    private val f32s = DxirType(F32, emptyList())

    @Test
    fun coarsenFunctionWrapsScalarRootInSingleCoarsenedOp() {
        // f(a) = a * a (straight-line, root-only leaf).
        val fn = DxirBuilder.function("square") {
            val a = param("a", f32s)
            val r = op(OpKind.MUL, listOf(a, a), f32s)
            listOf(r)
        }
        val coarsened = PhiCalculus.coarsenFunction(fn, engine = null)
        // Body should have exactly 1 op — a COARSENED wrapping the original.
        assertEquals(1, coarsened.body.size)
        val wrapOp = coarsened.body.single() as DxirOp
        assertEquals(OpKind.COARSENED, wrapOp.op)
        // Param + return types preserved.
        assertEquals(fn.params.map { it.type }, coarsened.params.map { it.type })
        assertEquals(fn.returns.map { it.type }, coarsened.returns.map { it.type })
    }

    @Test
    fun coarsenedFunctionIsNumericallyEquivalentToOriginal() {
        val fn = DxirBuilder.function("plus_one_times_two") {
            val a = param("a", f32s)
            val one = const(1f, f32s)
            val two = const(2f, f32s)
            val sum = op(OpKind.ADD, listOf(a, one), f32s)
            val r = op(OpKind.MUL, listOf(sum, two), f32s)
            listOf(r)
        }
        val coarsened = PhiCalculus.coarsenFunction(fn, engine = null)
        for (a in listOf(-5f, -1f, 0f, 1f, 3f, 10f)) {
            val expected = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(a)))[0][0]
            val actual = DxirInterpreter.evalFunction(coarsened, listOf(floatArrayOf(a)))[0][0]
            assertEquals(expected, actual, "coarsened primal diverged at a=$a")
        }
    }

    @Test
    fun coarsenedGradMatchesUncoarsenedGradAcrossSweep() {
        // f(a) = 2 * a + 3.  df/da = 2.
        val fn = DxirBuilder.function("affine") {
            val a = param("a", f32s)
            val two = const(2f, f32s)
            val three = const(3f, f32s)
            val scaled = op(OpKind.MUL, listOf(two, a), f32s)
            val shifted = op(OpKind.ADD, listOf(scaled, three), f32s)
            listOf(shifted)
        }
        val coarsened = PhiCalculus.coarsenFunction(fn, engine = null)
        val gradPlain = DxirReverseTransform.apply(fn)
        val gradCoarsened = DxirReverseTransform.apply(coarsened)
        for (a in listOf(-2f, 0f, 1f, 4f)) {
            val plain = DxirInterpreter.evalFunction(gradPlain, listOf(floatArrayOf(a)))[0][0]
            val coarsenedGrad = DxirInterpreter.evalFunction(gradCoarsened, listOf(floatArrayOf(a)))[0][0]
            assertEquals(plain, coarsenedGrad, "grad drift at a=$a: plain=$plain, coarsened=$coarsenedGrad")
        }
    }

    @Test
    fun coarsenFunctionReturnsInputUnchangedWhenRootHasRegions() {
        // f(a) = if (a > 0) a else -a — root has an IF, so root is NOT a leaf.
        val boolS = DxirType(Bool, emptyList())
        val fn = DxirBuilder.function("abs") {
            val a = param("a", f32s)
            val pred = op(OpKind.STEP, listOf(a), boolS)
            val negA = op(OpKind.NEG, listOf(a), f32s)
            val r = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region { yields(a) },
                elseRegion = region { yields(negA) },
            )
            listOf(r)
        }
        val coarsened = PhiCalculus.coarsenFunction(fn, engine = null)
        assertSame(fn, coarsened, "functions with regions should pass through unchanged in C.3b.3a")
    }

    @Test
    fun coarsenFunctionReturnsInputUnchangedWhenMultiReturn() {
        // Two-return function — not yet supported.
        val fn = DxirBuilder.function("two_ret") {
            val a = param("a", f32s)
            val b = param("b", f32s)
            val sum = op(OpKind.ADD, listOf(a, b), f32s)
            val prod = op(OpKind.MUL, listOf(a, b), f32s)
            listOf(sum, prod)
        }
        val coarsened = PhiCalculus.coarsenFunction(fn, engine = null)
        assertSame(fn, coarsened, "multi-return functions pass through unchanged in C.3b.3a")
    }

    @Test
    fun coarsenedOpCarriesReadsPrimalIndices() {
        // For `a*a`, the gradient_body references a twice (d/da = 2*a), so index 0
        // should appear in reads_primal_indices.
        val fn = DxirBuilder.function("square") {
            val a = param("a", f32s)
            val r = op(OpKind.MUL, listOf(a, a), f32s)
            listOf(r)
        }
        val coarsened = PhiCalculus.coarsenFunction(fn, engine = null)
        val wrapOp = coarsened.body.single() as DxirOp
        @Suppress("UNCHECKED_CAST")
        val reads = wrapOp.attrs["reads_primal_indices"] as Set<Int>
        assertTrue(0 in reads, "gradient body references operand a → index 0 must be in reads")
    }

    @Test
    fun coarsenedOpCarriesGradientBodyWithUpstreamParam() {
        val fn = DxirBuilder.function("double") {
            val a = param("a", f32s)
            val two = const(2f, f32s)
            val r = op(OpKind.MUL, listOf(two, a), f32s)
            listOf(r)
        }
        val coarsened = PhiCalculus.coarsenFunction(fn, engine = null)
        val wrapOp = coarsened.body.single() as DxirOp
        val gradBody = wrapOp.attrs["gradient_body"] as DxirFunction
        // Signature: (upstream: f32, a: f32) → (d_a: f32)
        assertEquals(2, gradBody.params.size, "gradient_body signature = (upstream, a)")
        assertEquals(f32s, gradBody.params[0].type, "upstream param type matches primal return")
        assertEquals(f32s, gradBody.params[1].type, "primal-operand param type matches")
        assertEquals(1, gradBody.returns.size, "one grad per primal operand")
    }

    @Test
    fun reverseTransformWithSeedAsParamProducesExpectedSignature() {
        // Standalone test of the DxirReverseTransform.apply(seedAsParam=true) option.
        val fn = DxirBuilder.function("sq") {
            val a = param("a", f32s)
            val r = op(OpKind.MUL, listOf(a, a), f32s)
            listOf(r)
        }
        val gradWithUpstream = DxirReverseTransform.apply(fn, seedAsParam = true)
        // Signature: (__upstream__, a) → (d_a)
        assertEquals(2, gradWithUpstream.params.size)
        assertEquals("__upstream__", gradWithUpstream.params[0].name)
        assertEquals("a", gradWithUpstream.params[1].name)
        // Evaluating at upstream=1, a=3: expect 2*1*3 = 6.
        val out = DxirInterpreter.evalFunction(
            gradWithUpstream,
            listOf(floatArrayOf(1f), floatArrayOf(3f)),
        )
        assertEquals(6f, out[0][0])
        // At upstream=2, a=3: expect 2*2*3 = 12.
        val out2 = DxirInterpreter.evalFunction(
            gradWithUpstream,
            listOf(floatArrayOf(2f), floatArrayOf(3f)),
        )
        assertEquals(12f, out2[0][0])
    }

    // ------------------------------------------------------------------------
    // §0.4.35 — C.3b.3b2: multi-SOI splicing inside IF branches
    // ------------------------------------------------------------------------

    @Test
    fun coarsenFunctionSplicesBothIfBranchesWithCoarsenedOps() {
        // f(x) = if (x > 0) x * x else x * x * x.
        // After coarsenFunction: IF structure preserved; each branch body replaced with a
        // single COARSENED op.
        val boolS = DxirType(io.tlaloc.core.Bool, emptyList())
        val fn = DxirBuilder.function("f") {
            val x = param("x", f32s)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val result = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region {
                    val sq = op(OpKind.MUL, listOf(x, x), f32s)
                    yields(sq)
                },
                elseRegion = region {
                    val sq = op(OpKind.MUL, listOf(x, x), f32s)
                    val cu = op(OpKind.MUL, listOf(sq, x), f32s)
                    yields(cu)
                },
            )
            listOf(result)
        }
        // sizeLimit = 3 — leaves fit (then=3, else=2), root doesn't (STEP+IF+3+2=7).
        // chooseSois promotes the small children as SOIs → both IF branches coarsen.
        // sizeLimit = 2 would mark the then-leaf large → splitOnReuses → fragment SOIs,
        // which C.3b.3b2's first-cut splice pass doesn't handle.
        val coarsened = PhiCalculus.coarsenFunction(fn, engine = null, sizeLimit = 3)
        // Check: the returned function has an IF whose branches each contain exactly one op,
        // and that op is a COARSENED.
        val ifOp = coarsened.body.filterIsInstance<DxirOp>().first { it.op == OpKind.IF }
        for ((idx, region) in ifOp.regions.withIndex()) {
            val block = region.blocks.single()
            assertEquals(1, block.body.size, "branch $idx body should be 1 op (the COARSENED)")
            val c = block.body.single() as DxirOp
            assertEquals(OpKind.COARSENED, c.op, "branch $idx body op must be COARSENED")
            // Yield must be the COARSENED's result.
            assertEquals(c, block.terminator.single(), "branch $idx yield must be the COARSENED op")
        }
    }

    @Test
    fun coarsenedIfBranchesProduceCorrectGradient() {
        // f(x) = if (x > 0) x * x else x * x * x.
        // d/dx = if (x > 0) 2x else 3x^2.  At x=3 (then) → 6.  At x=-2 (else) → 12.
        val boolS = DxirType(io.tlaloc.core.Bool, emptyList())
        val fn = DxirBuilder.function("f") {
            val x = param("x", f32s)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val result = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region {
                    val sq = op(OpKind.MUL, listOf(x, x), f32s)
                    yields(sq)
                },
                elseRegion = region {
                    val sq = op(OpKind.MUL, listOf(x, x), f32s)
                    val cu = op(OpKind.MUL, listOf(sq, x), f32s)
                    yields(cu)
                },
            )
            listOf(result)
        }
        val coarsened = PhiCalculus.coarsenFunction(fn, engine = null, sizeLimit = 3)
        val grad = DxirReverseTransform.apply(coarsened)

        // Forward correctness: coarsened primal matches original.
        assertEquals(
            DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(3f)))[0][0],
            DxirInterpreter.evalFunction(coarsened, listOf(floatArrayOf(3f)))[0][0],
        )
        // Gradient correctness: via coarsened path.
        val gradAtPos = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(3f)))
        assertEquals(6f, gradAtPos[0][0], "d/dx x^2 at x=3 = 6")
        val gradAtNeg = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(-2f)))
        assertEquals(12f, gradAtNeg[0][0], "d/dx x^3 at x=-2 = 3*4 = 12")
    }

    @Test
    fun coarsenedIfGradMatchesUncoarsenedGrad() {
        // Equivalence sweep: compare grad of the coarsened-IF function with the uncoarsened.
        val boolS = DxirType(io.tlaloc.core.Bool, emptyList())
        val buildFn = {
            DxirBuilder.function("f") {
                val x = param("x", f32s)
                val pred = op(OpKind.STEP, listOf(x), boolS)
                val result = ifOp(
                    cond = pred,
                    types = listOf(f32s),
                    thenRegion = region {
                        val two = const(2f, f32s)
                        val sq = op(OpKind.MUL, listOf(x, x), f32s)
                        val scaled = op(OpKind.MUL, listOf(two, sq), f32s)
                        yields(scaled)
                    },
                    elseRegion = region {
                        val neg = op(OpKind.NEG, listOf(x), f32s)
                        yields(neg)
                    },
                )
                listOf(result)
            }
        }
        val plain = buildFn()
        val coarsened = PhiCalculus.coarsenFunction(buildFn(), engine = null, sizeLimit = 4)
        val gradPlain = DxirReverseTransform.apply(plain)
        val gradCoarsened = DxirReverseTransform.apply(coarsened)
        for (xVal in listOf(-3f, -1f, 0.5f, 2f, 10f)) {
            val p = DxirInterpreter.evalFunction(gradPlain, listOf(floatArrayOf(xVal)))[0][0]
            val c = DxirInterpreter.evalFunction(gradCoarsened, listOf(floatArrayOf(xVal)))[0][0]
            assertEquals(p, c, "grad drift at x=$xVal: plain=$p, coarsened=$c")
        }
    }

    @Test
    fun coarsenFunctionPreservesWhileContainingPrimal() {
        // WHILE primal — root is NOT a leaf (has children), and leaves (cond + body) don't
        // match C.3b.3b2's IF-only criterion. Expected: fn passes through unchanged.
        val boolS = DxirType(io.tlaloc.core.Bool, emptyList())
        val i32s = DxirType(io.tlaloc.core.I32, emptyList())
        val fn = DxirBuilder.function("loop") {
            val x = param("x", f32s)
            val n = const(3, i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
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
        val result = PhiCalculus.coarsenFunction(fn, engine = null, sizeLimit = 2)
        // WHILE cond yields Bool (not coarsenable) and body yields multiple values (not coarsenable).
        // Expect the pass to produce no replacements → return fn unchanged.
        assertSame(fn, result, "WHILE primal should pass through unchanged in C.3b.3b2")
    }

    @Test
    fun coarsenFunctionPreservesEmptyIfBranches() {
        // f(x) = if (x > 0) x else x  — both branches have empty bodies (just yield x).
        // SoiIdentification's property-i filter rejects empty regions → no SOIs → pass through.
        val boolS = DxirType(io.tlaloc.core.Bool, emptyList())
        val fn = DxirBuilder.function("f") {
            val x = param("x", f32s)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val result = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region { yields(x) },
                elseRegion = region { yields(x) },
            )
            listOf(result)
        }
        val coarsened = PhiCalculus.coarsenFunction(fn, engine = null, sizeLimit = 2)
        assertSame(fn, coarsened, "empty-branch IF should pass through unchanged")
    }

    @Test
    fun reverseTransformSeedAsParamDisallowsIncludeForward() {
        val fn = DxirBuilder.function("sq") {
            val a = param("a", f32s)
            val r = op(OpKind.MUL, listOf(a, a), f32s)
            listOf(r)
        }
        try {
            DxirReverseTransform.apply(fn, includeForward = true, seedAsParam = true)
            kotlin.test.fail("expected IllegalArgumentException — includeForward + seedAsParam incompatible")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("incompatible"), "unexpected error: ${e.message}")
        }
    }
}
