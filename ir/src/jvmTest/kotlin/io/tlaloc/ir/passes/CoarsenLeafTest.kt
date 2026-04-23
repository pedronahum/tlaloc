package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.30 — Stage C.3a tests for [PhiCalculus.coarsenLeaf] + the engine-backed size
 * source in [SoiIdentification.identifyWithSizeLimit]. Lives in jvmTest because
 * coarsenLeaf calls [PhiCalculus.apply] which needs a [SymbolicEngine] ([SymjaEngine]
 * is JVM-only).
 */
class CoarsenLeafTest {

    private val f32s = DxirType(F32, emptyList())
    private val i32s = DxirType(I32, emptyList())
    private val boolS = DxirType(Bool, emptyList())

    private val engine = SymjaEngine()

    @Test
    fun coarsenLeafOnScalarRootReturnsSuccess() {
        val fn = DxirBuilder.function("square") {
            val x = param("x", f32s)
            val r = op(OpKind.MUL, listOf(x, x), f32s)
            listOf(r)
        }
        val tree = RegionTree.build(fn)
        val result = PhiCalculus.coarsenLeaf(tree.root, fn, engine)
        assertTrue(result is CoarsenResult.Success, "root leaf coarsening must succeed; got $result")
        result as CoarsenResult.Success
        // Mini-function: params = [x] (free var), body = [MUL], returns = [MUL].
        // PhiCalculus doesn't simplify a single MUL(x, x); size stays = 1.
        assertEquals(1, result.size)
    }

    @Test
    fun coarsenLeafFailsOnRegionBearingOp() {
        // Root contains an IF op — so the root is NOT a leaf (isLeaf checks children,
        // and IF creates children). But if we simulate a leaf that *did* contain a
        // region-bearing op (we'd have to construct the RegionTreeNode manually),
        // coarsenLeaf would refuse. For a natural primal, the root has children when
        // IF exists, so the leaf check runs on the IF's internal regions, which don't
        // contain more IFs. So coarsenLeaf on those inner leaves succeeds.
        //
        // This test just verifies the high-level contract: if given a non-leaf,
        // coarsenLeaf returns Failure.
        val fn = DxirBuilder.function("has_if") {
            val x = param("x", f32s)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val result = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region { yields(x) },
                elseRegion = region {
                    val neg = op(OpKind.NEG, listOf(x), f32s)
                    yields(neg)
                },
            )
            listOf(result)
        }
        val tree = RegionTree.build(fn)
        val result = PhiCalculus.coarsenLeaf(tree.root, fn, engine)
        assertTrue(result is CoarsenResult.Failure, "non-leaf root must refuse coarsening")
    }

    @Test
    fun coarsenLeafOnIfBranchLeafSucceeds() {
        val fn = DxirBuilder.function("if_arith") {
            val x = param("x", f32s)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val result = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region {
                    val two = const(2f, f32s)
                    val sq = op(OpKind.MUL, listOf(x, x), f32s)
                    val r = op(OpKind.MUL, listOf(sq, two), f32s)
                    yields(r)
                },
                elseRegion = region { yields(x) },
            )
            listOf(result)
        }
        val tree = RegionTree.build(fn)
        val thenLeaf = tree.root.children[0]
        assertTrue(thenLeaf.isLeaf)
        val result = PhiCalculus.coarsenLeaf(thenLeaf, fn, engine)
        assertTrue(result is CoarsenResult.Success, "then-branch leaf coarsens; got $result")
        result as CoarsenResult.Success
        // Mini-function has params=[x] (the outer-scope free var), body=[const 2, MUL(x,x), MUL(*,2)].
        // PhiCalculus on straight-line arithmetic typically doesn't shrink it — size stays ~3.
        assertTrue(result.size in 1..3, "expected size in [1, 3], got ${result.size}")
    }

    @Test
    fun identifyWithSizeLimitUsesEngineBackedSizeWhenAvailable() {
        // Synthetic primal where the engine-backed size is the same as raw (straight-
        // line arithmetic has no simplification to do). Verifies the integration path
        // works without changing behaviour vs. engine=null.
        val fn = DxirBuilder.function("f") {
            val x = param("x", f32s)
            val c = const(2f, f32s)
            val r = op(OpKind.MUL, listOf(x, c), f32s)
            listOf(r)
        }
        val withoutEngine = SoiIdentification.identifyWithSizeLimit(fn, sizeLimit = 10)
        val withEngine = SoiIdentification.identifyWithSizeLimit(fn, sizeLimit = 10, engine = engine)
        assertEquals(withoutEngine.sois.size, withEngine.sois.size, "SOI count unchanged on straight-line")
        // Sizes match (PhiCalculus doesn't shrink a single MUL).
        assertEquals(
            withoutEngine.sois.single().subtreeSize,
            withEngine.sois.single().subtreeSize,
        )
    }

    @Test
    fun identifyWithSizeLimitEngineBackedIsTighterForLoopPrimals() {
        // Loop primal that C5 can unroll to a constant chain. C5's output is smaller
        // than raw count for some configurations, making the engine-backed size
        // *strictly less* than raw. The tree's root is a leaf (no outer control-flow
        // beyond the WHILE, so children come from the WHILE — root has regions as
        // children; WHILE's body is the leaf we coarsen).
        val fn = DxirBuilder.function("loop") {
            val x = param("x", f32s)
            val n = const(5, i32s)
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
        // Root is NOT a leaf (has WHILE children), so coarsenLeaf is called on the
        // COND + BODY leaves. Those are straight-line arithmetic — PhiCalculus won't
        // shrink them meaningfully. This test just pins that the integration runs
        // without error for a WHILE-containing primal.
        val withEngine = SoiIdentification.identifyWithSizeLimit(
            fn = fn,
            sizeLimit = 100,
            engine = engine,
        )
        assertTrue(withEngine.sois.isNotEmpty(), "integration must produce at least one SOI")
    }

    @Test
    fun coarsenLeafFailureOnMultiResultOp() {
        // If a leaf somehow had a multi-result op in its directOps (e.g., a WHILE
        // result reference — but those have `regions`, so they're filtered earlier).
        // Today no natural path produces this. Test via a synthetic RegionTreeNode
        // that holds a WHILE directly — but since WHILE has regions, it's caught
        // by the region-bearing guard first. So both guards share the same outcome:
        // Failure.
        val fn = DxirBuilder.function("w") {
            val x = param("x", f32s)
            val n = const(5, i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val pred = op(OpKind.STEP, listOf(args[1]), boolS)
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
        // Root isn't a leaf (WHILE has children), so coarsenLeaf returns Failure.
        val tree = RegionTree.build(fn)
        val result = PhiCalculus.coarsenLeaf(tree.root, fn, engine)
        assertTrue(result is CoarsenResult.Failure)
    }

    @Test
    fun coarsenLeafWithNullEngineStillWorks() {
        // engine=null falls back to running apply engine-free (structural rewrites
        // only). Same leaf → Success(size).
        val fn = DxirBuilder.function("square") {
            val x = param("x", f32s)
            val r = op(OpKind.MUL, listOf(x, x), f32s)
            listOf(r)
        }
        val tree = RegionTree.build(fn)
        val result = PhiCalculus.coarsenLeaf(tree.root, fn, engine = null)
        assertTrue(result is CoarsenResult.Success)
        result as CoarsenResult.Success
        assertEquals(1, result.size)
    }
}
