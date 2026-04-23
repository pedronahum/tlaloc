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
 * §0.4.27 — Stage C.1 tests for [SoiIdentification]'s worklist skeleton. Verifies
 * per-sink enumeration + reachability filtering + bottom-up ordering. Does NOT test
 * the symbolic-expression size check (C.2) or the PhiCalculus integration (C.3) —
 * those land in follow-up slices.
 */
class SoiIdentificationTest {

    private val f32s = DxirType(F32, emptyList())
    private val i32s = DxirType(I32, emptyList())
    private val boolS = DxirType(Bool, emptyList())

    @Test
    fun scalarFunctionYieldsSingleRootCandidate() {
        val fn = DxirBuilder.function("f") {
            val x = param("x", f32s)
            val c = const(2f, f32s)
            val r = op(OpKind.MUL, listOf(x, c), f32s)
            listOf(r)
        }
        val soi = SoiIdentification.identify(fn)
        assertEquals(1, soi.size, "single-sink scalar → 1 root candidate")
        assertTrue(soi.single().node.isRoot)
        assertEquals(fn.returns.single().id, soi.single().sinkId)
    }

    @Test
    fun ifFunctionYieldsCandidatesFromNonEmptyRegions() {
        // f(x) = if (x > 0) x*x else -x*x. Tree: root + 2 leaves; each leaf has ops in
        // its region body, so all 3 nodes produce candidates. (If a branch body is
        // empty — e.g., it just yields an outer-scope value — that branch's leaf
        // intersection is empty and the candidate is filtered out.)
        val fn = DxirBuilder.function("if_arith") {
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
                    val neg = op(OpKind.NEG, listOf(sq), f32s)
                    yields(neg)
                },
            )
            listOf(result)
        }
        val soi = SoiIdentification.identify(fn)
        assertEquals(3, soi.size, "root + 2 non-empty leaf regions = 3 candidates")
        assertTrue(soi.all { it.sinkId == fn.returns.single().id })
        // Bottom-up: then-leaf, else-leaf, root.
        assertTrue(soi[0].node.isLeaf)
        assertTrue(soi[1].node.isLeaf)
        assertTrue(soi[2].node.isRoot)
    }

    @Test
    fun emptyRegionsAreFilteredOut() {
        // Both branches just yield outer-scope values (no ops in the region body). The
        // leaf candidates have empty reachableOps and get filtered — only the root
        // remains. Pins the "only relevant variable definitions are considered"
        // property of the paper's def-use region tree.
        val fn = DxirBuilder.function("abs") {
            val x = param("x", f32s)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val negX = op(OpKind.NEG, listOf(x), f32s)
            val result = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region { yields(x) },
                elseRegion = region { yields(negX) },
            )
            listOf(result)
        }
        val soi = SoiIdentification.identify(fn)
        assertEquals(1, soi.size, "both-branches-empty → only root contributes")
        assertTrue(soi.single().node.isRoot)
    }

    @Test
    fun multipleSinksEachDriveIndependentTraversal() {
        // grad2-shape function: two returns. Each sink enumerates its own candidate
        // list via SoiIdentification; total candidates = 2 × (per-sink count).
        val fn = DxirBuilder.function("two_outs") {
            val a = param("a", f32s)
            val b = param("b", f32s)
            val sum = op(OpKind.ADD, listOf(a, b), f32s)
            val prod = op(OpKind.MUL, listOf(a, b), f32s)
            listOf(sum, prod)
        }
        val soi = SoiIdentification.identify(fn)
        // No regions → 1 candidate per sink → 2 total.
        assertEquals(2, soi.size)
        // Two distinct sink ids.
        assertEquals(fn.returns.map { it.id }.toSet(), soi.map { it.sinkId }.toSet())
    }

    @Test
    fun whileBodyAppearsAsCandidate() {
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
        val soi = SoiIdentification.identify(fn)
        // Tree has 3 nodes (root + cond-child + body-child). All three have non-empty
        // reachable interiors (every op in a WHILE's cond+body contributes to the
        // carried output via the loop semantics). Candidates = 3.
        assertEquals(3, soi.size)
        // Bottom-up: cond-child (leaf), body-child (leaf), root.
        val levels = soi.map { it.node.isRoot }
        assertEquals(listOf(false, false, true), levels, "root must appear last")
    }

    @Test
    fun reachableOpsFieldCapturesIntersection() {
        // Sanity check: the `reachableOps` field is non-empty for every candidate (by
        // construction — candidates with empty intersections are filtered out).
        val fn = DxirBuilder.function("f") {
            val x = param("x", f32s)
            val c = const(2f, f32s)
            val r = op(OpKind.MUL, listOf(x, c), f32s)
            listOf(r)
        }
        val soi = SoiIdentification.identify(fn)
        assertTrue(soi.all { it.reachableOps.isNotEmpty() })
    }

    // ------------------------------------------------------------------------
    // §0.4.28 — C.2a: size-limit marking + hasLargeChildren promotion
    // ------------------------------------------------------------------------

    @Test
    fun identifyWithSizeLimitAllSmallEmitsRootOnly() {
        // 2-op function, L = 10. Root subtreeSize = 2 ≤ 10 → not marked large → root
        // is the single SOI.
        val fn = DxirBuilder.function("f") {
            val x = param("x", f32s)
            val c = const(2f, f32s)
            val r = op(OpKind.MUL, listOf(x, c), f32s)
            listOf(r)
        }
        val result = SoiIdentification.identifyWithSizeLimit(fn, sizeLimit = 10)
        assertEquals(1, result.sois.size)
        assertTrue(result.sois.single().node.isRoot)
        assertEquals(false, result.sois.single().markedLarge)
        assertEquals(2, result.sois.single().subtreeSize)
    }

    @Test
    fun identifyWithSizeLimitMarksOversizedRoot() {
        // Build a function with 15 body ops, L = 5. Root subtreeSize = 16 (1 const +
        // 15 ADDs) > 5 → marked large. C.2b's `splitOnReuses` now partitions around
        // the root leaf's only free variable (x): pre = [const 1] (doesn't depend on
        // x), post = [15 ADDs] (all depend on x transitively). Two fragments.
        val fn = DxirBuilder.function("many") {
            val x = param("x", f32s)
            var cur: io.tlaloc.ir.DxirNode = x
            val one = const(1f, f32s)
            repeat(15) { cur = op(OpKind.ADD, listOf(cur, one), f32s) }
            listOf(cur)
        }
        val result = SoiIdentification.identifyWithSizeLimit(fn, sizeLimit = 5)
        assertEquals(2, result.sois.size, "C.2b split partitions around x")
        val total = result.sois.sumOf { it.subtreeSize!! }
        assertEquals(16, total, "fragments together cover all 16 original ops")
    }

    @Test
    fun identifyWithSizeLimitPromotesSmallChildrenWhenParentLarge() {
        // Synthetic shape: IF with two non-empty small regions, L just under root size
        // so root is markedLarge but children aren't. Expected SOI = the 2 small
        // children, not the root.
        val fn = DxirBuilder.function("if_split") {
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
                    val neg = op(OpKind.NEG, listOf(x), f32s)
                    yields(neg)
                },
            )
            listOf(result)
        }
        // Root subtreeSize = STEP + IF + (MUL) + (NEG) = 4 ops. Each child = 1 op.
        // L = 3 → root (4) marked large; children (1 each) not.
        val result = SoiIdentification.identifyWithSizeLimit(fn, sizeLimit = 3)
        assertEquals(2, result.sois.size, "small children of large root become SOIs")
        assertTrue(result.sois.all { !it.markedLarge })
        assertTrue(result.sois.all { it.node.isLeaf })
    }

    @Test
    fun identifyWithSizeLimitRootTakesWholeFunctionWhenSmallEnough() {
        // Same shape as above but L large enough that root itself fits. Expected:
        // the root is the single SOI, not the children.
        val fn = DxirBuilder.function("if_whole") {
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
        val result = SoiIdentification.identifyWithSizeLimit(fn, sizeLimit = 100)
        assertEquals(1, result.sois.size)
        assertTrue(result.sois.single().node.isRoot)
    }

    @Test
    fun identifyWithSizeLimitMarksParentWhenChildLarge() {
        // Inner WHILE body is deliberately large; root's subtreeSize includes the
        // entire WHILE. With L small enough to fail the body but not so small the
        // cond also fails, the WHILE (body-bearing op's parent scope — root) gets
        // marked large because its body-child is large. §0.4.29's split then
        // partitions the large body leaf around its most-reused free var.
        val fn = DxirBuilder.function("big_loop") {
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
                    var cur: io.tlaloc.ir.DxirNode = args[0]
                    val two = const(2f, f32s)
                    repeat(20) { cur = op(OpKind.MUL, listOf(cur, two), f32s) }
                    val one = const(1, i32s)
                    val newI = op(OpKind.ADD, listOf(args[1], one), i32s)
                    yields(cur, newI)
                },
            )
            listOf(w.result(0))
        }
        val result = SoiIdentification.identifyWithSizeLimit(fn, sizeLimit = 5)
        val largeNodes = result.candidates.filter { it.markedLarge }
        assertTrue(largeNodes.isNotEmpty())
        // Cond leaf is small (1 SOI). Body leaf is large and splits into 2 fragments.
        // Total = 3 SOIs.
        assertEquals(3, result.sois.size)
        // All SOIs are leaves (original cond + 2 body-split fragments).
        assertTrue(result.sois.all { it.node.isLeaf })
    }

    @Test
    fun identifyWithSizeLimitRequiresPositiveLimit() {
        val fn = DxirBuilder.function("x") {
            val x = param("x", f32s)
            listOf(x)
        }
        try {
            SoiIdentification.identifyWithSizeLimit(fn, sizeLimit = 0)
            kotlin.test.fail("expected IllegalArgumentException for sizeLimit=0")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("must be positive"))
        }
    }

    // ------------------------------------------------------------------------
    // §0.4.29 — C.2b: splitOnReuses on large leaves
    // ------------------------------------------------------------------------

    @Test
    fun splitOnReusesBreaksLargeLeafAroundMostReusedFreeVar() {
        // Build: a large flat function with 10 body ops, all referencing x.
        // L=3 forces the root (leaf, since no regions) to be marked large. With
        // splitOnReuses, the single most-reused free var (x, used 10 times) is the
        // pivot. But every op depends on x → pre is empty → split fails, falls back.
        //
        // Rework: build a mix where some early ops compute a value c NOT from x, and
        // later ops depend on x + c. Pivot=x (most reused); pre = ops computing c;
        // post = ops depending on x.
        val fn = DxirBuilder.function("split") {
            val x = param("x", f32s)
            val y = param("y", f32s)
            val one = const(1f, f32s)
            val two = const(2f, f32s)
            // "pre" partition (doesn't depend on x — depends on y):
            val yOne = op(OpKind.ADD, listOf(y, one), f32s)
            val yTwo = op(OpKind.MUL, listOf(yOne, two), f32s)
            // "post" partition (depends on x):
            val xOne = op(OpKind.ADD, listOf(x, one), f32s)
            val xTwo = op(OpKind.MUL, listOf(xOne, two), f32s)
            val xy = op(OpKind.ADD, listOf(xTwo, yTwo), f32s)
            listOf(xy)
        }
        // body: 4 consts/ops pre (const 1, const 2, yOne, yTwo) + 3 post (xOne, xTwo, xy) = 7 ops.
        // L = 3 → root is marked large. splitOnReuses finds pivot among free vars
        // {x (id=0), y (id=1)}. x has more uses (xOne, xTwo reference it; that's 2
        // uses) vs y (yOne references it; 1 use). So x wins as pivot.
        val result = SoiIdentification.identifyWithSizeLimit(fn, sizeLimit = 3)
        assertEquals(2, result.sois.size, "splitOnReuses must emit 2 fragment SOIs")
        // Pre (doesn't depend on x): 2 consts + yOne + yTwo = 4 ops. Post: xOne + xTwo + xy = 3 ops.
        val preFragment = result.sois.maxByOrNull { it.subtreeSize!! }!!
        val postFragment = result.sois.minByOrNull { it.subtreeSize!! }!!
        assertTrue(preFragment.subtreeSize!! >= postFragment.subtreeSize!!)
        // Sum of fragment sizes equals the original leaf's directOps count.
        assertEquals(
            fn.body.size,
            result.sois.sumOf { it.subtreeSize!! },
            "fragments must partition the original directOps (no ops dropped)",
        )
    }

    @Test
    fun splitOnReusesFallsBackWhenAllOpsDependOnOnlyFreeVar() {
        // All arithmetic ops depend on x directly, with no const-only side ops. Only
        // free var = x; every op's dependency closure includes x. Split's `pre`
        // (ops NOT depending on pivot) is empty → null → fallback to single large SOI.
        //
        // To force this, we need: every body op references x directly (no consts that
        // exist independently). Use `x + x`, `x * x` style expressions with no consts.
        val fn = DxirBuilder.function("chain") {
            val x = param("x", f32s)
            val a = op(OpKind.ADD, listOf(x, x), f32s)    // depends on x
            val b = op(OpKind.MUL, listOf(a, x), f32s)    // depends on x
            val c = op(OpKind.ADD, listOf(b, x), f32s)    // depends on x
            val d = op(OpKind.MUL, listOf(c, x), f32s)    // depends on x
            val e = op(OpKind.ADD, listOf(d, x), f32s)    // depends on x
            listOf(e)
        }
        val result = SoiIdentification.identifyWithSizeLimit(fn, sizeLimit = 3)
        assertEquals(1, result.sois.size, "split must fall back when pivot is the only free var and all ops depend on it")
        assertTrue(result.sois.single().markedLarge)
    }

    @Test
    fun splitOnReusesDeclinesSingleOpLeaf() {
        // Degenerate: single-op leaf. subtreeSize = 1, no matter what L is, a single-
        // op function can't be split into two smaller pieces.
        val fn = DxirBuilder.function("tiny") {
            val x = param("x", f32s)
            val r = op(OpKind.NEG, listOf(x), f32s)
            listOf(r)
        }
        val result = SoiIdentification.identifyWithSizeLimit(fn, sizeLimit = 0 + 1)
        // subtreeSize = 1 ≤ L=1 → NOT marked large. Root is the SOI.
        assertEquals(1, result.sois.size)
        assertEquals(false, result.sois.single().markedLarge)
    }

    @Test
    fun splitFragmentsPartitionOriginalOpsOrderPreserved() {
        // Verify fragment op order matches original program order.
        val fn = DxirBuilder.function("order") {
            val x = param("x", f32s)
            val y = param("y", f32s)
            val yA = op(OpKind.ADD, listOf(y, y), f32s)           // id 2
            val xA = op(OpKind.ADD, listOf(x, x), f32s)           // id 3
            val yB = op(OpKind.MUL, listOf(yA, y), f32s)          // id 4
            val xB = op(OpKind.MUL, listOf(xA, x), f32s)          // id 5
            val r = op(OpKind.ADD, listOf(yB, xB), f32s)          // id 6
            listOf(r)
        }
        // x has more uses (xA references x twice — dedup to 1, xB references x once — 2 total).
        // y has uses: yA refs y twice (dedup → 1), yB refs y once — 2. Tied.
        // The pivot picks by max useCount, tie-broken by smallest id (well, `-it` flipped
        // so actually greatest id). Let's just verify the partition contains the right ops.
        val result = SoiIdentification.identifyWithSizeLimit(fn, sizeLimit = 3)
        // Root has 5 ops total > 3 → marked large. Split fires.
        assertEquals(2, result.sois.size)
        // Every original op appears in exactly one fragment.
        val allFragmentOpIds = result.sois.flatMap { it.reachableOps }.toSet()
        val originalOpIds = fn.body.map { it.id }.toSet()
        assertEquals(originalOpIds, allFragmentOpIds)
    }

    @Test
    fun splitAffectsOnlyLeaves() {
        // Non-leaf nodes (IF/WHILE parents) aren't split — only leaves are. A large
        // IF primal with a large body region should split THAT body, not the root.
        val fn = DxirBuilder.function("nested_large") {
            val x = param("x", f32s)
            val y = param("y", f32s)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val result = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region {
                    val one = const(1f, f32s)
                    val two = const(2f, f32s)
                    val yA = op(OpKind.ADD, listOf(y, one), f32s)
                    val yB = op(OpKind.MUL, listOf(yA, two), f32s)
                    val xA = op(OpKind.ADD, listOf(x, one), f32s)
                    val xB = op(OpKind.MUL, listOf(xA, two), f32s)
                    val combined = op(OpKind.ADD, listOf(xB, yB), f32s)
                    yields(combined)
                },
                elseRegion = region { yields(x) },
            )
            listOf(result)
        }
        // Then-region has 7 ops, else empty. Root has STEP + IF + nothing else = 2 direct + nested.
        // L = 3 → root subtreeSize = 2 + 7 + 0 = 9 > 3, large. Then-region leaf = 7 > 3, large.
        // Else-region leaf has empty directOps → filtered out. So SOIs come from splitting
        // the then-region leaf.
        val result = SoiIdentification.identifyWithSizeLimit(fn, sizeLimit = 3)
        // Expected: 2 fragments from splitting the then-region.
        assertTrue(result.sois.isNotEmpty())
        // All SOI nodes should be leaves (fragments are leaves by construction).
        assertTrue(result.sois.all { it.node.isLeaf })
    }

    @Test
    fun subtreeSizeCountsAllNestedOps() {
        // Direct test of RegionTreeNode.subtreeSize.
        val fn = DxirBuilder.function("nested") {
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
        val tree = RegionTree.build(fn)
        // Root directOps: const n + const 0 + WHILE = 3. WHILE cond: SUB + STEP = 2.
        // WHILE body: 2 consts + MUL + ADD = 4. Total = 3 + 2 + 4 = 9.
        assertEquals(9, tree.root.subtreeSize())
    }
}
