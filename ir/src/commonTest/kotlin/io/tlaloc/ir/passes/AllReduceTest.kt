package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.render.KotlinRenderRefusal
import io.tlaloc.ir.render.toKotlinSource
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.460 — Phase G3a: ALL_REDUCE from demoted refusal to real op, and
 * SHARD_CONSTRAINT's value-identity un-demotion. The oracles the slice
 * promised:
 *
 * 1. INTERPRETER SINGLE-REPLICA EXACTNESS — with `replica_groups` absent or
 *    `[[0]]`, all-reduce-sum is BIT-EXACT identity (single-process semantics:
 *    sum over a one-replica group). The multi-replica arm (×|group|, the SPMD
 *    replicated-value view) is exercised only via these unit semantics until
 *    G2b/G4 put real devices behind the groups — stated here, on the pins.
 * 2. JVP-VJP IDENTITY through an all-reduce-bearing loss — the reverse
 *    gradient dotted with the direction equals the forward tangent, and the
 *    gradient is hand-pinned (2·|group|·x for Σ all_reduce(x⊙x)).
 * 3. The SELF-ADJOINT structure: the gradient body carries the SAME
 *    ALL_REDUCE (same replica_groups) on the upstream.
 * 4. The named refusals: non-sum reductions (mean scales, max/min need
 *    subgradient routing), ragged/overlapping/zero-less groups, and the
 *    KotlinSourceRenderer's no-single-process-host-spelling refusal.
 *
 * Emission pins live in the stablehlo module's EmitterTest (MLIR shape).
 */
class AllReduceTest {

    private val f32s = DxirType(F32, emptyList())
    private val vec4 = DxirType(F32, listOf(4))

    private fun allReduceOf(groups: List<List<Int>>?, reduction: String? = null): DxirFunction =
        DxirBuilder.function("ar") {
            val x = param("x", vec4)
            val attrs = buildMap<String, Any> {
                if (groups != null) put("replica_groups", groups)
                if (reduction != null) put("reduction", reduction)
            }
            listOf(op(OpKind.ALL_REDUCE, listOf(x), vec4, attrs = attrs))
        }

    /** loss(x) = Σ all_reduce(x ⊙ x) — the all-reduce-bearing loss. */
    private fun allReduceLoss(groups: List<List<Int>>?): DxirFunction =
        DxirBuilder.function("ar_loss") {
            val x = param("x", vec4)
            val sq = op(OpKind.MUL, listOf(x, x), vec4)
            val attrs = if (groups != null) {
                mapOf<String, Any>("replica_groups" to groups)
            } else {
                emptyMap()
            }
            val ar = op(OpKind.ALL_REDUCE, listOf(sq), vec4, attrs = attrs)
            listOf(op(OpKind.SUM, listOf(ar), f32s))
        }

    private val x = floatArrayOf(0.5f, -1.25f, 2f, 0.03125f)

    // --- 1. Interpreter: single-replica exactness, multi-replica unit semantics. ---

    @Test
    fun interpreterSingleReplicaIsExactIdentity() {
        // Absent attr = the single-replica program [[0]]; both spellings are
        // BIT-EXACT identity (n == 1 takes the copyOf arm — no float math at all).
        val absent = DxirInterpreter.evalFunction(allReduceOf(null), listOf(x))[0]
        assertContentEquals(x, absent, "absent replica_groups must be exact identity")
        val explicit = DxirInterpreter.evalFunction(allReduceOf(listOf(listOf(0))), listOf(x))[0]
        assertContentEquals(x, explicit, "[[0]] must be exact identity")
    }

    @Test
    fun interpreterMultiReplicaScalesByReplicaZeroGroupSize() {
        // The SPMD replicated-value view: this process models replica 0, and
        // every member of its group holds the same value — sum = |group| × x.
        // UNIT SEMANTICS ONLY until G2b/G4 (no devices behind these groups).
        val twoGroups = DxirInterpreter.evalFunction(
            allReduceOf(listOf(listOf(0, 1), listOf(2, 3))),
            listOf(x),
        )[0]
        assertContentEquals(FloatArray(4) { x[it] * 2 }, twoGroups, "|group(0)| = 2 must scale ×2")
        val oneGroup = DxirInterpreter.evalFunction(
            allReduceOf(listOf(listOf(0, 1, 2, 3))),
            listOf(x),
        )[0]
        assertContentEquals(FloatArray(4) { x[it] * 4 }, oneGroup, "|group(0)| = 4 must scale ×4")
    }

    // --- The named attr refusals (shared parse — every layer says the same thing). ---

    @Test
    fun nonSumReductionsRefuseByName() {
        val max = assertFailsWith<IllegalStateException> {
            DxirInterpreter.evalFunction(allReduceOf(null, reduction = "max"), listOf(x))
        }
        assertTrue("ALL_REDUCE" in (max.message ?: ""), "must name the kind: ${max.message}")
        assertTrue("max" in (max.message ?: ""), "must name the reduction: ${max.message}")
        assertTrue("subgradient" in (max.message ?: ""), "must state the deferral story: ${max.message}")
        val mean = assertFailsWith<IllegalStateException> {
            DxirReverseTransform.apply(allReduceLossWithReduction("mean"))
        }
        assertTrue("mean" in (mean.message ?: ""), "must name the reduction: ${mean.message}")
        assertTrue("1/|group|" in (mean.message ?: ""), "must state the scaling story: ${mean.message}")
    }

    private fun allReduceLossWithReduction(reduction: String): DxirFunction =
        DxirBuilder.function("ar_red") {
            val x = param("x", vec4)
            val ar = op(
                OpKind.ALL_REDUCE, listOf(x), vec4,
                attrs = mapOf<String, Any>("reduction" to reduction),
            )
            listOf(op(OpKind.SUM, listOf(ar), f32s))
        }

    @Test
    fun malformedReplicaGroupsRefuseByName() {
        // parse()'s structural checks are `require` (IllegalArgumentException),
        // its deferral refusals `error` (IllegalStateException) — catch the
        // common supertype and pin the message, which is the contract.
        fun failing(groups: List<List<Int>>): String {
            val ex = assertFailsWith<RuntimeException> {
                DxirInterpreter.evalFunction(allReduceOf(groups), listOf(x))
            }
            return ex.message ?: ""
        }
        assertTrue("ragged" in failing(listOf(listOf(0, 1), listOf(2))), "ragged groups must refuse by name")
        assertTrue("disjoint" in failing(listOf(listOf(0, 1), listOf(1, 2))), "overlapping groups must refuse by name")
        assertTrue("replica 0" in failing(listOf(listOf(1, 2))), "groups without replica 0 must refuse by name")
    }

    // --- 2 + 3. Differentiability: self-adjoint VJP, linear JVP, and their identity. ---

    @Test
    fun reverseGradientCarriesTheSameAllReduce() {
        // Self-adjointness, structurally: d/dx Σ all_reduce(x) puts the SAME
        // collective (same replica_groups) on the upstream in the grad body.
        val groups = listOf(listOf(0, 1))
        val grad = DxirReverseTransform.apply(allReduceLoss(groups))
        val gradAllReduces = grad.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.ALL_REDUCE }
        assertTrue(gradAllReduces.isNotEmpty(), "grad body must carry an ALL_REDUCE (self-adjoint rule)")
        assertTrue(
            gradAllReduces.any { it.attrs["replica_groups"] == groups },
            "the adjoint collective must carry the SAME replica_groups: " +
                gradAllReduces.map { it.attrs },
        )
    }

    @Test
    fun singleReplicaGradientMatchesTheUnshardedProgram() {
        // With replica_count == 1 the whole differentiation story collapses to
        // the unsharded program's: d/dx Σ x⊙x = 2x, BIT-EXACT (the identity
        // arm does no float math and MUL's grad is the same 2·x either way).
        val grad = DxirReverseTransform.apply(allReduceLoss(null))
        val out = DxirInterpreter.evalFunction(grad, listOf(x))[0]
        assertContentEquals(FloatArray(4) { 2 * x[it] }, out, "single-replica grad must be 2x exactly")
    }

    @Test
    fun jvpVjpIdentityThroughAllReduceBearingLoss() {
        // loss = Σ all_reduce(x⊙x) over [[0,1]]: interpreter semantics give
        // loss = 2·Σx², grad = 4x (hand pin), and forward/reverse must agree:
        // ⟨∇f(x), v⟩ == jvp_f(x, v).tangent.
        val groups = listOf(listOf(0, 1))
        val fn = allReduceLoss(groups)
        val grad = DxirReverseTransform.apply(fn)
        val g = DxirInterpreter.evalFunction(grad, listOf(x))[0]
        assertContentEquals(FloatArray(4) { 4 * x[it] }, g, "grad of Σ all_reduce(x⊙x) over |group|=2 must be 4x")

        val v = floatArrayOf(0.11f, -0.23f, 0.37f, -0.41f)
        val jvp = DxirForwardTransform.apply(fn)
        val out = DxirInterpreter.evalFunction(jvp, listOf(x, v))
        var lossExpected = 0f
        for (i in 0 until 4) lossExpected += 2 * x[i] * x[i]
        assertEquals(lossExpected, out[0].single(), 1e-6f, "jvp value stream must match the loss")
        var dot = 0f
        for (i in 0 until 4) dot += g[i] * v[i]
        val tangent = out[1].single()
        assertTrue(
            abs(dot - tangent) < 1e-5f,
            "JVP-VJP identity: ⟨∇f, v⟩ = $dot must equal the forward tangent $tangent",
        )
    }

    // --- SHARD_CONSTRAINT: the value identity with metadata. ---

    @Test
    fun shardConstraintInterpretsAsIdentity() {
        val fn = DxirBuilder.function("sc") {
            val x = param("x", vec4)
            listOf(op(OpKind.SHARD_CONSTRAINT, listOf(x), vec4))
        }
        assertContentEquals(x, DxirInterpreter.evalFunction(fn, listOf(x))[0])
    }

    @Test
    fun shardConstraintDifferentiatesAsIdentity() {
        // loss = Σ shard_constraint(x⊙x): the constraint is a value identity,
        // so grad = 2x exactly and the forward tangent is 2⟨x, v⟩.
        val fn = DxirBuilder.function("sc_loss") {
            val x = param("x", vec4)
            val sq = op(OpKind.MUL, listOf(x, x), vec4)
            val sc = op(OpKind.SHARD_CONSTRAINT, listOf(sq), vec4)
            listOf(op(OpKind.SUM, listOf(sc), f32s))
        }
        val g = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x))[0]
        assertContentEquals(FloatArray(4) { 2 * x[it] }, g, "identity adjoint: grad must be 2x exactly")

        val v = floatArrayOf(1f, 0.5f, -2f, 0.25f)
        val out = DxirInterpreter.evalFunction(DxirForwardTransform.apply(fn), listOf(x, v))
        var expected = 0f
        for (i in 0 until 4) expected += 2 * x[i] * v[i]
        assertTrue(
            abs(expected - out[1].single()) < 1e-6f,
            "identity tangent: jvp must be 2⟨x,v⟩ = $expected, got ${out[1].single()}",
        )
    }

    // --- 4. The renderer story: named refusal vs identity arm. ---

    @Test
    fun rendererRefusesAllReduceByName() {
        val ex = assertFailsWith<KotlinRenderRefusal> { toKotlinSource(allReduceOf(null)) }
        val msg = ex.message ?: ""
        assertTrue("ALL_REDUCE" in msg, "must name the kind: $msg")
        assertTrue("per-replica" in msg, "must state the per-replica story: $msg")
    }

    @Test
    fun rendererRendersShardConstraintAsPassThrough() {
        val fn = DxirBuilder.function("sc_render") {
            val x = param("x", vec4)
            val sc = op(OpKind.SHARD_CONSTRAINT, listOf(x), vec4)
            listOf(op(OpKind.NEG, listOf(sc), vec4))
        }
        val src = toKotlinSource(fn)
        assertTrue("= x" in src, "the constraint must render as the identity pass-through:\n$src")
        assertTrue(".neg()" in src, "downstream ops must consume the pass-through:\n$src")
    }
}
