package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirAxisRef
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirDimSharding
import io.tlaloc.ir.DxirMesh
import io.tlaloc.ir.DxirMeshAxis
import io.tlaloc.ir.DxirSharding
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class GradShardingVerifyTest {

    private val mesh = DxirMesh("m", listOf(DxirMeshAxis("data", 8), DxirMeshAxis("model", 4)))
    private fun dataShard() = DxirSharding(
        "m",
        listOf(DxirDimSharding(listOf(DxirAxisRef.Full("data")))),
    )
    private fun modelShard() = DxirSharding(
        "m",
        listOf(DxirDimSharding(listOf(DxirAxisRef.Full("model")))),
    )

    // A representative unary forward: f(x) = relu(x). Returns just a dxir function.
    private fun unaryForward(paramSharding: DxirSharding?) = DxirBuilder.function("fwd") {
        declareMesh(mesh)
        val x = param("x", DxirType(F32, listOf(16)), sharding = paramSharding)
        val y = op(OpKind.RELU, listOf(x), DxirType(F32, listOf(16)))
        listOf(y)
    }

    private fun unaryGrad(returnSharding: DxirSharding?) = DxirBuilder.function("grad_fwd") {
        declareMesh(mesh)
        val x = param("x", DxirType(F32, listOf(16)), sharding = returnSharding)
        // Gradient function returns `∂L/∂x` with the SAME sharding as x in the identity-dual case.
        // The body is synthetic; only the return sharding matters here.
        val g = op(OpKind.NEG, listOf(x), DxirType(F32, listOf(16)), sharding = returnSharding)
        listOf(g)
    }

    @Test
    fun identityDualShardingIsValid() {
        val fwd = unaryForward(dataShard())
        val grad = unaryGrad(dataShard())
        // Wrap return sharding on the returned op, matching the forward param.
        val violations = GradShardingVerify.verify(fwd, wrapReturnSharding(grad, dataShard()))
        assertTrue(violations.isEmpty(), "expected no violations, got: $violations")
    }

    @Test
    fun bothUnshardedIsValid() {
        val fwd = unaryForward(null)
        val grad = unaryGrad(null)
        val violations = GradShardingVerify.verify(fwd, grad)
        assertTrue(violations.isEmpty())
    }

    @Test
    fun mismatchedShardingAxisIsFlagged() {
        val fwd = unaryForward(dataShard())
        val grad = unaryGrad(modelShard())
        val violations = GradShardingVerify.verify(fwd, wrapReturnSharding(grad, modelShard()))
        assertEquals(1, violations.size)
        assertTrue(violations.single().message.contains("does not match forward sharding"))
    }

    @Test
    fun forwardShardedGradUnshardedIsFlagged() {
        val fwd = unaryForward(dataShard())
        val grad = unaryGrad(null)   // missing sharding
        val violations = GradShardingVerify.verify(fwd, grad)
        assertEquals(1, violations.size)
    }

    @Test
    fun paramArityMismatchStopsFurtherChecks() {
        val fwd = DxirBuilder.function("fwd") {
            val a = param("a", DxirType(F32, listOf(4)))
            val b = param("b", DxirType(F32, listOf(4)))
            val c = op(OpKind.ADD, listOf(a, b), DxirType(F32, listOf(4)))
            listOf(c)
        }
        val grad = DxirBuilder.function("grad_fwd") {
            // Only one return — but fwd has two params.
            val a = param("a", DxirType(F32, listOf(4)))
            listOf(a)
        }
        val violations = GradShardingVerify.verify(fwd, grad)
        assertEquals(1, violations.size)
        assertTrue(violations.single().site == "function arity")
    }

    @Test
    fun typeMismatchIsFlagged() {
        val fwd = unaryForward(null)
        val grad = DxirBuilder.function("grad_fwd") {
            val x = param("x", DxirType(F32, listOf(16)))
            // Wrong shape on the gradient output.
            val g = op(OpKind.NEG, listOf(x), DxirType(F32, listOf(32)))
            listOf(g)
        }
        val violations = GradShardingVerify.verify(fwd, grad)
        assertEquals(1, violations.size)
        assertTrue(violations.single().message.contains("does not match forward param type"))
    }

    @Test
    fun multiParamFunctionsWithMatchingShardingsValidate() {
        val fwd = DxirBuilder.function("fwd") {
            declareMesh(mesh)
            val x = param("x", DxirType(F32, listOf(32, 8)), sharding = dataShard2d())
            val w = param("w", DxirType(F32, listOf(8, 4)), sharding = modelShard2d())
            val out = op(OpKind.MATMUL, listOf(x, w), DxirType(F32, listOf(32, 4)))
            listOf(out)
        }
        val grad = DxirBuilder.function("grad_fwd") {
            declareMesh(mesh)
            val x = param("x", DxirType(F32, listOf(32, 8)), sharding = dataShard2d())
            val w = param("w", DxirType(F32, listOf(8, 4)), sharding = modelShard2d())
            val gx = op(OpKind.NEG, listOf(x), DxirType(F32, listOf(32, 8)), sharding = dataShard2d())
            val gw = op(OpKind.NEG, listOf(w), DxirType(F32, listOf(8, 4)), sharding = modelShard2d())
            listOf(gx, gw)
        }
        val violations = GradShardingVerify.verify(fwd, grad)
        assertTrue(violations.isEmpty(), "violations: $violations")
    }

    @Test
    fun verifyOrThrowThrowsOnViolation() {
        val fwd = unaryForward(dataShard())
        val grad = unaryGrad(modelShard())
        assertFailsWith<IllegalStateException> {
            GradShardingVerify.verifyOrThrow(fwd, wrapReturnSharding(grad, modelShard()))
        }
    }

    @Test
    fun verifyOrThrowNoOpOnValidPair() {
        val fwd = unaryForward(dataShard())
        val grad = unaryGrad(dataShard())
        GradShardingVerify.verifyOrThrow(fwd, wrapReturnSharding(grad, dataShard()))
        // No exception = pass.
    }

    // --- helpers ---

    private fun dataShard2d() = DxirSharding(
        "m",
        listOf(
            DxirDimSharding(listOf(DxirAxisRef.Full("data"))),
            DxirDimSharding(emptyList()),
        ),
    )

    private fun modelShard2d() = DxirSharding(
        "m",
        listOf(
            DxirDimSharding(emptyList()),
            DxirDimSharding(listOf(DxirAxisRef.Full("model"))),
        ),
    )

    /**
     * Rebuilds `grad` replacing the sharding attached to its single return op. Lets
     * individual tests control the return sharding without reshuffling the builder closure.
     */
    private fun wrapReturnSharding(grad: io.tlaloc.ir.DxirFunction, sharding: DxirSharding?): io.tlaloc.ir.DxirFunction =
        DxirBuilder.function(grad.name) {
            grad.meshes.forEach { declareMesh(it) }
            val xp = grad.params.single()
            val x = param(xp.name, xp.type, sharding = xp.sharding)
            val g = op(OpKind.NEG, listOf(x), xp.type, sharding = sharding)
            listOf(g)
        }

    // --- v1: op-by-op collective duality rules ---

    private val t16 = DxirType(F32, listOf(16))

    private fun fnWithCollectives(name: String, vararg kinds: OpKind) =
        DxirBuilder.function(name) {
            val x = param("x", t16)
            var cur: io.tlaloc.ir.DxirNode = x
            for (k in kinds) {
                cur = op(k, listOf(cur), t16)
            }
            // Terminal non-collective op so the collectives aren't the last nodes (mirrors realistic fns).
            val out = op(OpKind.NEG, listOf(cur), t16)
            listOf(out)
        }

    @Test
    fun collectiveDuality_fwdAllReduce_gradIdentity_valid() {
        val fwd = fnWithCollectives("fwd", OpKind.ALL_REDUCE)
        val grad = fnWithCollectives("grad_fwd" /* no collectives */)
        val violations = GradShardingVerify.verifyCollectiveDuality(fwd, grad)
        assertTrue(violations.isEmpty(), "expected no violations, got: $violations")
    }

    @Test
    fun collectiveDuality_fwdAllGather_gradReduceScatter_valid() {
        val fwd = fnWithCollectives("fwd", OpKind.ALL_GATHER)
        val grad = fnWithCollectives("grad_fwd", OpKind.REDUCE_SCATTER)
        val violations = GradShardingVerify.verifyCollectiveDuality(fwd, grad)
        assertTrue(violations.isEmpty(), "expected no violations, got: $violations")
    }

    @Test
    fun collectiveDuality_fwdReduceScatter_gradAllGather_valid() {
        val fwd = fnWithCollectives("fwd", OpKind.REDUCE_SCATTER)
        val grad = fnWithCollectives("grad_fwd", OpKind.ALL_GATHER)
        val violations = GradShardingVerify.verifyCollectiveDuality(fwd, grad)
        assertTrue(violations.isEmpty(), "expected no violations, got: $violations")
    }

    @Test
    fun collectiveDuality_bothEmpty_valid() {
        val fwd = fnWithCollectives("fwd")
        val grad = fnWithCollectives("grad_fwd")
        val violations = GradShardingVerify.verifyCollectiveDuality(fwd, grad)
        assertTrue(violations.isEmpty())
    }

    @Test
    fun collectiveDuality_fwdAllReduce_gradAllReduce_invalid() {
        val fwd = fnWithCollectives("fwd", OpKind.ALL_REDUCE)
        val grad = fnWithCollectives("grad_fwd", OpKind.ALL_REDUCE)
        val violations = GradShardingVerify.verifyCollectiveDuality(fwd, grad)
        assertEquals(1, violations.size)
        // ALL_REDUCE's dual is identity, so the grad shouldn't have a collective here.
        assertTrue(
            violations.single().message.contains("expected dual sequence []") ||
                violations.single().site == "collective count",
            "got: ${violations.single()}",
        )
    }

    @Test
    fun collectiveDuality_fwdAllGather_gradAllGather_invalid() {
        val fwd = fnWithCollectives("fwd", OpKind.ALL_GATHER)
        val grad = fnWithCollectives("grad_fwd", OpKind.ALL_GATHER)
        val violations = GradShardingVerify.verifyCollectiveDuality(fwd, grad)
        assertEquals(1, violations.size)
        val msg = violations.single().message
        assertTrue(msg.contains("REDUCE_SCATTER") && msg.contains("ALL_GATHER"), "got: $msg")
    }

    @Test
    fun collectiveDuality_chainReversedDual_valid() {
        // forward sequence [AG, AR] → reversed duals: [adjoint(AR)=∅, adjoint(AG)=RS] → [RS].
        val fwd = fnWithCollectives("fwd", OpKind.ALL_GATHER, OpKind.ALL_REDUCE)
        val grad = fnWithCollectives("grad_fwd", OpKind.REDUCE_SCATTER)
        val violations = GradShardingVerify.verifyCollectiveDuality(fwd, grad)
        assertTrue(violations.isEmpty(), "expected no violations, got: $violations")
    }

    @Test
    fun collectiveDuality_chainWrongOrder_invalid() {
        // forward [AG, RS] → reversed duals [adjoint(RS)=AG, adjoint(AG)=RS] = [AG, RS].
        // But if grad lists [RS, AG] (wrong order), it should flag a per-position mismatch.
        val fwd = fnWithCollectives("fwd", OpKind.ALL_GATHER, OpKind.REDUCE_SCATTER)
        val grad = fnWithCollectives("grad_fwd", OpKind.REDUCE_SCATTER, OpKind.ALL_GATHER)
        val violations = GradShardingVerify.verifyCollectiveDuality(fwd, grad)
        assertTrue(violations.isNotEmpty(), "expected ordering violations")
        // Two per-index mismatches: got RS expected AG at [0], got AG expected RS at [1].
        assertEquals(2, violations.size, "got: $violations")
    }

    @Test
    fun adjointOf_rules() {
        assertEquals(null, GradShardingVerify.adjointOf(OpKind.ALL_REDUCE))
        assertEquals(OpKind.REDUCE_SCATTER, GradShardingVerify.adjointOf(OpKind.ALL_GATHER))
        assertEquals(OpKind.ALL_GATHER, GradShardingVerify.adjointOf(OpKind.REDUCE_SCATTER))
    }
}
