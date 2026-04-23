package io.tlaloc.stablehlo

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirAxisRef
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirDimSharding
import io.tlaloc.ir.DxirMesh
import io.tlaloc.ir.DxirMeshAxis
import io.tlaloc.ir.DxirModule
import io.tlaloc.ir.DxirSharding
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Pipes emitter output through `sdy-opt`; any syntax/verification error from the SDY
 * dialect fails the test. Self-skips if `sdy-opt` isn't on PATH so builds stay green
 * on boxes without Shardy installed.
 */
class SdyRoundTripTest {

    private fun requireSdyOrSkip() {
        assumeTrue(
            SdyOpt.available,
            "sdy-opt not on PATH — skipping SDY round-trip validation. Build Shardy from source.",
        )
    }

    private fun validate(mlir: String, label: String) {
        val result = SdyOpt.run(mlir)
        assertTrue(
            result.ok,
            "sdy-opt rejected '$label':\n  exit=${result.exitCode}\n  stderr=${result.stderr}\n--- input ---\n$mlir",
        )
        assertTrue(result.stdout.isNotEmpty(), "$label: sdy-opt output was empty")
    }

    @Test
    fun meshDeclRoundTrips() {
        requireSdyOrSkip()
        val fn = DxirBuilder.function("f") {
            declareMesh(DxirMesh("m", listOf(DxirMeshAxis("data", 8), DxirMeshAxis("model", 4))))
            val x = param("x", DxirType(F32, listOf(32, 16)))
            listOf(x)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "mesh decl only")
    }

    @Test
    fun shardConstraintRoundTripsBasic() {
        requireSdyOrSkip()
        val mesh = DxirMesh("m", listOf(DxirMeshAxis("data", 8), DxirMeshAxis("model", 4)))
        val sharding = DxirSharding(
            "m",
            listOf(
                DxirDimSharding(listOf(DxirAxisRef.Full("data"))),
                DxirDimSharding(listOf(DxirAxisRef.Full("model"))),
            ),
        )
        val fn = DxirBuilder.function("f") {
            declareMesh(mesh)
            val x = param("x", DxirType(F32, listOf(32, 16)))
            val y = op(
                OpKind.SHARD_CONSTRAINT, listOf(x), DxirType(F32, listOf(32, 16)),
                sharding = sharding,
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "SHARD_CONSTRAINT 2D mesh")
    }

    @Test
    fun shardConstraintWithReplicatedAxesRoundTrips() {
        requireSdyOrSkip()
        val mesh = DxirMesh(
            "m",
            listOf(DxirMeshAxis("data", 4), DxirMeshAxis("model", 2), DxirMeshAxis("pipe", 2)),
        )
        val sharding = DxirSharding(
            "m",
            dimShardings = listOf(
                DxirDimSharding(listOf(DxirAxisRef.Full("data"))),
                DxirDimSharding(emptyList()),
            ),
            replicated = listOf(DxirAxisRef.Full("model"), DxirAxisRef.Full("pipe")),
        )
        val fn = DxirBuilder.function("f") {
            declareMesh(mesh)
            val x = param("x", DxirType(F32, listOf(8, 4)))
            val y = op(
                OpKind.SHARD_CONSTRAINT, listOf(x), DxirType(F32, listOf(8, 4)),
                sharding = sharding,
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "SHARD_CONSTRAINT w/ replicated axes")
    }

    @Test
    fun shardConstraintWithOpenAndPriorityRoundTrips() {
        requireSdyOrSkip()
        val mesh = DxirMesh("m", listOf(DxirMeshAxis("a", 2), DxirMeshAxis("b", 2)))
        val sharding = DxirSharding(
            "m",
            listOf(
                DxirDimSharding(
                    listOf(DxirAxisRef.Full("a")),
                    closed = false,
                    priority = 0,
                ),
                DxirDimSharding(
                    listOf(DxirAxisRef.Full("b")),
                    closed = true,
                    priority = 1,
                ),
            ),
        )
        val fn = DxirBuilder.function("f") {
            declareMesh(mesh)
            val x = param("x", DxirType(F32, listOf(4, 4)))
            val y = op(
                OpKind.SHARD_CONSTRAINT, listOf(x), DxirType(F32, listOf(4, 4)),
                sharding = sharding,
            )
            listOf(y)
        }
        validate(
            DxirModule(listOf(fn)).toStablehlo(),
            "SHARD_CONSTRAINT w/ open-axis + priority",
        )
    }

    @Test
    fun manualComputationRoundTripsSimple() {
        requireSdyOrSkip()
        val mesh = DxirMesh("m", listOf(DxirMeshAxis("a", 4)))
        val shard = DxirSharding(
            "m",
            listOf(DxirDimSharding(listOf(DxirAxisRef.Full("a")))),
        )
        val fn = DxirBuilder.function("f") {
            declareMesh(mesh)
            val x = param("x", DxirType(F32, listOf(8)))
            val mc = op(
                OpKind.MANUAL_COMPUTATION, listOf(x),
                DxirType(F32, listOf(8)),
                attrs = mapOf(
                    "in_shardings" to listOf(shard),
                    "out_shardings" to listOf(shard),
                    "manual_axes" to listOf("a"),
                ),
                regions = listOf(
                    region {
                        // Per-device shard: 8/4 = 2 elements.
                        val local = arg(DxirType(F32, listOf(2)))
                        val r = op(OpKind.NEG, listOf(local), DxirType(F32, listOf(2)))
                        yields(r)
                    },
                ),
            )
            listOf(mc)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "MANUAL_COMPUTATION 1D negate")
    }

    @Test
    fun manualComputationTwoInputsRoundTrips() {
        requireSdyOrSkip()
        val mesh = DxirMesh("m", listOf(DxirMeshAxis("a", 2), DxirMeshAxis("b", 2)))
        val shardA = DxirSharding(
            "m",
            listOf(
                DxirDimSharding(listOf(DxirAxisRef.Full("a"))),
                DxirDimSharding(listOf(DxirAxisRef.Full("b"))),
            ),
        )
        val fn = DxirBuilder.function("f") {
            declareMesh(mesh)
            val q = param("q", DxirType(F32, listOf(4, 4)))
            val k = param("k", DxirType(F32, listOf(4, 4)))
            val mc = op(
                OpKind.MANUAL_COMPUTATION, listOf(q, k),
                DxirType(F32, listOf(4, 4)),
                attrs = mapOf(
                    "in_shardings" to listOf(shardA, shardA),
                    "out_shardings" to listOf(shardA),
                    "manual_axes" to listOf("a", "b"),
                ),
                regions = listOf(
                    region {
                        // 4/2 × 4/2 = 2×2 per-device shards.
                        val qLocal = arg(DxirType(F32, listOf(2, 2)))
                        val kLocal = arg(DxirType(F32, listOf(2, 2)))
                        val sum = op(OpKind.ADD, listOf(qLocal, kLocal), DxirType(F32, listOf(2, 2)))
                        yields(sum)
                    },
                ),
            )
            listOf(mc)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "MANUAL_COMPUTATION 2D add w/ 2D mesh")
    }

    @Test
    fun shardConstraintWithSubAxisRoundTrips() {
        requireSdyOrSkip()
        // Sub-axis "data":(2)4 means "starting at pre-size 2 within axis data, take size 4".
        val mesh = DxirMesh("m", listOf(DxirMeshAxis("data", 16)))
        val sharding = DxirSharding(
            "m",
            listOf(
                DxirDimSharding(listOf(DxirAxisRef.Sub("data", preSize = 2, size = 4))),
            ),
        )
        val fn = DxirBuilder.function("f") {
            declareMesh(mesh)
            val x = param("x", DxirType(F32, listOf(32)))
            val y = op(
                OpKind.SHARD_CONSTRAINT, listOf(x), DxirType(F32, listOf(32)),
                sharding = sharding,
            )
            listOf(y)
        }
        validate(DxirModule(listOf(fn)).toStablehlo(), "SHARD_CONSTRAINT w/ sub-axis")
    }
}
