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
 * Exercises Shardy's propagation pipeline on our emitter output. Per spec §9.6:
 *
 *   dxir → SDY emission → sdy-propagation-pipeline → StableHLO + explicit collectives
 *
 * We pipe the emitter output through `sdy-opt --sdy-propagation-pipeline` and assert that:
 *   (1) the tool accepts our SDY (no syntax/verification errors),
 *   (2) propagation infers shardings on compute ops we left unannotated,
 *   (3) those shardings are semantically sensible for the op and the input constraints.
 *
 * Self-skips if `sdy-opt` isn't on PATH.
 */
class SdyPropagationTest {

    private fun requireSdyOrSkip() {
        assumeTrue(
            SdyOpt.available,
            "sdy-opt not on PATH — skipping SDY propagation validation.",
        )
    }

    private fun propagate(mlir: String, label: String): String {
        val result = SdyOpt.run(mlir, "--sdy-propagation-pipeline")
        assertTrue(
            result.ok,
            "sdy-propagation-pipeline rejected '$label':\n  exit=${result.exitCode}\n  stderr=${result.stderr}\n--- input ---\n$mlir",
        )
        assertTrue(result.stdout.isNotEmpty(), "$label: propagation stdout was empty")
        return result.stdout
    }

    private fun fullMesh(axes: List<Pair<String, Int>>, name: String = "m"): DxirMesh =
        DxirMesh(name, axes.map { (n, s) -> DxirMeshAxis(n, s) })

    private fun shardFull(meshName: String, vararg dimAxes: List<String>): DxirSharding =
        DxirSharding(
            meshName,
            dimAxes.map { axes ->
                DxirDimSharding(axes.map { DxirAxisRef.Full(it) })
            },
        )

    // --- test cases ---

    @Test
    fun matmulDataModelParallelPropagatesToHybridOutput() {
        requireSdyOrSkip()
        // (B=32, D=16) × (D=16, C=8) → (32, 8). Constrain Q on "data", K on "model".
        // Shardy should infer output sharded on both axes [{data}, {model}].
        val mesh = fullMesh(listOf("data" to 8, "model" to 4))
        val shardQ = shardFull("m", listOf("data"), emptyList())
        val shardK = shardFull("m", emptyList(), listOf("model"))
        val fn = DxirBuilder.function("mm") {
            declareMesh(mesh)
            val q = param("q", DxirType(F32, listOf(32, 16)))
            val k = param("k", DxirType(F32, listOf(16, 8)))
            val qc = op(OpKind.SHARD_CONSTRAINT, listOf(q), DxirType(F32, listOf(32, 16)), sharding = shardQ)
            val kc = op(OpKind.SHARD_CONSTRAINT, listOf(k), DxirType(F32, listOf(16, 8)), sharding = shardK)
            val out = op(OpKind.MATMUL, listOf(qc, kc), DxirType(F32, listOf(32, 8)))
            listOf(out)
        }
        val propagated = propagate(DxirModule(listOf(fn)).toStablehlo(), "matmul DP×TP")

        // The dot_general op should now carry a sharding annotation with both "data" and "model".
        assertTrue(
            propagated.contains("stablehlo.dot_general") &&
                propagated.contains("sdy.sharding") &&
                propagated.contains("\"data\"") &&
                propagated.contains("\"model\""),
            "propagation output missing expected shardings on matmul:\n$propagated",
        )
        // Return type should also carry the inferred sharding.
        assertTrue(
            propagated.contains("[{\"data\"}, {\"model\"}]"),
            "return sharding should be [{data}, {model}]:\n$propagated",
        )
    }

    @Test
    fun elementwiseUnaryPreservesShardingThroughPropagation() {
        requireSdyOrSkip()
        // Input sharded on "data" → relu should inherit the same sharding.
        val mesh = fullMesh(listOf("data" to 4))
        val shardIn = shardFull("m", listOf("data"), emptyList())
        val fn = DxirBuilder.function("ew") {
            declareMesh(mesh)
            val x = param("x", DxirType(F32, listOf(8, 4)))
            val xc = op(OpKind.SHARD_CONSTRAINT, listOf(x), DxirType(F32, listOf(8, 4)), sharding = shardIn)
            val y = op(OpKind.RELU, listOf(xc), DxirType(F32, listOf(8, 4)))
            listOf(y)
        }
        val propagated = propagate(DxirModule(listOf(fn)).toStablehlo(), "elementwise relu")
        // After propagation, the maximum (relu's lowering) or final return should carry [{data}, {}].
        assertTrue(
            propagated.contains("[{\"data\"}, {}]"),
            "relu output should be sharded [{data}, {}]:\n$propagated",
        )
    }

    @Test
    fun chainedElementwiseOpsAllGetAnnotated() {
        requireSdyOrSkip()
        // Constrain input, then (x+x)*x. Every intermediate should get the same sharding.
        val mesh = fullMesh(listOf("data" to 8))
        val shardIn = shardFull("m", listOf("data"))
        val fn = DxirBuilder.function("chain") {
            declareMesh(mesh)
            val x = param("x", DxirType(F32, listOf(16)))
            val xc = op(OpKind.SHARD_CONSTRAINT, listOf(x), DxirType(F32, listOf(16)), sharding = shardIn)
            val a = op(OpKind.ADD, listOf(xc, xc), DxirType(F32, listOf(16)))
            val b = op(OpKind.MUL, listOf(a, xc), DxirType(F32, listOf(16)))
            listOf(b)
        }
        val propagated = propagate(DxirModule(listOf(fn)).toStablehlo(), "chain add+mul")
        // Every add/multiply line should carry a sharding annotation.
        val addSharded = Regex("stablehlo\\.add[^\\n]*sdy\\.sharding").containsMatchIn(propagated)
        val mulSharded = Regex("stablehlo\\.multiply[^\\n]*sdy\\.sharding").containsMatchIn(propagated)
        assertTrue(addSharded, "stablehlo.add must carry sdy.sharding after propagation:\n$propagated")
        assertTrue(mulSharded, "stablehlo.multiply must carry sdy.sharding after propagation:\n$propagated")
    }

    @Test
    fun reductionDropsReducedAxisSharding() {
        requireSdyOrSkip()
        // Input sharded on both dims, reduce along last dim. Output should only keep the
        // non-reduced dim's sharding.
        val mesh = fullMesh(listOf("data" to 4, "model" to 2))
        val shardIn = shardFull("m", listOf("data"), listOf("model"))
        val fn = DxirBuilder.function("red") {
            declareMesh(mesh)
            val x = param("x", DxirType(F32, listOf(8, 4)))
            val xc = op(OpKind.SHARD_CONSTRAINT, listOf(x), DxirType(F32, listOf(8, 4)), sharding = shardIn)
            // Reduce along dim 1 → output shape (8,), should be sharded on "data".
            val y = op(
                OpKind.SUM, listOf(xc), DxirType(F32, listOf(8)),
                attrs = mapOf("reduction_dims" to listOf(1)),
            )
            listOf(y)
        }
        val propagated = propagate(DxirModule(listOf(fn)).toStablehlo(), "reduce-sum last axis")
        assertTrue(
            propagated.contains("stablehlo.reduce"),
            "reduce op missing from output:\n$propagated",
        )
        // Output sharding on the single remaining dim should be `{data}`.
        assertTrue(
            propagated.contains("[{\"data\"}]"),
            "reduced output should be sharded [{data}]:\n$propagated",
        )
    }

    @Test
    fun propagationSurvivesFsdpStyleShardingPattern() {
        requireSdyOrSkip()
        // FSDP-ish: weights W sharded on "data" (first dim = output features).
        // Input X replicated. Output should be replicated on the activation dim that
        // matches the contracting/data axis — exercises propagation through matmul.
        val mesh = fullMesh(listOf("data" to 8))
        val shardW = shardFull("m", listOf("data"), emptyList())
        val fn = DxirBuilder.function("fsdp") {
            declareMesh(mesh)
            val x = param("x", DxirType(F32, listOf(16, 32)))
            val w = param("w", DxirType(F32, listOf(32, 64)))
            val wc = op(OpKind.SHARD_CONSTRAINT, listOf(w), DxirType(F32, listOf(32, 64)), sharding = shardW)
            val out = op(OpKind.MATMUL, listOf(x, wc), DxirType(F32, listOf(16, 64)))
            listOf(out)
        }
        val propagated = propagate(DxirModule(listOf(fn)).toStablehlo(), "FSDP pattern")
        assertTrue(
            propagated.contains("sdy.sharding") && propagated.contains("stablehlo.dot_general"),
            "propagation must annotate the matmul op:\n$propagated",
        )
    }
}
