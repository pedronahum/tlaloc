package io.tlaloc.stablehlo

import io.tlaloc.ir.DxirAxisRef
import io.tlaloc.ir.DxirDimSharding
import io.tlaloc.ir.DxirMesh
import io.tlaloc.ir.DxirMeshAxis
import io.tlaloc.ir.DxirSharding
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Round-trip tests: for every SDY form [SdyEmit] can emit, the parser should reconstruct
 * the exact original [DxirMesh] / [DxirSharding] value. These tests don't shell out to
 * `sdy-opt`; that's covered by the existing [SdyRoundTripTest] in jvmTest. This pair gives
 * us a pure-Kotlin contract that emit ∘ parse = identity.
 */
class SdyParseTest {

    // --- Mesh round-trip ---

    @Test
    fun parseSingleAxisMesh() {
        val mesh = DxirMesh("data_par", listOf(DxirMeshAxis("data", 8)))
        assertEquals(mesh, SdyParser.parseMeshDecl(mesh.toSdyDecl()))
    }

    @Test
    fun parseMultiAxisMesh() {
        val mesh = DxirMesh(
            "hybrid",
            listOf(DxirMeshAxis("data", 8), DxirMeshAxis("model", 4), DxirMeshAxis("pipeline", 2)),
        )
        assertEquals(mesh, SdyParser.parseMeshDecl(mesh.toSdyDecl()))
    }

    @Test
    fun parseMeshesInMultiMeshModule() {
        val m1 = DxirMesh("m1", listOf(DxirMeshAxis("x", 4)))
        val m2 = DxirMesh("m2", listOf(DxirMeshAxis("y", 2), DxirMeshAxis("z", 8)))
        val module = """
            module {
              ${m1.toSdyDecl()}
              ${m2.toSdyDecl()}
              func.func @f(%arg0: tensor<4xf32>) -> tensor<4xf32> { return %arg0 : tensor<4xf32> }
            }
        """.trimIndent()
        assertEquals(listOf(m1, m2), SdyParser.parseMeshesInModule(module))
    }

    // --- Sharding round-trip: dim shardings ---

    @Test
    fun parseFullyShardedDim() {
        val sh = DxirSharding(
            "m",
            listOf(
                DxirDimSharding(listOf(DxirAxisRef.Full("data"))),
                DxirDimSharding(listOf(DxirAxisRef.Full("model"))),
            ),
        )
        assertEquals(sh, SdyParser.parseSharding(sh.toSdyAttr()))
    }

    @Test
    fun parseEmptyDimSharding() {
        val sh = DxirSharding("m", listOf(DxirDimSharding(emptyList())))
        assertEquals(sh, SdyParser.parseSharding(sh.toSdyAttr()))
    }

    @Test
    fun parseReplicated() {
        val sh = DxirSharding(
            "m",
            listOf(DxirDimSharding(emptyList())),
            replicated = listOf(DxirAxisRef.Full("data"), DxirAxisRef.Full("model")),
        )
        assertEquals(sh, SdyParser.parseSharding(sh.toSdyAttr()))
    }

    @Test
    fun parseOpenDimSharding() {
        val sh = DxirSharding(
            "m",
            listOf(DxirDimSharding(axes = listOf(DxirAxisRef.Full("data")), closed = false)),
        )
        assertEquals(sh, SdyParser.parseSharding(sh.toSdyAttr()))
    }

    @Test
    fun parseOpenEmptyDimSharding() {
        val sh = DxirSharding(
            "m",
            listOf(DxirDimSharding(axes = emptyList(), closed = false)),
        )
        assertEquals(sh, SdyParser.parseSharding(sh.toSdyAttr()))
    }

    @Test
    fun parsePriorityAnnotation() {
        val sh = DxirSharding(
            "m",
            listOf(
                DxirDimSharding(listOf(DxirAxisRef.Full("data")), priority = 0),
                DxirDimSharding(listOf(DxirAxisRef.Full("model")), priority = 2),
            ),
        )
        assertEquals(sh, SdyParser.parseSharding(sh.toSdyAttr()))
    }

    @Test
    fun parseSubAxisSharding() {
        val sh = DxirSharding(
            "m",
            listOf(DxirDimSharding(listOf(DxirAxisRef.Sub("data", preSize = 4, size = 2)))),
        )
        assertEquals(sh, SdyParser.parseSharding(sh.toSdyAttr()))
    }

    @Test
    fun parseMultipleAxesPerDim() {
        val sh = DxirSharding(
            "m",
            listOf(
                DxirDimSharding(listOf(DxirAxisRef.Full("data"), DxirAxisRef.Full("model"))),
                DxirDimSharding(emptyList()),
            ),
        )
        assertEquals(sh, SdyParser.parseSharding(sh.toSdyAttr()))
    }

    @Test
    fun parseCombinedOpenSubAxisReplicatedPriority() {
        // Exercise the full grammar in a single sharding.
        val sh = DxirSharding(
            meshName = "hybrid",
            dimShardings = listOf(
                DxirDimSharding(
                    axes = listOf(DxirAxisRef.Full("data")),
                    closed = false,
                    priority = 1,
                ),
                DxirDimSharding(
                    axes = listOf(DxirAxisRef.Sub("model", preSize = 4, size = 2)),
                    closed = true,
                    priority = null,
                ),
            ),
            replicated = listOf(DxirAxisRef.Full("other")),
        )
        assertEquals(sh, SdyParser.parseSharding(sh.toSdyAttr()))
    }

    // --- Negative cases ---

    @Test
    fun rejectsMalformedMesh() {
        assertFailsWith<IllegalStateException> {
            SdyParser.parseMeshDecl("sdy.mesh name = <[]>") // missing @
        }
    }

    @Test
    fun rejectsTrailingInputAfterSharding() {
        val input = """<@m, [{"data"}]> trailing"""
        val ex = assertFailsWith<IllegalArgumentException> {
            SdyParser.parseSharding(input)
        }
        assertTrue(ex.message!!.contains("trailing"), "got: ${ex.message}")
    }
}
