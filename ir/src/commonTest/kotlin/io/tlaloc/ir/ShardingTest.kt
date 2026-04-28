package io.tlaloc.ir

import io.tlaloc.core.F32
import io.tlaloc.core.MeshSpec
import io.tlaloc.core.Spec
import io.tlaloc.core.partitionSpec
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ShardingTest {

    private val f32mat = DxirType(F32, listOf(32, 16))

    @Test
    fun meshConvertsFromCoreToDxir() {
        val core = MeshSpec.of("data" to 8, "model" to 4, name = "m")
        val dxir = core.toDxir()
        assertEquals("m", dxir.name)
        assertEquals(listOf(DxirMeshAxis("data", 8), DxirMeshAxis("model", 4)), dxir.axes)
    }

    @Test
    fun partitionSpecConvertsDimShardingsOpenAndClosed() {
        val spec = partitionSpec(
            Spec.On(listOf("data"), open = true, priority = 1),
            Spec.Replicated,
        )
        val sharding = spec.toDxir("m")
        assertEquals(2, sharding.dimShardings.size)
        assertEquals(
            DxirDimSharding(listOf(DxirAxisRef.Full("data")), closed = false, priority = 1),
            sharding.dimShardings[0],
        )
        assertEquals(
            DxirDimSharding(emptyList(), closed = true),
            sharding.dimShardings[1],
        )
    }

    @Test
    fun subOnSpecLowersToSubAxisRef() {
        val s = partitionSpec(Spec.SubOn("data", preSize = 2, size = 4)).toDxir("m")
        assertEquals(
            DxirDimSharding(listOf(DxirAxisRef.Sub("data", 2, 4))),
            s.dimShardings.single(),
        )
    }

    @Test
    fun nodeCarriesOptionalSharding() {
        val mesh = DxirMesh("m", listOf(DxirMeshAxis("data", 8), DxirMeshAxis("model", 4)))
        val sharding = DxirSharding(
            meshName = "m",
            dimShardings = listOf(
                DxirDimSharding(listOf(DxirAxisRef.Full("data"))),
                DxirDimSharding(listOf(DxirAxisRef.Full("model"))),
            ),
        )
        val fn = DxirBuilder.function("sharded") {
            declareMesh(mesh)
            val x = param("x", f32mat, sharding = sharding)
            val y = op(OpKind.RELU, listOf(x), f32mat, sharding = sharding)
            listOf(y)
        }
        assertEquals(sharding, fn.params.single().sharding)
        assertEquals(sharding, fn.body.filterIsInstance<DxirOp>().single().sharding)
        assertEquals(listOf(mesh), fn.meshes)
    }

    @Test
    fun functionRejectsShardingAgainstUndeclaredMesh() {
        val sharding = DxirSharding("ghostmesh", listOf(DxirDimSharding(emptyList())))
        assertFailsWith<IllegalArgumentException> {
            DxirBuilder.function("bad") {
                val x = param("x", f32mat, sharding = sharding)
                listOf(x)
            }
        }
    }

    @Test
    fun shardConstraintOpIsAvailable() {
        val mesh = DxirMesh("m", listOf(DxirMeshAxis("data", 8)))
        val sharding = DxirSharding("m", listOf(DxirDimSharding(listOf(DxirAxisRef.Full("data")))))
        val fn = DxirBuilder.function("withConstraint") {
            declareMesh(mesh)
            val x = param("x", DxirType(F32, listOf(16)))
            val pinned = op(OpKind.SHARD_CONSTRAINT, listOf(x), x.type, sharding = sharding)
            listOf(pinned)
        }
        val pinned = fn.body.filterIsInstance<DxirOp>().single()
        assertEquals(OpKind.SHARD_CONSTRAINT, pinned.op)
        assertEquals(sharding, pinned.sharding)
    }

    @Test
    fun printerIncludesMeshAndShardingAnnotations() {
        val mesh = DxirMesh("m", listOf(DxirMeshAxis("data", 8)))
        val sharding = DxirSharding("m", listOf(DxirDimSharding(listOf(DxirAxisRef.Full("data")))))
        val fn = DxirBuilder.function("f") {
            declareMesh(mesh)
            val x = param("x", DxirType(F32, listOf(16)), sharding = sharding)
            listOf(x)
        }
        val out = DxirModule(listOf(fn)).pretty()
        assertTrue(out.contains("mesh @m = [data=8]"), "mesh decl missing: $out")
        assertTrue(out.contains("@m[{data}]"), "sharding annotation missing: $out")
    }

    @Test
    fun duplicateMeshDeclarationRejected() {
        val mesh = DxirMesh("m", listOf(DxirMeshAxis("data", 2)))
        assertFailsWith<IllegalArgumentException> {
            DxirBuilder.function("dup") {
                declareMesh(mesh)
                declareMesh(mesh)
                val x = param("x", DxirType(F32, listOf(4)))
                listOf(x)
            }
        }
    }
}
