package io.tlaloc.maestro.serving

import io.tlaloc.autograd.BucketLadders
import io.tlaloc.autograd.boundedProgram
import io.tlaloc.autograd.broadcastAlong
import io.tlaloc.autograd.div
import io.tlaloc.autograd.mean
import io.tlaloc.autograd.specOf
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.core.Bounded
import io.tlaloc.core.DimBound
import io.tlaloc.core.F32
import io.tlaloc.core.Named
import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.SeqLen
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.io.JsonException
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

object MaxSeqExport : DimBound(8)

class BoundedProgramExportTest {

    private val hidden = 3

    private fun maskedMean() = boundedProgram(
        "masked_mean",
        listOf(specOf<Rank2<Named<SeqLen, Bounded<MaxSeqExport>>, Sym>>(F32, hidden)),
        specOf<Rank1<Sym>>(F32, hidden),
    ) { xs, ctx ->
        val x = xs[0]
        (x * x.broadcastAlong<Shape>(ctx.validMask(MaxSeqExport), 0)).sum<Rank1<Sym>>(intArrayOf(0)) /
            ctx.validLength(MaxSeqExport)
    }

    private val ladders = BucketLadders(mapOf(MaxSeqExport to listOf(2, 4, 8)))

    @Test
    fun `export writes one body per bucket and a manifest that reads back`() {
        val dir = Files.createTempDirectory("tlaloc-bounded-export")
        try {
            val m = BoundedProgramExport.export(maskedMean(), dir, ladders)
            assertEquals(BoundedManifest.SCHEMA_VERSION, m.schemaVersion)
            assertEquals(listOf(BoundDecl("MaxSeqExport", 8, listOf(2, 4, 8))), m.bounds)
            assertEquals(listOf("x0", "validMask_MaxSeqExport", "validLength_MaxSeqExport"), m.inputs.map { it.name })
            assertEquals(listOf("DATA", "VALID_MASK", "VALID_LENGTH"), m.inputs.map { it.role })
            assertEquals(listOf(AxisDecl(bound = "MaxSeqExport"), AxisDecl(size = hidden)), m.inputs[0].axes)
            assertEquals(listOf(AxisDecl(size = hidden)), m.outputs.single().axes)
            assertEquals(listOf("MaxSeqExport2", "MaxSeqExport4", "MaxSeqExport8"), m.entries.map { it.id })
            assertTrue(m.paddingCheck.maxDifference <= 1e-5)
            assertEquals(listOf(1, 2, 3, 4, 5, 8), m.paddingCheck.sizes.map { it.getValue("MaxSeqExport") })

            for (e in m.entries) {
                val body = Files.readString(dir.resolve(e.bodyPath))
                val n = e.sizes.getValue("MaxSeqExport")
                assertTrue("func.func @main(" in body || "func.func public @main(" in body, body.take(300))
                assertTrue("tensor<${n}x${hidden}xf32>" in body, "bucket $n body:\n$body")
                assertTrue("tensor<${n}xf32>" in body, "bucket $n body has its mask:\n$body")
                assertTrue("?" !in body.substringBefore("{"), "static shapes only")
                assertTrue(Files.exists(dir.resolve(e.programPath)))
            }
            assertEquals(m, BoundedProgramExport.load(dir))
            assertEquals(mapOf("MaxSeqExport" to 4), m.bucketsFor(mapOf("MaxSeqExport" to 3)))
            assertEquals(mapOf("MaxSeqExport" to 2), m.bucketsFor(mapOf("MaxSeqExport" to 1)))
            assertFailsWith<IllegalArgumentException> { m.bucketsFor(mapOf("MaxSeqExport" to 9)) }
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun `export refuses a program whose padded result differs from the exact one`() {
        val unmasked = boundedProgram(
            "unmasked_mean",
            listOf(specOf<Rank2<Bounded<MaxSeqExport>, Sym>>(F32, hidden)),
            specOf<Rank1<Sym>>(F32, hidden),
        ) { xs, _ -> xs[0].mean<Shape>(intArrayOf(0)) }
        val dir = Files.createTempDirectory("tlaloc-bounded-refused")
        try {
            val e = assertFailsWith<IllegalArgumentException> { BoundedProgramExport.export(unmasked, dir, ladders) }
            assertTrue("must read ctx.validMask or ctx.validLength; export refused" in e.message!!, e.message)
            assertTrue(!Files.exists(dir.resolve(BoundedManifest.FILE_NAME)))
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun `the reader refuses other versions, unknown keys and roles, and incomplete entries`() {
        val dir = Files.createTempDirectory("tlaloc-bounded-reader")
        try {
            val json = BoundedProgramExport.export(maskedMean(), dir, ladders).toJson()
            fun refused(text: String, fragment: String) {
                val e = assertFailsWith<Exception> { BoundedManifest.fromJson(text) }
                assertTrue(fragment in e.message!!, "expected '$fragment' in: ${e.message}")
            }
            refused(json.replace("tlaloc-bounded-v1", "tlaloc-bounded-v2"), "schemaVersion 'tlaloc-bounded-v2' is not one this reader knows")
            refused(json.replaceFirst("\"name\":", "\"extra\":1,\"name\":"), "unknown key(s) [extra]")
            refused(json.replace("\"role\":\"VALID_MASK\"", "\"role\":\"ATTENTION_BIAS\""), "unknown role 'ATTENTION_BIAS'")
            refused(json.replace("\"buckets\":[2,4,8]", "\"buckets\":[2,4,6,8]"), "missing [{MaxSeqExport=6}]")
            refused(json.replace("{\"size\":3}", "{\"size\":3,\"bound\":\"MaxSeqExport\"}"), "exactly one of 'size' and 'bound'")
            // The language-model reader names the other format instead of a missing field.
            val e = assertFailsWith<JsonException> { ServingManifest.fromJson(json) }
            assertTrue("bounded-program artifact, not a language-model serving artifact" in e.message!!, e.message)
            // And the bounded loader names a serving artifact.
            Files.move(dir.resolve(BoundedManifest.FILE_NAME), dir.resolve(ServingManifest.FILE_NAME))
            val l = assertFailsWith<IllegalArgumentException> { BoundedProgramExport.load(dir) }
            assertTrue("a language-model serving artifact, not a bounded-program artifact" in l.message!!, l.message)
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun `every committed serving manifest still loads, and the committed bounded examples load`() {
        // Written before bounded programs existed (tlaloc-serving-v2 and v3); the reader must
        // take them unchanged.
        val repo = Path.of("..", "triton", "examples", "model_repository").toAbsolutePath().normalize()
        val serving = Files.walk(repo).use { s ->
            s.filter { it.fileName.toString().startsWith("tlaloc-serving") && it.toString().endsWith(".json") }.toList()
        }
        assertTrue(serving.size >= 6, "found $serving")
        for (f in serving) {
            val m = ServingManifest.fromJson(Files.readString(f))
            assertTrue(m.schemaVersion in ServingManifest.READABLE_VERSIONS, "$f")
        }
        for (name in listOf("bounded_mean", "bounded_softmax")) {
            val m = BoundedProgramExport.load(repo.resolve(name).resolve("1"))
            assertEquals(listOf(BoundDecl("ExampleMaxSeq", 16, listOf(4, 8, 16))), m.bounds)
        }
    }

    @Test
    fun `load refuses a body that does not match its hash`() {
        val dir = Files.createTempDirectory("tlaloc-bounded-hash")
        try {
            val m = BoundedProgramExport.export(maskedMean(), dir, ladders)
            Files.writeString(dir.resolve(m.entries[1].bodyPath), "tampered")
            val e = assertFailsWith<IllegalArgumentException> { BoundedProgramExport.load(dir) }
            assertTrue("does not match its hash" in e.message!!, e.message)
        } finally {
            dir.toFile().deleteRecursively()
        }
    }
}
