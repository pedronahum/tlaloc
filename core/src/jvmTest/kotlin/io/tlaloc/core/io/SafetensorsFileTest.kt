package io.tlaloc.core.io

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.468 (Phase H2) — the jvm side of weight ingestion: opening a file,
 * reading one tensor's byte range out of it, and routing a SHARDED checkpoint
 * through `model.safetensors.index.json`.
 */
class SafetensorsFileTest {

    private fun f32(vararg v: Float): ByteArray {
        val out = ByteArray(v.size * 4)
        for ((i, x) in v.withIndex()) {
            var b = x.toRawBits()
            for (j in 0 until 4) { out[i * 4 + j] = (b and 0xFF).toByte(); b = b ushr 8 }
        }
        return out
    }

    private fun writeFile(path: Path, vararg tensors: Triple<String, List<Int>, ByteArray>): Path {
        val sb = StringBuilder("{")
        var off = 0L
        for ((i, t) in tensors.withIndex()) {
            if (i > 0) sb.append(',')
            val end = off + t.third.size
            sb.append("\"").append(t.first).append("\":{\"dtype\":\"F32\",\"shape\":[")
                .append(t.second.joinToString(",")).append("],\"data_offsets\":[")
                .append(off).append(',').append(end).append("]}")
            off = end
        }
        sb.append('}')
        val h = sb.toString().toByteArray(Charsets.UTF_8)
        val data = tensors.fold(ByteArray(0)) { a, t -> a + t.third }
        val out = ByteArray(8 + h.size + data.size)
        var n = h.size.toLong()
        for (i in 0 until 8) { out[i] = (n and 0xFF).toByte(); n = n ushr 8 }
        h.copyInto(out, 8)
        data.copyInto(out, 8 + h.size)
        Files.createDirectories(path.parent)
        Files.write(path, out)
        return path
    }

    @Test
    fun readsOneTensorsRangeWithoutReadingTheRest() {
        val dir = Files.createTempDirectory("tlaloc-st")
        val p = writeFile(
            dir.resolve("model.safetensors"),
            Triple("a", listOf(2, 2), f32(1f, 2f, 3f, 4f)),
            Triple("b", listOf(3), f32(10f, 20f, 30f)),
        )
        SafetensorsFile.open(p).use { f ->
            assertEquals(setOf("a", "b"), f.names)
            assertEquals(listOf(10f, 20f, 30f), f.load("b").toF32Array().toList())
            assertEquals(listOf(1f, 2f, 3f, 4f), f.load("a").toF32Array().toList())
            assertEquals(listOf(2, 2), f.load("a").dims.toList())
            assertEquals(2, f.loadAll().size)
        }
    }

    @Test
    fun aFileShorterThanItsHeaderClaimsIsRefused() {
        val dir = Files.createTempDirectory("tlaloc-st")
        val p = writeFile(dir.resolve("model.safetensors"), Triple("a", listOf(4), f32(1f, 2f, 3f, 4f)))
        val all = Files.readAllBytes(p)
        Files.write(p, all.copyOfRange(0, all.size - 6))
        val e = assertFailsWith<JsonException> { SafetensorsFile.open(p) }
        assertTrue("only" in e.message!! || "data buffer" in e.message!!, e.message!!)
    }

    @Test
    fun theShardedIndexRoutesEachTensorToItsShard() {
        val dir = Files.createTempDirectory("tlaloc-st")
        writeFile(dir.resolve("model-00001-of-00002.safetensors"), Triple("a", listOf(2), f32(1f, 2f)))
        writeFile(dir.resolve("model-00002-of-00002.safetensors"), Triple("b", listOf(2), f32(3f, 4f)))
        Files.write(
            dir.resolve("model.safetensors.index.json"),
            ("{\"metadata\":{\"total_size\":16},\"weight_map\":" +
                "{\"a\":\"model-00001-of-00002.safetensors\"," +
                "\"b\":\"model-00002-of-00002.safetensors\"}}").toByteArray(Charsets.UTF_8),
        )
        SafetensorsIndex.openCheckpoint(dir).use { src ->
            assertEquals(setOf("a", "b"), src.names)
            assertEquals(listOf(1f, 2f), src.load("a").toF32Array().toList())
            assertEquals(listOf(3f, 4f), src.load("b").toF32Array().toList())
        }
    }

    @Test
    fun openCheckpointPrefersTheIndexAndFallsBackToTheSingleFile() {
        val dir = Files.createTempDirectory("tlaloc-st")
        writeFile(dir.resolve("model.safetensors"), Triple("w", listOf(1), f32(7f)))
        SafetensorsIndex.openCheckpoint(dir).use { src ->
            assertEquals(listOf(7f), src.load("w").toF32Array().toList())
        }
        val empty = Files.createTempDirectory("tlaloc-st-empty")
        val e = assertFailsWith<JsonException> { SafetensorsIndex.openCheckpoint(empty) }
        assertTrue("neither" in e.message!!, e.message!!)
    }

    @Test
    fun anIndexMayNotNameAFileOutsideItsOwnDirectory() {
        val dir = Files.createTempDirectory("tlaloc-st")
        Files.write(
            dir.resolve("model.safetensors.index.json"),
            "{\"weight_map\":{\"a\":\"../../etc/passwd\"}}".toByteArray(Charsets.UTF_8),
        )
        val e = assertFailsWith<JsonException> { SafetensorsIndex.open(dir.resolve("model.safetensors.index.json")) }
        assertTrue("bare shard filename" in e.message!!, e.message!!)
    }

    @Test
    fun anUnknownTensorNamesTheIndexItCameFrom() {
        val dir = Files.createTempDirectory("tlaloc-st")
        Files.write(
            dir.resolve("model.safetensors.index.json"),
            "{\"weight_map\":{\"a\":\"s.safetensors\"}}".toByteArray(Charsets.UTF_8),
        )
        SafetensorsIndex.open(dir.resolve("model.safetensors.index.json")).use { idx ->
            val e = assertFailsWith<JsonException> { idx.load("nope") }
            assertTrue("nope" in e.message!! && "index.json" in e.message!!, e.message!!)
        }
    }
}
