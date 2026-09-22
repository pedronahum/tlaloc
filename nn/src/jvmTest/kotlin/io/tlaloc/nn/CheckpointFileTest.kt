package io.tlaloc.nn

import io.tlaloc.core.RandomKey
import io.tlaloc.core.hostF32
import io.tlaloc.core.io.SafetensorsFile
import io.tlaloc.core.split
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.502 — the filesystem half of model persistence: `saveCheckpoint` /
 * `loadCheckpoint` on a real path, and the two properties that are only
 * observable once a file is involved — that the bytes on disk are a valid
 * safetensors file to a reader that knows nothing about `:nn`, and that the
 * write leaves no partial file behind.
 */
class CheckpointFileTest {

    private fun tmp(): Path = Files.createTempDirectory("tlaloc-nn-ckpt-").also {
        it.toFile().deleteOnExit()
    }

    private fun model() = Sequential(
        Dense(3, 4, RandomKey.fromSeed(31).split(2)[0]),
        ReluLayer,
        BatchNorm(numFeatures = 4),
    )

    @Test
    fun aModelSavesToAPathAndLoadsBackBitIdentically() {
        val dir = tmp()
        val original = model()
        val opt = Adam(learningRate = 0.01f)
        val written = saveCheckpoint(
            dir.resolve("run/model.safetensors"),
            original,
            opt,
            opt.initialState(),
            mapOf("run" to "unit-test"),
        )
        assertTrue(Files.isRegularFile(written), "$written")
        assertEquals(dir.resolve("run/model.safetensors").toAbsolutePath(), written)

        val snapshot = loadCheckpoint(written)
        assertEquals(mapOf("run" to "unit-test"), snapshot.metadata)
        val restored = snapshot.restore(model())
        for (i in original.parameters.indices) {
            val a = original.parameters[i].tensor.hostF32()
            val b = restored.parameters[i].tensor.hostF32()
            for (j in a.indices) {
                assertEquals(a[j].toRawBits(), b[j].toRawBits(), "${original.parameters[i].key}[$j]")
            }
        }
        assertEquals(0, snapshot.restoreOptimizerState(opt).stepCount)
    }

    @Test
    fun theSavedFileIsReadableByTheFormatReaderWithNoKnowledgeOfNn() {
        val dir = tmp()
        val path = saveCheckpoint(dir.resolve("model.safetensors"), model())
        SafetensorsFile.open(path).use { f ->
            assertTrue(f.names.any { it.startsWith("param.") }, "${f.names}")
            assertTrue(f.names.any { it.startsWith("buffer.") }, "${f.names}")
            assertEquals("1", f.metadata[ModelCheckpoint.VERSION_KEY])
            // And the tensor really is there, at the right shape.
            assertContentEquals(intArrayOf(3, 4), f.load("param.0.w").dims)
        }
    }

    @Test
    fun theWriteLeavesNoPartialFileBehindAndOverwritesInPlace() {
        val dir = tmp()
        val path = dir.resolve("model.safetensors")
        saveCheckpoint(path, model())
        val first = Files.size(path)
        // A second save over the same path: the directory must hold exactly
        // one file afterwards — no `.partial` temporary survives a successful
        // write, which is what makes the atomic-rename path safe to retry.
        saveCheckpoint(path, Sequential(Dense(3, 4, RandomKey.fromSeed(99).split(2)[0])))
        Files.list(dir).use { s ->
            assertEquals(listOf("model.safetensors"), s.map { it.fileName.toString() }.sorted().toList())
        }
        assertTrue(Files.size(path) < first, "the smaller model must have replaced the larger file")
    }

    @Test
    fun checkpointTensorsNamesEveryTensorTheFileWouldHold() {
        val m = model()
        val names = checkpointTensors(m).map { it.name }
        assertEquals(
            listOf(
                "param.0.w", "param.0.b", "param.2.gamma", "param.2.beta",
                "buffer.2.runningN", "buffer.2.runningSum", "buffer.2.runningSumOfSquares",
            ),
            names,
        )
    }

    @Test
    fun loadingSomethingThatIsNotAFileIsRefusedByName() {
        val dir = tmp()
        val e = assertFailsWith<IllegalArgumentException> { loadCheckpoint(dir) }
        assertTrue("not a regular file" in e.message!!, e.message!!)
    }
}
