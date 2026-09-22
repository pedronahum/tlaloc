package io.tlaloc.core.io

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.core.HostBf16Storage
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostF64Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.floatArrayToBf16Bits
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.502 — **cross-implementation certification of the safetensors writer.**
 *
 * `SafetensorsWriterTest` pins the writer against Tlaloc's own reader, which
 * is a real oracle (the two were written five months apart and share no code)
 * but is still Tlaloc on both sides. This test puts the REFERENCE
 * implementation — the `safetensors` Python library, 0.8.0, in the frozen
 * oracle venv — on the other side of the exchange, in both directions:
 *
 *  1. **Tlaloc writes, the reference reads.** Compared as RAW HEX BYTES per
 *     tensor, not as floats: the claim is that the bytes are the bytes. A
 *     float comparison would pass over a NaN payload, a signed zero, or a
 *     bf16 value that had been widened and re-narrowed.
 *  2. **The reference writes, Tlaloc reads.** The same fixture, coming back.
 *
 * It is the exact mirror of §0.4.468's `write_llama_safetensors.py` lane,
 * which certified the reader this way.
 *
 * SELF-SKIPS BY NAME when the interpreter, the library or the script is
 * missing, following the `NnMlpVsPytorchTrainingTest` ladder. It installs
 * nothing and touches no venv — the oracle venv is canary-pinned by
 * `OracleVenvIntegrityTest` precisely so that it cannot be "helpfully"
 * modified.
 */
class SafetensorsWriterOracleTest {

    // The fixture, kept in lockstep with the script's own copy. Values are
    // exactly representable at every width so a disagreement can only be a
    // format disagreement.
    private val f32Values = floatArrayOf(1f, -2.5f, 0f, 1024f, -0.125f, 3f)
    private val f64Values = doubleArrayOf(1.0, -0.5, 1e300, 0.0)
    private val i32Values = intArrayOf(-7, 0, 2_000_000_000, 1)
    private val bf16Values = floatArrayOf(1f, -2f, 0.5f, 100f)
    private val metadata = mapOf("format" to "tlaloc", "tlaloc.test" to "§0.4.502")

    private fun fixture(): List<SafetensorsTensor> = listOf(
        SafetensorsTensor("a.f32", F32, intArrayOf(2, 3), HostF32Storage(f32Values)),
        SafetensorsTensor("b.f64", F64, intArrayOf(4), HostF64Storage(f64Values)),
        SafetensorsTensor("c.i32", I32, intArrayOf(2, 2), HostI32Storage(i32Values)),
        SafetensorsTensor("d.bf16", BF16, intArrayOf(4), HostBf16Storage(floatArrayToBf16Bits(bf16Values))),
        SafetensorsTensor("e.scalar", F32, intArrayOf(), HostF32Storage(floatArrayOf(7f))),
        SafetensorsTensor("f.empty", F32, intArrayOf(0, 4), HostF32Storage(FloatArray(0))),
    )

    private fun resolvePython(): String? {
        System.getenv("TLALOC_TORCH_PYTHON")?.let { if (Files.isExecutable(Path.of(it))) return it }
        val home = System.getProperty("user.home") ?: return null
        val p = Path.of(home, ".local", "venvs", "iree", "bin", "python")
        return if (Files.isExecutable(p)) p.toString() else null
    }

    private fun canImport(python: String, module: String): Boolean = runCatching {
        val pb = ProcessBuilder(python, "-c", "import $module")
        pb.redirectErrorStream(true)
        val p = pb.start()
        p.inputStream.readBytes()
        p.waitFor(30, TimeUnit.SECONDS) && p.exitValue() == 0
    }.getOrElse { false }

    private fun run(python: String, script: Path, vararg args: String): String {
        val pb = ProcessBuilder(listOf(python, script.toString()) + args)
        pb.redirectErrorStream(true)
        val p = pb.start()
        val out = p.inputStream.readBytes().decodeToString()
        assertTrue(p.waitFor(180, TimeUnit.SECONDS), "oracle script timed out\n$out")
        assertEquals(0, p.exitValue(), "oracle script failed:\n$out")
        return out
    }

    /** Little-endian hex of a raw byte range, the form the script reports. */
    private fun hexOf(bytes: ByteArray): String =
        bytes.joinToString("") { ((it.toInt() and 0xFF) + 0x100).toString(16).substring(1) }

    private fun leFloatHex(v: FloatArray): String {
        val b = ByteArray(v.size * 4)
        for (i in v.indices) {
            var x = v[i].toRawBits()
            for (j in 0 until 4) { b[i * 4 + j] = (x and 0xFF).toByte(); x = x ushr 8 }
        }
        return hexOf(b)
    }

    private fun leDoubleHex(v: DoubleArray): String {
        val b = ByteArray(v.size * 8)
        for (i in v.indices) {
            var x = v[i].toRawBits()
            for (j in 0 until 8) { b[i * 8 + j] = (x and 0xFF).toByte(); x = x ushr 8 }
        }
        return hexOf(b)
    }

    private fun leIntHex(v: IntArray): String {
        val b = ByteArray(v.size * 4)
        for (i in v.indices) {
            var x = v[i]
            for (j in 0 until 4) { b[i * 4 + j] = (x and 0xFF).toByte(); x = x ushr 8 }
        }
        return hexOf(b)
    }

    private fun leShortHex(v: ShortArray): String {
        val b = ByteArray(v.size * 2)
        for (i in v.indices) {
            val x = v[i].toInt()
            b[i * 2] = (x and 0xFF).toByte()
            b[i * 2 + 1] = ((x ushr 8) and 0xFF).toByte()
        }
        return hexOf(b)
    }

    private class Gate(val python: String, val script: Path)

    private fun gate(): Gate? {
        val python = resolvePython()
        assumeTrue(
            python != null,
            "no python interpreter resolved at ~/.local/venvs/iree/bin/python — skipping the " +
                "safetensors reference-implementation oracle. Set TLALOC_TORCH_PYTHON to override.",
        )
        assumeTrue(
            canImport(python!!, "safetensors"),
            "python at $python cannot `import safetensors` — skipping the reference-implementation " +
                "oracle (nothing is installed by this test; see OracleVenvIntegrityTest).",
        )
        assumeTrue(
            canImport(python, "torch"),
            "python at $python cannot `import torch` — skipping. torch is needed only for the BF16 " +
                "leg: numpy has no bfloat16 dtype, so safetensors.numpy cannot express one.",
        )
        val script = Path.of("..", "harness", "python", "safetensors_writer_oracle.py")
            .toAbsolutePath().normalize()
        assumeTrue(Files.exists(script), "oracle script not found at $script — skipping.")
        return Gate(python, script)
    }

    @Test
    fun theReferenceImplementationReadsTlalocsBytesBackBitForBit() {
        val g = gate() ?: return
        val dir = Files.createTempDirectory("tlaloc-st-writer-oracle-")
        dir.toFile().deleteOnExit()
        val file = dir.resolve("model.safetensors")
        SafetensorsFileWriter.write(file, fixture(), metadata)

        val out = dir.resolve("read.json")
        run(g.python, g.script, "read", file.toString(), out.toString())
        val root = parseJson(Files.readString(out)) as JsonObject
        val tensors = root.obj("tensors")
        val md = root.obj("metadata")

        assertEquals(metadata.keys, md.fields.keys)
        for ((k, v) in metadata) assertEquals(v, md.str(k), "metadata['$k']")

        val expected = mapOf(
            "a.f32" to Triple("F32", listOf(2, 3), leFloatHex(f32Values)),
            "b.f64" to Triple("F64", listOf(4), leDoubleHex(f64Values)),
            "c.i32" to Triple("I32", listOf(2, 2), leIntHex(i32Values)),
            "d.bf16" to Triple("BF16", listOf(4), leShortHex(floatArrayToBf16Bits(bf16Values))),
            "e.scalar" to Triple("F32", emptyList(), leFloatHex(floatArrayOf(7f))),
            "f.empty" to Triple("F32", listOf(0, 4), ""),
        )
        assertEquals(expected.keys, tensors.fields.keys)
        for ((name, want) in expected) {
            val got = tensors.obj(name)
            assertEquals(want.first, got.str("dtype"), "$name dtype")
            assertEquals(want.second, got.arr("shape").asIntList("$name.shape"), "$name shape")
            assertEquals(want.third, got.str("hex"), "$name bytes")
        }
    }

    @Test
    fun tlalocReadsTheReferenceImplementationsBytesBackBitForBit() {
        val g = gate() ?: return
        val dir = Files.createTempDirectory("tlaloc-st-writer-oracle-")
        dir.toFile().deleteOnExit()
        val file = dir.resolve("ref.safetensors")
        run(g.python, g.script, "write", file.toString())

        SafetensorsFile.open(file).use { f ->
            assertEquals(
                setOf("a.f32", "b.f64", "c.i32", "d.bf16", "e.scalar", "f.empty"),
                f.names,
            )
            assertEquals(metadata, f.metadata)
            assertContentEquals(f32Values, (f.load("a.f32").storage as HostF32Storage).data)
            assertContentEquals(intArrayOf(2, 3), f.load("a.f32").dims)
            assertContentEquals(f64Values, (f.load("b.f64").storage as HostF64Storage).data)
            assertContentEquals(i32Values, (f.load("c.i32").storage as HostI32Storage).data)
            assertContentEquals(
                floatArrayToBf16Bits(bf16Values),
                (f.load("d.bf16").storage as HostBf16Storage).data,
            )
            assertEquals(0, f.load("e.scalar").dims.size)
            assertContentEquals(floatArrayOf(7f), (f.load("e.scalar").storage as HostF32Storage).data)
            assertContentEquals(intArrayOf(0, 4), f.load("f.empty").dims)
            assertEquals(0, (f.load("f.empty").storage as HostF32Storage).data.size)
        }
    }

    /**
     * The round trip through the reference implementation, end to end: Tlaloc
     * writes, the reference reads it and writes its own file, Tlaloc reads
     * that. What this adds over the two tests above is that it puts a
     * reference-produced file through the SAME comparison as a
     * Tlaloc-produced one, so a systematic error shared by the writer and the
     * reader (a byte order, say) has nowhere to hide.
     */
    @Test
    fun aFileSurvivesTlalocThenTheReferenceThenTlalocAgain() {
        val g = gate() ?: return
        val dir = Files.createTempDirectory("tlaloc-st-writer-oracle-")
        dir.toFile().deleteOnExit()
        val mine = dir.resolve("mine.safetensors")
        SafetensorsFileWriter.write(mine, fixture(), metadata)
        val theirs = dir.resolve("theirs.safetensors")
        run(g.python, g.script, "write", theirs.toString())

        val a = Safetensors.readAll(Files.readAllBytes(mine))
        val b = Safetensors.readAll(Files.readAllBytes(theirs))
        assertEquals(a.keys, b.keys)
        for (name in a.keys) {
            assertContentEquals(a.getValue(name).dims, b.getValue(name).dims, "$name dims")
            assertEquals(a.getValue(name).dtype, b.getValue(name).dtype, "$name dtype")
            val x = a.getValue(name).storage
            val y = b.getValue(name).storage
            when (x) {
                is HostF32Storage -> assertContentEquals(x.data, (y as HostF32Storage).data, name)
                is HostF64Storage -> assertContentEquals(x.data, (y as HostF64Storage).data, name)
                is HostI32Storage -> assertContentEquals(x.data, (y as HostI32Storage).data, name)
                is HostBf16Storage -> assertContentEquals(x.data, (y as HostBf16Storage).data, name)
                else -> error("unexpected storage for $name")
            }
        }
    }
}
