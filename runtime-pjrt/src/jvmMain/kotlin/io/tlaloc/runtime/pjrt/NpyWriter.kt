package io.tlaloc.runtime.pjrt

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption

/**
 * Minimal NPY-format writer for f32 tensors (NPY 1.0). Same implementation as
 * `io.tlaloc.runtime.iree.NpyWriter`; duplicated here so `:runtime-pjrt` doesn't
 * cross-depend on `:runtime-iree` for one helper. If a third runtime appears,
 * this should factor out to a shared `:harness-npyio` module — for now the
 * 60-line duplication is the lower-friction choice.
 */
internal object NpyWriter {

    fun writeFloat32(path: Path, data: FloatArray, shape: List<Int>) {
        val expected = if (shape.isEmpty()) 1 else shape.fold(1) { a, b -> a * b }
        require(data.size == expected) {
            "NpyWriter: shape product $expected != data.size ${data.size} (shape=$shape)"
        }

        val shapeStr = when {
            shape.isEmpty() -> "()"
            shape.size == 1 -> "(${shape.single()},)"
            else -> shape.joinToString(", ", "(", ")")
        }
        val headerDict = "{'descr': '<f4', 'fortran_order': False, 'shape': $shapeStr, }"

        val preamble = 6 + 2 + 2  // magic + version + header_len
        val unpaddedTotal = preamble + headerDict.length + 1
        val paddedTotal = (unpaddedTotal + 63) / 64 * 64
        val paddingNeeded = paddedTotal - unpaddedTotal
        val finalHeader = headerDict + " ".repeat(paddingNeeded) + "\n"
        require(finalHeader.length <= 0xFFFF) { "NpyWriter: header too long for v1.0" }

        Files.newOutputStream(
            path,
            StandardOpenOption.CREATE,
            StandardOpenOption.TRUNCATE_EXISTING,
            StandardOpenOption.WRITE,
        ).use { out ->
            out.write(byteArrayOf(0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(), 'M'.code.toByte(), 'P'.code.toByte(), 'Y'.code.toByte()))
            out.write(byteArrayOf(0x01, 0x00))
            val hlen = finalHeader.length
            out.write(byteArrayOf((hlen and 0xFF).toByte(), ((hlen shr 8) and 0xFF).toByte()))
            out.write(finalHeader.toByteArray(Charsets.US_ASCII))
            val bb = ByteBuffer.allocate(data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            for (v in data) bb.putFloat(v)
            out.write(bb.array())
        }
    }

    /**
     * Reads a NPY 1.0 / 2.0 f32 file into a [FloatArray]. Tolerates the
     * minimal subset jax / numpy `np.save` / `np.savez` produces: little-endian,
     * C-order, dtype `<f4`. Errors loudly on unsupported variants.
     */
    fun readFloat32(path: Path): FloatArray {
        val bytes = Files.readAllBytes(path)
        require(bytes.size >= 10 && bytes[0] == 0x93.toByte()) {
            "NpyWriter: ${path} is not a valid .npy file (bad magic)"
        }
        require(bytes[1] == 'N'.code.toByte() && bytes[2] == 'U'.code.toByte()) {
            "NpyWriter: ${path} bad NUMPY signature"
        }
        val major = bytes[6].toInt()
        val headerLen: Int = when (major) {
            1 -> (bytes[8].toInt() and 0xFF) or ((bytes[9].toInt() and 0xFF) shl 8)
            2 -> (bytes[8].toInt() and 0xFF) or
                ((bytes[9].toInt() and 0xFF) shl 8) or
                ((bytes[10].toInt() and 0xFF) shl 16) or
                ((bytes[11].toInt() and 0xFF) shl 24)
            else -> error("NpyWriter: unsupported NPY major version $major")
        }
        val headerStart = if (major == 1) 10 else 12
        val header = String(bytes, headerStart, headerLen, Charsets.US_ASCII)
        require("'descr': '<f4'" in header || "'descr': '<f4 '" in header) {
            "NpyWriter: ${path} header lacks little-endian f32 descr; got: $header"
        }
        require("'fortran_order': False" in header) {
            "NpyWriter: ${path} fortran_order=True is unsupported; need C-order"
        }
        val dataStart = headerStart + headerLen
        val nFloats = (bytes.size - dataStart) / 4
        val bb = ByteBuffer.wrap(bytes, dataStart, nFloats * 4).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(nFloats) { bb.float }
    }
}
