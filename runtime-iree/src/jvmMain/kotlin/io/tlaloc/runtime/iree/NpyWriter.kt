package io.tlaloc.runtime.iree

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption

/**
 * Minimal NPY-format writer for f32 tensors. NPY 1.0 layout:
 *
 *   magic ("\x93NUMPY", 6 bytes) +
 *   version ("\x01\x00", 2 bytes) +
 *   HEADER_LEN (uint16 little-endian, 2 bytes) +
 *   header (ASCII Python dict literal, padded with spaces, terminated with '\n') +
 *   raw data (little-endian f32, row-major / C-order)
 *
 * The total preamble (magic + version + HEADER_LEN + header) is padded to a
 * multiple of 64 bytes so the data section is aligned for `numpy.load` to mmap
 * efficiently.
 *
 * `iree-run-module` and `iree-benchmark-module` accept `--input=@<path>.npy`
 * directly, which means a host that needs to dispatch with hundreds of MB of
 * inputs (e.g. a medium-shaped LlamaDecoder forward) can sidestep the textual
 * `<shape>xf32=v0,v1,…` form and the OS argv / flagfile-size limits that
 * come with it. Used by both the §0.4.289+ PyTorch-comparison test (npy as
 * the cross-language hand-off format) and the §0.4.294+ medium-config
 * benchmark (npy as the input-marshalling format for IREE's tools).
 */
object NpyWriter {

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
        val unpaddedTotal = preamble + headerDict.length + 1  // +1 for trailing '\n'
        val paddedTotal = (unpaddedTotal + 63) / 64 * 64
        val paddingNeeded = paddedTotal - unpaddedTotal
        val finalHeader = headerDict + " ".repeat(paddingNeeded) + "\n"
        require(finalHeader.length <= 0xFFFF) { "NpyWriter: header too long for v1.0; switch to v2.0" }

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
}
