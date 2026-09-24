package io.tlaloc.core.io

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption

// §0.4.502 — the jvm side of saving a checkpoint: putting bytes on a disk.
// Everything about the safetensors FORMAT is in commonMain
// (SafetensorsWriter.kt); this file holds a path and nothing else that could
// be wrong — the same division SafetensorsFile.kt draws on the reading side.

/**
 * Write a safetensors file.
 *
 * **Atomically, via a sibling temporary file and a rename.** A checkpoint is
 * written at the one moment a training run is most likely to be interrupted,
 * and a partially-written `model.safetensors` is worse than no file at all:
 * its header is complete and its data buffer is short, so it looks valid to
 * anything that only reads the header and fails deep inside a decode loop for
 * anything that does not. The temporary file is created in the DESTINATION'S
 * OWN DIRECTORY, because `ATOMIC_MOVE` across filesystems is not atomic and
 * is not even guaranteed to work.
 *
 * REJECTED: `Files.write(path, bytes)` directly — one line shorter, and it
 * makes the interrupted-save case silently corrupt instead of simply absent.
 * REJECTED: writing through a channel in chunks — [SafetensorsWriter.encode]
 * already materialises the whole file, so
 * chunking the write would only stage the same bytes twice.
 */
object SafetensorsFileWriter {

    fun write(
        path: Path,
        tensors: List<SafetensorsTensor>,
        metadata: Map<String, String> = emptyMap(),
    ): Path = writeBytes(path, SafetensorsWriter.encode(tensors, metadata))

    /** [write]'s pre-encoded form, for a caller that already holds the bytes. */
    fun writeBytes(path: Path, bytes: ByteArray): Path {
        val target = path.toAbsolutePath()
        val dir = target.parent
            ?: throw JsonException("safetensors writer: $path has no parent directory")
        Files.createDirectories(dir)
        val tmp = Files.createTempFile(dir, ".${target.fileName}", ".partial")
        try {
            Files.write(tmp, bytes)
            try {
                Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
            } catch (_: java.nio.file.AtomicMoveNotSupportedException) {
                // Named, not swallowed: some filesystems (and some container
                // overlay mounts) refuse ATOMIC_MOVE outright. Falling back
                // to a plain replace keeps the save working and loses the
                // crash-safety guarantee; the caller is not told because
                // there is nothing it could do differently, and the reason is
                // recorded here.
                Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING)
            }
        } finally {
            Files.deleteIfExists(tmp)
        }
        return target
    }
}
