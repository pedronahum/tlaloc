package io.tlaloc.nn

import io.tlaloc.core.io.SafetensorsFile
import io.tlaloc.core.io.SafetensorsFileWriter
import io.tlaloc.core.io.SafetensorsTensor
import java.nio.file.Files
import java.nio.file.Path

// §0.4.502 — the jvm side of model persistence: a path instead of a
// ByteArray. The format lives in commonMain (Checkpoint.kt); this file holds
// nothing but the filesystem, the same division the safetensors reader and
// writer draw in `:core`.

/**
 * Save [model] (and optionally an optimizer's state) to [path].
 *
 * The write is ATOMIC — a temporary sibling plus a rename, see
 * [SafetensorsFileWriter] — because a checkpoint is written at the one moment
 * a run is most likely to be interrupted, and a half-written checkpoint whose
 * header is complete is worse than no checkpoint at all.
 *
 * Returns the absolute path actually written, so a caller can log it.
 */
fun saveCheckpoint(
    path: Path,
    model: Trainable<*>,
    metadata: Map<String, String> = emptyMap(),
): Path = SafetensorsFileWriter.writeBytes(path, ModelCheckpoint.encode(model, metadata))

/** [saveCheckpoint]'s resumable form: the model plus [optimizer]'s [state]. */
fun <S> saveCheckpoint(
    path: Path,
    model: Trainable<*>,
    optimizer: Optimizer<S>,
    state: S,
    metadata: Map<String, String> = emptyMap(),
): Path = SafetensorsFileWriter.writeBytes(
    path,
    ModelCheckpoint.encode(model, optimizer, state, metadata),
)

/**
 * Read a checkpoint from [path].
 *
 * Goes through [SafetensorsFile] — the per-tensor reader — rather than
 * slurping the file, for the reason recorded there: a checkpoint is the
 * largest file a training process handles and reading it whole doubles peak
 * memory at the moment memory is scarcest.
 */
fun loadCheckpoint(path: Path): ModelSnapshot {
    require(Files.isRegularFile(path)) { "loadCheckpoint: $path is not a regular file" }
    return SafetensorsFile.open(path).use { f ->
        ModelCheckpoint.fromParts(f.loadAll(), f.metadata)
    }
}

/**
 * The tensors a checkpoint of [model] would contain, without encoding them —
 * for a caller that wants to write them into a file of its own arrangement, or
 * to count them in a log line. The names carry the checkpoint's own prefixes.
 */
fun checkpointTensors(model: Trainable<*>): List<SafetensorsTensor> = buildList {
    for (p in model.parameters) {
        add(SafetensorsTensor.of(ModelCheckpoint.PARAM_PREFIX + p.key, p.tensor))
    }
    if (model is Stateful<*>) {
        for (b in model.buffers) {
            add(SafetensorsTensor.of(ModelCheckpoint.BUFFER_PREFIX + b.key, b.tensor))
        }
    }
}
