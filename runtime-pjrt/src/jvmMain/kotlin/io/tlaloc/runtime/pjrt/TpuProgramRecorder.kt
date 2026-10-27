package io.tlaloc.runtime.pjrt

import io.tlaloc.runtime.pjrt.ffm.CompileOptions
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest

/**
 * With `TLALOC_TPU_PROGRAMS_DIR` set, every [PjrtSession] writes each program it compiles, as a
 * TPU session would send it, into that directory: `<name>-<hash>.mlir`, and beside it
 * `<name>-<hash>.options` with its replicas, partitions and partitioner (`key=value` lines). A
 * program compiled from a DxirFunction is written in its TPU lowering whatever the session's
 * target; StableHLO text is written as given. `scripts/tpu/aot-check.sh` runs the device tests
 * this way and compiles what they wrote with XLA:TPU, no TPU attached.
 */
internal object TpuProgramRecorder {
    val dir: Path? = System.getenv("TLALOC_TPU_PROGRAMS_DIR")?.takeIf { it.isNotBlank() }?.let { Path.of(it) }

    fun record(name: String, mlir: String, options: CompileOptions, into: Path? = dir) {
        val dir = into ?: return
        val hash = MessageDigest.getInstance("SHA-256").digest(mlir.toByteArray())
            .take(6).joinToString("") { "%02x".format(it) }
        val base = "${name.replace(Regex("[^A-Za-z0-9_.-]"), "_").take(80)}-$hash"
        val file = dir.resolve("$base.mlir")
        if (Files.exists(file)) return
        Files.createDirectories(dir)
        Files.writeString(
            dir.resolve("$base.options"),
            "replicas=${options.numReplicas}\npartitions=${options.numPartitions}\n" +
                "spmd=${options.useSpmdPartitioning}\nshardy=${options.useShardyPartitioner}\n",
        )
        Files.writeString(file, mlir)
    }
}
