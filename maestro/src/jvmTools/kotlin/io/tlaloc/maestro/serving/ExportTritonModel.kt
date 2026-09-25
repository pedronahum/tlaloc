package io.tlaloc.maestro.serving

import java.nio.file.Path

/**
 * Entry point of
 * `./gradlew :maestro:exportTritonModel -PartifactDir=… -PoutDir=<repository> [-PmodelName=…]
 * [-PkvMode=sequence|client] [-PmaxSequenceIdleMicros=…] [-PmaxQueueDelayMicros=…]`.
 *
 * Writes an existing serving artifact into a Triton model repository through
 * [TritonModelRepository.write]. Arguments, positionally: the artifact
 * directory, the repository directory, the model name (default: the artifact
 * directory's name), the KV mode (default `sequence`), and the two
 * [TritonModelRepository.SequenceOptions] values.
 */
fun main(args: Array<String>) {
    require(args.size >= 2) {
        "usage: ExportTritonModelKt <artifactDir> <repositoryDir> [modelName] [sequence|client] " +
            "[maxSequenceIdleMicros] [maxQueueDelayMicros]"
    }
    fun arg(i: Int): String? = args.getOrNull(i)?.takeIf { it.isNotBlank() }
    val artifact = Path.of(args[0]).toAbsolutePath()
    val repository = Path.of(args[1]).toAbsolutePath()
    val name = arg(2) ?: artifact.fileName.toString()
    val kv = when (val k = arg(3) ?: "sequence") {
        "sequence" -> TritonModelRepository.KvMode.SEQUENCE
        "client" -> TritonModelRepository.KvMode.CLIENT
        else -> throw IllegalArgumentException("kvMode must be 'sequence' or 'client', got '$k'")
    }
    val defaults = TritonModelRepository.SequenceOptions()
    val options = TritonModelRepository.SequenceOptions(
        maxSequenceIdleMicros = arg(4)?.toLong() ?: defaults.maxSequenceIdleMicros,
        maxQueueDelayMicros = arg(5)?.toLong() ?: defaults.maxQueueDelayMicros,
    )
    val model = TritonModelRepository.write(artifact, repository, name, kv, options)
    println("wrote Triton model '$name' (${kv.name.lowercase()} mode) to $model")
}
