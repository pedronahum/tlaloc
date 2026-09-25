package io.tlaloc.maestro.serving

import java.nio.file.Path

/**
 * Entry point of
 * `./gradlew :maestro:exportTritonModel -PartifactDir=… -PoutDir=<repository> [-PmodelName=…]`.
 *
 * Writes an existing serving artifact into a Triton model repository through
 * [TritonModelRepository.write]. Arguments, positionally: the artifact
 * directory, the repository directory, and the model name (default: the
 * artifact directory's name).
 */
fun main(args: Array<String>) {
    require(args.size >= 2) {
        "usage: ExportTritonModelKt <artifactDir> <repositoryDir> [modelName]"
    }
    val artifact = Path.of(args[0]).toAbsolutePath()
    val repository = Path.of(args[1]).toAbsolutePath()
    val name = args.getOrNull(2)?.takeIf { it.isNotBlank() } ?: artifact.fileName.toString()
    val model = TritonModelRepository.write(artifact, repository, name)
    println("wrote Triton model '$name' to $model")
}
