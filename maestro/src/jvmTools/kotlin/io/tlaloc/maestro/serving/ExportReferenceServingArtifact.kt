package io.tlaloc.maestro.serving

/**
 * Entry point of `./gradlew :maestro:exportServingArtifact -PoutDir=<dir>`.
 *
 * Lives in the `tools` compilation, which is not published: the `:maestro` jar
 * carries the exporter ([ReferenceDecodeGraph.exportTo]) but no `main`.
 * It runs at build time; after it returns, the directory is the deployment.
 */
fun main(args: Array<String>) {
    val dir = java.nio.file.Path.of(args.firstOrNull() ?: "build/serving-artifact")
    val manifest = ReferenceDecodeGraph.exportTo(dir)
    println("wrote ${manifest.entries.size} entries to ${dir.toAbsolutePath()}")
    for (e in manifest.entries) println("  ${e.entryId}  ${e.bodyPath}  ${e.cacheKey}")
}
