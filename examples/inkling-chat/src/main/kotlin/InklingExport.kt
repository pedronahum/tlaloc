import io.tlaloc.nn.InklingCheckpoint
import io.tlaloc.runtime.pjrt.serving.InklingServingExport
import java.nio.file.Path

/**
 * Writes Inkling-2-Mini as a serving artifact for the Triton backend (InklingServingExport):
 *
 *   inkling-export <snapshot dir> <artifact dir> [--contexts 2048,8192] [--batches 1,2,4]
 *                  [--prefill-chunks 16,256] [--slots 4]
 *
 * Then `./gradlew :maestro:exportTritonModel -PartifactDir=<artifact dir> -PoutDir=<repository>
 * -PmodelName=inkling -Pdecoupled=true` makes the Triton model repository (see the README).
 */
fun main(args: Array<String>) {
    fun opt(name: String) = args.indexOf(name).takeIf { it >= 0 }?.let { args.getOrNull(it + 1) }
    val pos = args.filterIndexed { i, a -> !a.startsWith("--") && (i == 0 || !args[i - 1].startsWith("--")) }
    require(pos.size == 2) { "usage: inkling-export <snapshot dir> <artifact dir> [--contexts 2048,8192] [--batches 1,2,4] [--prefill-chunks 16,256] [--slots 4]" }
    val t0 = System.nanoTime()
    InklingCheckpoint(Path.of(pos[0])).use { ck ->
        val m = InklingServingExport.export(
            Path.of(pos[1]), ck,
            contexts = (opt("--contexts") ?: "2048,8192").split(",").map { it.trim().toInt() },
            decodeBatches = (opt("--batches") ?: "1,2,4").split(",").map { it.trim().toInt() },
            prefillChunks = (opt("--prefill-chunks") ?: "16,256").split(",").map { it.trim().toInt() },
            numSlots = opt("--slots")?.toInt() ?: 4,
        ) { System.err.println(it) }
        System.err.println("inkling-export: ${m.entries.size} entries, ${m.weights.table.size} weights in %.0f s".format((System.nanoTime() - t0) / 1e9))
    }
}
