package io.tlaloc.runtime.pjrt.serving

import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.parseJson
import io.tlaloc.maestro.serving.ServingManifest
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.PjrtTarget
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit

/**
 * `:runtime-pjrt:runtimeParity` — which runtimes accept or refuse each serving artifact,
 * **observed rather than remembered**.
 *
 *     ./gradlew :runtime-pjrt:runtimeParity --args="<artifactDir> [<artifactDir> ...]"
 *
 * The November audit's DOC-7 records that the README's "three runtimes load it without a JVM"
 * is false: the framework-free Python runtime and the vLLM plugin refuse a v3 artifact by its
 * schema version, and `ServingModel` refuses windowed and quantized KV. A hand-written table of
 * that rots; this prints what the runtimes actually say, with their own words.
 *
 * Three lanes are exercised: [ServingModel] on CUDA, [ServingModel] on the XLA CPU plugin (the
 * lane that needs no GPU), and the framework-free Python runtime through its own
 * `ServingArtifact` constructor, whose schema gate fires before any device work. A lane that
 * cannot run here is reported UNAVAILABLE with the reason — never as a refusal it did not make.
 * The Triton backend needs a running server and the vLLM plugin needs vllm; both are out of this
 * tool's reach and it says so rather than guessing.
 *
 * A v2 artifact is genuinely LOADED, weights and all, so pass small fixtures for a quick pass;
 * the refusals for v3 and later happen before any device work and cost nothing.
 */

private fun attempt(block: () -> Unit): String = try {
    block()
    "ACCEPTED"
} catch (e: Throwable) {
    "refused: ${e.message?.replace('\n', ' ')?.take(200) ?: e::class.simpleName}"
}

private fun jvmLane(dir: Path, target: PjrtTarget): String = attempt {
    ServingModel.load(dir, target).use { }
}

private val PYTHON_DRIVER = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
try:
    import tlaloc_serve
except Exception as e:
    print("UNAVAILABLE: cannot import tlaloc_serve: %s" % str(e)[:160]); sys.exit(0)
root = Path(sys.argv[1])
try:
    m = json.loads((root / "tlaloc-serving.json").read_text())
    a = tlaloc_serve.ServingArtifact(root, m, platform="cuda")
    print("ACCEPTED")
except Exception as e:
    print("refused: %s" % str(e).replace("\n", " ")[:200])
""".trimIndent()

private fun pythonLane(harness: Path, dir: Path): String {
    if (!Files.isRegularFile(harness.resolve("tlaloc_serve.py"))) {
        return "UNAVAILABLE: no tlaloc_serve.py under $harness"
    }
    return try {
        val p = ProcessBuilder("python3", "-c", PYTHON_DRIVER, dir.toString(), harness.toString())
            .redirectErrorStream(true).start()
        val out = p.inputStream.bufferedReader().readText().trim()
        if (!p.waitFor(120, TimeUnit.SECONDS)) {
            p.destroyForcibly()
            return "UNAVAILABLE: the Python runtime did not answer in 120 s"
        }
        out.ifEmpty { "UNAVAILABLE: the Python runtime printed nothing (exit ${p.exitValue()})" }
    } catch (e: Exception) {
        "UNAVAILABLE: ${e::class.simpleName}: ${e.message?.take(120)}"
    }
}

fun main(args: Array<String>) {
    require(args.isNotEmpty()) {
        "usage: RuntimeParityKt <artifactDir> [<artifactDir> ...] " +
            "(the Python harness is found at ../harness/python, or \$TLALOC_HARNESS_PYTHON_DIR)"
    }
    val harness = Path.of(
        System.getenv("TLALOC_HARNESS_PYTHON_DIR") ?: "../harness/python",
    ).toAbsolutePath().normalize()
    val cudaHere = PjrtBinaries.available && PjrtBinaries.cudaAvailable
    val cpuHere = PjrtBinaries.cpuPluginPath != null

    println("runtime parity — ${args.size} artifact(s), ${java.time.LocalDate.now()}")
    println("lanes: ServingModel/CUDA ${if (cudaHere) "available" else "UNAVAILABLE (no CUDA plugin or device)"}, " +
        "ServingModel/CPU ${if (cpuHere) "available" else "UNAVAILABLE (no XLA CPU plugin)"}, " +
        "framework-free Python ${if (Files.isRegularFile(harness.resolve("tlaloc_serve.py"))) "available" else "UNAVAILABLE"}")
    println("not exercised here: the Triton backend (needs a running server) and the vLLM plugin (needs vllm)")

    for (a in args) {
        val dir = Path.of(a).toAbsolutePath().normalize()
        val file = dir.resolve(ServingManifest.FILE_NAME)
        if (!Files.isRegularFile(file)) {
            println("\n== $dir\n   not a serving artifact: no ${ServingManifest.FILE_NAME}")
            continue
        }
        val schema = ((parseJson(Files.readString(file)) as JsonObject)["schemaVersion"] as? JsonString)?.value
        val label = if (dir.fileName.toString() == "artifact") "${dir.parent?.fileName}/${dir.fileName}" else dir.fileName.toString()
        println("\n== $label  ($schema)")
        println("   ServingModel/CUDA : ${if (cudaHere) jvmLane(dir, PjrtTarget.Cuda) else "UNAVAILABLE: no CUDA plugin or device"}")
        println("   ServingModel/CPU  : ${if (cpuHere) jvmLane(dir, PjrtTarget.LlvmCpu) else "UNAVAILABLE: no XLA CPU plugin"}")
        println("   Python ctypes     : ${pythonLane(harness, dir)}")
    }
}
