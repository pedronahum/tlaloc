package io.tlaloc.maestro.serving

import io.tlaloc.core.io.JsonException
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.name
import kotlin.io.path.readText
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * The shared `tlaloc-bounded-v1` fixtures (harness/bounded-manifest-conformance): every
 * valid manifest reads, and every invalid one is refused. The Python reader
 * (`tlaloc_bounded_test.py`) and the Triton backend's (`bounded_manifest_test.cc`) are
 * tested against the same files.
 */
class BoundedManifestConformanceTest {
    private val fixtures: Path =
        Path.of("..", "harness", "bounded-manifest-conformance").toAbsolutePath().normalize()

    private fun files(kind: String): List<Path> {
        val all = Files.list(fixtures.resolve(kind)).use { s -> s.filter { it.name.endsWith(".json") }.sorted().toList() }
        assertTrue(all.isNotEmpty(), "no $kind fixtures in $fixtures")
        return all
    }

    @Test
    fun `every valid fixture reads`() {
        for (f in files("valid")) {
            try {
                BoundedManifest.fromJson(f.readText())
            } catch (e: Exception) {
                fail("${f.name} was refused: ${e.message}")
            }
        }
    }

    @Test
    fun `every invalid fixture is refused`() {
        val accepted = files("invalid").filter { f ->
            try {
                BoundedManifest.fromJson(f.readText())
                true
            } catch (e: JsonException) {
                false
            } catch (e: IllegalArgumentException) {
                false
            }
        }
        assertTrue(accepted.isEmpty(), "accepted: ${accepted.map { it.name }}")
    }
}
