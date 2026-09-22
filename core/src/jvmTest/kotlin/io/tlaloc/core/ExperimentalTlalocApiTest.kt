package io.tlaloc.core

import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.505 (Tier 4, item 4) — the opt-in marker, pinned in the only place it can be:
 * the class files.
 *
 * **Why this test reads bytes instead of calling `getAnnotation`.** An opt-in marker
 * must be `AnnotationRetention.BINARY` — `SOURCE` would make the requirement
 * invisible to anyone compiling against the published jar, which is every consumer.
 * Kotlin's `BINARY` maps to Java's `RetentionPolicy.CLASS`, and `Class.getAnnotations()`
 * returns **only** `RUNTIME`-retained annotations. So the correct retention is exactly
 * the one reflection cannot see. Writing the test the obvious way produced three
 * green-looking assertions over an empty array; this version asserts what a
 * consumer's *compiler* reads, which is the constant pool.
 *
 * What this file does NOT establish is the compiler refusing an un-opted-in
 * consumer. That needs a second compilation and lives in `:compiler-plugin`'s
 * `ExperimentalApiOptInTest`, which runs a real `K2JVMCompiler` with no `-opt-in`
 * argument. Both halves are here on purpose: a marker at the wrong level still
 * compiles, and a marker nobody applied still passes a byte-level test of itself.
 */
class ExperimentalTlalocApiTest {

    @Test
    fun `the marker declares a RequiresOptIn requirement at ERROR level`() {
        val pool = constantPoolOf(ExperimentalTlalocApi::class.java)
        assertTrue(
            "Lkotlin/RequiresOptIn;" in pool,
            "ExperimentalTlalocApi must carry @RequiresOptIn, or it is an ordinary annotation " +
                "that marks nothing and refuses nobody",
        )
        assertTrue(
            "Lkotlin/RequiresOptIn\$Level;" in pool && "ERROR" in pool,
            "the requirement must be ERROR level. A WARNING-level marker on a provisional " +
                "API is advice a build log swallows, and §0.4.499 spent a whole tier moving " +
                "Tlaloc's diagnostics off that setting",
        )
        assertTrue(
            "WARNING" !in pool,
            "nothing in this annotation should mention WARNING; if it does, the level was " +
                "changed and this test is reading the wrong thing",
        )
    }

    @Test
    fun `the marker's message names where a reader can check the claim`() {
        val pool = constantPoolOf(ExperimentalTlalocApi::class.java)
        assertTrue(
            "docs/CAPABILITIES.md" in pool,
            "the refusal must point at the file that GRADES the capability, so 'provisional' " +
                "is checkable rather than an adjective",
        )
        assertTrue(
            "shape" in pool,
            "the message must say what may change — shape, not merely signature. That is the " +
                "whole difference from the blanket alpha promise in COMPATIBILITY.md",
        )
    }

    @Test
    fun `the marker is BINARY retained, so a consumer's compiler can see it`() {
        val pool = constantPoolOf(ExperimentalTlalocApi::class.java)
        assertTrue(
            "Ljava/lang/annotation/Retention;" in pool && "CLASS" in pool,
            "Kotlin's AnnotationRetention.BINARY compiles to Java's RetentionPolicy.CLASS. " +
                "SOURCE retention would put the requirement in our sources and nowhere a " +
                "consumer's compiler looks",
        )
    }

    @Test
    fun `the four-worlds taxonomy carries the marker`() {
        listOf(
            KernelScope::class.java,
            OrchestrationScope::class.java,
            ProgramScope::class.java,
            ClusterScope::class.java,
            Tlaloc::class.java,
            BufferHandle::class.java,
            HandleRef::class.java,
        ).forEach { cls ->
            assertTrue(
                MARKER_DESCRIPTOR in constantPoolOf(cls),
                "${cls.simpleName} is part of the four-worlds taxonomy, whose own KDoc scopes " +
                    "it to v1 and names context parameters as where a multi-scope op would send " +
                    "it — it must carry @ExperimentalTlalocApi",
            )
        }
    }

    @Test
    fun `the certified surface does NOT carry the marker`() {
        // The counter-assertion, and the reason the marker is worth anything. An
        // annotation applied to everything teaches a reader to add `-opt-in=` once and
        // stop reading it. Every class here is a ✅ row in docs/CAPABILITIES.md with a
        // named gate behind it.
        listOf(
            DTensor::class.java,
            Shape::class.java,
            DType::class.java,
            Tensors::class.java,
            SparseTensor::class.java,
            MeshSpec::class.java,
        ).forEach { cls ->
            assertTrue(
                MARKER_DESCRIPTOR !in constantPoolOf(cls),
                "${cls.simpleName} is certified with a named gate; marking it experimental " +
                    "would make the marker noise",
            )
        }
    }

    private companion object {
        private const val MARKER_DESCRIPTOR = "Lio/tlaloc/core/ExperimentalTlalocApi;"

        /**
         * The class file's bytes as Latin-1 text. Every string a class file carries —
         * annotation type descriptors, enum constant names, annotation string values —
         * is a modified-UTF8 constant-pool entry, so a substring search over the raw
         * bytes is a sound (if blunt) way to ask "is this in the class file".
         */
        private fun constantPoolOf(cls: Class<*>): String {
            val path = cls.name.replace('.', '/') + ".class"
            val stream = cls.classLoader.getResourceAsStream(path)
                ?: error("could not read $path from the classpath; the test cannot check anything")
            return stream.use { String(it.readBytes(), Charsets.ISO_8859_1) }
        }
    }
}
