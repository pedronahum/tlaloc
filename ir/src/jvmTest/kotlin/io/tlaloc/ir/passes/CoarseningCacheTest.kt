package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertSame

/**
 * §0.4.26 — tests for the Stage B.3 coarsening cache.
 *
 *  - [InMemoryCoarseningCache]: lifecycle within one JVM (put/get/size/clear).
 *  - [DiskCoarseningCache]: persistence across fresh instances (simulated reopen),
 *    directory layout (two-level hash prefix), CAS-version invalidation (same hash +
 *    different casVersion = cache miss), corrupt-file resilience.
 *  - [NoOpCoarseningCache]: always-miss semantics (never caches).
 */
class CoarseningCacheTest {

    private val f32s = DxirType(F32, emptyList())
    private val i32s = DxirType(I32, emptyList())

    private fun simplePrimal(): DxirFunction = DxirBuilder.function("square") {
        val x = param("x", f32s)
        val mul = op(OpKind.MUL, listOf(x, x), f32s)
        listOf(mul)
    }

    private fun otherPrimal(): DxirFunction = DxirBuilder.function("cube") {
        val x = param("x", f32s)
        val sq = op(OpKind.MUL, listOf(x, x), f32s)
        val cb = op(OpKind.MUL, listOf(sq, x), f32s)
        listOf(cb)
    }

    @Test
    fun inMemoryCachePutGetRoundTrip() {
        val cache = InMemoryCoarseningCache()
        val fn = simplePrimal()
        assertEquals(0, cache.size())
        assertNull(cache.get("missing"))

        val key = DxirCanonical.hash(fn)
        cache.put(key, fn)
        assertEquals(1, cache.size())

        val retrieved = cache.get(key)
        assertNotNull(retrieved)
        assertEquals(DxirCanonical.serialise(fn), DxirCanonical.serialise(retrieved))
    }

    @Test
    fun inMemoryGetOrComputeRunsComputeOnlyOnMiss() {
        val cache = InMemoryCoarseningCache()
        val fn = simplePrimal()
        val coarsened = otherPrimal() // stand-in for "PhiCalculus result"

        var computeCount = 0
        val r1 = cache.getOrCompute(fn) { computeCount++; coarsened }
        val r2 = cache.getOrCompute(fn) { computeCount++; coarsened }
        val r3 = cache.getOrCompute(fn) { computeCount++; coarsened }

        assertEquals(1, computeCount, "compute fired more than once despite cache hit")
        assertEquals(DxirCanonical.serialise(coarsened), DxirCanonical.serialise(r1))
        assertEquals(DxirCanonical.serialise(coarsened), DxirCanonical.serialise(r2))
        assertEquals(DxirCanonical.serialise(coarsened), DxirCanonical.serialise(r3))
    }

    @Test
    fun noOpCacheNeverHits() {
        val cache = NoOpCoarseningCache
        val fn = simplePrimal()
        val coarsened = otherPrimal()

        var computeCount = 0
        repeat(3) {
            cache.getOrCompute(fn) { computeCount++; coarsened }
        }
        assertEquals(3, computeCount, "NoOp cache must compute on every call")
        // Same object-ness confirmation: the NoOp singleton is a single instance.
        assertSame(NoOpCoarseningCache, NoOpCoarseningCache)
    }

    @Test
    fun diskCachePersistsAcrossReopen() {
        val dir = Files.createTempDirectory("tlaloc-cache-test")
        try {
            val fn = simplePrimal()
            val coarsened = otherPrimal()
            val casV = "test-v1"

            // Session 1: put.
            val c1 = DiskCoarseningCache(dir, casV)
            val key = DxirCanonical.hash(fn)
            c1.put(key, coarsened)

            // Session 2: a brand-new instance at the same dir + casV should see the entry.
            val c2 = DiskCoarseningCache(dir, casV)
            val retrieved = c2.get(key)
            assertNotNull(retrieved, "reopened disk cache must see prior put")
            assertEquals(
                DxirCanonical.serialise(coarsened),
                DxirCanonical.serialise(retrieved),
            )
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun diskCacheCasVersionBumpInvalidates() {
        val dir = Files.createTempDirectory("tlaloc-cache-test")
        try {
            val fn = simplePrimal()
            val coarsened = otherPrimal()
            val key = DxirCanonical.hash(fn)

            val cOld = DiskCoarseningCache(dir, "symja-3.0.0-tlaloc-0.4.25")
            cOld.put(key, coarsened)
            assertNotNull(cOld.get(key), "same-version read must hit")

            val cNew = DiskCoarseningCache(dir, "symja-3.0.1-tlaloc-0.4.26")
            assertNull(cNew.get(key), "CAS-version bump must invalidate")
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun diskCacheUsesTwoLevelHashPrefixDirectory() {
        val dir = Files.createTempDirectory("tlaloc-cache-test")
        try {
            val fn = simplePrimal()
            val casV = "v"
            val cache = DiskCoarseningCache(dir, casV)
            val key = DxirCanonical.hash(fn)
            cache.put(key, simplePrimal())

            val prefix = key.substring(0, 2)
            val prefixDir = dir.resolve(prefix)
            assertEquals(true, Files.isDirectory(prefixDir), "expected prefix dir '$prefix'")
            val entryFile = prefixDir.resolve("$key-$casV.dxir")
            assertEquals(true, Files.isRegularFile(entryFile), "expected entry file under prefix")
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun diskCacheCorruptFileIsHandledGracefully() {
        val dir = Files.createTempDirectory("tlaloc-cache-test")
        try {
            val fn = simplePrimal()
            val casV = "v"
            val cache = DiskCoarseningCache(dir, casV)
            val key = DxirCanonical.hash(fn)
            val prefix = key.substring(0, 2)
            val file: Path = dir.resolve(prefix).also { Files.createDirectories(it) }
                .resolve("$key-$casV.dxir")
            Files.writeString(file, "not-a-valid-canonical-form")

            val result = cache.get(key)
            assertNull(result, "corrupt cache entry must read as miss")
            assertEquals(false, Files.exists(file), "corrupt entry must be deleted on miss")
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun distinctFunctionsProduceDistinctCacheEntries() {
        val dir = Files.createTempDirectory("tlaloc-cache-test")
        try {
            val cache = DiskCoarseningCache(dir, "v")
            val a = simplePrimal()
            val b = otherPrimal()

            cache.put(DxirCanonical.hash(a), a)
            cache.put(DxirCanonical.hash(b), b)

            val getA = cache.get(DxirCanonical.hash(a))
            val getB = cache.get(DxirCanonical.hash(b))
            assertNotNull(getA); assertNotNull(getB)
            assertEquals(DxirCanonical.serialise(a), DxirCanonical.serialise(getA))
            assertEquals(DxirCanonical.serialise(b), DxirCanonical.serialise(getB))
        } finally {
            dir.toFile().deleteRecursively()
        }
    }
}
