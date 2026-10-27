package io.tlaloc.runtime.pjrt

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.bf16BitsToFloat
import io.tlaloc.core.floatToBf16Bits
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.runtime.pjrt.ffm.CompileOptions
import io.tlaloc.runtime.pjrt.ffm.PjrtExecutableCache
import org.junit.jupiter.api.Assumptions.assumeFalse
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class PjrtExecutableCacheTest {

    private val identity = PjrtExecutableCache.Identity(
        pluginPath = "/opt/pjrt/plugin.so", pluginSize = 1234, pluginMtimeSeconds = 1_700_000_000,
        platformName = "cuda", platformVersion = "cuda 13000", deviceKind = "NVIDIA GB10", xlaFlags = "",
    )

    @Test
    fun keyIsPinnedSoTheTritonBackendCanComputeTheSameOne() {
        // triton/backend/test/executable_cache_test.cc pins the same hex for the same inputs.
        val key = PjrtExecutableCache.key(identity, CompileOptions.DEFAULT.toProtoBytes(), "module {}")
        assertEquals(KEY_GOLDEN, key)
    }

    @Test
    fun everyIdentityFieldOptionsAndProgramChangeTheKey() {
        val base = PjrtExecutableCache.key(identity, CompileOptions.DEFAULT.toProtoBytes(), "module {}")
        val variants = listOf(
            identity.copy(pluginPath = "/other.so"), identity.copy(pluginSize = 1235),
            identity.copy(pluginMtimeSeconds = 1), identity.copy(platformName = "tpu"),
            identity.copy(platformVersion = "x"), identity.copy(deviceKind = "TPU v6 lite"),
            identity.copy(xlaFlags = "--xla_gpu_autotune_level=0"),
        ).map { PjrtExecutableCache.key(it, CompileOptions.DEFAULT.toProtoBytes(), "module {}") }
        variants.forEach { assertNotEquals(base, it) }
        assertNotEquals(base, PjrtExecutableCache.key(identity, CompileOptions.spmd(listOf(0, 1)).toProtoBytes(), "module {}"))
        assertNotEquals(base, PjrtExecutableCache.key(identity, CompileOptions.DEFAULT.toProtoBytes(), "module { }"))
    }

    @Test
    fun aTruncatedOrAlteredEntryIsRefused(@TempDir dir: Path) {
        val f = dir.resolve("x.pjrt")
        Files.write(f, "TLXC0001".toByteArray())
        assertFailsWith<IllegalArgumentException> { PjrtExecutableCache.readEntry(f) }
        Files.write(f, ByteArray(80))
        assertFailsWith<IllegalArgumentException> { PjrtExecutableCache.readEntry(f) }
    }

    private fun affine() = DxirBuilder.function("cache_affine") {
        val x = param("x", DxirType(F32, listOf(4)))
        val y = op(OpKind.MUL, listOf(x, x), DxirType(F32, listOf(4)))
        listOf(op(OpKind.ADD, listOf(y, x), DxirType(F32, listOf(4))))
    }

    /** `x * x + c·x`: one program per [c], so each compiles to its own entry. */
    private fun scaled(c: Float) = DxirBuilder.function("cache_scaled") {
        val x = param("x", DxirType(F32, listOf(4)))
        val t = DxirType(F32, listOf(4))
        val k = op(OpKind.BROADCAST, listOf(const(c, DxirType(F32, emptyList()))), t, attrs = mapOf("broadcast_dimensions" to emptyList<Int>()))
        listOf(op(OpKind.ADD, listOf(op(OpKind.MUL, listOf(x, x), t), op(OpKind.MUL, listOf(k, x), t)), t))
    }

    private fun entries(dir: Path) = Files.list(dir).use { s -> s.filter { it.toString().endsWith(".pjrt") }.toList() }

    @Test
    fun aBoundedCacheEvictsTheLeastRecentlyUsedEntries(@TempDir dir: Path) {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        assumeFalse(TestBackend.isTpu || TestBackend.isMetal || TestBackend.isCerebras, "CUDA lane")
        val x = floatArrayOf(1f, -2f, 0.5f, 3f)
        // Measure one entry, then allow two and a half.
        PjrtSession(target = PjrtTarget.Cuda, diskCache = PjrtExecutableCache(dir.resolve("probe"))).use { it.runOn(scaled(0f), listOf(x)) }
        val size = Files.size(entries(dir.resolve("probe")).single())
        val cache = PjrtExecutableCache(dir.resolve("c"), maxBytes = size * 5 / 2)
        PjrtSession(target = PjrtTarget.Cuda, diskCache = cache).use { s ->
            s.runOn(scaled(1f), listOf(x))
            Thread.sleep(20)
            s.runOn(scaled(2f), listOf(x))
            Thread.sleep(20)
            s.runOn(scaled(3f), listOf(x))
        }
        assertEquals(1, cache.evictedCount)
        assertEquals(2, entries(dir.resolve("c")).size)
        // The first program was evicted: a new session compiles it again; the third loads.
        val again = PjrtExecutableCache(dir.resolve("c"), maxBytes = size * 5 / 2)
        PjrtSession(target = PjrtTarget.Cuda, diskCache = again).use { s ->
            s.runOn(scaled(3f), listOf(x))
            s.runOn(scaled(1f), listOf(x))
        }
        assertEquals(1, again.hitCount)
        assertEquals(1, again.missCount)
    }

    @Test
    fun aSharedDirectoryServesAnotherMachinesCompile(@TempDir dir: Path) {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        assumeFalse(TestBackend.isTpu || TestBackend.isMetal || TestBackend.isCerebras, "CUDA lane")
        val x = floatArrayOf(1f, -2f, 0.5f, 3f)
        val shared = dir.resolve("shared")
        // "Machine" one compiles and publishes.
        val one = PjrtExecutableCache(dir.resolve("one"), shared = shared, publish = true)
        val a = PjrtSession(target = PjrtTarget.Cuda, diskCache = one).use { it.runOn(affine(), listOf(x)) }
        assertEquals(1, one.missCount)
        assertEquals(1, entries(shared).size)
        // "Machine" two has an empty local cache: it loads the shared entry and keeps a copy.
        val two = PjrtExecutableCache(dir.resolve("two"), shared = shared)
        val b = PjrtSession(target = PjrtTarget.Cuda, diskCache = two).use { it.runOn(affine(), listOf(x)) }
        assertEquals(0, two.missCount)
        assertEquals(1, two.sharedHitCount)
        assertEquals(1, entries(dir.resolve("two")).size)
        assertContentEquals(a[0], b[0])
        val env = mapOf("TLALOC_PJRT_CACHE_DIR" to "$dir/env", "TLALOC_PJRT_CACHE_MAX_BYTES" to "1000000", "TLALOC_PJRT_CACHE_SHARED_DIR" to shared.toString(), "TLALOC_PJRT_CACHE_PUBLISH" to "1")
        val fromEnv = PjrtExecutableCache.fromEnv { env[it] }!!
        assertEquals(1_000_000L, fromEnv.maxBytes)
        assertTrue(fromEnv.publish && fromEnv.shared == shared)
    }

    @Test
    fun aSecondSessionLoadsTheFirstOnesExecutable(@TempDir dir: Path) {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        assumeFalse(TestBackend.isTpu || TestBackend.isMetal || TestBackend.isCerebras, "CUDA lane")
        val x = floatArrayOf(1f, -2f, 0.5f, 3f)

        val first = PjrtExecutableCache(dir)
        val a = PjrtSession(target = PjrtTarget.Cuda, diskCache = first).use { it.runOn(affine(), listOf(x)) }
        assertEquals(1, first.missCount)
        assertEquals(0, first.hitCount)
        assertEquals(1, Files.list(dir).use { s -> s.filter { it.toString().endsWith(".pjrt") }.count() })

        val second = PjrtExecutableCache(dir)
        val b = PjrtSession(target = PjrtTarget.Cuda, diskCache = second).use { it.runOn(affine(), listOf(x)) }
        assertEquals(0, second.missCount)
        assertEquals(1, second.hitCount)
        assertContentEquals(a[0], b[0])
        assertContentEquals(floatArrayOf(2f, 2f, 0.75f, 12f), b[0])
    }

    /**
     * Sessions that start together on one cache (the ranks of one host, or a server's compile
     * threads): one compiles, the others wait on the entry's lock and load what it wrote.
     */
    @Test
    fun sessionsStartingTogetherCompileOnce(@TempDir dir: Path) {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        assumeFalse(TestBackend.isTpu || TestBackend.isMetal || TestBackend.isCerebras, "CUDA lane")
        val x = floatArrayOf(1f, -2f, 0.5f, 3f)
        val cache = PjrtExecutableCache(dir)
        val start = java.util.concurrent.CountDownLatch(1)
        val results = java.util.Collections.synchronizedList(mutableListOf<FloatArray>())
        val threads = List(3) {
            Thread {
                PjrtSession(target = PjrtTarget.Cuda, diskCache = cache).use { s ->
                    start.await()
                    results += s.runOn(affine(), listOf(x))[0]
                }
            }
        }
        threads.forEach { it.start() }
        start.countDown()
        threads.forEach { it.join() }
        assertEquals(3, results.size)
        assertEquals(1, cache.missCount, "one compile")
        assertEquals(2, cache.hitCount, "the others load it")
        for (r in results) assertContentEquals(floatArrayOf(2f, 2f, 0.75f, 12f), r)
    }

    @Test
    fun anEntryThePluginRefusesIsReplaced(@TempDir dir: Path) {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        assumeFalse(TestBackend.isTpu || TestBackend.isMetal || TestBackend.isCerebras, "CUDA lane")
        val x = floatArrayOf(1f, 2f, 3f, 4f)
        PjrtSession(target = PjrtTarget.Cuda, diskCache = PjrtExecutableCache(dir)).use { it.runOn(affine(), listOf(x)) }
        val entry = Files.list(dir).use { s -> s.filter { it.toString().endsWith(".pjrt") }.findFirst().get() }
        // A well-formed entry whose payload is not an executable.
        val junk = "not an executable".toByteArray()
        val header = java.nio.ByteBuffer.allocate(48).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            .put("TLXC0001".toByteArray()).putLong(junk.size.toLong())
            .put(java.security.MessageDigest.getInstance("SHA-256").digest(junk))
        Files.write(entry, header.array() + junk)

        val cache = PjrtExecutableCache(dir)
        val out = PjrtSession(target = PjrtTarget.Cuda, diskCache = cache).use { it.runOn(affine(), listOf(x)) }
        assertEquals(1, cache.rejectedCount)
        assertEquals(1, cache.missCount)
        assertContentEquals(floatArrayOf(2f, 6f, 12f, 20f), out[0])
        assertTrue(PjrtExecutableCache.readEntry(entry).size > 1000, "the entry was rewritten with a real executable")
    }

    /**
     * An override reaches XLA. XLA-CUDA folds an f32→bf16→f32 convert pair (the bf16 smoke test
     * measured it); with `xla_allow_excess_precision = false` the narrowing must happen.
     */
    @Test
    fun excessPrecisionOffKeepsTheNarrowing() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        assumeFalse(TestBackend.isTpu || TestBackend.isMetal || TestBackend.isCerebras, "CUDA lane")
        val v = DxirType(F32, listOf(2))
        val fn = DxirBuilder.function("convert_pair") {
            val x = param("x", v)
            val b = op(OpKind.CAST, listOf(x), DxirType(BF16, listOf(2)))
            listOf(op(OpKind.CAST, listOf(b), v))
        }
        val x = floatArrayOf(1.005f, -3.14159f)
        val snapped = FloatArray(2) { bf16BitsToFloat(floatToBf16Bits(x[it])) }

        val folded = PjrtSession(target = PjrtTarget.Cuda, diskCache = null).use { it.runOn(fn, listOf(x)) }
        assertContentEquals(x, folded[0], "default: XLA folds the convert pair")

        val strict = CompileOptions.DEFAULT
            .withOverride("xla_allow_excess_precision", CompileOptions.OptionValue.Flag(false))
        val kept = PjrtSession(target = PjrtTarget.Cuda, compileOptions = strict, diskCache = null)
            .use { it.runOn(fn, listOf(x)) }
        assertContentEquals(snapped, kept[0], "xla_allow_excess_precision=false: the bf16 narrowing happens")
    }

    /** XLA parses the multi-device form (here over the one device) and an optimization level. */
    @Test
    fun spmdShardyAndAnOptimizationLevelCompileOnOneDevice() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        assumeFalse(TestBackend.isTpu || TestBackend.isMetal || TestBackend.isCerebras, "CUDA lane")
        val x = floatArrayOf(1f, 2f, 3f, 4f)
        for (opts in listOf(
            CompileOptions.spmd(listOf(0L)),
            CompileOptions(optimizationLevel = CompileOptions.EffortLevel.O1),
        )) {
            val out = PjrtSession(target = PjrtTarget.Cuda, compileOptions = opts, diskCache = null)
                .use { it.runOn(affine(), listOf(x)) }
            assertContentEquals(floatArrayOf(2f, 6f, 12f, 20f), out[0], "$opts")
        }
    }

    private companion object {
        const val KEY_GOLDEN = "136d3eaa56d6eb4660560b0b28713d2a40864162121e469bb842471d448e3dc7"
    }
}
