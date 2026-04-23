package io.tlaloc.ir.passes

import io.tlaloc.ir.DxirFunction
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption

/**
 * §0.4.26 — cache interface for storing the result of an expensive coarsening pass
 * (typically [PhiCalculus.apply]) keyed by the input function's canonical SHA-256 hash
 * (from [DxirCanonical.hash]). Implementations vary in where they persist:
 *
 *  - [InMemoryCoarseningCache]: lives for one JVM invocation; mainly for tests.
 *  - [DiskCoarseningCache]: on-disk under `<baseDir>/<hashPrefix>/<hash>-<casVersion>.dxir`.
 *    Two-level hash-prefix directory (first 2 hex chars) keeps a single dir from ballooning.
 *  - [NoOpCoarseningCache]: the null object — every `get` misses, every `put` is a no-op.
 *    Useful as a default when caching is disabled.
 *
 * The CAS version is baked into the [put] key + re-checked on [get] so a toolchain bump
 * (Symja upgrade, rewrite-rule change, emitter tweak) invalidates stale entries without
 * an explicit purge. Pruning stale entries by timestamp (plan §5.4 — age > 30 days) is
 * deferred until cache growth becomes measurable.
 *
 * Concurrent-access safety is minimal: a single Gradle daemon may run compilations
 * sequentially, and disk writes use `move(REPLACE_EXISTING)` so a reader never observes
 * a partial file. True multi-process concurrency would want a file-lock layer; not
 * worth the complexity until a benchmark exposes contention.
 */
interface CoarseningCache {
    /** Return the cached coarsened function for [key], or null on miss. */
    fun get(key: String): DxirFunction?

    /** Store [value] under [key]. Subsequent [get] with the same [key] returns it. */
    fun put(key: String, value: DxirFunction)

    /**
     * Memoise [compute] against the canonical hash of [input]. Common entry point used
     * by [io.tlaloc.plugin.TlalocIrGenerationExtension] — callers don't need to thread
     * the hash computation themselves.
     */
    fun getOrCompute(input: DxirFunction, compute: () -> DxirFunction): DxirFunction {
        val key = DxirCanonical.hash(input)
        val cached = get(key)
        if (cached != null) return cached
        val result = compute()
        put(key, result)
        return result
    }
}

/** No-op: every [get] misses, every [put] drops. The default when caching is disabled. */
object NoOpCoarseningCache : CoarseningCache {
    override fun get(key: String): DxirFunction? = null
    override fun put(key: String, value: DxirFunction) {}
    override fun getOrCompute(input: DxirFunction, compute: () -> DxirFunction): DxirFunction =
        compute()
}

/** Transient in-memory cache; lives for one JVM. Thread-safe via `synchronized`. */
class InMemoryCoarseningCache : CoarseningCache {
    private val store = HashMap<String, String>()

    override fun get(key: String): DxirFunction? {
        val text = synchronized(store) { store[key] } ?: return null
        return DxirCanonical.deserialise(text)
    }

    override fun put(key: String, value: DxirFunction) {
        val text = DxirCanonical.serialise(value)
        synchronized(store) { store[key] = text }
    }

    /** Test-only: drop all entries. */
    fun clear() { synchronized(store) { store.clear() } }

    /** Test-only: observable entry count. */
    fun size(): Int = synchronized(store) { store.size }
}

/**
 * File-backed cache at `<baseDir>/<hashPrefix2>/<hash>-<casVersion>.dxir`. The CAS
 * version is part of the on-disk filename so a toolchain bump yields a different file
 * path — old entries stick around until pruning lands, but won't be read.
 */
class DiskCoarseningCache(
    private val baseDir: Path,
    private val casVersion: String,
) : CoarseningCache {

    init {
        Files.createDirectories(baseDir)
    }

    private fun fileFor(key: String): Path {
        require(key.length >= 2) { "cache key too short: '$key'" }
        val prefix = key.substring(0, 2)
        val dir = baseDir.resolve(prefix)
        Files.createDirectories(dir)
        return dir.resolve("$key-$casVersion.dxir")
    }

    override fun get(key: String): DxirFunction? {
        val file = fileFor(key)
        if (!Files.isRegularFile(file)) return null
        val text = Files.readString(file, Charsets.UTF_8)
        return try {
            DxirCanonical.deserialise(text)
        } catch (_: Throwable) {
            // Corrupt entry (interrupted write, format drift). Drop + miss.
            runCatching { Files.deleteIfExists(file) }
            null
        }
    }

    override fun put(key: String, value: DxirFunction) {
        val file = fileFor(key)
        val tmp = Files.createTempFile(file.parent, key, ".tmp")
        val text = DxirCanonical.serialise(value)
        Files.writeString(tmp, text, Charsets.UTF_8)
        Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
    }
}
