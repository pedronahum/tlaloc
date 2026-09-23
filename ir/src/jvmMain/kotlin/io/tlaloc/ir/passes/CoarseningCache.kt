package io.tlaloc.ir.passes

import io.tlaloc.ir.DxirFunction
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.time.Duration

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
 * an explicit purge. Stale entries are pruned at [DiskCoarseningCache] instantiation
 * (§0.4.109): files whose CAS-version suffix doesn't match the current one OR whose
 * mtime is older than 30 days are deleted before the cache becomes live.
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
 * path. At instantiation, [pruneStaleEntries] sweeps the cache directory and deletes
 * (a) any `.dxir` file whose embedded CAS-version doesn't match [casVersion] (the
 * toolchain has moved on; the entry can no longer be read) and (b) any file whose
 * last-modified time predates [maxAge] (default 30 days, per plan §5.4).
 *
 * Sweeping at instantiation rather than on every read keeps the hot path cost-free:
 * compilations that hit a warm cache pay only the read; pruning amortises across the
 * whole compiler invocation. Pruning failures (locked file, permission denied) are
 * tolerated silently — pruning is best-effort and never blocks the cache from working.
 */
class DiskCoarseningCache(
    private val baseDir: Path,
    private val casVersion: String,
    private val maxAge: Duration = DEFAULT_MAX_AGE,
) : CoarseningCache {

    init {
        Files.createDirectories(baseDir)
        pruneStaleEntries()
    }

    private fun fileFor(key: String): Path {
        require(key.length >= 2) { "cache key too short: '$key'" }
        val prefix = key.substring(0, 2)
        val dir = baseDir.resolve(prefix)
        Files.createDirectories(dir)
        return dir.resolve("$key-$casVersion.dxir")
    }

    /**
     * §0.4.109 — sweep the cache directory at startup, deleting any `.dxir` entry that:
     *  - has a CAS-version suffix not matching [casVersion] (a toolchain bump made the
     *    entry unreadable), OR
     *  - has an mtime older than [maxAge] (cold entry; reclaim space).
     *
     * Pruning is best-effort: any IO failure on a single file is swallowed so a locked
     * or permission-denied file doesn't poison the whole cache. The walk doesn't recurse
     * past the two-level prefix layout — the cache only writes one level deep, so any
     * deeper nesting is foreign and left alone.
     */
    private fun pruneStaleEntries() {
        if (!Files.isDirectory(baseDir)) return
        val cutoffMillis = System.currentTimeMillis() - maxAge.toMillis()
        val prefixDirs: List<Path> = Files.list(baseDir).use { stream ->
            stream.filter { Files.isDirectory(it) }.toList()
        }
        for (prefixDir in prefixDirs) {
            val files: List<Path> = try {
                Files.list(prefixDir).use { it.toList() }
            } catch (_: Exception) {
                continue
            }
            for (file in files) {
                if (!Files.isRegularFile(file)) continue
                val name = file.fileName.toString()
                if (!name.endsWith(DXIR_EXT)) continue
                val stale = isCasVersionMismatch(name) || isOlderThanCutoff(file, cutoffMillis)
                if (stale) {
                    runCatching { Files.deleteIfExists(file) }
                }
            }
        }
    }

    private fun isCasVersionMismatch(filename: String): Boolean {
        val withoutExt = filename.removeSuffix(DXIR_EXT)
        val dashIdx = withoutExt.indexOf('-')
        if (dashIdx < 0) return true // unrecognised layout — treat as stale
        val fileVersion = withoutExt.substring(dashIdx + 1)
        return fileVersion != casVersion
    }

    private fun isOlderThanCutoff(file: Path, cutoffMillis: Long): Boolean {
        val mtime = try {
            Files.getLastModifiedTime(file).toMillis()
        } catch (_: Exception) {
            return false
        }
        return mtime < cutoffMillis
    }

    override fun get(key: String): DxirFunction? {
        val file = fileFor(key)
        if (!Files.isRegularFile(file)) return null
        val text = Files.readString(file, Charsets.UTF_8)
        return try {
            DxirCanonical.deserialise(text)
        } catch (_: Exception) {
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

    companion object {
        /** §0.4.109 — default cache-entry lifetime. Per plan §5.4 ("age > 30 days"). */
        val DEFAULT_MAX_AGE: Duration = Duration.ofDays(30)

        private const val DXIR_EXT: String = ".dxir"
    }
}
