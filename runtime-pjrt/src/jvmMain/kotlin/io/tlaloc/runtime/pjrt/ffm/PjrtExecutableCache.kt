package io.tlaloc.runtime.pjrt.ffm

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Compiled executables kept on disk, so a program is compiled once per machine rather than
 * once per process.
 *
 * An entry is the plugin's own serialization (`PJRT_Executable_Serialize`), loaded back with
 * `PJRT_Executable_DeserializeAndLoad`. It carries the kernels XLA chose when it compiled,
 * autotuning included, so a loaded executable runs what the first compile ran.
 *
 * The key is the SHA-256 of [Identity] (the plugin file, platform name and version, device
 * kind, `XLA_FLAGS`), the [CompileOptions] bytes and the program text. The Triton backend
 * computes the same key (`triton/backend/executable_cache.cc`), so the two share a directory.
 *
 * Within the JVM a lock per key, and across processes a file lock per key, make one caller
 * compile while the others wait and then load its entry. An entry that cannot be read or
 * loaded (truncated, written by another plugin build) is deleted and compiled again.
 *
 * File layout: the magic `TLXC0001`, the payload length (8 bytes, little endian), the
 * payload's SHA-256, then the payload.
 */
class PjrtExecutableCache(val dir: Path) {

    /** What, besides the program and its options, decides whether bytes can be loaded. */
    data class Identity(
        val pluginPath: String,
        val pluginSize: Long,
        val pluginMtimeSeconds: Long,
        val platformName: String,
        val platformVersion: String,
        val deviceKind: String,
        val xlaFlags: String,
    ) {
        companion object {
            fun of(plugin: Path, client: PjrtClient, device: PjrtDevice, env: (String) -> String? = System::getenv): Identity {
                val abs = plugin.toAbsolutePath().normalize()
                return Identity(
                    pluginPath = abs.toString(),
                    pluginSize = Files.size(abs),
                    pluginMtimeSeconds = Files.getLastModifiedTime(abs).toMillis() / 1000,
                    platformName = client.platformName(),
                    platformVersion = client.platformVersion,
                    deviceKind = client.deviceKind(device),
                    xlaFlags = env("XLA_FLAGS") ?: "",
                )
            }
        }
    }

    private val hits = AtomicLong()
    private val misses = AtomicLong()
    private val rejected = AtomicLong()
    private val keyLocks = ConcurrentHashMap<String, ReentrantLock>()

    /** Entries loaded from disk. */
    val hitCount: Long get() = hits.get()

    /** Programs compiled (and written) because no usable entry existed. */
    val missCount: Long get() = misses.get()

    /** Entries found but unreadable or refused by the plugin, then replaced. */
    val rejectedCount: Long get() = rejected.get()

    init {
        Files.createDirectories(dir)
    }

    /**
     * The executable for [program], loaded from disk when an entry exists, otherwise compiled
     * with [options] and written. The caller owns the result.
     */
    fun getOrCompile(
        client: PjrtClient,
        identity: Identity,
        program: String,
        options: CompileOptions = CompileOptions.DEFAULT,
    ): PjrtLoadedExecutable {
        if (!client.supportsSerialization) return client.compile(program, options)
        val key = key(identity, options.toProtoBytes(), program)
        val entry = dir.resolve("$key.pjrt")
        return keyLocks.computeIfAbsent(key) { ReentrantLock() }.withLock {
            FileChannel.open(dir.resolve("$key.lock"), StandardOpenOption.CREATE, StandardOpenOption.WRITE).use { ch ->
                ch.lock().use {
                    load(client, entry)?.let { hits.incrementAndGet(); return@withLock it }
                    val exe = client.compile(program, options)
                    misses.incrementAndGet()
                    runCatching { write(entry, client.serialize(exe)) }
                    exe
                }
            }
        }
    }

    private fun load(client: PjrtClient, entry: Path): PjrtLoadedExecutable? {
        if (!Files.exists(entry)) return null
        val payload = runCatching { readEntry(entry) }.getOrNull()
        val exe = payload?.let { runCatching { client.deserialize(it) }.getOrNull() }
        if (exe == null) {
            rejected.incrementAndGet()
            runCatching { Files.deleteIfExists(entry) }
        }
        return exe
    }

    private fun write(entry: Path, payload: ByteArray) {
        val tmp = Files.createTempFile(dir, entry.fileName.toString(), ".tmp")
        try {
            val header = ByteBuffer.allocate(MAGIC.size + 8 + 32).order(ByteOrder.LITTLE_ENDIAN)
            header.put(MAGIC).putLong(payload.size.toLong()).put(sha256(payload))
            Files.newOutputStream(tmp).use { out -> out.write(header.array()); out.write(payload) }
            Files.move(tmp, entry, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        } finally {
            Files.deleteIfExists(tmp)
        }
    }

    companion object {
        private val MAGIC = "TLXC0001".toByteArray(Charsets.US_ASCII)
        private const val KEY_VERSION = "tlaloc-pjrt-cache-v1"

        /** The cache at `TLALOC_PJRT_CACHE_DIR`, or null when it is unset or empty. */
        fun fromEnv(env: (String) -> String? = System::getenv): PjrtExecutableCache? =
            env("TLALOC_PJRT_CACHE_DIR")?.trim()?.takeIf { it.isNotEmpty() }?.let { PjrtExecutableCache(Path.of(it)) }

        /**
         * The entry name: SHA-256, as lowercase hex, over length-prefixed fields (8-byte little
         * endian length, then the bytes): the key version, the identity's seven fields in
         * declaration order (numbers in decimal), the options bytes, the program text.
         */
        fun key(identity: Identity, optionsBytes: ByteArray, program: String): String {
            val md = MessageDigest.getInstance("SHA-256")
            fun field(b: ByteArray) {
                md.update(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(b.size.toLong()).array())
                md.update(b)
            }
            fun field(s: String) = field(s.toByteArray(Charsets.UTF_8))
            field(KEY_VERSION)
            with(identity) {
                field(pluginPath); field(pluginSize.toString()); field(pluginMtimeSeconds.toString())
                field(platformName); field(platformVersion); field(deviceKind); field(xlaFlags)
            }
            field(optionsBytes)
            field(program)
            return md.digest().joinToString("") { "%02x".format(it) }
        }

        internal fun readEntry(entry: Path): ByteArray {
            val all = Files.readAllBytes(entry)
            val headerSize = MAGIC.size + 8 + 32
            require(all.size >= headerSize) { "cache entry $entry is shorter than its header" }
            val buf = ByteBuffer.wrap(all).order(ByteOrder.LITTLE_ENDIAN)
            val magic = ByteArray(MAGIC.size).also { buf.get(it) }
            require(magic.contentEquals(MAGIC)) { "cache entry $entry has no TLXC0001 magic" }
            val size = buf.getLong()
            val digest = ByteArray(32).also { buf.get(it) }
            require(size == (all.size - headerSize).toLong()) { "cache entry $entry is truncated" }
            val payload = all.copyOfRange(headerSize, all.size)
            require(sha256(payload).contentEquals(digest)) { "cache entry $entry fails its checksum" }
            return payload
        }

        private fun sha256(b: ByteArray): ByteArray = MessageDigest.getInstance("SHA-256").digest(b)
    }
}
