package io.tlaloc.endpoint

import io.tlaloc.tokenizer.ChatOptions
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/** A conversation a live sequence holds: its messages with the reply and how they were rendered. */
internal class ChatState(val messages: List<RequestMessage>, val options: ChatOptions, val reply: String, val endText: String?)

/** A backend sequence and the tokens its KV holds. */
internal class Live(val id: Long) {
    var kv: IntArray = IntArray(0)
    var pending: Int? = null // the last generated token, not yet run
    var chat: ChatState? = null
    var busy = true
    var used = System.nanoTime()
}

/**
 * The live sequences, matched to requests by token prefix. Triton holds at most [limit]
 * sequences and queues a new one until another ends, so a new sequence ends the least recently
 * used idle one when [limit] are live; a request waits while every sequence is busy. The ids are
 * kept in [stateFile] so that the next endpoint ends the sequences this one left open.
 */
internal class Sequences(
    private val triton: SequenceBackend,
    private val limit: Int,
    private val stateFile: Path,
    private val maxWaiting: Int = Int.MAX_VALUE,
    private val queueTimeoutMillis: Long = Long.MAX_VALUE / 1_000_000,
) {
    /** Too many requests wait for a sequence, or this one waited too long. */
    class Busy(message: String) : Exception(message)

    private val lock = ReentrantLock()
    private val changed = lock.newCondition()
    private val live = ArrayList<Live>()
    private var nextId = (System.currentTimeMillis() / 1000 % 100_000) * 100 + 1
    private var waiting = 0

    init {
        try {
            if (Files.isRegularFile(stateFile)) {
                val ids = Files.readString(stateFile).trim().removeSurrounding("[", "]").split(',').mapNotNull { it.trim().toLongOrNull() }
                for (id in ids) end(Live(id))
            }
        } catch (e: Exception) {
            System.err.println("ending the sequences in $stateFile: ${e.message}")
        }
        save()
    }

    private fun save() {
        try {
            Files.createDirectories(stateFile.toAbsolutePath().parent)
            Files.writeString(stateFile, live.joinToString(",", "[", "]") { it.id.toString() })
        } catch (e: Exception) {
            System.err.println("cannot write $stateFile: ${e.message}")
        }
    }

    class Acquired(val live: Live, val send: IntArray, val start: Boolean)

    /**
     * The idle sequence with the longest KV prefix of [ids], else one that [extend] continues
     * (the tokens to send, or null), else a new sequence.
     */
    fun acquire(ids: IntArray, extend: ((Live) -> IntArray?)?): Acquired {
        var evict: Live? = null
        val acquired = lock.withLock { take(ids, extend) { evict = it } }
        // Not waited for: Triton queues the new sequence's START until the END frees a slot.
        evict?.let { e -> Thread.ofVirtual().start { end(e) } }
        return acquired
    }

    private fun take(ids: IntArray, extend: ((Live) -> IntArray?)?, evicted: (Live) -> Unit): Acquired {
        var left = java.util.concurrent.TimeUnit.MILLISECONDS.toNanos(queueTimeoutMillis)
        while (true) {
            var best: Live? = null
            for (s in live) {
                val n = s.kv.size
                if (!s.busy && n in 1 until ids.size && (best == null || n > best.kv.size) && startsWith(ids, s.kv)) best = s
            }
            if (best != null) {
                best.busy = true
                return Acquired(best, ids.copyOfRange(best.kv.size, ids.size), false)
            }
            if (extend != null) {
                for (s in live.filter { !it.busy && it.chat != null }.sortedByDescending { it.kv.size }) {
                    val send = extend(s) ?: continue
                    s.busy = true
                    return Acquired(s, send, false)
                }
            }
            val idle = live.filter { !it.busy }.minByOrNull { it.used }
            if (live.size < limit || idle != null) {
                if (live.size >= limit && idle != null) {
                    live.remove(idle)
                    evicted(idle)
                }
                val s = Live(nextId++)
                live += s
                save()
                return Acquired(s, ids, true)
            }
            if (waiting >= maxWaiting) throw Busy("too many requests are waiting")
            if (left <= 0) throw Busy("no sequence became free in $queueTimeoutMillis ms")
            waiting++
            try {
                left = changed.awaitNanos(left)
            } finally {
                waiting--
            }
        }
    }

    /**
     * Ends up to [count] idle sequences other than [keep], the least recently used first, for a
     * sequence the backend refused KV pages; the number ended (0 when none is idle).
     */
    fun evictIdle(keep: Live, count: Int = 1): Int {
        val victims = lock.withLock {
            live.filter { !it.busy && it !== keep }.sortedBy { it.used }.take(count).also {
                if (it.isNotEmpty()) {
                    live.removeAll(it.toSet())
                    save()
                    changed.signalAll()
                }
            }
        }
        for (v in victims) end(v)
        return victims.size
    }

    /** The idle sequences. */
    val idle: Int get() = lock.withLock { live.count { !it.busy } }

    fun release(s: Live, kv: IntArray, pending: Int?, chat: ChatState?) = lock.withLock {
        s.kv = kv
        s.pending = pending
        s.chat = chat
        s.busy = false
        s.used = System.nanoTime()
        changed.signalAll()
    }

    /** Forgets [s] and ends it in the backend (its KV no longer matches what is recorded). */
    fun drop(s: Live) {
        lock.withLock {
            live.remove(s)
            save()
            changed.signalAll()
        }
        end(s)
    }

    fun end(s: Live) {
        try {
            triton.end(s.id)
        } catch (e: Exception) {
            System.err.println("ending sequence ${s.id}: ${e.message?.take(200)}")
        }
    }

    /** Ends every live sequence (at shutdown). */
    fun endAll() {
        val all = lock.withLock { live.toList().also { live.clear(); save() } }
        for (s in all) end(s)
    }

    val size: Int get() = lock.withLock { live.size }

    private fun startsWith(a: IntArray, prefix: IntArray): Boolean {
        if (prefix.size > a.size) return false
        for (i in prefix.indices) if (a[i] != prefix[i]) return false
        return true
    }
}
