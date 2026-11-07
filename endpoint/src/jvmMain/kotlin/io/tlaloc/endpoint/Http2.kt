package io.tlaloc.endpoint

import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.IOException
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

class Http2Exception(message: String, cause: Throwable? = null) : IOException(message, cause)

/** What arrives on a stream. */
internal sealed interface StreamEvent {
    class Headers(val fields: List<Pair<String, String>>, val endStream: Boolean) : StreamEvent
    class Data(val bytes: ByteArray, val endStream: Boolean) : StreamEvent
    class Reset(val code: Int) : StreamEvent
    class Failed(val error: IOException) : StreamEvent
}

/**
 * A client HTTP/2 connection over cleartext TCP with prior knowledge (h2c), as gRPC servers
 * accept it. One reader thread reads frames and hands each stream's headers and data to its
 * queue; writes take a lock. Flow control both ways: what this side sends waits for the
 * server's windows, and this side opens its receive windows to 1 GiB and tops them up as
 * data is read.
 */
internal class Http2Connection(host: String, port: Int, connectTimeoutMillis: Int = 10_000) : AutoCloseable {
    private val socket = Socket()
    private val input: DataInputStream
    private val output: BufferedOutputStream
    private val writeLock = ReentrantLock()
    private val state = ReentrantLock()
    private val windowOpened = state.newCondition()
    private val streams = HashMap<Int, Stream>()
    private val decoder = HpackDecoder()
    private val authority = "$host:$port"
    private var nextStreamId = 1
    private var sendWindow = 65_535L // the connection's
    private var initialStreamWindow = 65_535L
    private var maxFrameSize = 16_384
    private var maxConcurrentStreams = Int.MAX_VALUE
    private var receivedSinceUpdate = 0L
    @Volatile private var failure: IOException? = null

    val isOpen: Boolean get() = failure == null && !socket.isClosed

    inner class Stream internal constructor(val id: Int) {
        val events = LinkedBlockingQueue<StreamEvent>()
        internal var sendWindow = initialStreamWindow
        internal var receivedSinceUpdate = 0L
        internal var localClosed = false

        /** Sends [bytes] as DATA frames, waiting for flow-control windows. */
        fun sendData(bytes: ByteArray, endStream: Boolean) = this@Http2Connection.sendData(this, bytes, endStream)

        /** Ends the stream with RST_STREAM (CANCEL by default). */
        fun cancel(code: Int = CANCEL) {
            val known = state.withLock { (streams.remove(id) != null).also { windowOpened.signalAll() } }
            if (known) runCatching { writeFrame(TYPE_RST_STREAM, 0, id, int32(code)) }
        }

        /** Forgets the stream once its last event has been read. */
        fun closed() {
            state.withLock {
                streams.remove(id)
                windowOpened.signalAll()
            }
        }
    }

    init {
        socket.tcpNoDelay = true
        socket.connect(InetSocketAddress(host, port), connectTimeoutMillis)
        input = DataInputStream(BufferedInputStream(socket.getInputStream(), 1 shl 16))
        output = BufferedOutputStream(socket.getOutputStream(), 1 shl 16)
        writeLock.withLock {
            output.write(PREFACE)
            // ENABLE_PUSH 0, INITIAL_WINDOW_SIZE 1 GiB
            val settings = ByteArrayOutputStream()
            settings.write(setting(2, 0))
            settings.write(setting(4, RECEIVE_WINDOW))
            frame(TYPE_SETTINGS, 0, 0, settings.toByteArray())
            frame(TYPE_WINDOW_UPDATE, 0, 0, int32(RECEIVE_WINDOW - 65_535))
            output.flush()
        }
        Thread.ofPlatform().daemon().name("h2-reader-$authority").start { readLoop() }
    }

    /** Opens a stream with [headers] (pseudo-headers first); its events arrive on [Stream.events]. */
    fun open(headers: List<Pair<String, String>>, endStream: Boolean = false): Stream {
        val block = HpackEncoder.encode(headers)
        val s: Stream
        state.withLock {
            failure?.let { throw Http2Exception("connection to $authority failed", it) }
            while (streams.size >= maxConcurrentStreams) {
                windowOpened.await()
                failure?.let { throw Http2Exception("connection to $authority failed", it) }
            }
            if (nextStreamId > 0x7fff_ffff - 2) throw Http2Exception("stream ids exhausted")
            s = Stream(nextStreamId)
            nextStreamId += 2
            streams[s.id] = s
            // Under the state lock so stream ids reach the wire in order.
            writeLock.withLock {
                var flags = if (endStream) FLAG_END_STREAM else 0
                if (block.size <= maxFrameSize) {
                    frame(TYPE_HEADERS, flags or FLAG_END_HEADERS, s.id, block)
                } else {
                    frame(TYPE_HEADERS, flags, s.id, block.copyOfRange(0, maxFrameSize))
                    var p = maxFrameSize
                    while (p < block.size) {
                        val e = minOf(block.size, p + maxFrameSize)
                        frame(TYPE_CONTINUATION, if (e == block.size) FLAG_END_HEADERS else 0, s.id, block.copyOfRange(p, e))
                        p = e
                    }
                }
                output.flush()
            }
        }
        if (endStream) s.localClosed = true
        return s
    }

    private fun sendData(s: Stream, bytes: ByteArray, endStream: Boolean) {
        var p = 0
        do {
            val n: Int
            state.withLock {
                while (true) {
                    failure?.let { throw Http2Exception("connection to $authority failed", it) }
                    if (!streams.containsKey(s.id)) throw Http2Exception("stream ${s.id} is closed")
                    if (p == bytes.size || (sendWindow > 0 && s.sendWindow > 0)) break
                    windowOpened.await()
                }
                n = minOf(bytes.size - p, maxFrameSize, sendWindow.coerceAtMost(Int.MAX_VALUE.toLong()).toInt(), s.sendWindow.coerceAtMost(Int.MAX_VALUE.toLong()).toInt())
                sendWindow -= n
                s.sendWindow -= n
            }
            val last = p + n == bytes.size
            writeFrame(TYPE_DATA, if (last && endStream) FLAG_END_STREAM else 0, s.id, bytes.copyOfRange(p, p + n))
            p += n
        } while (p < bytes.size)
        if (endStream) s.localClosed = true
    }

    private fun writeFrame(type: Int, flags: Int, streamId: Int, payload: ByteArray) = writeLock.withLock {
        frame(type, flags, streamId, payload)
        output.flush()
    }

    private fun frame(type: Int, flags: Int, streamId: Int, payload: ByteArray) {
        val n = payload.size
        output.write(n ushr 16)
        output.write(n ushr 8)
        output.write(n)
        output.write(type)
        output.write(flags)
        output.write(int32(streamId))
        output.write(payload)
    }

    private fun readLoop() {
        var headerStream = 0
        var headerEnd = false
        val headerBlock = ByteArrayOutputStream()
        try {
            while (true) {
                val len = (input.readUnsignedByte() shl 16) or (input.readUnsignedByte() shl 8) or input.readUnsignedByte()
                val type = input.readUnsignedByte()
                val flags = input.readUnsignedByte()
                val streamId = input.readInt() and 0x7fff_ffff
                val payload = ByteArray(len)
                input.readFully(payload)
                if (headerStream != 0 && type != TYPE_CONTINUATION) throw Http2Exception("expected CONTINUATION")
                when (type) {
                    TYPE_DATA -> {
                        var body = payload
                        if (flags and FLAG_PADDED != 0) body = unpad(payload)
                        val end = flags and FLAG_END_STREAM != 0
                        val s = state.withLock { streams[streamId] }
                        receiveCredit(s, len)
                        s?.events?.put(StreamEvent.Data(body, end))
                        if (end) s?.closed()
                    }
                    TYPE_HEADERS -> {
                        var p = 0
                        var end = len
                        if (flags and FLAG_PADDED != 0) {
                            p = 1
                            end -= payload[0].toInt() and 0xff
                        }
                        if (flags and FLAG_PRIORITY != 0) p += 5
                        headerBlock.reset()
                        headerBlock.write(payload, p, end - p)
                        headerStream = streamId
                        headerEnd = flags and FLAG_END_STREAM != 0
                        if (flags and FLAG_END_HEADERS != 0) {
                            deliverHeaders(headerStream, headerBlock.toByteArray(), headerEnd)
                            headerStream = 0
                        }
                    }
                    TYPE_CONTINUATION -> {
                        if (streamId != headerStream) throw Http2Exception("CONTINUATION on stream $streamId, not $headerStream")
                        headerBlock.write(payload)
                        if (flags and FLAG_END_HEADERS != 0) {
                            deliverHeaders(headerStream, headerBlock.toByteArray(), headerEnd)
                            headerStream = 0
                        }
                    }
                    TYPE_RST_STREAM -> {
                        val code = int32(payload, 0)
                        val s = state.withLock { streams.remove(streamId).also { windowOpened.signalAll() } }
                        s?.events?.put(StreamEvent.Reset(code))
                    }
                    TYPE_SETTINGS -> if (flags and FLAG_ACK == 0) {
                        state.withLock {
                            for (k in 0 until len / 6) {
                                val id = ((payload[6 * k].toInt() and 0xff) shl 8) or (payload[6 * k + 1].toInt() and 0xff)
                                val v = int32(payload, 6 * k + 2).toLong() and 0xffff_ffffL
                                when (id) {
                                    3 -> maxConcurrentStreams = v.coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
                                    4 -> {
                                        val delta = v - initialStreamWindow
                                        initialStreamWindow = v
                                        for (s in streams.values) s.sendWindow += delta
                                    }
                                    5 -> maxFrameSize = v.toInt()
                                }
                            }
                            windowOpened.signalAll()
                        }
                        writeFrame(TYPE_SETTINGS, FLAG_ACK, 0, ByteArray(0))
                    }
                    TYPE_PING -> if (flags and FLAG_ACK == 0) writeFrame(TYPE_PING, FLAG_ACK, 0, payload)
                    TYPE_GOAWAY -> {
                        val last = int32(payload, 0) and 0x7fff_ffff
                        val code = int32(payload, 4)
                        val debug = String(payload, 8, len - 8, Charsets.UTF_8)
                        val e = Http2Exception("server sent GOAWAY (code $code, last stream $last) $debug".trim())
                        // Streams up to `last` may still finish; later ones never will.
                        val dropped = state.withLock {
                            failure = failure ?: e
                            streams.values.filter { it.id > last }.also { d -> d.forEach { streams.remove(it.id) } }
                        }
                        dropped.forEach { it.events.put(StreamEvent.Failed(e)) }
                    }
                    TYPE_WINDOW_UPDATE -> {
                        val inc = (int32(payload, 0) and 0x7fff_ffff).toLong()
                        state.withLock {
                            if (streamId == 0) sendWindow += inc else streams[streamId]?.let { it.sendWindow += inc }
                            windowOpened.signalAll()
                        }
                    }
                    TYPE_PUSH_PROMISE -> throw Http2Exception("PUSH_PROMISE although push is disabled")
                    else -> {} // PRIORITY and unknown types are ignored
                }
            }
        } catch (e: IOException) {
            fail(if (socket.isClosed && failure == null) Http2Exception("connection closed") else e)
        } catch (e: RuntimeException) {
            fail(Http2Exception("HTTP/2 protocol error: ${e.message}", e))
        }
    }

    private fun deliverHeaders(streamId: Int, block: ByteArray, endStream: Boolean) {
        val fields = decoder.decode(block) // always decoded: it updates the dynamic table
        val s = state.withLock { streams[streamId] }
        s?.events?.put(StreamEvent.Headers(fields, endStream))
        if (endStream) s?.closed()
    }

    private fun receiveCredit(s: Stream?, n: Int) {
        if (n == 0) return
        var conn = 0L
        var stream = 0L
        state.withLock {
            receivedSinceUpdate += n
            if (receivedSinceUpdate >= RECEIVE_WINDOW / 2) {
                conn = receivedSinceUpdate
                receivedSinceUpdate = 0
            }
            if (s != null) {
                s.receivedSinceUpdate += n
                if (s.receivedSinceUpdate >= RECEIVE_WINDOW / 2) {
                    stream = s.receivedSinceUpdate
                    s.receivedSinceUpdate = 0
                }
            }
        }
        if (conn > 0) writeFrame(TYPE_WINDOW_UPDATE, 0, 0, int32(conn.toInt()))
        if (stream > 0 && s != null) writeFrame(TYPE_WINDOW_UPDATE, 0, s.id, int32(stream.toInt()))
    }

    private fun fail(e: IOException) {
        val open = state.withLock {
            failure = failure ?: e
            windowOpened.signalAll()
            streams.values.toList().also { streams.clear() }
        }
        open.forEach { it.events.put(StreamEvent.Failed(e)) }
        runCatching { socket.close() }
    }

    override fun close() {
        runCatching { writeFrame(TYPE_GOAWAY, 0, 0, int32(0) + int32(0)) }
        fail(Http2Exception("connection closed"))
    }

    val target: String get() = authority

    companion object {
        private val PREFACE = "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n".toByteArray(Charsets.US_ASCII)
        private const val RECEIVE_WINDOW = 1 shl 30
        const val TYPE_DATA = 0
        const val TYPE_HEADERS = 1
        const val TYPE_RST_STREAM = 3
        const val TYPE_SETTINGS = 4
        const val TYPE_PUSH_PROMISE = 5
        const val TYPE_PING = 6
        const val TYPE_GOAWAY = 7
        const val TYPE_WINDOW_UPDATE = 8
        const val TYPE_CONTINUATION = 9
        const val FLAG_END_STREAM = 0x1
        const val FLAG_ACK = 0x1
        const val FLAG_END_HEADERS = 0x4
        const val FLAG_PADDED = 0x8
        const val FLAG_PRIORITY = 0x20
        const val CANCEL = 0x8

        private fun setting(id: Int, value: Int) = byteArrayOf((id ushr 8).toByte(), id.toByte()) + int32(value)

        fun int32(v: Int) = byteArrayOf((v ushr 24).toByte(), (v ushr 16).toByte(), (v ushr 8).toByte(), v.toByte())

        fun int32(b: ByteArray, off: Int) = ((b[off].toInt() and 0xff) shl 24) or ((b[off + 1].toInt() and 0xff) shl 16) or
            ((b[off + 2].toInt() and 0xff) shl 8) or (b[off + 3].toInt() and 0xff)

        private fun unpad(payload: ByteArray): ByteArray {
            val pad = payload[0].toInt() and 0xff
            if (pad >= payload.size) throw Http2Exception("padding longer than the frame")
            return payload.copyOfRange(1, payload.size - pad)
        }
    }
}
