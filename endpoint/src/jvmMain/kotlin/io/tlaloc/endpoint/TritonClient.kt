package io.tlaloc.endpoint

import java.io.ByteArrayOutputStream
import java.io.IOException
import java.net.URLDecoder
import java.util.concurrent.TimeUnit

/** A request parameter for Triton: its value as one of InferParameter's fields. */
sealed interface TritonParameter {
    data class Bool(val value: Boolean) : TritonParameter
    data class Int64(val value: Long) : TritonParameter
    data class Text(val value: String) : TritonParameter
    data class Real(val value: Double) : TritonParameter
}

class TritonException(message: String, cause: Throwable? = null) : IOException(message, cause)

/** One request's responses: each step's tokens, then null after the final one. */
interface TokenStream : AutoCloseable {
    /** The next step's tokens, or null after the final response; [TritonException] for an error. */
    fun next(timeoutMillis: Long = Long.MAX_VALUE): IntArray?

    /** Cancels the request if it has not finished. */
    override fun close()
}

/** Where the endpoint's sequences live: a decoupled sequence-mode model. */
interface SequenceBackend {
    /** Appends [tokens] to [sequenceId] and streams the generation request [parameters] ask for. */
    fun generate(
        sequenceId: Long,
        tokens: IntArray,
        start: Boolean,
        end: Boolean = false,
        parameters: Map<String, TritonParameter> = emptyMap(),
    ): TokenStream

    /** Ends [sequenceId], freeing what the backend holds for it. */
    fun end(sequenceId: Long)
}

/**
 * A client for a decoupled sequence-mode Tlaloc model in Triton, over gRPC
 * (`inference.GRPCInferenceService/ModelStreamInfer`). Each [generate] opens its own stream, so
 * closing it cancels that request alone. Connections are shared and reopened when one fails.
 */
class TritonClient(private val host: String, private val port: Int, val model: String) : SequenceBackend, AutoCloseable {
    private val lock = Any()
    private var connection: Http2Connection? = null

    private fun connection(): Http2Connection = synchronized(lock) {
        connection?.takeIf { it.isOpen } ?: Http2Connection(host, port).also { connection = it }
    }

    /**
     * One request on [sequenceId]: [tokens] appended, then the backend's generation (a request
     * parameter `max_tokens` above 0) streamed back as [Generation.next] returns each step's
     * tokens. A request without tokens and with [end] ends the sequence.
     */
    override fun generate(
        sequenceId: Long,
        tokens: IntArray,
        start: Boolean,
        end: Boolean,
        parameters: Map<String, TritonParameter>,
    ): Generation {
        val request = ProtoWriter().apply {
            string(1, model)
            fun param(key: String, p: TritonParameter) = message(4) {
                string(1, key)
                message(2) {
                    when (p) {
                        is TritonParameter.Bool -> bool(1, p.value)
                        is TritonParameter.Int64 -> int64(2, p.value)
                        is TritonParameter.Text -> string(3, p.value)
                        is TritonParameter.Real -> double(4, p.value)
                    }
                }
            }
            param("sequence_id", TritonParameter.Int64(sequenceId))
            param("sequence_start", TritonParameter.Bool(start))
            param("sequence_end", TritonParameter.Bool(end))
            param("triton_enable_empty_final_response", TritonParameter.Bool(true))
            for ((k, v) in parameters) param(k, v)
            message(5) {
                string(1, "TOKENS")
                string(2, "INT32")
                packed(3, longArrayOf(1, tokens.size.toLong()))
            }
            message(6) { string(1, "NEXT_TOKENS") }
            val raw = ByteArray(4 * tokens.size)
            for ((i, t) in tokens.withIndex()) {
                raw[4 * i] = t.toByte()
                raw[4 * i + 1] = (t ushr 8).toByte()
                raw[4 * i + 2] = (t ushr 16).toByte()
                raw[4 * i + 3] = (t ushr 24).toByte()
            }
            bytes(7, raw)
        }.bytes()
        val c: Http2Connection
        val stream: Http2Connection.Stream
        try {
            c = connection()
            stream = c.open(listOf(
                ":method" to "POST",
                ":scheme" to "http",
                ":path" to "/inference.GRPCInferenceService/ModelStreamInfer",
                ":authority" to c.target,
                "content-type" to "application/grpc",
                "te" to "trailers",
                "user-agent" to "tlaloc-endpoint",
            ))
        } catch (e: IOException) {
            throw TritonException("Triton at $host:$port: ${e.message}", e)
        }
        val framed = ByteArray(5 + request.size)
        framed[1] = (request.size ushr 24).toByte()
        framed[2] = (request.size ushr 16).toByte()
        framed[3] = (request.size ushr 8).toByte()
        framed[4] = request.size.toByte()
        request.copyInto(framed, 5)
        try {
            // The request side stays open: closing it would end the gRPC stream before the
            // responses, and Triton reads a half-closed stream as finished.
            stream.sendData(framed, endStream = false)
        } catch (e: IOException) {
            stream.cancel()
            throw TritonException("sending to $model: ${e.message}", e)
        }
        return Generation(stream)
    }

    /** Ends [sequenceId], waiting for the backend's final response. */
    override fun end(sequenceId: Long) {
        try {
            generate(sequenceId, IntArray(0), start = false, end = true, parameters = emptyMap()).use { g -> while (g.next() != null) Unit }
        } catch (e: TritonException) {
            // A sequence Triton already ended wants START again: START and END free what the backend holds.
            if ("START flag" !in (e.message ?: "")) throw e
            generate(sequenceId, IntArray(0), start = true, end = true, parameters = emptyMap()).use { g -> while (g.next() != null) Unit }
        }
    }

    override fun close() {
        synchronized(lock) {
            connection?.close()
            connection = null
        }
    }

    /** The responses of one request, read in order; [close] cancels it if it has not finished. */
    class Generation internal constructor(private val stream: Http2Connection.Stream) : TokenStream {
        private val buffer = ByteArrayOutputStream()
        private var pending = ByteArray(0)
        private var finished = false
        private var headersSeen = false

        /**
         * The next step's tokens, or null after the final response. Throws [TritonException] for
         * an error from Triton or the backend, or after [timeoutMillis] without a response.
         */
        override fun next(timeoutMillis: Long): IntArray? {
            while (!finished) {
                message()?.let { m ->
                    val r = response(m)
                    if (r.final) finished = true
                    if (r.tokens != null) return r.tokens
                    if (finished) return null
                    continue
                }
                val e = stream.events.poll(timeoutMillis, TimeUnit.MILLISECONDS)
                    ?: throw TritonException("no response for $timeoutMillis ms")
                when (e) {
                    is StreamEvent.Headers -> {
                        val h = e.fields.toMap()
                        h[":status"]?.let { if (it != "200") fail("HTTP status $it") }
                        val status = h["grpc-status"]
                        if (status != null && status != "0") {
                            fail(URLDecoder.decode(h["grpc-message"] ?: "grpc-status $status", Charsets.UTF_8))
                        }
                        if (headersSeen && status == null && e.endStream) fail("the stream ended without grpc-status")
                        headersSeen = true
                        if (e.endStream) {
                            finished = true
                            stream.closed()
                            if (status == null) fail("the stream ended before the final response")
                            return null
                        }
                    }
                    is StreamEvent.Data -> {
                        buffer.write(e.bytes)
                        if (e.endStream) {
                            stream.closed()
                            if (!finished) fail("the stream ended before the final response")
                        }
                    }
                    is StreamEvent.Reset -> fail("the server reset the stream (code ${e.code})")
                    is StreamEvent.Failed -> fail(e.error.message ?: "connection failed", e.error)
                }
            }
            return null
        }

        private fun fail(message: String, cause: Throwable? = null): Nothing {
            finished = true
            stream.cancel()
            throw TritonException(message, cause)
        }

        /** The next complete gRPC message in what has arrived, or null. */
        private fun message(): ByteArray? {
            if (buffer.size() > 0) {
                pending += buffer.toByteArray()
                buffer.reset()
            }
            if (pending.size < 5) return null
            val n = Http2Connection.int32(pending, 1)
            if (pending[0].toInt() != 0) fail("compressed gRPC messages are not supported")
            if (pending.size < 5 + n) return null
            val m = pending.copyOfRange(5, 5 + n)
            pending = pending.copyOfRange(5 + n, pending.size)
            return m
        }

        private class Response(val tokens: IntArray?, val final: Boolean)

        private fun response(m: ByteArray): Response {
            var error: String? = null
            var infer: ByteArray? = null
            for (f in protoFields(m)) {
                when (f.number) {
                    1 -> error = f.string.takeIf { it.isNotEmpty() }
                    2 -> infer = f.bytes
                }
            }
            if (error != null) fail(error)
            var final = false
            var tokens: IntArray? = null
            var outputIndex = -1
            var nextIndex = -1
            val raw = ArrayList<ByteArray>()
            for (f in protoFields(infer ?: return Response(null, false))) {
                when (f.number) {
                    4 -> { // parameters: map<string, InferParameter>
                        var key = ""
                        var value: ByteArray? = null
                        for (e in protoFields(f.bytes!!)) if (e.number == 1) key = e.string else if (e.number == 2) value = e.bytes
                        if (key == "triton_final_response" && value != null) {
                            final = protoFields(value).any { it.number == 1 && it.varint != 0L }
                        }
                    }
                    5 -> {
                        outputIndex++
                        val name = protoFields(f.bytes!!).firstOrNull { it.number == 1 }?.string
                        if (name == "NEXT_TOKENS") nextIndex = outputIndex
                    }
                    6 -> raw += f.bytes!!
                }
            }
            if (nextIndex >= 0 && nextIndex < raw.size) {
                val b = raw[nextIndex]
                tokens = IntArray(b.size / 4) { i ->
                    (b[4 * i].toInt() and 0xff) or ((b[4 * i + 1].toInt() and 0xff) shl 8) or
                        ((b[4 * i + 2].toInt() and 0xff) shl 16) or ((b[4 * i + 3].toInt() and 0xff) shl 24)
                }
            }
            return Response(tokens, final)
        }

        /**
         * Cancels the request if its final response has not arrived; otherwise half-closes the
         * stream, after which Triton sends its status and ends it.
         */
        override fun close() {
            if (!finished) {
                finished = true
                stream.cancel()
            } else {
                try {
                    stream.sendData(ByteArray(0), endStream = true)
                } catch (_: IOException) {
                    // already ended by the server
                }
            }
        }
    }
}
