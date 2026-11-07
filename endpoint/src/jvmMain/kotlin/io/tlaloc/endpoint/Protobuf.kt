package io.tlaloc.endpoint

import java.io.ByteArrayOutputStream

/** Protocol Buffers wire format, the parts Triton's inference messages use. */
internal class ProtoWriter {
    private val out = ByteArrayOutputStream()

    fun bytes(): ByteArray = out.toByteArray()

    private fun varint(v: Long) {
        var x = v
        while (x and 0x7fL.inv() != 0L) {
            out.write(((x and 0x7f) or 0x80).toInt())
            x = x ushr 7
        }
        out.write(x.toInt())
    }

    private fun tag(field: Int, wire: Int) = varint(((field shl 3) or wire).toLong())

    fun int64(field: Int, v: Long) {
        tag(field, 0)
        varint(v)
    }

    fun bool(field: Int, v: Boolean) = int64(field, if (v) 1 else 0)

    fun double(field: Int, v: Double) {
        tag(field, 1)
        val bits = java.lang.Double.doubleToRawLongBits(v)
        for (i in 0 until 8) out.write((bits ushr (8 * i)).toInt() and 0xff)
    }

    fun bytes(field: Int, v: ByteArray) {
        tag(field, 2)
        varint(v.size.toLong())
        out.write(v)
    }

    fun string(field: Int, v: String) = bytes(field, v.toByteArray(Charsets.UTF_8))

    fun message(field: Int, build: ProtoWriter.() -> Unit) = bytes(field, ProtoWriter().apply(build).bytes())

    /** A packed repeated int64. */
    fun packed(field: Int, values: LongArray) {
        val inner = ProtoWriter()
        for (v in values) inner.varint(v)
        bytes(field, inner.bytes())
    }
}

/** One field as read: its number, wire type and value (a varint, a fixed 64-bit word or bytes). */
internal class ProtoField(val number: Int, val wire: Int, val varint: Long, val bytes: ByteArray?) {
    val string: String get() = String(bytes!!, Charsets.UTF_8)
    val double: Double get() = java.lang.Double.longBitsToDouble(varint)
}

internal fun protoFields(b: ByteArray): List<ProtoField> {
    val out = ArrayList<ProtoField>()
    var p = 0
    fun varint(): Long {
        var v = 0L
        var shift = 0
        while (true) {
            if (p >= b.size) throw IllegalArgumentException("protobuf varint runs past the message")
            val x = b[p++].toInt() and 0xff
            v = v or ((x and 0x7f).toLong() shl shift)
            if (x and 0x80 == 0) return v
            shift += 7
            if (shift > 63) throw IllegalArgumentException("protobuf varint too long")
        }
    }
    while (p < b.size) {
        val key = varint()
        val number = (key ushr 3).toInt()
        when (val wire = (key and 7).toInt()) {
            0 -> out += ProtoField(number, wire, varint(), null)
            1 -> {
                var v = 0L
                for (i in 0 until 8) v = v or ((b[p + i].toLong() and 0xff) shl (8 * i))
                p += 8
                out += ProtoField(number, wire, v, null)
            }
            2 -> {
                val n = varint().toInt()
                if (n < 0 || p + n > b.size) throw IllegalArgumentException("protobuf field $number runs past the message")
                out += ProtoField(number, wire, 0, b.copyOfRange(p, p + n))
                p += n
            }
            5 -> {
                var v = 0L
                for (i in 0 until 4) v = v or ((b[p + i].toLong() and 0xff) shl (8 * i))
                p += 4
                out += ProtoField(number, wire, v, null)
            }
            else -> throw IllegalArgumentException("protobuf wire type $wire is not supported")
        }
    }
    return out
}
