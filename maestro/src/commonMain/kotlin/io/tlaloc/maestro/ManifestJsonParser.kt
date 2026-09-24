package io.tlaloc.maestro

/**
 * Minimal recursive-descent JSON parser tailored to
 * the [ProgramManifest] schema. Does **not** aspire to be a general JSON
 * library — accepts only the exact shape produced by `ProgramManifest.toJson()`
 * and `TypeDescriptor.toJson()`.
 *
 * Hand-rolled vs. pulling in
 * `kotlinx.serialization` — the schema is small, stable, and entirely
 * internal to Tlaloc's manifest pipeline. A 120-line targeted parser is
 * cheaper than the dependency. Richer serialization (back-compat-aware
 * versioning, polymorphic decoding) would warrant revisiting that choice.
 */
internal class ManifestJsonParser(private val text: String) {
    private var pos = 0

    fun parse(): ProgramManifest {
        skipWs()
        expect('{')
        var name: String? = null
        var inputs: List<TypeDescriptor>? = null
        var outputs: List<TypeDescriptor>? = null
        var meshRequirement: String? = null
        var bodyHash: String? = null
        var shardingSpec: List<String>? = null
        var backendMatrix: List<BackendTarget>? = null
        while (true) {
            skipWs()
            if (peek() == '}') break
            val key = readString()
            skipWs()
            expect(':')
            skipWs()
            when (key) {
                "name" -> name = readString()
                "inputs" -> inputs = readTypeDescriptorArray()
                "outputs" -> outputs = readTypeDescriptorArray()
                "meshRequirement" -> meshRequirement = readString()
                "bodyHash" -> bodyHash = readString()
                "shardingSpec" -> shardingSpec = readStringArray()
                "backendMatrix" -> backendMatrix = readBackendTargetArray()
                else -> error("unknown manifest key: $key at pos $pos")
            }
            skipWs()
            if (peek() == ',') {
                pos++
            } else {
                break
            }
        }
        skipWs()
        expect('}')
        return ProgramManifest(
            name = requireNotNull(name) { "manifest missing 'name'" },
            inputs = requireNotNull(inputs) { "manifest missing 'inputs'" },
            outputs = requireNotNull(outputs) { "manifest missing 'outputs'" },
            meshRequirement = requireNotNull(meshRequirement) { "manifest missing 'meshRequirement'" },
            bodyHash = requireNotNull(bodyHash) { "manifest missing 'bodyHash'" },
            shardingSpec = shardingSpec ?: emptyList(),
            backendMatrix = backendMatrix ?: emptyList(),
        )
    }

    /**
     * Parse a single [BackendTarget] JSON object. Public to allow
     * standalone deserialization (e.g., for tooling that consumes a
     * single target row out of an external manifest file).
     */
    internal fun parseBackendTarget(): BackendTarget = readBackendTarget()

    private fun readBackendTarget(): BackendTarget {
        skipWs()
        expect('{')
        var vendor: String? = null
        var arch: String? = null
        var kernelName: String? = null
        var kvQuantDtype: String? = null
        var costMicroseconds: Double? = null
        while (true) {
            skipWs()
            if (peek() == '}') break
            val key = readString()
            skipWs()
            expect(':')
            skipWs()
            when (key) {
                "vendor" -> vendor = readString()
                "arch" -> arch = readString()
                "kernelName" -> kernelName = readNullableString()
                "kvQuantDtype" -> kvQuantDtype = readNullableString()
                "costMicroseconds" -> costMicroseconds = readNullableDouble()
                else -> error("unknown BackendTarget key: $key at pos $pos")
            }
            skipWs()
            if (peek() == ',') {
                pos++
            } else {
                break
            }
        }
        skipWs()
        expect('}')
        return BackendTarget(
            vendor = requireNotNull(vendor) { "BackendTarget missing 'vendor'" },
            arch = requireNotNull(arch) { "BackendTarget missing 'arch'" },
            kernelName = kernelName,
            kvQuantDtype = kvQuantDtype,
            costMicroseconds = costMicroseconds,
        )
    }

    private fun readBackendTargetArray(): List<BackendTarget> {
        skipWs()
        expect('[')
        skipWs()
        if (peek() == ']') {
            pos++
            return emptyList()
        }
        val out = mutableListOf<BackendTarget>()
        while (true) {
            out += readBackendTarget()
            skipWs()
            when (peek()) {
                ',' -> pos++
                ']' -> {
                    pos++
                    return out
                }
                else -> error("expected ',' or ']' at pos $pos")
            }
        }
    }

    private fun readNullableString(): String? {
        skipWs()
        return if (text.startsWith("null", pos)) {
            pos += 4
            null
        } else {
            readString()
        }
    }

    private fun readNullableDouble(): Double? {
        skipWs()
        if (text.startsWith("null", pos)) {
            pos += 4
            return null
        }
        val start = pos
        if (peek() == '-') pos++
        while (pos < text.length && (text[pos].isDigit() || text[pos] == '.' ||
                text[pos] == 'e' || text[pos] == 'E' || text[pos] == '+' || text[pos] == '-')) {
            pos++
        }
        return text.substring(start, pos).toDouble()
    }

    /**
     * Parse a top-level [TypeDescriptor] JSON object. Public to enable
     * standalone descriptor parsing for cross-pod handles in
     * `SerializedBufferHandle`.
     */
    internal fun parseTypeDescriptor(): TypeDescriptor = readTypeDescriptor()

    private fun readTypeDescriptor(): TypeDescriptor {
        skipWs()
        expect('{')
        var dtype: String? = null
        var dims: List<Int>? = null
        var axisNames: List<String?>? = null
        while (true) {
            skipWs()
            if (peek() == '}') break
            val key = readString()
            skipWs()
            expect(':')
            skipWs()
            when (key) {
                "dtype" -> dtype = readString()
                "dims" -> dims = readIntArray()
                "axisNames" -> axisNames = readNullableStringArray()
                else -> error("unknown TypeDescriptor key: $key")
            }
            skipWs()
            if (peek() == ',') {
                pos++
            } else {
                break
            }
        }
        skipWs()
        expect('}')
        return TypeDescriptor(
            dtype = requireNotNull(dtype),
            dims = requireNotNull(dims),
            axisNames = axisNames ?: emptyList(),
        )
    }

    private fun readTypeDescriptorArray(): List<TypeDescriptor> {
        skipWs()
        expect('[')
        val out = mutableListOf<TypeDescriptor>()
        skipWs()
        if (peek() == ']') {
            pos++
            return emptyList()
        }
        while (true) {
            out += readTypeDescriptor()
            skipWs()
            when (peek()) {
                ',' -> pos++
                ']' -> {
                    pos++
                    return out
                }
                else -> error("expected ',' or ']' at pos $pos")
            }
        }
    }

    private fun readStringArray(): List<String> {
        skipWs()
        expect('[')
        skipWs()
        if (peek() == ']') {
            pos++
            return emptyList()
        }
        val out = mutableListOf<String>()
        while (true) {
            skipWs()
            out += readString()
            skipWs()
            when (peek()) {
                ',' -> pos++
                ']' -> {
                    pos++
                    return out
                }
                else -> error("expected ',' or ']' at pos $pos")
            }
        }
    }

    private fun readNullableStringArray(): List<String?> {
        skipWs()
        expect('[')
        skipWs()
        if (peek() == ']') {
            pos++
            return emptyList()
        }
        val out = mutableListOf<String?>()
        while (true) {
            skipWs()
            if (text.startsWith("null", pos)) {
                out += null
                pos += 4
            } else {
                out += readString()
            }
            skipWs()
            when (peek()) {
                ',' -> pos++
                ']' -> {
                    pos++
                    return out
                }
                else -> error("expected ',' or ']' at pos $pos")
            }
        }
    }

    private fun readIntArray(): List<Int> {
        skipWs()
        expect('[')
        skipWs()
        if (peek() == ']') {
            pos++
            return emptyList()
        }
        val out = mutableListOf<Int>()
        while (true) {
            skipWs()
            val start = pos
            if (peek() == '-') pos++
            while (pos < text.length && text[pos].isDigit()) pos++
            out += text.substring(start, pos).toInt()
            skipWs()
            when (peek()) {
                ',' -> pos++
                ']' -> {
                    pos++
                    return out
                }
                else -> error("expected ',' or ']' at pos $pos")
            }
        }
    }

    private fun readString(): String {
        skipWs()
        expect('"')
        val sb = StringBuilder()
        while (pos < text.length) {
            val c = text[pos]
            if (c == '"') {
                pos++
                return sb.toString()
            }
            if (c == '\\' && pos + 1 < text.length) {
                when (val e = text[pos + 1]) {
                    '"' -> sb.append('"')
                    '\\' -> sb.append('\\')
                    'n' -> sb.append('\n')
                    'r' -> sb.append('\r')
                    't' -> sb.append('\t')
                    'u' -> {
                        val code = text.substring(pos + 2, pos + 6).toInt(16)
                        sb.append(code.toChar())
                        pos += 4
                    }
                    else -> sb.append(e)
                }
                pos += 2
            } else {
                sb.append(c)
                pos++
            }
        }
        error("unterminated string at pos $pos")
    }

    private fun skipWs() {
        while (pos < text.length && text[pos].isWhitespace()) pos++
    }

    private fun peek(): Char = if (pos < text.length) text[pos] else ' '

    private fun expect(c: Char) {
        if (pos >= text.length || text[pos] != c) error("expected '$c' at pos $pos, got '${peek()}'")
        pos++
    }
}
