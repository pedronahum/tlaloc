package io.tlaloc.stablehlo

/**
 * [s] as an MLIR string literal, quoted, escaped the way MLIR prints string
 * attributes: printable ASCII other than `"` and `\` verbatim, `\` as `\\`,
 * and every other UTF-8 byte (including `"`) as `\XX` in upper-case hex. A
 * JSON `backend_config` therefore prints as `"{\22custom_call_config\22: ...}"`,
 * the spelling JAX's exported modules carry.
 */
internal fun mlirStringLiteral(s: String): String {
    val bytes = s.encodeToByteArray()
    val out = StringBuilder(bytes.size + 2)
    out.append('"')
    for (b in bytes) {
        val c = b.toInt() and 0xFF
        when {
            c == '\\'.code -> out.append("\\\\")
            c == '"'.code || c < 0x20 || c > 0x7E -> {
                out.append('\\')
                out.append(HEX[c shr 4])
                out.append(HEX[c and 0xF])
            }
            else -> out.append(c.toChar())
        }
    }
    out.append('"')
    return out.toString()
}

private const val HEX = "0123456789ABCDEF"
