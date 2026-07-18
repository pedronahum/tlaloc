package io.tlaloc.kptx

/**
 * KPTX v2.2 (§0.4.339) — opcode-agnostic parser for canonical-format
 * PTX (plan task 10). Inverse of [emitPtx]: for any text in the
 * canonical style, `parsePtx(text).emitPtx() == text` **byte-for-byte**
 * — pinned by the round-trip corpus (the five real v1 kernels,
 * §0.4.328–337, copied verbatim).
 *
 * Opcode-agnostic by construction: an instruction line is
 * `[@[!]%guard ] opcode [op1, op2, ...];[    // comment]` and the
 * opcode is stored as its uninterpreted dotted string — the parser
 * never consults an opcode table, so unknown/future instructions parse
 * fine. Per-opcode validation is the ISA spec's job (task 11).
 *
 * **Strict, not lenient.** The parser accepts exactly the canonical
 * format (the emitter's output grammar) and throws [PtxParseException]
 * with a 1-based line number otherwise. Leniency would silently break
 * the byte-identity contract: an input the parser "helpfully" accepts
 * but the emitter respells is a corpus bug that strictness surfaces
 * immediately.
 *
 * Operand surface matches the IR (v1 kernels): registers (incl.
 * special `%ctaid.x` forms), immediates (kept as exact spellings),
 * `[base]` / `[base+offset]` memory operands, bare symbols. Vector
 * operands (`{a, b}`) are out of scope until task 14 needs them.
 */
class PtxParseException(val line: Int, message: String) :
    RuntimeException("PTX parse error at line $line: $message")

fun parsePtx(text: String): PtxModule {
    return PtxParser(text.split("\n")).parse()
}

private class PtxParser(private val lines: List<String>) {
    private var i = 0

    private fun fail(message: String): Nothing = throw PtxParseException(i + 1, message)

    private fun current(): String =
        if (i < lines.size) lines[i] else fail("unexpected end of input")

    private fun advance(): String = current().also { i++ }

    private fun expectPrefix(prefix: String): String {
        val line = current()
        if (!line.startsWith(prefix)) fail("expected `$prefix…`, got `$line`")
        i++
        return line.removePrefix(prefix)
    }

    fun parse(): PtxModule {
        val version = expectPrefix(".version ")
        val target = expectPrefix(".target ")
        val addressSize = expectPrefix(".address_size ").toIntOrNull()
            ?: fail("`.address_size` value is not an integer")

        val kernels = ArrayList<PtxKernel>()
        while (true) {
            // Canonical form: exactly one blank line before each kernel;
            // the file ends after the last kernel's `}` (split leaves one
            // trailing empty string for the final newline).
            if (i >= lines.size) fail("expected blank line before kernel or end of file")
            if (current().isEmpty() && i == lines.size - 1) { i++; break } // trailing newline
            if (current().isNotEmpty()) fail("expected blank line before kernel, got `${current()}`")
            i++
            kernels.add(parseKernel())
            if (i == lines.size) break // no trailing newline
        }
        if (kernels.isEmpty()) fail("module contains no kernels")
        return PtxModule(kernels, version, target, addressSize)
    }

    private fun parseKernel(): PtxKernel {
        val header = advance()
        if (!header.startsWith(".visible .entry ") || !header.endsWith("(")) {
            i--; fail("expected `.visible .entry <name>(`, got `$header`")
        }
        val name = header.removePrefix(".visible .entry ").removeSuffix("(")

        val params = ArrayList<PtxParam>()
        while (true) {
            val line = current()
            if (line == ")") { i++; break }
            val body = line.removePrefix("    .param ")
            if (body == line) fail("expected `    .param .<type> <name>[,]` or `)`, got `$line`")
            val trimmed = body.removeSuffix(",")
            val parts = trimmed.split(" ")
            if (parts.size != 2 || !parts[0].startsWith(".")) fail("malformed param `$line`")
            params.add(PtxParam(parts[0], parts[1]))
            i++
        }
        if (advance() != "{") { i--; fail("expected `{`") }

        val body = ArrayList<PtxStmt>()
        while (true) {
            val line = advance()
            when {
                line == "}" -> return PtxKernel(name, params, body)
                line.isEmpty() -> body.add(PtxBlank)
                line.startsWith("    // ") -> body.add(PtxComment(line.removePrefix("    // ")))
                line.startsWith("    .reg ") -> body.add(parseRegDecl(line))
                line.startsWith("    .shared ") -> body.add(parseSharedDecl(line))
                line.startsWith("    ") -> body.add(parseInst(line.removePrefix("    ")))
                line.endsWith(":") && !line.startsWith(" ") ->
                    body.add(PtxLabel(line.removeSuffix(":")))
                else -> { i--; fail("unrecognized body line `$line`") }
            }
        }
    }

    private fun parseRegDecl(line: String): PtxRegDecl {
        // `    .reg .pred %p<4>;`
        val m = Regex("""^    \.reg (\.\w+) (%[a-z]+)<(\d+)>;$""").matchEntire(line)
            ?: run { i--; fail("malformed `.reg` declaration `$line`") }
        return PtxRegDecl(m.groupValues[1], m.groupValues[2], m.groupValues[3].toInt())
    }

    private fun parseSharedDecl(line: String): PtxSharedDecl {
        // `    .shared .align 4 .b8 sdata[1024];`
        val m = Regex("""^    \.shared \.align (\d+) \.b8 (\w+)\[(\d+)];$""").matchEntire(line)
            ?: run { i--; fail("malformed `.shared` declaration `$line`") }
        return PtxSharedDecl(m.groupValues[1].toInt(), m.groupValues[2], m.groupValues[3].toInt())
    }

    private fun parseInst(content: String): PtxInst {
        var rest = content
        // Trailing comment: `;    // text` (canonical four-space separator).
        var comment: String? = null
        val semi = rest.indexOf(';')
        if (semi < 0) { i--; fail("instruction missing `;` in `$content`") }
        val after = rest.substring(semi + 1)
        when {
            after.isEmpty() -> {}
            after.startsWith("    // ") -> comment = after.removePrefix("    // ")
            else -> { i--; fail("unexpected text after `;` in `$content`") }
        }
        rest = rest.substring(0, semi)

        var guard: PtxGuard? = null
        if (rest.startsWith("@")) {
            val sp = rest.indexOf(' ')
            if (sp < 0) { i--; fail("guard without instruction in `$content`") }
            val g = rest.substring(1, sp)
            guard = if (g.startsWith("!")) PtxGuard(g.substring(1), negated = true) else PtxGuard(g)
            if (!guard.reg.startsWith("%")) { i--; fail("guard register must start with `%` in `$content`") }
            rest = rest.substring(sp + 1)
        }

        val sp = rest.indexOf(' ')
        if (sp < 0) return PtxInst(rest, emptyList(), guard, comment)
        val opcode = rest.substring(0, sp)
        val operands = rest.substring(sp + 1).split(", ").map { parseOperand(it) }
        return PtxInst(opcode, operands, guard, comment)
    }

    private fun parseOperand(s: String): PtxOperand = when {
        s.isEmpty() -> { i--; fail("empty operand") }
        s.startsWith("[") && s.endsWith("]") -> {
            val inner = s.substring(1, s.length - 1)
            val plus = inner.indexOf('+')
            if (plus < 0) PtxMem(inner)
            else {
                val off = inner.substring(plus + 1).toIntOrNull()
                    ?: run { i--; fail("non-integer memory offset in `$s`") }
                PtxMem(inner.substring(0, plus), off)
            }
        }
        s.startsWith("%") -> PtxReg(s)
        s[0].isDigit() || s[0] == '-' -> PtxImm(s)
        else -> PtxSym(s)
    }
}
